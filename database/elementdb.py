from constants import *
import cv2
import numpy as np
import shelve
import gui
import os
import ransac

def normalize(points):
    """ Normalize a collection of points in
    homogeneous coordinates so that last row = 1. """
    for row in points:
        row /= points[-1]
    return points

def make_homog(points):
    """ Convert a set of points (dim*n array) to
    homogeneous coordinates. """
    return np.vstack((points,np.ones((1,points.shape[1]))))

def homography(fp, tp):
    """A h = b
    h = (A^T A)^-1 (A^T b)"""
    ### condition
    def condition(pts):
        m = np.mean(pts[:2], axis=1)
        maxstd = np.max(np.std(pts[:2], axis=1)) + 1e-9
        C = np.diag([1/maxstd, 1/maxstd, 1])
        C[0,2] = -m[0]/maxstd
        C[1,2] = -m[1]/maxstd
        return C
    C1 = condition(fp)
    fp = (C1 @ fp)
    C2 = condition(tp)
    tp = (C2 @ tp)
    ### Calc homog (scale and translation only)
    nbr_correspondences = fp.shape[1]
    def h():
        A = np.zeros((2*nbr_correspondences,4))
        b = np.zeros((2*nbr_correspondences))
        for i in range(nbr_correspondences):
            x, y = fp[0, i], fp[1, i]
            A[2*i] =    [x, 1, 0, 0]
            A[2*i+1] =  [0, 0, y, 1]
            b[2*i] = tp[0, i]
            b[2*i+1] = tp[1, i]
        m1 = (A.T @ A)
        m2 = (A.T @ b)
        m1 = np.linalg.inv(m1)
        h = m1 @ m2
        h = np.array([
            [h[0], 0, h[1]],
            [0, h[2], h[3]],
            [0, 0, 1]
        ])
        # print(h)
        return h
    def hSVD():
        A = np.zeros((2*nbr_correspondences,5))
        for i in range(nbr_correspondences):
            x, y = fp[0, 1], fp[1, i]
            w, z = tp[0, i], tp[1, i]
            A[2*i] =    [x, 1, 0, 0, w]
            A[2*i+1] =  [0, 0, y, 1, z]
        U,S,V = np.linalg.svd(A)
        h = V[8].reshape((3,3))
        return h
    h = h()
    ### decondition
    h = np.dot(np.linalg.inv(C2),np.dot(h,C1))
    # normalize and return
    return h / h[2,2]

class RansacModel(object):
    """ Class for testing homography fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC"""
    def __init__(self,debug=False):
        self.debug = debug
    def fit(self, data):
        """ Fit homography to four selected correspondences. """
        # transpose to fit H_from_points()
        data = data.T
        # from points
        fp = data[:3,:4]
        # target points
        tp = data[3:,:4]
        # fit homography and return
        return homography(fp,tp)

    def get_error( self, data, H):
        """ Apply homography to all correspondences,
        return error for each transformed point. """
        data = data.T
        # from points
        fp = data[:3]
        # target points
        tp = data[3:]
        # transform fp
        # fp_transformed = np.linalg.inv(H).dot(fp)
        fp_transformed = (H @ fp)
        # normalize hom. coordinates
        # for i in range(3):
        #     fp_transformed[i] /= fp_transformed[2]
        # return error per point
        diff = (tp-fp_transformed)
        # diff2 = np.square(diff)
        diff2 = diff**2
        res = np.sqrt(np.sum(diff2,axis=0))
        return res

def H_from_ransac(fp,tp,model,maxiter=1000,match_threshold=10):
    """ Robust estimation of homography H from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).
    input: fp,tp (3*n arrays) points in hom. coordinates. """
    # group corresponding points
    data = np.hstack((fp,tp))
    # compute H and return
    # NOTE: got rid of data transpose
    H, ransac_data = ransac.ransac(data,model,4,maxiter,match_threshold,MINIMUM_RANSAC_FEATURES,return_all=True)
    return H, ransac_data['inliers']

AKAZE_THRESHHOLD = 3e-4
MINIMUM_RANSAC_FEATURES = 10

def featureTemplateMatch(image, template):
    kaze = cv2.AKAZE_create(cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, threshold=AKAZE_THRESHHOLD)
    kp1, des1 = kaze.detectAndCompute(image, None)
    kp2, des2 = kaze.detectAndCompute(template, None)
    gui.showFeatures(image, template, kp1, kp2)

    # FLANN_INDEX_KDTREE = 0
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50)   # or pass empty dictionary

    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfm.match(des1, des2)
    gui.showFeatureMatches(image, template, kp1, kp2, matches)

    good = matches
    # nn_match_ratio = 0.8
    # good = []
    # for m, n in matches:
    #     if m.distance < nn_match_ratio*n.distance:
    #         good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            # kp1[m.queryIdx].size
            # NOTE: query and train idx might be backwards
            [kp1[m.queryIdx].pt + (1,) for m in good]).reshape(-1, 3)
        dst_pts = np.float32(
            # kp2[m.trainIdx].size
            [kp2[m.trainIdx].pt + (1,) for m in good]).reshape(-1, 3)

        H, inliers = H_from_ransac(src_pts, dst_pts, RansacModel(debug=True))
        # print(inliers)
        # print(H)
        indices = [good[idx] for idx in inliers]
        gui.showFeatureMatches(image, template, kp1, kp2, indices)


def findFeatureMatch(image, template):
    template = cv2.imread('./templates/view_char.png', 0)          # queryImage
    image = cv2.imread('screenshot.png', 0)  # trainImage

    kaze = cv2.KAZE_create(upright=True)
    kp1, des1 = kaze.detectAndCompute(template, None)
    kp2, des2 = kaze.detectAndCompute(image, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    nn_match_ratio = 0.7
    good = []
    for m, n in matches:
        if m.distance < nn_match_ratio*n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # print(trans, scale)
        # _, rot, trans, norm = cv2.decomposeHomographyMat(M, np.identity(3))
        # M = np.float32(trans)
        matchesMask = mask.ravel().tolist()

        h, w = template.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.transform(pts, M)
        dst = cv2.perspectiveTransform(pts, M)

        image = cv2.polylines(image, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(template, kp1, image, kp2, good, None, **draw_params)

    gui.showImage(img3)


def getDetectionField(image, template):
    h, w, c = template.shape
    if c == 4:
        color, a = template[:, :, :-1], template[:, :, -1]
        alpha = a  # cv2.merge([a] * (c-1))
        method = cv2.TM_CCORR_NORMED
    else:
        color = template
        alpha = None
        method = cv2.TM_CCOEFF_NORMED
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    # gui.showImage(image)
    # gui.showImage(color)
    # gui.showImage(alpha)
    field = cv2.matchTemplate(image, color, method, None, alpha)
    # showImage(field)
    # cv2.imwrite("field.png", field)
    # showBestMatch(image, template, field)
    return field


def findTemplate(image, template):
    res = getDetectionField(image, template)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
    return maxLoc[::-1], maxVal


def findTemplateScaleInvariant(image, template):
    ws, hs = image.shape[:2]
    h, w = template.shape[:2]
    found = None
    print("Starting scale-invariant search...")
    for scale in np.linspace(0.5, 2.0, 100)[::-1]:
        resized = cv2.resize(template, (int(w * scale), int(h * scale)))
        if resized.shape[0] > ws or resized.shape[1] > hs:
            continue

        maxLoc, maxVal = findTemplate(image, resized)

        if found is None or maxVal > found[1]:
            found = (maxLoc, maxVal, scale)
    return found


def findTemplates(image, template, tolerance=0.05):
    h, w = template.shape[:2]
    res = getDetectionField(image, template)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
    points = np.where(res >= maxVal - tolerance)
    # points = points[::-1]
    # for pt in zip(*points):
    #     cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    # showImage(res)
    # showImage(image)
    return list(map(lambda p: (p[0] + h // 2, p[1] + w // 2), zip(*points)))


class Selection:
    def __init__(self, y, x, h, w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.center = (y+h//2, x+w//2)
        self.index = (slice(y,y+h), slice(x,x+w))

    def __str__(self):
        return "<Selection " + ("x: %d y: %d w: %d h: %d " % (
            self.x,
            self.y,
            self.w,
            self.h
        )) + ">"


class ElementDB:

    def __init__(self):
        if CLEAN:
            self.db = shelve.open(INDICES_CACHE_NAME, flag='n')
        self.db = shelve.open(INDICES_CACHE_NAME)

    def __del__(self):
        self.db.close()

    def get(self, cachename):
        return self.db[cachename]

    def exists(self, source_img, element_img, cachename="", threshhold=0.8):
        if cachename != "" and cachename in self.db:
            val = self.simpleImageMatch(
                source_img[self.db[cachename]], element_img)[1]
        else:
            s, val = self.complexImageMatch(source_img, element_img)[1]
            if cachename != "" and val >= threshhold:
                self.db[cachename] = s
        return val >= threshhold

    def find(self, source_img, element_img, cachename=""):
        """Returns None if element does not exist in source, 
        otherwise returns the coordinates of match

        Arguments:
            source_img {ndarray} -- Source
            element_img {ndarray} -- Template

        Keyword Arguments:
            cachename {str} -- Name to store results for faster lookup (default: {""})

        Returns:
            Selection -- Selection object
        """
        if cachename != "" and cachename in self.db:
            s = self.db[cachename]
        else:
            s = self.complexImageMatch(source_img, element_img)[0]
            if cachename != "":
                self.db[cachename] = s
        return s

    def findAll(self, source_img, element_img):
        return findTemplates(source_img, element_img)

    def simpleImageMatch(self, source_img, element_img):
        element_img = cv2.resize(element_img, source_img.shape[:2])
        h, w = element_img[:2]
        pos, val = findTemplate(source_img, element_img)
        y, x = pos
        return Selection(y, x, h, w), val

    def complexImageMatch(self, source_img, element_img):
        h, w = element_img.shape[:2]
        pos, val, scale = findTemplateScaleInvariant(source_img, element_img)
        h = int(h*scale)
        w = int(w*scale)
        y, x = pos
        return Selection(y, x, h, w), val

    def findMap(self, original_image, cachename):
        """Specialized function for finding a map element on screen 
        
        Arguments:
            original_image {ndarray} -- Image
            cachename {str} -- cache name
        
        Returns:
            Selection -- Selection object
        """
        if cachename in self.db:
            return (self.db[cachename])
        image = original_image.copy()
        bestMatch = None
        for f in os.listdir(MAP_DIRECTORY):
            filename = os.fsdecode(f)
            if filename.endswith(".png"):
                name = filename[:-4]
                template = cv2.imread(os.path.join(MAP_DIRECTORY, filename))
                hi, wi, ci = image.shape
                ht, wt, ct = template.shape
                if wi != wt:
                    template = cv2.resize(template,
                                          (wi, int(ht * wi / wt)),
                                          interpolation=cv2.INTER_LINEAR)  # scale to match width

                pos, value = findTemplate(image, template)

                if bestMatch is None or value > bestMatch[-1]:
                    bestMatch = (name, value)
        self.db[cachename] = Selection(
            pos[0], pos[1], template.shape[0], template.shape[1])
        return self.db[cachename]


if __name__ == "__main__":
    from PIL import ImageFont
    font = ImageFont.truetype("./templates/Fire_Emblem_Heroes_Font.ttf")

    template = cv2.imread('./templates/small_fight_btn.png', 0)
    image = cv2.imread('screenshot.png', 0)
    featureTemplateMatch(image, template)
    exit(0)
    finder = ElementDB()
    img1 = cv2.imread('./templates/view_char.png', -1)
    img2 = cv2.imread('screenshot.png', -1)
    var = findTemplates(img2, img1)
    print(var)
    p = finder.find(img2, img1, "test3")
    print(p)
    p = finder.find(img2, img1, "test3")
    print(p)
    s = finder.get("test3")
    gui.showImage(img2[s.index])

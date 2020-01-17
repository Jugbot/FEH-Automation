from constants import *
import cv2
import numpy as np
import shelve
import gui




def findFeatureMatch(img2, img1):
    img1 = cv2.imread('./templates/view_char.png', 0)          # queryImage
    img2 = cv2.imread('screenshot.png', 0)  # trainImage

    # Initiate SIFT detector
    # sift = cv2.SIFT()
    sift = cv2.KAZE_create(upright=True)
    # cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

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

        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1],
                          [w-1, 0]]).reshape(-1, 1, 2)
        # dst = cv2.transform(pts, M)
        dst = cv2.perspectiveTransform(pts, M)

        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" %
              (len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    gui.showImage(img3)


def getDetectionField(image, template):
    h, w, c = template.shape
    if c == 4:
        color, a = template[:,:,:-1], template[:,:,-1]
        alpha = a # cv2.merge([a] * (c-1))
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

    def __index__(self):
        return (slice(y,y+h), slice(x,x+w))
    
    def center(self):
        return (self.y + self.h // 2, self.x + self.w // 2)

    


class ElementDB:
    THRESHHOLD = 0.8

    def __init__(self):
        if CLEAN:
            self.db = shelve.open(INDICES_CACHE_NAME, flag='n')
        self.db = shelve.open(INDICES_CACHE_NAME)

    def __del__(self):
        self.db.close()

    def find(self, source_img, element_img, cachename=""):
        """Returns None if element does not exist in source, 
        otherwise returns the coordinates of match

        Arguments:
            source_img {ndarray} -- Source
            element_img {ndarray} -- Template

        Keyword Arguments:
            cachename {str} -- Name to store results for faster lookup (default: {""})

        Returns:
            tuple -- Position of center of element
        """
        if cachename != "" and cachename in self.db:
            s = self.db[cachename]
            if self.simpleImageMatch(source_img[s], element_img) is not None:
                return s
        else:
            s = self.complexImageMatch(source_img, element_img)
            if cachename != "" and s is not None:
                self.db[cachename] = s
                return s
        return None

    def findAll(self, source_img, element_img):
        return findTemplates(source_img, element_img)

    def findQuick(self, source_img, element_img, cachename=""):
        """Like find() except will return the cached value 
        even if it doesn't exist in the image
        """
        if cachename != "" and cachename in self.db:
            return self.center(self.db[cachename])
        else:
            return self.find(source_img, element_img, cachename)

    def simpleImageMatch(self, source_img, element_img):
        element_img = cv2.resize(element_img, source_img.shape[:2])
        h, w = element_img[:2]
        pos, val = findTemplate(source_img, element_img)
        y, x = pos
        if val < self.THRESHHOLD:
            return None
        return (slice(y,y+h), slice(x,x+w))

    def complexImageMatch(self, source_img, element_img):
        h, w = element_img.shape[:2]
        pos, val, scale = findTemplateScaleInvariant(source_img, element_img)
        h = int(h*scale)
        w = int(w*scale)
        y, x = pos
        if val < self.THRESHHOLD:
            return None
        return (slice(y,y+h), slice(x,x+w))

    def getSlice(self, cachename):
        if cachename in self.db:
            return self.db[cachename]
        return None

    def getCenter(self, cachename):
        if cachename in self.db:
            return self.center(self.db[cachename])
        return None

    def findMap(self, original_image, cachename):
        """Takes a screenshot and returns the matching image name index range

        Arguments:
            image {image} -- full screenshot

        Returns:
            tuple -- (name, index)
        """
        if cachename in self.db:
            return self.center(self.db[cachename])
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
        self.db[cachename] = (slice(pos[0], pos[0] + template.shape[0]), slice(pos[1], pos[1] + template.shape[1]))
        return self.center(self.db[cachename])


if __name__ == "__main__":
    finder = ElementDB()
    img1 = cv2.imread('./templates/view_char.png', -1)
    img2 = cv2.imread('screenshot.png', -1)
    var = findTemplates(img2, img1)
    print(var)
    print("begin")
    p = finder.find(img2,img1, "test2")
    print(p)
    p = finder.find(img2, img1, "test2")
    print(p)
    gui.showImage(img2[finder.getSlice("test2")])

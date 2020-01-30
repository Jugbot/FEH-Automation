from constants import *
import cv2
import numpy as np
import shelve
import gui
import os
import ransac
from PIL import Image, ImageFont, ImageDraw


def normalize(points):
    """ Normalize a collection of points in
    homogeneous coordinates so that last row = 1. """
    for row in points:
        row /= points[-1]
    return points5


def make_homog(points):
    """ Convert a set of points (dim*n array) to
    homogeneous coordinates. """
    return np.vstack((points, np.ones((1, points.shape[1]))))


def homography(fp, tp):
    # condition
    def condition(pts):
        m = np.mean(pts[:2], axis=1)
        maxstd = np.max(np.std(pts[:2], axis=1)) + 1e-9
        C = np.diag([1/maxstd, 1/maxstd, 1])
        C[0, 2] = -m[0]/maxstd
        C[1, 2] = -m[1]/maxstd
        return C
    C1 = condition(fp)
    fp = (C1 @ fp)
    C2 = condition(tp)
    tp = (C2 @ tp)
    # Calc homog (scale and translation only)
    nbr_correspondences = fp.shape[1]

    def h():
        """A h = b
        h = (A^T A)^-1 (A^T b)"""
        A = np.zeros((2*nbr_correspondences, 4))
        b = np.zeros((2*nbr_correspondences))
        for i in range(nbr_correspondences):
            x, y = fp[0, i], fp[1, i]
            A[2*i] = [x, 1, 0, 0]
            A[2*i+1] = [0, 0, y, 1]
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

    def h2():
        """A h = b (keep ratio scale)
        h = (A^T A)^-1 (A^T b)"""
        A = np.zeros((2*nbr_correspondences, 3))
        b = np.zeros((2*nbr_correspondences))
        for i in range(nbr_correspondences):
            x, y = fp[0, i], fp[1, i]
            A[2*i] = [x, 1, 0]
            A[2*i+1] = [y, 0, 1]
            b[2*i] = tp[0, i]
            b[2*i+1] = tp[1, i]
        m1 = (A.T @ A)
        m2 = (A.T @ b)
        m1 = np.linalg.inv(m1)
        h = m1 @ m2
        h = np.array([
            [h[0], 0, h[1]],
            [0, h[0], h[2]],
            [0, 0, 1]
        ])
        # print(h)
        return h

    def hSVD():
        A = np.zeros((2*nbr_correspondences, 5))
        for i in range(nbr_correspondences):
            x, y = fp[0, 1], fp[1, i]
            w, z = tp[0, i], tp[1, i]
            A[2*i] = [x, 1, 0, 0, w]
            A[2*i+1] = [0, 0, y, 1, z]
        U, S, V = np.linalg.svd(A)
        h = V[8].reshape((3, 3))
        return h
    h = h2()
    # decondition
    h = np.dot(np.linalg.inv(C2), np.dot(h, C1))
    # normalize and return
    return h / h[2, 2]


class RansacModel(object):
    """ Class for testing homography fit with ransac.py from
    http://www.scipy.org/Cookbook/RANSAC"""

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """ Fit homography to four selected correspondences. """
        # transpose to fit H_from_points()
        data = data.T
        # from points
        fp = data[:3, :4]
        # target points
        tp = data[3:, :4]
        # fit homography and return
        return homography(fp, tp)

    def get_error(self, data, H):
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
        res = np.sqrt(np.sum(diff2, axis=0))
        return res


def H_from_ransac(fp, tp, model, maxiter=1000, match_threshold=10):
    """ Robust estimation of homography H from point
    correspondences using RANSAC (ransac.py from
    http://www.scipy.org/Cookbook/RANSAC).
    input: fp,tp (3*n arrays) points in hom. coordinates. """
    # group corresponding points
    data = np.hstack((fp, tp))
    # compute H and return
    # NOTE: got rid of data transpose
    H, ransac_data = ransac.ransac(
        data, model, 4, maxiter, match_threshold, MINIMUM_RANSAC_FEATURES, return_all=True)
    return H, ransac_data['inliers']


AKAZE_THRESHHOLD = 3e-4
MINIMUM_RANSAC_FEATURES = 10


def KAZE_kpm(image, template):
    kaze = cv2.KAZE_create(upright=True)
    kp1, des1 = kaze.detectAndCompute(image, None)
    kp2, des2 = kaze.detectAndCompute(template, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    nn_match_ratio = 0.8
    good = []
    for m, n in matches:
        if m.distance < nn_match_ratio*n.distance:
            good.append(m)

    gui.showFeatureMatches(image, template, kp1, kp2, good)
    return kp1, kp2, good


def BF_match(des1, des2):
    global bfm
    if bfm is None:
        bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    return bfm.match(des1, des2)


def ORB_features(image):
    global orb
    if orb is None:
        orb = cv2.ORB_create(nfeatures=10000, firstLevel=3)
    return orb.detectAndCompute(image, None)


def ORB_kpm(image, template):
    orb = cv2.ORB_create(nfeatures=10000, firstLevel=3)
    kp1, des1 = orb.detectAndCompute(image, None)
    kp2, des2 = orb.detectAndCompute(template, None)
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfm.match(des1, des2)
    # gui.showFeatureMatches(image, template, kp1, kp2, matches)
    return kp1, kp2, matches


def AKAZE_kpm(image, template):
    akaze = cv2.AKAZE_create(
        cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT, threshold=AKAZE_THRESHHOLD)
    kp1, des1 = akaze.detectAndCompute(image, None)
    kp2, des2 = akaze.detectAndCompute(template, None)
    bfm = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfm.match(des1, des2)
    # gui.showFeatureMatches(image, template, kp1, kp2, matches)
    return kp1, kp2, matches


def combinedKeypointMatches(*args):
    kp1 = [k for o in args for k in o[0]]
    kp2 = [k for o in args for k in o[1]]
    matches = []
    lastSize1 = 0
    lastSize2 = 0
    for ka, kb, mlst in args:
        for m in mlst:
            m.queryIdx += lastSize1
            m.trainIdx += lastSize2
            matches.append(m)
        lastSize1 = len(ka)
        lastSize2 = len(kb)
    return kp1, kp2, matches


def featureTemplateMatch(image, template):
    kp1, kp2, matches = combinedKeypointMatches(
        ORB_kpm(image, template), AKAZE_kpm(image, template))
    # gui.showFeatures(image, template, kp1, kp2)

    src_pts = np.float32([kp1[m.queryIdx].pt + (1,)
                          for m in matches]).reshape(-1, 3)
    dst_pts = np.float32([kp2[m.trainIdx].pt + (1,)
                          for m in matches]).reshape(-1, 3)

    H, inliers = H_from_ransac(dst_pts, src_pts, RansacModel(debug=True))
    indices = [matches[idx] for idx in inliers]
    # gui.showFeatureMatches(image, template, kp1, kp2, indices, H)
    # cv2.waitKey()
    # tuple((H @ np.array((0, 0, 1))).astype(int))
    maxLoc = (H[0, -1], H[1, -1])
    maxVal = len(inliers) / len(matches)
    scale = H[0, 0]
    return (maxLoc, maxVal, scale)


if __name__ == "__main__":
    template = cv2.imread('./templates/view_char.png', -1)
    image = cv2.imread('screenshot.png', -1)
    from timeit import timeit
    print("exec time: ", timeit(lambda: print(
        featureTemplateMatch(image, template)), number=1))

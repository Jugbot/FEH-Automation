import os
import shelve

import cv2
import numpy as np
import skimage.feature

import gui
import ransac
from constants import *


def getDetectionField(image, template, grayscale=True):
    h, w, c = template.shape
    if c == 4:
        color, a = template[:, :, :-1], template[:, :, -1]
        if grayscale:
            alpha = a
        else:
            alpha = cv2.merge([a] * (c-1))
        method = cv2.TM_CCORR_NORMED
    else:
        color = template
        alpha = None
        method = cv2.TM_CCOEFF_NORMED
    if grayscale:
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
    for scale in np.logspace(-1, 1, num=16 + 1, base=2)[::-1]:
        # num must be odd since range is inclusive (and you want midpoint = 1.0)
        resized = cv2.resize(template, (int(w * scale), int(h * scale)))
        if resized.shape[0] > ws or resized.shape[1] > hs:
            continue

        maxLoc, maxVal = findTemplate(image, resized)

        if found is None or maxVal > found[1]:
            found = (maxLoc, maxVal, scale)
    return found


def findTemplates(image, template, tolerance=0.05):
    # gui.showImage(image)
    # gui.showImage(template)
    # cv2.waitKey()
    res = getDetectionField(image, template, grayscale=False)
    res *= 255
    cv2.imwrite("field.png", res)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
    points = skimage.feature.peak_local_max(
        res, threshold_rel=1.0-tolerance, indices=True)
    return points


if __name__ == "__main__":
    template = cv2.imread('./templates/view_char.png', -1)
    image = cv2.imread('screenshot.png', -1)
    from timeit import timeit
    print("exec time: ", timeit(lambda: print(
        findTemplateScaleInvariant(image, template)), number=1))

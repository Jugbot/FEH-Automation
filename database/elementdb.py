import os
import shelve

import cv2
import numpy as np
import skimage.feature

import gui
import ransac
from constants import *
from vision.template import *
from vision.feature import featureTemplateMatch


class Selection:
    def __init__(self, y, x, h, w):
        self.x = int(x)
        self.y = int(y)
        self.h = int(h)
        self.w = int(w)
        self.center = (int(y+h//2), int(x+w//2))
        self.index = (slice(y, y+h), slice(x, x+w))

    def __str__(self):
        return "<Selection " + ("x: %d y: %d w: %d h: %d " % (
            self.x,
            self.y,
            self.w,
            self.h
        )) + ">"

    def splitX(self, t):
        if type(t) is int:
            t = (t,)
        s = 0
        res = []
        for val in t:
            res.append(Selection(self.y, self.x + s, self.h, val - s))
            s = val
        res.append(Selection(self.y, self.x + val, self.h, self.w - val))
        return res

    def splitY(self, t):
        pass


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

    def findFrameX(self, source_img, element_img, cachename=""):
        """Finds the rectangle containing the beginning and end
        of an image (the source_img is split in two horizontally)

        Arguments:
            source_img {ndarray} -- Image
            element_img {ndarray} -- Image

        Keyword Arguments:
            cachename {str} -- Cachename (default: {""})

        Returns:
            list -- list of Selections
        """
        if cachename != "" and cachename in self.db:
            s = self.db[cachename]
        else:
            h, w = element_img.shape[:2]
            left = element_img[:, :w//2]
            right = element_img[:, w//2:]
            a1 = findTemplates(source_img, left)[::-1]
            a2 = findTemplates(source_img, right)[::-1]
            # results ordered by height first
            k1 = 0
            k2 = 0
            s = []
            while k1 < len(a1) and k2 < len(a2):
                y1, y2 = a1[k1, 0], a2[k2, 0]
                x1, x2 = a1[k1, 1], a2[k2, 1]
                if y2 < y1:
                    if k2 < len(a2) - 1:
                        k2 += 1
                    else:
                        break
                elif y1 < y2:
                    if k1 < len(a1) - 1:
                        k1 += 1
                    else:
                        break
                else:  # y1 == y2:
                    s.append(Selection(y1, x1, h, w//2 + (x2 - x1)))
                    k1 += 1
                    k2 += 1
            if cachename != "":
                self.db[cachename] = s
        return s

    def findAll(self, source_img, element_img, cachename=""):
        if cachename != "" and cachename in self.db:
            s = self.db[cachename]
        else:
            a = findTemplates(source_img, element_img)
            h, w = element_img.shape[:2]
            s = [Selection(t[0], t[1], h, w) for t in a]
            if cachename != "":
                self.db[cachename] = s
        return s

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
                    bestMatch = (name, pos, value)
        pos = bestMatch[1]
        self.db[cachename] = Selection(
            pos[0], pos[1], template.shape[0], template.shape[1])
        return self.db[cachename]


if __name__ == "__main__":
    template = cv2.imread('./templates/frame_HM.png', -1)
    image = cv2.imread('screenshot_char.png', -1)
    finder = ElementDB()
    selection = finder.findFrameX(image, template)
    gui.showSelections(image, selection)
    cv2.waitKey()
    print(selection)
    exit(0)
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

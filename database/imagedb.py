import os
import shelve

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gui
import ransac
from constants import *
from vision.feature import featureTemplateMatch


class ImageDB:

    def __init__(self):
        if CLEAN:
            self.db = shelve.open(INDICES_CACHE_NAME, flag='n')
        self.db = shelve.open(INDICES_CACHE_NAME)
        if CLEAN or len(self.db):
            pass

    def __del__(self):
        self.db.close()

    def loadFolder(self, path):
        print("loading image library %s " % path, end='')
        for f in os.listdir(path):
            fname = os.fsdecode(f)
            if fname.endswith('.png'):
                name = fname[:-4]
                if not self.has(name):
                    self.add(name, cv2.imread(os.path.join(
                        MAP_DIRECTORY, fname), cv2.IMREAD_GRAYSCALE))
        print('done')

    def loadWords(self, lst):
        # font = ImageFont.truetype(FONT_FILE)
        ft = cv2.freetype.createFreeType2()
        for word in lst:
            d = ImageDraw.Draw(img)
            d.text((20, 20), 'Hello', fill=(255, 0, 0))

    def get(self, cachename):
        return self.db[cachename]

    def match(self, image):
        """Gives the name of the best fit image in DB

        Arguments:
            image {ndarray} -- Image

        Returns:
            str -- Key for matching image in DB
        """
        _, desc = ORB_features(image)
        best = None
        for key, data in self.db.items():
            matches = BF_match(desc, data)
            if best is None or best[-1] > len(matches):
                best = (key, len(matches))
        return best[0]

    def similarity(self, image1, image2):
        kp1, kp2, matches = ORB_kpm(image1, image2)
        return len(matches)

    def add(self, name, image):
        kp, desc = ORB_features(image)
        self.db[name] = desc

    def has(self, name):
        return name in self.db


if __name__ == "__main__":
    img = strToImage("1234567890")
    cv2.imshow("", img)
    cv2.waitKey()
    IDB = ImageDB()

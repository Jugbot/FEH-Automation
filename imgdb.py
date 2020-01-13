import pickle
import numpy as np
from constants import *
import cv2
import gui


class ImageDB:
    def __init__(self, save_path, descriptors=[]):
        super().__init__()
        self.descriptors = descriptors
        try:
            with open(save_path, 'rb') as f:
                self.db = pickle.load(f)
        except IOError:
            self.db = []

    def add(self, image_path):
        for d in self.descriptors:
            pass

    def close():
        with open(save_path, 'wb') as f:
            pass


def chi2_distance(self, histA, histB, eps=1e-10):
    # compute the chi-squared distance
    d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                      for (a, b) in zip(histA, histB)])

    # return the chi-squared distance
    return d


def mse(imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(np.prod(imageA.shape))

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def sampleGrid(image, size):
    r, c = size
    h, w = image.shape[:2]
    out = np.ndarray(size, np.uint8)
    dx = w/c
    dy = h/r
    for x in range(c):
        for y in range(r):
            out[y, x] = np.mean(
                image[int(dy * y): int(dy * y + dy), int(dx * x): int(dx * x + dx)])
    return out


class MapImageDB:
    def __init__(self, save_path):
        super().__init__()
        try:
            with open(save_path, 'rb') as f:
                self.db = pickle.load(f)
        except IOError:
            self.db = {}

    @staticmethod
    def sample(image):
        h, w = image.shape[:2]
        ratio = w/h
        if (MAP_RATIO_N - ratio) + (MAP_RATIO_L - ratio) > 0:
            # NORMAL SIZE-ISH
            sample = sampleGrid(image, MAP_DIMENSIONS_N)
        else:
            sample = sampleGrid(image, MAP_DIMENSIONS_L)
        return sample

    def add(self, image, name):
        sample = MapImageDB.sample(image)
        self.db[name] = sample

    def search(self, image):
        sample = MapImageDB.sample(image)
        gui.showImage(sample)

        best = None
        for name, entry in self.db.items():
            if entry.shape == sample.shape:
                val = mse(entry, sample)
                if best is None or val < best[-1]:
                    best = (name, val)

        return best[0]

    def has(self, name):
        return name in self.db

import os
mapdb = MapImageDB(MAP_DIRECTORY + "/db.ppy")
# for root, dirs, files in os.walk(MAP_DIRECTORY):
for f in os.listdir(MAP_DIRECTORY):
    fname = os.fsdecode(f)
    # for fname in files:
    if fname.endswith('.png'):
        name = fname[:-4]
        if not mapdb.has(name):
            print(name)
            mapdb.add(cv2.imread(os.path.join(MAP_DIRECTORY, fname), cv2.IMREAD_GRAYSCALE), name)

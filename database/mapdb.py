import pickle
import numpy as np
from constants import *
import shelve
import cv2
import gui
import os
from util import mse
from database.elementdb import findTemplate

def createTerrainData(original_image):
    print("""
    ### tile types ###
    0 plain
    1 wall
    2 forest
    3 mountain/gap
    4 defensive plain
    5 defensive forest
    6 wall health 1
    7 wall health 2
    8 wall health 3
    """)
    r, c = 8, 6
    data = np.zeros((r, c), dtype=np.uint8)
    cv2.namedWindow("tagging", cv2.WINDOW_NORMAL)

    def cycleValue(event, x, y, flags, param):
        if (event == cv2.EVENT_MOUSEWHEEL or event == cv2.EVENT_LBUTTONUP):
            image = original_image.copy()
            h, w = image.shape[:2]
            j, i = int(y / h * r), int(x / w * c)
            data[j, i] += 1
            data[j, i] %= 10
            gui.showImageMapData(image, data)
            cv2.imshow('tagging', image)
    cv2.setMouseCallback("tagging", cycleValue)
    cv2.imshow('tagging', original_image)
    cv2.waitKey()
    return data


def getMapData(name, board_img=None):
    try:
        if CLEAN:
            raise IOError
        terrain_data = np.loadtxt(os.path.join(MAP_DIRECTORY, name + '.csv'), delimiter=",", dtype=np.uint8)
    except IOError as e:
        if board_img is None:
            raise e
        terrain_data = createTerrainData(board_img)
        np.savetxt(os.path.join(MAP_DIRECTORY, name + '.csv'), terrain_data, delimiter=",", fmt="%d")
    return terrain_data




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


class MapDB:
    def __init__(self):
        super().__init__()
        if CLEAN:
            self.db = shelve.open(MAP_CACHE_NAME, flag='n')
        self.db = shelve.open(MAP_CACHE_NAME)
        ### preload ###
        print("loading map assets... ", end='')
        for f in os.listdir(MAP_DIRECTORY):
            fname = os.fsdecode(f)
            if fname.endswith('.png'):
                name = fname[:-4]
                if not self.has(name):
                    self.add(name, cv2.imread(os.path.join(
                        MAP_DIRECTORY, fname), cv2.IMREAD_GRAYSCALE))
                elif not os.path.exists(os.path.join(MAP_DIRECTORY, name + '.csv')):
                    getMapData(name, cv2.imread(os.path.join(
                        MAP_DIRECTORY, fname), cv2.IMREAD_GRAYSCALE))
        print('done')

    def __del__(self):
        self.db.close()

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

    def add(self, name, image):
        sample = MapDB.sample(image)
        self.db[name] = sample

    def search(self, image):
        """Gives the name of the map from the image
        
        Arguments:
            image {ndarray} -- Image
        
        Returns:
            str -- name
        """
        sample = MapDB.sample(image)

        best = None
        for name, entry in self.db.items():
            if entry.shape == sample.shape:
                val = mse(entry, sample)
                if best is None or val < best[-1]:
                    best = (name, val)

        return best[0]

    def data(self, name, image=None):
        getMapData(name, image)
        return getMapData(name)

    def has(self, name):
        return name in self.db

if __name__ == "__main__":
    mdb = MapDB()

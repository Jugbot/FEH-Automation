# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pickle
from constants import *
import os
import gui
from imgdb import mapdb
# from imageloader import templates





# def selectRegion(image):
#     cv2.namedWindow('select', cv2.WINDOW_NORMAL)
#     selection = cv2.selectROI('select', image, True)
#     cv2.destroyWindow('select')
#     return selection


# def getSubImage(image, template, selection):
#     x1, y1 = selection
#     x2, y2 = x1 + template.shape[1], y1 + template.shape[0]
#     board_img = image[y1:y2, x1:x2]
#     return board_img

### map functions


def findEntities(image, map_data):
    tile_type = 8
    for path in ['./templates/friendly_healthbar.bmp', './templates/enemy_healthbar.bmp']:
        healthbar_template = cv2.imread(path)
        positions = findTemplates(image, healthbar_template)
        h, w, c = image.shape
        r, c = map_data.shape
        for p in zip(*positions):
            x = int(p[0]*c/w)
            y = int(p[1]*r/h)
            map_data[y, x] = tile_type
        tile_type = 9

    gui.showImageMapData(image, map_data)

    return map_data

def loadEntity():
    pass

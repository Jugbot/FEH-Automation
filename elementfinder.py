# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
# from imageloader import templates

def debug(image):
    # h, w = image.shape[:2]
    # image = cv2.resize(image, (int(0.4*w), int(0.4*h)))
    cv2.namedWindow('D E B U G', cv2.WINDOW_NORMAL)
    cv2.imshow('D E B U G', image)
    cv2.waitKey()

def getDetectionField(image, template):
    h, w, c = template.shape
    if c - 1 == image.shape[2]:
        color, a = template[:-1], template[-1]
        # a = np.array(a, dtype=np.float32)
        alpha = cv2.merge([a] * (c-1))
        method = cv2.TM_CCORR_NORMED
    else:
        color = template
        alpha = None
        method = cv2.TM_CCOEFF_NORMED
    # image = cv2.Canny(image, 100, 200)
    # color = cv2.Canny(color, 100, 200)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    return cv2.matchTemplate(image, color, method, None, alpha)

def findTemplate(image, template):
    h, w, c = template.shape
    res = getDetectionField(image, template)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
    return (maxLoc[0], maxLoc[1], maxLoc[0] + w, maxLoc[1] + h)

def findTemplates(image, template, tolerance=0.05):
    h, w, c = template.shape
    res = getDetectionField(image, template)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
    loc = np.where( res >= maxVal - tolerance)
    points = loc[::-1]
    for pt in zip(*points):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    debug(res)
    debug(image)
    return points



# for name, template in templates.items():
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# color = np.ndarray((w, h, 3), np.uint8)
# alpha = np.ndarray((w, h, 1), np.uint8)
# cv2.mixChannels([template], [color, alpha], [0,0,1,1,2,2,3,3])
# color = np.repeat(color, 3, axis=2)
# TM_CCOEFF_NORMED -> TM_CCORR_NORMED 
###
def getTemplateScale(image, template):
    ws, hs, cs = image.shape
    h, w, c = template.shape
    found = None
    print("Starting scale-invariant search...")
    for scale in np.linspace(0.5, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = cv2.resize(template, (int(w * scale), int(h * scale)) )
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] > ws or resized.shape[1] > hs:
            break

        result = getDetectionField(image, resized)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, scale)
    return found


def selectRegion(image):
    cv2.namedWindow('select', cv2.WINDOW_NORMAL)
    selection = cv2.selectROI('select', image, True)
    cv2.destroyWindow('select')
    return selection

def debugBoard(image, map):
    h, w, c = image.shape
    for ix,iy in np.ndindex(map.shape):
        tile = map[ix, iy]
        cv2.putText(image, str(tile), (int(ix * w / 6), int((iy+1) * h / 8)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 10)
        cv2.putText(image, str(tile), (int(ix * w / 6), int((iy+1) * h / 8)), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    debug(image)

img_rgb = cv2.imread('screenshot.png')
template = cv2.imread('./templates/snow_map.png')
selection = findTemplate(img_rgb, template)
x1, y1, x2, y2 = selection
print(selection)
board_img = img_rgb[y1:y2, x1:x2]
debug(board_img)
### tile types ###
# 0 plain
# 1 wall
# 2 forest
# 3 mountain/gap
# 4 defensive plain
# 5 defensive forest
# 8 friend
# 9 enemy
# note friend & enemy are both plain types
snow_map = np.array([
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [2, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 2, 0, 0],
    [0, 2, 1, 0, 0, 0, 1, 0]
], np.uint8)

# template = templates['template.png']
# h, w, c = template.shape
# (maxVal, maxLoc, scale) = getTemplateScale(img_rgb, template)
# resized = cv2.resize(template, (int(w * scale), int(h * scale)) )
# tolerance = 0.1
# findTemplates(img_rgb, resized, maxVal - tolerance)

tile_type = 8
for path in ['./templates/friendly_healthbar.bmp', './templates/enemy_healthbar.bmp']:
    healthbar_template = cv2.imread(path)
    positions = findTemplates(board_img, healthbar_template)
    h, w, c = board_img.shape
    for p in zip(*positions):
        x = int(p[0]*6/w)
        y = int(p[1]*8/h)
        snow_map[x, y] = tile_type
    tile_type = 9


debugBoard(board_img, snow_map)



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






def getDetectionField(image, template):
    h, w, c = template.shape
    if c - 1 == image.shape[2]:
        color, a = template[:-1], template[-1]
        alpha = cv2.merge([a] * (c-1))
        method = cv2.TM_CCORR_NORMED
    else:
        color = template
        alpha = None
        method = cv2.TM_CCOEFF_NORMED
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
    field = cv2.matchTemplate(image, color, method, None, alpha)
    # showImage(field)
    # cv2.imwrite("field.png", field)
    # showBestMatch(image, template, field)
    return field


def findTemplate(image, template):
    h, w, c = template.shape
    res = getDetectionField(image, template)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
    return maxLoc, maxVal


def getSimilarity(image, template):pass


def findTemplates(image, template, tolerance=0.05):
    h, w, c = template.shape
    res = getDetectionField(image, template)
    (_, maxVal, _, maxLoc) = cv2.minMaxLoc(res)
    loc = np.where(res >= maxVal - tolerance)
    points = loc[::-1]
    for pt in zip(*points):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    # showImage(res)
    # showImage(image)
    return points


def getTemplateScale(image, template):
    ws, hs, cs = image.shape
    h, w, c = template.shape
    found = None
    print("Starting scale-invariant search...")
    for scale in np.linspace(0.5, 1.0, 20)[::-1]:
        resized = cv2.resize(template, (int(w * scale), int(h * scale)))
        if resized.shape[0] > ws or resized.shape[1] > hs:
            break

        result = getDetectionField(image, resized)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, scale)
    return found


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

def createTerrainData(original_image):
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
    # snow_map = np.array([
    #     [0, 0, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 1, 0],
    #     [2, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 2, 0, 0],
    #     [0, 2, 1, 0, 0, 0, 1, 0]
    # ], np.uint8)
    r, c = 8, 6
    data = np.zeros((r, c), dtype=np.uint8)
    cv2.namedWindow("tagging", cv2.WINDOW_NORMAL)

    def cycleValue(event, x, y, flags, param):
        if (event == cv2.EVENT_MOUSEWHEEL or event == cv2.EVENT_LBUTTONUP):
            image = original_image.copy()
            h, w, _ = image.shape
            j, i = int(y / h * r), int(x / w * c)
            data[j, i] += 1
            data[j, i] %= 10
            showImageMapData(image, data)
            cv2.imshow('tagging', image)
    cv2.setMouseCallback("tagging", cycleValue)
    cv2.imshow('tagging', original_image)
    cv2.waitKey()
    return data


def findMapMatch(original_image):
    """Takes a screenshot of the board and returns the matching image name index range
    
    Arguments:
        ndarray {board_img} -- battlefield capture
    
    Returns:
        str -- name
    """
    gui.showImage(original_image)
    return mapdb.search(original_image)
    

def findMapMatchBlind(original_image):
    """Takes a screenshot and returns the matching image name index range
    
    Arguments:
        image {image} -- full screenshot
    
    Returns:
        tuple -- (name, index)
    """
    image = original_image.copy()
    bestMatch = None
    for f in os.listdir(MAP_DIRECTORY):
        filename = os.fsdecode(f)
        if filename.endswith(".png"):
            # filename = "S2054.png"  # TODO
            name = filename[:-4]
            print(name)
            template = cv2.imread(os.path.join(MAP_DIRECTORY, filename))
            hi, wi, ci = image.shape
            ht, wt, ct = template.shape
            if wi != wt:
                template = cv2.resize(template, 
                (wi, int(ht * wi / wt)), 
                interpolation=cv2.INTER_LINEAR)  # scale to match width

            pos, value = findTemplate(image, template)

            if bestMatch is None or value > bestMatch[-1]:
                bestMatch = (name, (slice(pos[0], pos[0] + template.shape[0]), slice(pos[1], pos[1] + template.shape[1])), value)
            # break  # TODO
    return bestMatch[:-1]


def getMapData(board_img, name):
    try:
        terrain_data = np.load(os.path.join(MAP_DIRECTORY, name + '.npy'))
    except IOError:
        terrain_data = createTerrainData(board_img)
        np.save(os.path.join(MAP_DIRECTORY, name + '.npy'), terrain_data)
    return terrain_data


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

def loadIndices():
    try:
        with open(INDICES_CACHE_NAME + '.ppy', 'rb') as f:
            indices = pickle.load(f)
    except IOError:
        indices = {}
    return indices

def saveIndices(data):
    with open(INDICES_CACHE_NAME + '.ppy', 'xb') as f:
        indices = pickle.dump(data, f)

### character data

def loadEntity():
    pass

if __name__ == "__main__":
    screen_capture = cv2.imread('screenshot.png')
    indices = loadIndices()
    if "board_normal" in indices:
        selection = indices["board_normal"]
        name = findMapMatch(screen_capture[selection])
    else:
        name, selection = findMapMatchBlind(screen_capture)
        indices["board_normal"] = selection
    saveIndices(indices)

    print("found", name)
    board_image = screen_capture[selection]
    data = getMapData(board_image, name)
    data = findEntities(board_image, data) # TODO: keep entity info in different place
    # for each entity, 
    # click entity on board then full screen view
    loadEntity() # read full-screen info

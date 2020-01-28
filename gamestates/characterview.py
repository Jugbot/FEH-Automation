from gamestates._base import GameStateBase
from constants import *
import cv2
import time
import numpy as np
import gui
import pytesseract
import util
from database import elemDB, Selection


def readTXT(image, select=None, numeric=False):
    image = ~np.logical_or.reduce([np.all(image == t, axis=-1) for t in select])
    gui.showImage((image * 255).astype('uint8'))
    config = '--psm 7'
    if numeric: 
        config += ' -c tessedit_char_whitelist=0123456789-.+'
    return pytesseract.image_to_string(image, config=config)

def readTXTExperimental(image):
    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("./frozen_east_text_detection.pb")
    
    orig = image.copy()
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    h, w = image.shape[:2]
    nh = h - (h % 32)
    nw = w - (w % 32)
    rW = w / float(nw)
    rH = h / float(nh)

    blob = cv2.dnn.blobFromImage(image, 1.0, (nw, nh),
        (123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    
    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    min_confidence = 0.8

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = util.non_max_suppression(np.array(rects), probs=confidences)

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # show the output image
    # gui.showImage(orig)

class CharacterView(GameStateBase):
    def __init__(self, image):
        super().__init__(image)
        data = {
            #
        }
        frame = cv2.imread("%s/frame_skill.png" % TEMPLATES_DIR, -1)
        frames = elemDB.findFrameX(image, frame, "frames")
        # gui.showSelections(image, frames)
        assert(len(frames) == 11)
        tl, tr, hp, atk, spd, df, skillA, res, skillB, sp, skillC = frames
        statBisection = 200 # half
        skillBisection = 106
        lvl = Selection(tl.y, tl.x, tl.h, tr.x - tl.x)
        lvlBisection = (92, 170, 320)
        dflowers = Selection(tr.y, tr.x, tr.h, (tl.x + tl.w) - tr.x)
        dflowersBisection = 100
        exp = Selection(tr.y, (tl.x + tl.w), tr.h, (tr.x + tr.w) - (tl.x + tl.w))
        frame = cv2.imread("%s/frame_HM.png" % TEMPLATES_DIR, -1)
        hm = elemDB.findFrameX(image, frame, "frame_HM")[0]
        frame = cv2.imread("%s/frame_seal.png" % TEMPLATES_DIR, -1)
        seal = elemDB.findFrameX(image, frame, "frame_seal")[0]
        frame = cv2.imread("%s/frame_weap.png" % TEMPLATES_DIR, -1)
        weapon = elemDB.findFrameX(image, frame, "frame_weapon")[0]
        frame = cv2.imread("%s/frame_assist.png" % TEMPLATES_DIR, -1)
        assist = elemDB.findFrameX(image, frame, "frame_assist")[0]
        frame = cv2.imread("%s/frame_special.png" % TEMPLATES_DIR, -1)
        special = elemDB.findFrameX(image, frame, "frame_special")[0]
        ### get parts of data in selected ###
        weapon_skill, weapon = weapon.splitX(skillBisection) # TODO: find refined subtype from weapon_skill
        _, assist = assist.splitX(skillBisection)
        _, special = special.splitX(skillBisection)
        _, skillA = skillA.splitX(skillBisection)
        _, skillB = skillB.splitX(skillBisection)
        _, skillC = skillC.splitX(skillBisection)
        _, seal = seal.splitX(skillBisection)
        _, hp = hp.splitX(statBisection)
        _, atk = atk.splitX(statBisection)
        _, spd = spd.splitX(statBisection)
        _, df = df.splitX(statBisection)
        _, res = res.splitX(statBisection)
        _, sp = sp.splitX(statBisection)
        _, hm = hm.splitX(statBisection)
        weapon_type, _, lvl, move_type_and_cosmetic = lvl.splitX(lvlBisection)
        _, exp = exp.splitX(statBisection)
        white = (255,255,255)
        limegreen = (69, 244, 129)
        gold = (149, 249, 254)
        data["weapon"] = readTXT(image[weapon.index], (white, (70, 245, 130)))
        data["assist"] = readTXT(image[assist.index], (white,))
        data["special"] = readTXT(image[special.index], (white,))
        data["skillA"] = readTXT(image[skillA.index], (white,))
        data["skillB"] = readTXT(image[skillB.index], (white,))
        data["skillC"] = readTXT(image[skillC.index], (white,))
        data["seal"] = readTXT(image[seal.index], (white,))
        data["hp"] = readTXT(image[hp.index], (gold,limegreen), True)
        data["atk"] = readTXT(image[atk.index], (gold,limegreen), True)
        data["spd"] = readTXT(image[spd.index], (gold,limegreen), True)
        data["def"] = readTXT(image[df.index], (gold,limegreen), True)
        data["res"] = readTXT(image[res.index], (gold,limegreen), True)
        data["sp"] = readTXT(image[sp.index], (gold,limegreen), True)
        data["hm"] = readTXT(image[hm.index], (gold,limegreen), True)
        data["lvl"] = readTXT(image[lvl.index], ((254,254,254), limegreen), True)
        if data["lvl"] == "40":
            data["exp"] = "MAX"
        else:
            data["exp"] = readTXT(image[exp.index], (white,), True)
        print(*data.items(), sep="\n")
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)

if __name__ == "__main__":
    image = cv2.imread('screenshot_char.png', -1)
    CharacterView(image)
    cv2.waitKey()
    # img = cv2.imread('screenshot_char_thresh.png', 0)
    # _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    # readTXT(img)
    # gui.showImage(img)
    # txt = pytesseract.image_to_string(img, config=r'--psm 3')
    # print(txt)
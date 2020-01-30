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
    image = ~np.logical_or.reduce(
        [np.all(image == t, axis=-1) for t in select])
    gui.showImage((image * 255).astype('uint8'))
    config = '--psm 7'
    if numeric:
        config += ' -c tessedit_char_whitelist=0123456789-.+'
    return pytesseract.image_to_string(image, config=config)


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
        statBisection = 200  # half
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
        white = (255, 255, 255)
        limegreen = (69, 244, 129)
        gold = (149, 249, 254)
        data["weapon"] = readTXT(image[weapon.index], (white, (70, 245, 130)))
        data["assist"] = readTXT(image[assist.index], (white,))
        data["special"] = readTXT(image[special.index], (white,))
        data["skillA"] = readTXT(image[skillA.index], (white,))
        data["skillB"] = readTXT(image[skillB.index], (white,))
        data["skillC"] = readTXT(image[skillC.index], (white,))
        data["seal"] = readTXT(image[seal.index], (white,))
        data["hp"] = readTXT(image[hp.index], (gold, limegreen), True)
        data["atk"] = readTXT(image[atk.index], (gold, limegreen), True)
        data["spd"] = readTXT(image[spd.index], (gold, limegreen), True)
        data["def"] = readTXT(image[df.index], (gold, limegreen), True)
        data["res"] = readTXT(image[res.index], (gold, limegreen), True)
        data["sp"] = readTXT(image[sp.index], (gold, limegreen), True)
        data["hm"] = readTXT(image[hm.index], (gold, limegreen), True)
        data["lvl"] = readTXT(image[lvl.index], ((254, 254, 254), limegreen), True)
        if data["lvl"] == "40":
            data["exp"] = "MAX"
        else:
            data["exp"] = readTXT(image[exp.index], (white,), True)
        # print(*data.items(), sep="\n")
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

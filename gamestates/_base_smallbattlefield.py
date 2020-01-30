import cv2

import gui
from constants import *
from database import elemDB, mapDB

from gamestates._base import GameStateBase

MAP_KEY = "small_battlefield"
FRIENDLY_HEALTHBAR = cv2.imread('%s/friendly_healthbar.bmp' % TEMPLATES_DIR)
ENEMY_HEALTHBAR = cv2.imread('%s/enemy_healthbar.bmp' % TEMPLATES_DIR)


def toGrid(x, y, mx, my):
    return int(MAP_DIMENSIONS_N[0] * x / mx), int(MAP_DIMENSIONS_N[1] * y / my)


def findEntities(image):
    entities = {}
    h, w = image.shape[:2]
    ally_bars = elemDB.findAll(image, FRIENDLY_HEALTHBAR)
    # gui.showSelections(image, ally_bars)
    entities["allies"] = list(map(lambda s: toGrid(s.x, s.y, w, h), ally_bars))
    enemy_bars = elemDB.findAll(image, ENEMY_HEALTHBAR)
    # gui.showSelections(image, enemy_bars)
    entities["enemies"] = list(map(lambda s: toGrid(s.x, s.y, w, h), enemy_bars))
    return entities


class SmallBattlefield(GameStateBase):
    def __init__(self, image):
        super().__init__(image)
        s = elemDB.findMap(image, MAP_KEY)
        mapArea = image[s.index]
        name = mapDB.search(mapArea)
        data = {
            "mapData": mapDB.data(name),
            "positions": findEntities(mapArea)
        }
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)

    def selectTile(x, y):
        assert((0, 0) <= (x, y) <= MAP_DIMENSIONS_N)

    def doubleTapTile(x, y):
        assert((0, 0) <= (x, y) <= MAP_DIMENSIONS_N)

    def dragTile(x1, y1, x2, y2):
        p1 = (x1, y1)
        p2 = (x2, y2)
        assert((0, 0) <= p1 <= MAP_DIMENSIONS_N)
        assert((0, 0) <= p2 <= MAP_DIMENSIONS_N)
        assert(p1 != p2)


if __name__ == "__main__":
    img = cv2.imread('screenshot.png', -1)
    o = SmallBattlefield(img)
    print(*o.parsed_data.items(), sep='\n')
    cv2.waitKey()

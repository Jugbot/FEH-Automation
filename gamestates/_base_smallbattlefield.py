from constants import MAP_DIMENSIONS_N
from database import mapDB 
from database import elemDB 
from gamestates._base import GameStateBase

MAP_KEY = "small_battlefield"

def findEntities(image):
    entities = {}
    path = './templates/friendly_healthbar.bmp'    
    entities["allies"] = elemDB.findAll(image, cv2.imread(path))
    path = './templates/enemy_healthbar.bmp'
    entities["enemies"] = elemDB.findAll(image, cv2.imread(path))
    return entities

class SmallBattlefield(GameStateBase):
    def __init__(self, image):
        super().__init__(image)
        s = elemDB.findMap(image, MAP_KEY)
        mapArea = image[s.index].copy()
        name = mapDB.search(mapArea)
        data = {
            "mapData": mapDB.data(name, mapArea),
            "entities": findEntities(image)
        }
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)
        
    def selectTile(x, y):
        assert((0,0) <= (x, y) <= MAP_DIMENSIONS_N)

    def doubleTapTile(x, y):
        assert((0,0) <= (x, y) <= MAP_DIMENSIONS_N)

    def dragTile(x1, y1, x2, y2):
        p1 = (x1, y1)
        p2 = (x2, y2)
        assert((0,0) <= p1 <= MAP_DIMENSIONS_N)
        assert((0,0) <= p2 <= MAP_DIMENSIONS_N)
        assert(p1 != p2)

if __name__ == "__main__":
    import cv2
    img = cv2.imread('screenshot.png', -1)
    o = SmallBattlefield(img)
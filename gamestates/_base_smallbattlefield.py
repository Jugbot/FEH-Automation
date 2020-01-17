from constants import MAP_DIMENSIONS_N
from database import mapdb 
from database import elementdb 

MAP_KEY = "small_battlefield"

class SmallBattlefield(GameStateBase):
    def __init__(self, image):
        super().__init__(image, data)
        if not elementdb.has(MAP_KEY):
            elementdb.find(image, mapImage, MAP_KEY)
        s = elementdb.getSlice(MAP_KEY)
        data={"mapData": mapdb.getMapData(MAP_KEY, image[s])}
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


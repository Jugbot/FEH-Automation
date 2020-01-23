from gamestates._base_smallbattlefield import SmallBattlefield
from gamestates._base_battleui import BattleUI

class TrainingTower(BattleUI, SmallBattlefield):
    def __init__(self, image):
        super().__init__(image, data)
        data={}
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)
        
    """Logic & rules for TT mode
    """
    def battle(self):
        pass
from gamestates._base import GameStateBase 

class BattleUI(GameStateBase):
    def __init__(self, image):
        # super().__init__(image, data)
        data={}
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)

    isFighting = False

    def selectFight(self):
        if self.isFighting: return
        self.isFighting = True

    def selectSwap(self):
        if not self.isFighting: return
        self.isFighting = False

    def selectAutoBattle(self):
        print("bruh")
        if not self.isFighting: return
        self.isFighting = False

    def selectDangerArea(self):
        pass

    def selectOther(self):
        pass

    def selectCharacterInfo(self):
        # might not be available depending on selection
        pass
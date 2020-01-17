

class BattleConfirm(GameStateBase):
    def __init__(self, image):
        super().__init__(image, data)
        data={}
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)
        
    def selectFight():
        pass

    def selectBack():
        pass

    def selectEditTeams():
        pass

    def selectEquipSkillSets():
        pass

    def selectPreviewMap():
        pass

    def selectMissions():
        pass
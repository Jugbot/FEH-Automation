from gamestates._base_mainview import MainView


class BattleSelect(MainView):
    def __init__(self, image):
        super().__init__(image, data)
        data = {}
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)

    def selectTrainingTower():
        pass

    def selectAetherKeeps():
        pass

    def selectSpecialMaps():
        pass

    def selectStoryMaps():
        pass

    def selectColiseum():
        pass

    def selectEvents():
        pass

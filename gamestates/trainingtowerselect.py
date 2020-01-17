from gamestates._base_stageselect import StageSelect

class TrainingTowerSelect(StageSelect):
    def __init__(self, image):
        super().__init__(image, data)
        data={}
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)

    def selectStratumTier(tier):
        assert(0 <= tier <= 10)

    def selectStratumLvl(level):
        level = int(level)
        assert(0 <= level <= 40)

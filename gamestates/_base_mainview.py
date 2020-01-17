class MainView(GameStateBase):
    def __init__(self, image):
        super().__init__(image, data)
        data={}
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        valid = True
        return valid and self.super().valid(image)
        
    def selectHome():
        pass
    
    def selectBattle():
        pass

    def selectAllies():
        pass

    def selectSummon():
        pass

    def selectShop():
        print("wtf you doin boi")

    def selectMisc():
        pass
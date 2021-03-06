
class GameStateBase:
    parsed_data = {}
    def __init__(self, image):
        """Parse image and store the data in parsed_data with data
        
        Arguments:
            image {[type]} -- [description]
            data {[type]} -- [description]
        """
        data={}
        self.parsed_data.update(data)

    @staticmethod
    def valid(image):
        """Confirm that image matches state.
        
        Returns:
            bool -- Match or not a match
        """
        valid = True
        return valid and self.super().valid(image)
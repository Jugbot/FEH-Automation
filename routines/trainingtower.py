from scraper import *
import cv2
import pickle

def routine(state):
    if state == "battle_menu":
        selectTrainingTower()
    elif state == "TT":
        pass

def run():
    screen_capture = cv2.imread('screenshot_char.png')
    layouts = loadLayouts()
    state_name = getGamestate()

    if state_name == "board_view":
        if state_name in layouts:
            selection = layouts[state_name]
            name = findMapMatch(screen_capture[selection])
        else:
            print('Performing first-time analysis... ')
            name, selection = findMapMatchBlind(screen_capture)
            print('done')
            layouts[state_name] = selection

        print("Board name: ", name)
        board_image = screen_capture[selection]
        data = getMapData(board_image, name)
        data = findEntities(board_image, data) # TODO: keep entity info in different place
        # for each entity, 
        # click entity on board then full screen view
        loadEntity() # read full-screen info
        saveLayouts(layouts)

if __name__ == "__main__":
    run()

import cv2
import numpy as np

def showImage(image):
    cv2.namedWindow('D E B U G', cv2.WINDOW_NORMAL)
    cv2.imshow('D E B U G', image)
    cv2.waitKey()


def showBestMatch(original_image, template, field):
    image = original_image.copy()
    h, w, c = template.shape
    pt = cv2.minMaxLoc(field)[-1]
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.namedWindow('Selection', cv2.WINDOW_NORMAL)
    cv2.imshow('Selection', image)
    cv2.waitKey()


def showImageMapData(image, map):
    h, w, c = image.shape
    hm, wm = map.shape
    for iy, ix in np.ndindex(map.shape):
        tile = map[iy, ix]
        cv2.putText(image, str(tile), 
            (int(ix * w / wm), int((iy+1) * h / hm)), 
            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 10)
        cv2.putText(image, str(tile), 
            (int(ix * w / wm), int((iy+1) * h / hm)), 
            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
import cv2
import numpy as np


def showImage(image):
    cv2.namedWindow('D E B U G', cv2.WINDOW_NORMAL)
    cv2.imshow('D E B U G', image)
    cv2.waitKey()


def showBestMatch(original_image, template, field):
    image = original_image.copy()
    h, w = template.shape[:2]
    pt = cv2.minMaxLoc(field)[-1]
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    cv2.namedWindow('Selection', cv2.WINDOW_NORMAL)
    cv2.imshow('Selection', image)
    cv2.waitKey()


def showImageMapData(image, mapData):
    f = 5
    h, w = image.shape[:2]
    s = w/1080
    hm, wm = mapData.shape
    for iy, ix in np.ndindex(mapData.shape):
        tile = mapData[iy, ix]
        cv2.line(image,
                 (int(ix * w / wm), int((iy+1) * h / hm)),
                 (int((ix+1) * w / wm), int((iy+1) * h / hm)),
                 (255, 255, 255))
        cv2.line(image,
                 (int((ix+1) * w / wm), int((iy) * h / hm)),
                 (int((ix+1) * w / wm), int((iy+1) * h / hm)),
                 (255, 255, 255))
        cv2.putText(image, str(tile),
                    (int(ix * w / wm) + f, int((iy+1) * h / hm) - f),
                    cv2.FONT_HERSHEY_COMPLEX, 2*s, (0, 0, 0), 4)
        cv2.putText(image, str(tile),
                    (int(ix * w / wm) + f, int((iy+1) * h / hm) - f),
                    cv2.FONT_HERSHEY_COMPLEX, 2*s, (255, 255, 255), 1)

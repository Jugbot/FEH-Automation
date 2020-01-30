import cv2
import numpy as np

wid = 0
def getId():
    global wid
    wid += 1
    return str(wid)

def showImage(image):
    name = 'D E B U G ' + getId()
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(name, image)

def showPoints(image, arr):
    image = image.copy()
    for p in arr:
        cv2.drawMarker(image, tuple(p[::-1]), (0,0,255))
    showImage(image)

def showFeatures(image, template, kp1, kp2):
    graph1 = cv2.drawKeypoints(image,kp1,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    graph2 = cv2.drawKeypoints(template,kp2,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    name1 = 'A Features' + getId()
    cv2.namedWindow(name1, cv2.WINDOW_NORMAL)
    cv2.imshow(name1, graph1)
    name2 = 'B Features ' + getId()
    cv2.namedWindow(name2, cv2.WINDOW_NORMAL)
    cv2.imshow(name2, graph2)

def showSelections(image, selections):
    image = image.copy()
    for s in selections:
        cv2.rectangle(image, (s.x, s.y), (s.x+s.w,s.y+s.h), (0,0,255), 2)
    showImage(image)


def showFeatureMatches(image, template, kp1, kp2, matches, H=None):
    if H is not None:
        h, w = template.shape[:2]
        p1 = np.array((0, 0, 1))
        p2 = np.array((w, h, 1))
        p1 = tuple((H @ p1).astype(int))
        p2 = tuple((H @ p2).astype(int))
        cv2.rectangle(image,p1[:2],p2[:2],(0,0,255),2)
    graph = cv2.drawMatches(image,kp1,template,kp2,matches,None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    name = 'Matches ' + getId()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, graph)

def showBestMatch(original_image, template, field):
    image = original_image.copy()
    h, w = template.shape[:2]
    pt = cv2.minMaxLoc(field)[-1]
    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
    name = 'Selection ' + getId()
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)


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

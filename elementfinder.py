# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imageloader import templates

def debug(image):
    cv2.imshow('name', image)
    cv2.waitKey()

img_rgb = cv2.imread('test.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# for name, template in templates.items():
template = templates['Btn_Back.0.png']
# template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h, c = template.shape
r,g,b,a = cv2.split(template)
color = cv2.merge((r,g,b))
alpha = cv2.merge((a,a,a))
# color = np.ndarray((w, h, 3), np.uint8)
# alpha = np.ndarray((w, h, 1), np.uint8)
# cv2.mixChannels([template], [color, alpha], [0,0,1,1,2,2,3,3])
# color = np.repeat(color, 3, axis=2)
# TM_CCOEFF_NORMED -> TM_CCORR_NORMED 
res = cv2.matchTemplate(img_rgb, color, cv2.TM_CCORR_NORMED, None, alpha)
threshold = 0.9
loc = np.where( res >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

    


cv2.imshow('res.png', template)
cv2.waitKey()
cv2.imshow('res.png', img_rgb)
cv2.waitKey()
cv2.imshow('res.png', res)
cv2.waitKey()
cv2.imwrite('res.png',img_rgb)

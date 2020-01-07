# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
import cv2
import numpy as np
from matplotlib import pyplot as plt
from imageloader import templates

img_rgb = cv2.imread('test.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# for name, template in templates.items():
template = cv2.cvtColor(templates['Btn_Back.0.png'], cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
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

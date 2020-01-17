from appium import webdriver
from appium.webdriver.common.mobileby import MobileBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import cv2
import io
import base64
import time
import numpy as np
import gui

userName = "USERNAME"
accessKey = "ACCESS_KEY"

desired_caps = {
  "automationName": "UIAutomator2",
  "platformName": "Android",
  "deviceName": "Pixel 3a",
  "platformVersion": "10.0",
  "appPackage": "com.nintendo.zaba",
  "noReset": True,
  "appActivity": "org.cocos2dx.cpp.AppActivity",
  "deviceId": "192.168.1.2:5555"
}

driver = webdriver.Remote("http://localhost:4723/wd/hub", desired_caps)
# driver = webdriver.Remote("0.0.0.0:5037", desired_caps)
print(driver.currentActivity())
hierarchy = driver.page_source
with open('appLayout.xml', 'w+') as f:
    f.write(hierarchy)
    f.close()
print(hierarchy)

search_results = driver.find_elements_by_class_name("android.view.View")
assert(len(search_results) > 0)
o = search_results[0].rect
x = o["x"]
y = o["y"]
w = o["width"]
h = o["height"]
screenshotBase64 = driver.get_screenshot_as_base64()
nparr = np.fromstring(base64.standard_b64decode(screenshotBase64), np.uint8)
screenshot = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
# screenshot = screenshot[y:y+h,x:x+w]
gui.showImage(screenshot)
cv2.waitKey()

driver.quit()

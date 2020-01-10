from appium import webdriver
from appium.webdriver.common.mobileby import MobileBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

userName = "USERNAME"
accessKey = "ACCESS_KEY"

desired_caps = {
  "automationName": "UIAutomator2",
  "platformName": "Android",
  "deviceName": "Pixel 3a",
  "platformVersion": "10.0",
  "appPackage": "com.nintendo.zaba",
  "noReset": True,
  "appActivity": "org.cocos2dx.cpp.AppActivity"
}

driver = webdriver.Remote("0.0.0.0:4723", desired_caps)

search_element = WebDriverWait(driver, 30).until(
    EC.element_to_be_clickable((MobileBy.ACCESSIBILITY_ID, "Search Wikipedia"))
)
search_element.click()

search_input = WebDriverWait(driver, 30).until(
    EC.element_to_be_clickable((MobileBy.ID, "org.wikipedia.alpha:id/search_src_text"))
)
search_input.send_keys("BrowserStack")
time.sleep(5)

search_results = driver.find_elements_by_class_name("android.widget.TextView")
assert(len(search_results) > 0)

driver.quit()

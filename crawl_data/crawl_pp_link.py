from selenium import webdriver
from selenium.webdriver.chrome.options import Options as chromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from pathlib import Path
from seleniumwire import webdriver as wireWebdriver

chrome_path = Path(r'path/to/chrome.exe')
chrome_driver_path = Path(r'path/to/chrome_driver/dir')

options = chromeOptions()
options.add_argument('--headless')  # use headless mode
options.add_argument('--disable-gpu')  # we only want pp link, so ignore other irrelevant elements
options.add_argument('--disable-images')  
options.add_argument('--disable-extensions')
options.add_argument('--disable-infobars')
options.page_load_strategy = 'eager'  # set page loading strategy to 'eager'

options.binary_location = str(chrome_path)  # set the chrome exe and chrome driver path
chrome_service = ChromeService(executable_path=str(chrome_driver_path)) 
        
def search_for_pp_link(homepage_link):
    try:
        driver = wireWebdriver.Chrome(options=options, service=chrome_service)
        driver.get(homepage_link)
        time.sleep(2)  # leave 2s for browser to load link
        pp_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.XPATH,
                    "//a[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'privacy')]"))
            # search for clikcable element with 'privacy' contained
            # (By.XPATH, "//a[contains(., 'privacy')]"))
        )
        pp_link = pp_element.get_attribute('href')
    except Exception as message:
        pass
    driver.quit()
    return pp_link

from selenium import webdriver
from selenium.webdriver.firefox.options import Options

def init_driver() -> webdriver.Firefox:
    opts = Options()
    opts.add_argument("--headless")

    return webdriver.Firefox(options=opts)

def close_driver(driver: webdriver.Firefox) -> None:
    driver.close()

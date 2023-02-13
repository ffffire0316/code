from selenium import webdriver

chrome_options=webdriver.ChromeOptions
chrome_options.headless=True
chrome=webdriver.Chrome(chrome_options=chrome_options)
page=chrome.get("www.baidu.com")
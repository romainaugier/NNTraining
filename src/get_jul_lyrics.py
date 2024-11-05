import typing
import scrap
import io
import os
import shutil
import traceback
import threading
import time

from concurrent.futures import ThreadPoolExecutor

from queue import Queue

from selenium import webdriver
from selenium.webdriver.common.by import By

SONG_LIST_BASE_URL = "https://www.paroles.net/jul-{}"
NUM_PAGES = 10

GET_MUTEX = threading.Lock()

def fix_chars(string: str) -> str:
    string = string.replace('’', '\'')
    string = string.replace('ù', 'u')
    string = string.replace('û', 'u')
    string = string.replace('à', 'a')
    string = string.replace('À', 'a')
    string = string.replace('â', 'a')
    string = string.replace('Â', 'A')
    string = string.replace('ê', 'e')
    string = string.replace('é', 'e')
    string = string.replace('É', 'E')
    string = string.replace('è', 'e')
    string = string.replace('È', 'E')
    string = string.replace('œ', 'oe')
    string = string.replace('ô', 'o')
    string = string.replace('Ô', 'O')
    string = string.replace('ó', 'o')
    string = string.replace('ç', 'c')
    string = string.replace('Ç', 'C')
    string = string.replace('î' , 'i')
    string = string.replace('ï' , 'i')
    string = string.replace('¿', '')

    return string

def get_song_links(driver: webdriver.Firefox) -> typing.List[str]:
    print("Getting JUL lyrics links")

    links = list()

    for i in range(2, 11):
        url = SONG_LIST_BASE_URL.format(i)
        driver.get(url)

        songs = driver.find_elements(By.CLASS_NAME, "song-name")

        for song in songs:
            children = song.find_elements(By.CSS_SELECTOR, "*")
            for child in children:
                href = child.get_attribute("href")

                if href is not None and "jul" in href:
                    links.append(href)

    print(f"Got {len(links)} lyrics links")

    return links

def get_song_lyrics(driver: webdriver.Firefox, url: str) -> str:
    driver.get(url)

    song_text = driver.find_element(By.CLASS_NAME, "song-text")

    lyrics = song_text.get_attribute("innerText")

    if lyrics is None:
        return ""

    return lyrics

def get_lyrics(links_queue: "Queue[str]", buffer: io.StringIO, counter: typing.List[int]) -> None:
    driver = scrap.init_driver()

    while not links_queue.empty():
        link = links_queue.get()

        try:
            lyrics = get_song_lyrics(driver, link)
            lyrics_clean = "\n".join([line for line in lyrics.splitlines()[1:] if line.strip("\n") != ""])
            # lyrics_clean = fix_chars(lyrics_clean)

            with GET_MUTEX:
                lyrics_buffer.write(lyrics_clean)
                lyrics_buffer.write("\n")
                lyrics_buffer.write("\n")
                counter[0] += 1

        except Exception:
            print("Error")

    scrap.close_driver(driver)

if __name__ == "__main__":
    driver = scrap.init_driver()

    links = get_song_links(driver)

    links_queue = Queue()

    for link in links:
        links_queue.put(link)

    lyrics_buffer = io.StringIO()

    counter = [0]

    threadpool = ThreadPoolExecutor()

    for _ in range(int(os.cpu_count() / 4)):
        threadpool.submit(get_lyrics, links_queue, lyrics_buffer, counter)

    while not links_queue.empty():
        with GET_MUTEX:
            print(f"\rDownloaded [{counter[0]}/{len(links)}]", end='', flush=True)
        time.sleep(5)

    scrap.close_driver(driver)

    print("Saving lyrics")

    with open("lyrics.txt", "w", encoding="utf-8") as file:
        lyrics_buffer.seek(0)
        shutil.copyfileobj(lyrics_buffer, file)

    print("Saved lyrics")

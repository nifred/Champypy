#! /usr/bin/env python3
# coding: utf-8

import json
import os
import aiohttp
import asyncio
import async_timeout
import pandas as pd
import numpy as np
from extract_mushroom_image import main

# Constant pathname
FILEPATH = os.path.dirname(os.path.abspath(__file__))
DIRPATH = os.path.abspath(FILEPATH)

# function that gets url to be used in mushroom observer api
def get_url(name):
    url = "https://mushroomobserver.org/api/"\
    "images?&name={}&format=json".format(name)
    return url

# function that gets img url from mushroom observer website
def get(img_list):
    img_urls = []
    for i in img_list:
        img_url = 'https://images.mushroomobserver.org/320/{}.jpg'.format(i)
        img_urls.append(img_url)
    return img_urls

# asynchronous function that gets image id from mushroom observer api
async def get_img_url(name, url, session):
    async with session.get(url) as rp:
        try:
            value = await rp.json()
            df.at[name, 'img'] = get(value['results'])
        except Exception as e:
            print('error', e, name)
        return await rp.json()

# asynchronous function calls get_img_url function under a semaphore
async def bound_fetch(semaphore, name, url, session):
    async with semaphore:
        return await get_img_url(name, url, session)

# asynchronous function using asyncio to run async request
async def run(url_list):
    tasks = []
    semaphore = asyncio.Semaphore(9)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
        for name, url in url_list.items():
            task = asyncio.ensure_future(
                bound_fetch(semaphore, name, url, session))
            tasks.append(task)
        responses = asyncio.gather(*tasks)
        return await responses

# launch the script
if __name__ == "__main__":
    with open(LIST) as json_list:
        mushroom_list = json.load(json_list)
    url_list = {}
    for name_fr, name_la in mushroom_list.items():
        url_list[name_fr] = get_url(name_la)
    df = pd.DataFrame.from_dict(
        url_list, orient='index', columns=['url'])
    df['img'] = np.empty((len(df), 0)).tolist()
    loop = asyncio.get_event_loop()
    future = asyncio.ensure_future(run(url_list))
    loop.run_until_complete(future)
    main(df, DIRPATH)

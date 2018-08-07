#! /usr/bin/env python3
# coding: utf-8

import json
import os
import aiohttp
import asyncio
import async_timeout
import pandas as pd
import numpy as np
from unidecode import unidecode

# asynchronous function downloads images from mushroom observer
async def download_img(url, path_name, session):
    async with session.get(url) as rp:
        try:
            filename = url.split('/')[-1]
            pathname = os.path.join(path_name, filename)
            if not os.path.exists(pathname):
                with open(pathname, 'wb') as file:
                    img = await rp.content.read()
                    file.write(img)
            else:
                img = None
        except Exception as e:
            print('error', e, name)
            return {'data':img, 'url':rp.url.raw_name}

# asynchronous function calls download_img function under a semaphore
async def bound_fetch(semaphore, url, path_name, session):
    async with semaphore:
        return await download_img(url, path_name, session)

# asynchronous function using asyncio to run async request
async def run(df):
    tasks = []
    semaphore = asyncio.Semaphore(9)

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
        for index, row in df.iterrows():
            for url in row['img']:
                task = asyncio.ensure_future(
                    bound_fetch(
                        semaphore, url, row['path_name'], session))
                tasks.append(task)
        responses = asyncio.gather(*tasks)
        return await responses

# function that creates local directories
def create_directory(name, dirpath):
    path_name = os.path.join(dirpath, unidecode(name.replace(' ', '_').lower()))
    if not os.path.exists(path_name):
        os.mkdir(path_name)
    return path_name

# main function
def main(df, dirpath):
        DIRPATH = os.path.join(dirpath, 'db')
        LIST = os.path.join(DIRPATH, 'mushroom_selectionned_list.json')
        if not os.path.exists(DIRPATH):
            os.mkdir(DIRPATH)
        path_name = []
        for name in df.index.values:
            path_name.append(create_directory(name, DIRPATH))
        df['path_name'] = path_name
        loop = asyncio.get_event_loop()
        future = asyncio.ensure_future(run(df))
        loop.run_until_complete(future)
        df.to_csv(os.path.join(DIRPATH, 'mushroom_list_img_url.csv'))

#! /usr/bin/env python3
# coding: utf-8
import requests
import json
import sys
import os
import pandas as pd
from pandas.io.json import json_normalize
from urllib import request
import numpy as np

FILEPATH = os.path.dirname(os.path.abspath(__file__))
DIRPATH = os.path.abspath(os.path.join(FILEPATH, os.pardir))
BDPATH = os.path.join(DIRPATH, 'bd2')

"""
 URL from mushroom observer api which returns a json file filled by
 mushroom names
"""
URLNAME = "https://mushroomobserver.org/api/names?&misspellings&=no&is_deprecated=False&detail=low&format=json"

class ExtractName:
    """ Initialize ExtractName """
    def __init__(self, session, length):
        self.length = length
        self.session = session

    def nameExtraction(self):
        j=1
        mushroomNames = []
        while j <= self.length:
            # Get the HTTPS response
            response = session.get(
                'https://mushroomobserver.org/api/names?&misspellings=no&is_deprecated=False&detail=low&page=%s&format=json' % (j,)
                )
            # Extract the HTTPS response in json format
            responseJson = transformInJson(response)
            # Extract the response results tag
            responseJsonResults = responseJson['results']
            df = pd.DataFrame.from_dict(
                json_normalize(
                    responseJsonResults),
                orient='columns')
            mushroomNames.append(df)
            updateProgress("Extracting mushroom names", j/self.length)
            j += 1
            df = pd.concat(mushroomNames, sort=True)
        return df

class ExtractImage:
    def __init__(self, length, name, session):
        self.name = name
        self.length = length
        self.session = session

    def urlExtraction(self):
        k=0
        url = []
        if self.length:
            for page in range(1, self.length+1):
                imageURL = 'https://mushroomobserver.org/api/images?&name=%s&page=%s&format=json' % (self.name, page)
                response = session.get(imageURL)
                responseJson = transformInJson(response)
                try:
                    responseJsonResults = responseJson['results']
                    newDir = os.path.abspath(os.path.join(BDPATH, self.name))
                    if not os.path.exists(newDir):
                        os.mkdir(newDir)
                    for id in responseJsonResults:
                        image_url = "https://images.mushroomobserver.org/320/%s.jpg" % id
                        filename = image_url.split('/')[-1]
                        savePath = os.path.join(newDir, filename)
                        if not os.path.exists(savePath):
                            request.urlretrieve(
                                image_url, os.path.join(
                                    newDir,
                                    image_url.split('/')[-1]))
                        url.append(image_url)
                        k += 1
                        updateProgress(
                            "Extracting {}".format(
                                self.name),
                            k/responseJson['number_of_records'])
                    return url
                except Exception as e:
                    return np.nan
            else:
                return np.nan

def transformInJson(response):
    return response.json()

def getPageNumber(session, URL):
    response = session.get(URL)
    responseJson = response.json()
    try:
        length = responseJson['number_of_pages']
    except:
        length = False
    return length

def updateProgress(job, progress):
    length = 20
    block = int(round(length*progress))
    message = "\r{0}: [{1}] {2}%".format(
        job, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: message += " Done\r\n"
    sys.stdout.write(message)
    sys.stdout.flush()

if __name__ == "__main__":
    if not os.path.exists(BDPATH):
        os.mkdir(BDPATH)
    session = requests.Session()
    numberOfNamesPages = getPageNumber(session, URLNAME)
    Extraction = ExtractName(session, numberOfNamesPages)
    name = Extraction.nameExtraction()
    name.drop_duplicates(subset='name', inplace=True)
    name = name.iloc[:5]

    name['image_url'] = name.apply(
        lambda row: ExtractImage(getPageNumber(
            session,
            'https://mushroomobserver.org/api/images?&name=%s&format=json' % row['name']),
                                 row['name'], session).urlExtraction(), axis=1)
    name.to_csv('test.csv')

#! /usr/bin/env python3
# coding: utf-8
import requests
import json
import sys
import os
import pandas as pd
from pandas.io.json import json_normalize
from gevent.pool import Pool
import numpy as np
# FILEPATH = os.path.dirname(os.path.abspath(__file__))
# DIRPATH = os.path.abspath(os.path.join(FILEPATH, os.pardir))
# BDPATH = os.path.join(DIRPATH, 'bd')
# MNPATH = os.path.join(DIRPATH, 'mushroomDBDone.data')
# ALLNAMEPATH = os.path.join(DIRPATH, 'mushroomNames.data')
# MUSHROOMDB = []
# MUSHROOMALLNAMES = []

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
                'https://mushroomobserver.org/api/names?&misspellings&=no&is_deprecated=False&detail=low&page=%s&format=json' % (j,)
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

    def imageExtraction(self):
        k=0
        url = []
        if self.length:
            for page in range(1, self.length+1):
                imageURL = 'https://mushroomobserver.org/api/images?&name=%s&page=%s&format=json' % (self.name, page)
                response = session.get(imageURL)
                responseJson = transformInJson(response)
                try:
                    responseJsonResults = responseJson['results']
                    for id in responseJsonResults:
                        url.append("https://images.mushroomobserver.org/320/%s.jpg" % id)
                        # imgData = requests.get(url).content
                        # filename = os.path.join(self.newDir, url.split('/')[-1])
                        # if not os.path.exists(filename):
                        #     with open(filename, 'wb') as image:
                        #         image.write(imgData)
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
#
#
#
def transformInJson(response):
    return response.json()

def getPageNumber(session, URL):
    response = session.get(URL)
    responseJson = response.json()
    try:
        length = 1
    except:
        length = False
    return length
#
def updateProgress(job, progress):
    length = 20
    block = int(round(length*progress))
    message = "\r{0}: [{1}] {2}%".format(
        job, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: message += " Done\r\n"
    sys.stdout.write(message)
    sys.stdout.flush()

# def saveData(filename, object):
#     with open(filename, 'wb') as file:
#         pickle.dump(object, file)
#
# def loadData(filename):
#     with open(filename, 'rb') as file:
#         return pickle.load(file)

if __name__ == "__main__":
    session = requests.Session()
    numberOfNamesPages = getPageNumber(session, URLNAME)
    Extraction = ExtractName(session, numberOfNamesPages)
    name = Extraction.nameExtraction()
    name.drop_duplicates(subset='name', inplace=True)
    name = name.iloc[:5]
    print(name)
    name.to_csv('test.csv')
    name['image_url'] = name.apply(
        lambda row: ExtractImage(getPageNumber(
            session,
            'https://mushroomobserver.org/api/images?&name=%s&format=json' % row['name']),
                                 row['name'], session).imageExtraction(), axis=1)
    print(name[['name', 'image_url']])
    # if not os.path.exists(BDPATH):
    #     os.mkdir(BDPATH)
    #
    # if os.path.exists(MNPATH):
    #     MUSHROOMDB = loadData(MNPATH)
    #
    # if os.path.exists(ALLNAMEPATH):
    #     MUSHROOMALLNAMES = loadData(ALLNAMEPATH)
    #
    # if not MUSHROOMALLNAMES:
    #     numberOfNamesPages = getPageNumber(URLNAME)
    #     Extraction = ExtractName(numberOfNamesPages)
    #     nameExtracts = Extraction.nameExtraction()
    #     uniqueNames = removeDuplicate(nameExtracts)
    #     saveData('mushroomNames.data', uniqueNames)
    # else:
    #
    #     if MUSHROOMDB:
    #         uniqueNames = list(set(MUSHROOMALLNAMES) - set(MUSHROOMDB))
    #     else:
    #         uniqueNames = MUSHROOMALLNAMES
    # try:
    #     for name in uniqueNames:
    #         try:
    #             imageURL = 'https://mushroomobserver.org/api/images?&name=%s&format=json' % name
    #             numberOfImagesPages = getPageNumber(imageURL)
    #             image = ExtractImage(name, numberOfImagesPages)
    #             image.imageExtraction()
    #             MUSHROOMDB.append(name)
    #         except Exception as e:
    #             print(e)
    # except KeyboardInterrupt:
    #     saveData('mushroomDBDone.data', MUSHROOMDB)
    #
    # saveData('mushroomDBDone.data', MUSHROOMDB)

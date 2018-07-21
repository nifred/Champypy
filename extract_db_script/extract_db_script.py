#! /usr/bin/env python3
# coding: utf-8
import requests
import json
import sys
import os

FILEPATH = os.path.dirname(os.path.abspath(__file__))
DIRPATH = os.path.abspath(os.path.join(FILEPATH, os.pardir))
BDPATH = os.path.join(DIRPATH, 'bd')
if not os.path.exists(BDPATH):
    os.mkdir(BDPATH)


"""
 URL from mushroom observer api which returns a json file filled by
 mushroom names
"""
URLNAME = "https://mushroomobserver.org/api/names?&detail=high&format=json"

class ExtractName:
    """ Initialize ExtractName """
    def __init__(self, length):
        self.length = length

    def nameExtraction(self):
        j=1
        while j <= self.length:
            # Get the HTTPS response
            response = requests.get(
                'https://mushroomobserver.org/api/names?&detail=high&page=%s&format=json' % j
                )
            # Extract the HTTPS response in json format
            responseJson = transformInJson(response)
            # Extract the response results tag
            responseJsonResults = responseJson['results']
            # Initialize mushroomNames list to get mushroom name from API
            mushroomNames = []
            for mushroom in responseJsonResults:
                if not (mushroom['deprecated'] or mushroom['misspelled']):
                    mushroomNames.append(mushroom['name'].lower())
            updateProgress("Extracting mushroom names", j/self.length)
            j +=1
        return mushroomNames

class ExtractImage:
    def __init__(self, name):
        self.name = name
        self.newDir = os.path.abspath(os.path.join(BDPATH, self.name))
        if not os.path.exists(self.newDir):
            os.mkdir(self.newDir)


    def imageExtraction(self):
        imageURL = 'https://mushroomobserver.org/api/images?&name=%s&format=json' % self.name
        response = requests.get(imageURL)
        responseJson = transformInJson(response)
        try:
            k=0
            responseJsonResults = responseJson['results']
            for id in responseJsonResults:
                url = "https://mushroomobserver.nyc3.digitaloceanspaces.com/orig/%s.jpg" % id
                imgData = requests.get(url).content
                filename = os.path.join(self.newDir, url.split('/')[-1])
                if not os.path.exists(filename):
                    with open(filename, 'wb') as image:
                        image.write(imgData)
                updateProgress(
                    "Extracting {}".format(
                        self.name),
                    k/responseJson['number_of_records'])
                k +=1
        except Exception as e:
            print(e)



def transformInJson(response):
    return response.json()

def removeDuplicate(List):
    getUnique = set(List)
    return list(getUnique)

def getPageNumber(URL):
    response = requests.get(URL)
    responseJson = response.json()
    length = responseJson['number_of_pages']
    return 1

def updateProgress(job, progress):
    length = 20
    block = int(round(length*progress))
    message = "\r{0}: [{1}] {2}%".format(
        job, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: message += " Done\r\n"
    sys.stdout.write(message)
    sys.stdout.flush()

if __name__ == "__main__":
    numberOfNamesPages = getPageNumber(URLNAME)
    Extraction = ExtractName(numberOfNamesPages)
    nameExtracts = Extraction.nameExtraction()
    uniqueNames = removeDuplicate(nameExtracts)
    with open('mushroomNames.txt', 'w') as f:
        f.write("\n".join(uniqueNames))
    for name in uniqueNames:
        image = ExtractImage(name)
        image.imageExtraction()

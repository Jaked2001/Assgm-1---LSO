import numpy as np
import math   
import os
import random
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt




def selectFiles(nSmall, nMedium, nLarge):
    """
    Select a set number of files at random from the directory. It differientiate between the different sizes.

    Parameters
    ----------
    nSmall : integer
        How many small cities are being requested
    nMedium : integer
        How many medium cities are being requested
    nLarge : integer
        How many large cities are being requested

    Returns
    -------
    selectedPaths : list of strings
        a list of all paths selected.

    """

    size = list((nSmall, nMedium, nLarge))

    folderPath_small = "Instances/Small"
    folderPath_medium = "Instances/Medium"
    folderPath_large = "Instances/Large"

    pathList = list((folderPath_small, folderPath_medium, folderPath_large))

    selectedPaths = []
    for i in range(len(pathList)):
        # Select random files in each ith directory
        files = os.listdir(pathList[i])
        print(files)
        randomFiles = random.sample(files, size[i])

        for file in randomFiles:
            # Add the file name to the first part of the directory path
            fullPath = os.path.join(pathList[i], file)
            selectedPaths.append(fullPath)

    return(selectedPaths)

def generateRandomList(size, amount):
    rList = sorted(random.sample(range(size), amount))
    return rList

def generate_output(fileName):
    name = fileName + '.csv'
    fileName = open(name, 'w')
    fileName.write("File,Starting point, Cost\n")
    return fileName

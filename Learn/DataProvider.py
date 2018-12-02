#!/usr/bin/python
# -*- coding: UTF-8 -*-
from multiprocessing import Pool, Manager
from math import log
import operator
from os import listdir
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

DIGIT_TRAIN_PATCH = "/Users/zdh/Develepment/Game/machinelearninginaction/Ch02/trainingDigits/"
DIGIT_TEST_PATCH = "/Users/zdh/Develepment/Game/machinelearninginaction/Ch02/testDigits/"

DIGIT_JUMP_COUNT = 20


def digitData(isTraining):
    if isTraining:
        folderPath = DIGIT_TRAIN_PATCH
    else:
        folderPath = DIGIT_TEST_PATCH
    trainingFileList = listdir(folderPath)  # load the training set
    m = len(trainingFileList)
    hwLabels = []
    yArray = []
    trainingMat = []
    for i in range(m):
        if (i % DIGIT_JUMP_COUNT != 0):
            continue
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        valueLabel = []
        for i in range(10):
            if classNumStr == i:
                valueLabel.append(1)
            else:
                valueLabel.append(0)
        yArray.append(valueLabel)
        hwLabels.append(classNumStr)
        trainingMat.append(readOnDigitFile(folderPath + fileNameStr))

    return array(hwLabels), array(trainingMat), array(yArray)


def readOnDigitFile(fileNameStr):
    returnMat = []
    fr = open(fileNameStr)
    for i in range(32):
        lineStr = fr.readline()
        oneLine = []
        for j in range(32):
            oneLine.append(int(lineStr[j]))
        returnMat.append(oneLine)
    fr.close()
    return returnMat


def printWrongDigitClassFile(resultPath):
    fr = open(resultPath)
    resultLine = fr.readline()
    resultArray = resultLine.split(',')
    trainingFileList = listdir(DIGIT_TEST_PATCH)
    m = len(trainingFileList)
    count = 0
    for i in range(m):
        if (i % DIGIT_JUMP_COUNT != 0):
            continue
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if int(resultArray[count]) != classNumStr:
            print fileNameStr, ":", resultArray[count]
        count+=1

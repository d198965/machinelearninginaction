#!/usr/bin/python
# -*- coding: UTF-8 -*-
from numpy import *
import numpy as np
from Learn import DataProvider
import Learn.gbdt.GBDT as gt

gt.MAX_VALUE = 1.0
gt.MIN_VALUE = 0.0
gt.STEP_COUNT = 50

def adbTest():
    labelMat, allMat = DataProvider.readMushRooms()
    # halfMat,labelHalfMat,lastMat,lastHalfLabel = halfData(labelMat, allMat)
    halfMat,labelHalfMat,lastMat,lastHalfLabel = getRanomData(labelMat, allMat)
    resultTrees = gt.gbdtClassify(halfMat, labelHalfMat, 1, 0, 1, 20)
    m, n = shape(halfMat)
    resultClassify = zeros((m, 1))
    for i in range(len(resultTrees)):
        temTree = resultTrees[i][0]
        resultClassify[lastMat[:, temTree[1]] < temTree[0]] += temTree[2]
        resultClassify[lastMat[:, temTree[1]] >= temTree[0]] += temTree[3]

    errorCount = 0
    pErrorCount = 0
    eErrorCount = 0
    for index in range(len(lastHalfLabel)):
        if (lastHalfLabel[index] - (gt.MAX_VALUE + gt.MIN_VALUE) / 2) * (
                    resultClassify[index][0] - (gt.MAX_VALUE + gt.MIN_VALUE) / 2) < 0:
            if(lastHalfLabel[index] == 0):
                eErrorCount+=1
            else:
                pErrorCount +=1
            errorCount += 1
    print("LP:",len(lastHalfLabel[lastHalfLabel>0.5]))
    print("LE:",len(lastHalfLabel[lastHalfLabel<0.5]))
    print (100.0 * (1-float(errorCount) / m), "p:",pErrorCount,"e:",eErrorCount)


def getRanomData(labelMat, allMat):
    halfMat = []
    labelHalfMat = []
    lastMat = []
    lastHalfLabel = []

    for index in range(0, len(labelMat)):
        if index % 2 == 0:
            halfMat.append(allMat[index])
            labelHalfMat.append(labelMat[index])
        else:
            lastMat.append(allMat[index])
            lastHalfLabel.append(labelMat[index])
    halfMat = np.array(halfMat)
    lastMat = np.array(lastMat)
    return halfMat, np.array(labelHalfMat), lastMat, np.array(lastHalfLabel)

def halfData(labelMat, allMat):
    halfMat = allMat[0:len(allMat) / 2]
    labelHalfMat = labelMat[0:len(labelMat) / 2]
    lastMat = allMat[len(allMat) / 2:len(allMat)]
    lastHalfLabel = labelMat[len(allMat) / 2:len(allMat)]

    return halfMat,labelHalfMat,lastMat,lastHalfLabel

adbTest()

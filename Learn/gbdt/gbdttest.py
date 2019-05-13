#!/usr/bin/python
# -*- coding: UTF-8 -*-
from numpy import *
import numpy as np
import Learn.gbdt.GBDT as gt

gt.MIN_VALUE = 10.0
gt.MAX_VALUE = 20.0
# 读取数据
def readHorese(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    lineSplit = arrayOLines[0].strip('\n').replace(',', '').replace('\t', ' ').split(' ')
    column = len(lineSplit) - 1
    returnMat = zeros((0, column))
    labelMat = []
    for index in range(len(arrayOLines)):
        line = arrayOLines[index]
        datas = line.strip('\n').replace(',', '').replace('\t', ' ').split(' ')
        newRow = []
        for index in range(column):
            newRow.append(double(datas[index]))

        returnMat = np.row_stack((returnMat, newRow))

        if double(datas[column]) == 0:
            labelMat.append(gt.MIN_VALUE)
        else:
            labelMat.append(gt.MAX_VALUE)
    return returnMat, np.array(labelMat)



def adbTest():
    allMat, labelMat = readHorese("horseColicTraining.txt")
    resultTrees = gt.gbdtClassify(allMat, labelMat, 1, 0, 1, 20)

    allTestMat, labelTestMat = readHorese("horseColicTraining.txt")
    m, n = shape(allTestMat)
    resultClassify = zeros((m, 1))
    for i in range(len(resultTrees)):
        temTree = resultTrees[i][0]
        resultClassify[allTestMat[:, temTree[1]] < temTree[0]] += temTree[2]
        resultClassify[allTestMat[:, temTree[1]] >= temTree[0]] += temTree[3]

    errorCount = 0
    for index in range(m):
        if (labelTestMat[index] - (gt.MAX_VALUE + gt.MIN_VALUE) / 2) * (resultClassify[index][0] - (gt.MAX_VALUE + gt.MIN_VALUE) / 2) < 0:
            errorCount += 1
    print errorCount

adbTest()
#!/usr/bin/python
# -*- coding: UTF-8 -*-
from math import log
import operator
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import numpy.linalg as nlg
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

MAX_VALUE = 20.0
MIN_VALUE = 10.0
STEP_COUNT = 50

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
            labelMat.append(MIN_VALUE)
        else:
            labelMat.append(MAX_VALUE)
    return returnMat, np.array(labelMat)


# 返回当前最优单层决策树
# 使用两层决策树效果会更好
def calculateWeakClassifier(dataMat, classifyLabel):
    m, n = shape(dataMat)
    onePieceTree = []
    minError = inf
    # 先从维度遍历
    for dim in range(n):
        # 获取该维度的最大值和最小值,用于判断
        maxValue = max(dataMat[:, dim])
        minValue = min(dataMat[:, dim])
        # 每一个维度设定10个判断值
        step = (maxValue - minValue) * 1.0 / STEP_COUNT
        for temStep in range(STEP_COUNT):
            classifyValue = minValue + temStep * step
            # 计算左右节点的方差和
            leftValue = classifyLabel[dataMat[:, dim] < classifyValue]
            leftLength = len(leftValue)
            if leftLength == 0:
                leftLength = 1
            leftAvg = sum(leftValue) * 1.0 / leftLength
            leftArr = sum(np.dot((leftValue - leftAvg).T, leftValue - leftAvg))

            rightValue = classifyLabel[dataMat[:, dim] >= classifyValue]
            rightLength = len(rightValue)
            if rightLength == 0:
                rightLength = 1
            rightAvg = sum(rightValue) * 1.0 / rightLength
            rightArr = sum(np.dot((rightValue - rightAvg).T, rightValue - rightAvg))
            if leftArr + rightArr < minError:
                minError = leftArr + rightArr
                onePieceTree = []
                onePieceTree.append(classifyValue)
                onePieceTree.append(dim)
                onePieceTree.append(leftAvg)
                onePieceTree.append(rightAvg)
    return onePieceTree, minError  # 单层决策树


def gbdtClassify(attrMat, labelMat, maxDeep, leafMaxError, alpha, classifierMaxNumber):
    result = []
    m, n = shape(attrMat)
    for i in range(classifierMaxNumber):
        onePieceTree, minError = calculateWeakClassifier(attrMat, labelMat)
        result.append([onePieceTree])
        if minError <= leafMaxError:
            break
        # 更新labelMat
        labelMat[attrMat[:, onePieceTree[1]] < onePieceTree[0]] -= onePieceTree[2]
        labelMat[attrMat[:, onePieceTree[1]] >= onePieceTree[0]] -= onePieceTree[3]
    return result


def adbTest():
    allMat, labelMat = readHorese("horseColicTraining.txt")
    resultTrees = gbdtClassify(allMat, labelMat, 1, 0, 1, 20)

    allTestMat, labelTestMat = readHorese("horseColicTraining.txt")
    m, n = shape(allTestMat)
    resultClassify = zeros((m, 1))
    for i in range(len(resultTrees)):
        temTree = resultTrees[i][0]
        resultClassify[allTestMat[:, temTree[1]] < temTree[0]] += temTree[2]
        resultClassify[allTestMat[:, temTree[1]] >= temTree[0]] += temTree[3]

    errorCount = 0
    for index in range(m):
        if (labelTestMat[index] - (MAX_VALUE + MIN_VALUE) / 2) * (resultClassify[index][0] - (MAX_VALUE + MIN_VALUE) / 2) < 0:
            errorCount += 1
    print errorCount


adbTest()

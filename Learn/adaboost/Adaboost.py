#!/usr/bin/python
# -*- coding: UTF-8 -*-
from math import log
import operator
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import numpy.linalg as nlg
import operator
from os import listdir


# 计算分类列表
# dim第几维,classifyValue 分类的值, dataMata 数据集,leftRight 判断方向
# 在计算分类值是,用用dataMat中dim维的值与classifyValue作比较,与leftRight方向一致的为1,否则为-1,
def classifyTheValue(dim, classifyValue, dataMat, leftRight):
    resultMat = ones((shape(dataMat)[0], 1))
    if leftRight == 'ld':
        resultMat[dataMat[:, dim] >= classifyValue] = -1.0
    else:
        resultMat[dataMat[:, dim] < classifyValue] = -1.0
    return resultMat


# 返回当前最优单层决策树
def calculateWeakClassifier(dataMat, classifyLabel, D):
    m, n = shape(dataMat)
    onePieceTree = []
    minError = inf  #
    classResult = []  # 单层最优弱分类结果
    # 先从维度遍历
    for dim in range(n):
        # 获取该维度的最大值和最小值,用于判断
        maxValue = max(dataMat[:, dim])
        minValue = min(dataMat[:, dim])
        # 每一个维度设定10个判断值
        step = (maxValue - minValue) * 1.0 / 10
        for temStep in range(10):
            for leftRight in ['ld', 'rt']:
                classifyValue = minValue + temStep * step
                resultMat = classifyTheValue(dim, classifyValue, dataMat, leftRight)
                # 计算误差值
                compaireResult = zeros((m, 1))
                compaireResult[resultMat[:, 0] != classifyLabel] = 1
                weightError = np.dot(D.T, compaireResult)
                if weightError < minError:
                    minError = weightError
                    classResult = resultMat.copy()
                    onePieceTree = []
                    onePieceTree.append(0.5 * log((1 - weightError[0][0]) / weightError[0][0]))
                    onePieceTree.append(classifyValue)
                    onePieceTree.append(dim)
                    onePieceTree.append(leftRight)
    return onePieceTree, minError, classResult


def adaboost(dataMat, labelMat, classifierMaxNumber):
    result = []
    m, n = shape(dataMat)
    D = ones((m, 1)) / (m * 1.0)
    for i in range(classifierMaxNumber):
        onePieceTree, minError, classResult = calculateWeakClassifier(dataMat, labelMat, D)
        result.append([onePieceTree])
        if minError == 0:
            break
        # 计算DM下一个分类器样本权重
        temMinError = max(minError, 1e-16)
        D[classResult[:, 0] != labelMat] = D[classResult[:, 0] != labelMat] / (2 * temMinError)
        D[classResult[:, 0] == labelMat] = D[classResult[:, 0] == labelMat] / (2 * (1 - temMinError))
    return result


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
            labelMat.append(-1)
        else:
            labelMat.append(1)
    return returnMat, labelMat


def adbTest():
    allMat, labelMat = readHorese("horseColicTraining.txt")
    resultTrees = adaboost(allMat, labelMat, 100)

    allTestMat, labelTestMat = readHorese("horseColicTest.txt")
    m, n = shape(allTestMat)
    resultClassify = zeros((m, 1))
    for i in range(len(resultTrees)):
        temTree = resultTrees[i][0]
        if temTree[3] == 'ld':
            resultClassify[allTestMat[:, temTree[2]] > temTree[1]] -= temTree[0]
            resultClassify[allTestMat[:, temTree[2]] <= temTree[1]] += temTree[0]
        else:
            resultClassify[allTestMat[:, temTree[2]] < temTree[1]] -= temTree[0]
            resultClassify[allTestMat[:, temTree[2]] >= temTree[1]] += temTree[0]

    errorCount = 0
    for index in range(m):
        if labelTestMat[index] * resultClassify[index][0] < 0:
            errorCount += 1
    print errorCount


adbTest()

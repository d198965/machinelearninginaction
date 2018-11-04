#!/usr/bin/python
# -*- coding: UTF-8 -*-
from math import log
import operator
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import numpy.linalg as nlg


# 估计第三个用户对第四件商品的评分
def loadData():
    # 行代表一个用户对六件商品的评分1-10, 0代表没有评过分, 列表示一件商品被若干个用户评分
    M = [[3, 7, 4, 9, 9, 7],
         [7, 0, 5, 3, 8, 8],
         [7, 5, 5, 0, 8, 4],
         [5, 6, 8, 5, 9, 8],
         [5, 8, 8, 8, 10, 9],
         [7, 7, 0, 4, 7, 8]]
    return np.array(M)

# maxK 取最相近的几个用户评分
def baseOnUserCollaborative(M, rowIndex, colomunIndex, maxK):
    rowCount, columnCount = shape(M)
    nomalizedMat, aveMat = normalizedData(M)
    # rowIndex用户与其他用户相关性
    targetRow = nomalizedMat[rowIndex]
    fenZiMat = nomalizedMat * mat(targetRow).transpose()
    fenMuMat = nomalizedMat * mat(nomalizedMat).transpose()
    fenMuSqrtMat = zeros((rowCount, 1))
    targetRowSqrt = sqrt(targetRow * mat(targetRow).transpose())

    for row in range(0, rowCount):
        fenMuSqrtMat[row] = sqrt(fenMuMat[row, row]) * targetRowSqrt

    relationMat = fenZiMat / fenMuSqrtMat
    sortArray = []
    for row in range(0, rowCount):
        sortArray.append(sum(relationMat[row][0]))

    sortIndexArray = np.argsort(sortArray)
    sumWeight = 0
    sumRelationScore = 0
    for index in range(rowCount - maxK - 1, rowCount):
        if index >= 0 and sortIndexArray[index] != rowIndex:
            # 参与计算
            sumWeight += relationMat[sortIndexArray[index]]
            sumRelationScore += nomalizedMat[sortIndexArray[index]][colomunIndex] * relationMat[sortIndexArray[index]]
    print aveMat[rowIndex] + sumRelationScore / sumWeight


def baseOnGoodCollaborative(M, rowIndex, colomunIndex, maxK):
    calculateMat = M.T
    baseOnUserCollaborative(calculateMat, colomunIndex, rowIndex, maxK)
    return


def normalizedData(M):
    rowCount, columnCount = shape(M)
    aveMat = zeros((rowCount, 1))
    resultMat = zeros((rowCount, columnCount))
    # 计算时过滤掉0
    for row in range(0, rowCount):
        effectCoulumnCount = len(np.nonzero(M[row])[0])
        temAve = sum(M[row]) * 1.0 / effectCoulumnCount
        aveMat[row, 0] = temAve
        for column in range(0, columnCount):
            if M[row, column] != 0:
                resultMat[row, column] = M[row, column] - temAve
    return resultMat, aveMat


baseOnUserCollaborative(loadData(), 1, 1, 2)
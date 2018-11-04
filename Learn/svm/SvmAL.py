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


class cacheE:
    def __init__(self, dataMat, labelClass, C, error):
        self.X = dataMat
        self.Y = labelClass
        self.C = C
        self.error = error
        m, n = shape(dataMat)
        self.cacheE = zeros((m, 2))
        self.alpha = zeros((m, 1))
        self.b = 0


def loadDataSet(fileName):
    dataMat = []
    labelMat = []
    allMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
        allMat.append([float(lineArr[0]), float(lineArr[1]), float(lineArr[2])])
    return dataMat, labelMat, allMat


def plotMap():
    dataMat, labelMat, allMat = loadDataSet('svmTestData.txt')
    oneMat = zeros((0, 3))
    twoMat = zeros((0, 3))
    for temLine in allMat:
        if temLine[2] == 1:
            oneMat = np.row_stack((oneMat, temLine))
        else:
            twoMat = np.row_stack((twoMat, temLine))

    alpha, b = caculateSVM(40, 0.001, 0.6) # 标准算法,使用max|e1-e2|选a2,收敛更快
    # alpha, b = standardSvm(40, 0.001, 0.6) # 随机算法,随机选取a2,收敛慢

    print alpha[alpha > 0]
    print "d:", b
    w = calcWs(alpha, dataMat, labelMat)
    print w[0]
    # z= kx+b => z = -(wx+b)w1
    x = np.linspace(min(twoMat[:, 0]), max(oneMat[:, 0]))
    y = -(x * w[0, 0] + b[0, 0]) / w[1, 0]

    plt.figure(1)  # 创建图表1
    plt.plot(oneMat[:, 0], oneMat[:, 1], 'bo')
    plt.plot(twoMat[:, 0], twoMat[:, 1], 'ro')
    plt.plot(x, y)
    plt.title("图表")
    plt.show()


def caculateSVM(maxIter, error, C):
    dataMat, labelMat, allMat = loadDataSet('svmTestData.txt')
    dataMat = mat(dataMat)
    labelMat = mat(labelMat).transpose()
    # wTx+b => y  [a c]Tx+b => y
    r, c = shape(dataMat)
    alpha = zeros((r, 1))
    b = 0
    iter = 0
    while (iter < maxIter):
        pairChangeSize = 0
        for index in range(r):
            fx1 = float(multiply(alpha, labelMat).T * (dataMat * dataMat[index, :].T)) + b
            e1 = fx1 - float(labelMat[index])
            if ((e1 * labelMat[index] < -error and alpha[index] < C) or (
                                e1 * labelMat[index] > error and alpha[index] > 0)):
                oldA1 = alpha[index].copy()
                a2Row = selectJrand(index, r)
                i = index
                j = a2Row
                oldA2 = alpha[a2Row].copy()
                fx2 = float(multiply(alpha, labelMat).T * (dataMat * dataMat[a2Row, :].T)) + b
                e2 = fx2 - labelMat[a2Row]
                k12 = dataMat[index, :] * dataMat[a2Row, :].T
                k11 = dataMat[index, :] * dataMat[index, :].T
                k22 = dataMat[a2Row, :] * dataMat[a2Row, :].T

                L = 0
                H = 0
                if (labelMat[a2Row] * labelMat[index] == -1):
                    L = max(0, oldA2 - oldA1)
                    H = min(C, C + oldA2 - oldA1)
                else:
                    L = max(0, oldA2 + oldA1 - C)
                    H = min(C, oldA2 + oldA1)

                if (k11 + k22 - 2.0 * k12) <= 0:
                    continue

                a2 = oldA2 + labelMat[a2Row] * (e1 - e2) / (k11 + k22 - 2.0 * k12)

                a2 = clipAlpha(a2, H, L)
                a1 = (oldA1 * labelMat[index] + (oldA2 - a2) * labelMat[a2Row]) / labelMat[
                    index]

                if (abs(a2 - oldA2) < 0.00001): print "j not moving enough"; continue

                alpha[index] = a1
                alpha[a2Row] = a2

                b1 = -e1 - labelMat[index] * k11 * (alpha[index] - oldA1) - labelMat[a2Row] * k12 * (
                    alpha[a2Row] - oldA2) + b
                b2 = -e2 - labelMat[a2Row] * k22 * (alpha[a2Row] - oldA2) - labelMat[index] * k12 * (
                    alpha[index] - oldA1) + b

                if (0 < alpha[index]) and (C > alpha[index]):
                    b = b1
                elif (0 < alpha[a2Row]) and (C > alpha[a2Row]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                pairChangeSize += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, index, pairChangeSize)
        if (pairChangeSize == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter

    return alpha, b


def selectJrand(i, m):
    j = i  # we want to select any J not equal to i
    while (j == i):
        j = int(random.uniform(0, m))
    return j


def standardSvm(maxIter, error, C):
    dataMat, labelMat, allMat = loadDataSet('svmTestData.txt')
    dataMat = mat(dataMat)
    labelMat = mat(labelMat).transpose()
    cache = cacheE(dataMat, labelMat, C, error)
    # wTx+b => y  [a c]Tx+b => y
    r, c = shape(dataMat)
    iter = 0
    while (iter < maxIter):
        pairChangeSize = 0
        for index in range(r):
            e1 = calculateEi(index,cache)
            if ((e1 * cache.Y[index] < -error and cache.alpha[index] < C) or (
                                e1 * cache.Y[index] > error and cache.alpha[index] > 0)):
                oldA1 = cache.alpha[index].copy()

                a2Row,e2 = selectJByE(index, e1, cache)

                i = index
                j = a2Row
                oldA2 = cache.alpha[a2Row].copy()
                # fx2 = float(multiply(cache.alpha, labelMat).T * (dataMat * dataMat[a2Row, :].T)) + cache.b
                # e2 = fx2 - labelMat[a2Row]
                k12 = cache.X[index, :] * cache.X[a2Row, :].T
                k11 = cache.X[index, :] * cache.X[index, :].T
                k22 = cache.X[a2Row, :] * cache.X[a2Row, :].T
                L = 0
                H = 0
                if (cache.Y[a2Row] * cache.Y[index] == -1):
                    L = max(0, oldA2 - oldA1)
                    H = min(C, C + oldA2 - oldA1)
                else:
                    L = max(0, oldA2 + oldA1 - C)
                    H = min(C, oldA2 + oldA1)

                if (k11 + k22 - 2.0 * k12) <= 0:
                    continue

                a2 = oldA2 + cache.Y[a2Row] * (e1 - e2) / (k11 + k22 - 2.0 * k12)

                a2 = clipAlpha(a2, H, L)
                a1 = (oldA1 * cache.Y[index] + (oldA2 - a2) * cache.Y[a2Row]) / cache.Y[index]

                if (abs(a2 - oldA2) < 0.00001): print "j not moving enough"; continue

                cache.alpha[index] = a1
                updateCahce(index, cache)

                cache.alpha[a2Row] = a2
                updateCahce(a2Row, cache)

                b1 = -e1 - cache.Y[index] * k11 * (cache.alpha[index] - oldA1) - cache.Y[a2Row] * k12 * (
                    cache.alpha[a2Row] - oldA2) + cache.b
                b2 = -e2 - cache.Y[a2Row] * k22 * (cache.alpha[a2Row] - oldA2) - cache.Y[index] * k12 * (
                    cache.alpha[index] - oldA1) + cache.b

                if (0 < cache.alpha[index]) and (C > cache.alpha[index]):
                    cache.b = b1
                elif (0 < cache.alpha[a2Row]) and (C > cache.alpha[a2Row]):
                    cache.b = b2
                else:
                    cache.b = (b1 + b2) / 2.0

                pairChangeSize += 1
                print "iter: %d i:%d, pairs changed %d" % (iter, index, pairChangeSize)
        if (pairChangeSize == 0):
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter

    return cache.alpha, cache.b


def selectJByE(i, e1, cacheE):
    maxIndex = 0
    maxValue = -1
    cacheE.cacheE[i] = [1, e1]
    cacheE.cacheE[5] = [1, e1]
    validEcacheList = nonzero(cacheE.cacheE[:, 0])[0]
    if (len(validEcacheList) > 1):
        for index in range(len(validEcacheList)):
            if (validEcacheList[index] != i and abs(cacheE.cacheE[validEcacheList[index]][1] - e1) >= maxValue):
                maxValue = abs(cacheE.cacheE[validEcacheList[index]][1] - e1)
                maxIndex = validEcacheList[index]
    else:
        maxIndex = selectJrand(i, len(cacheE.cacheE))

    e2 = calculateEi(maxIndex,cacheE)
    return maxIndex,e2

def calculateEi(index,cache):
    fx1 = float(multiply(cache.alpha, cache.Y).T * (cache.X * cache.X[index, :].T)) + cache.b
    e1 = fx1 - float(cache.Y[index])
    return e1

def updateCahce(i, cacheE):
    fx1 = float(multiply(cacheE.alpha, cacheE.Y).T * (cacheE.X * cacheE.X[i, :].T)) + cacheE.b
    cacheE.cacheE[i] = fx1 - float(cacheE.Y[i])


def clipAlpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


plotMap()

# alpha, b = caculateSVM(40, 0.001, 0.6)

# b, alpha = smoSimple(0.6, 0.001, 40)
#
# print alpha[alpha > 0]
# print "d:", b

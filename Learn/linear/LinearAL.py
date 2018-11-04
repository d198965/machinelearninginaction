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


def openFile():
    fr = open("linear")
    arrayOLines = fr.readlines()
    returnMat = zeros((len(arrayOLines), 2))
    for index in range(len(arrayOLines)):
        line = arrayOLines[index]
        datas = line.strip('\n').replace(',', '').replace('\t', ' ').split(' ')
        returnMat[index][0] = int(datas[0])
        returnMat[index][1] = int(datas[1])
    returnMat.sort(axis=0)
    return returnMat


def plotMap():
    matMap = openFile()
    x = matMap[:, 0]
    y = matMap[:, 1]
    plt.figure(1)  # 创建图表1
    plt.plot(x, y, 'bo')
    plt.title("图表")
    plt.show()


def plotLinear(a, b, minX, maxX):
    matMap = openFile()
    ox = matMap[:, 0]
    oy = matMap[:, 1]

    plt.figure(1)  # 创建图表1
    x = np.linspace(minX, maxX)
    y = x * a + b
    plt.plot(x, y)
    plt.plot(ox, oy, 'bo')
    plt.title("图表")
    plt.show()


# 针对一元回归计算方法计算参数
def lsmSum():
    # ax+b=y
    # ax+b-y=0
    matMap = openFile()
    x = matMap[:, 0]
    y = matMap[:, 1]

    n = len(x) * 1.0
    sumXY = (x * y).sum()
    sumX = x.sum()
    sumY = y.sum()
    sumX2 = (x * x).sum()

    a = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX ** 2)
    b = (sumX2 * sumY - sumX * sumXY) / (n * sumX2 - sumX ** 2)

    print a, b
    plotLinear(a, b, x[0], x[len(x) - 1])


# 多元线性回归计算方法计算参数
def lsmMatrix():
    # ax+b=y
    # ax+b-y=0
    matMap = openFile()
    x = matMap[:, 0]
    d = ones((len(x), 1))
    x = np.column_stack((x, d))
    y = matMap[:, 1]
    A = np.dot(np.dot(nlg.inv(np.dot(x.T, x)), x.T), y)

    print A
    plotLinear(A[0], A[1], x[0, 0], x[len(x) - 1, 0])


# 梯度下降法
def gradFollow(A, x, y):
    while True:
        print A
        dab = np.dot(np.dot(x.T, x), A) - np.dot(x.T, y)
        print dab
        if np.dot(dab, dab.T) < 0.1:
            print A
            plotLinear(A[0], A[1], x[0, 0], x[len(x) - 1, 0])
            return
        A = A - [0.00002, 0.0001] * dab


def startGradFollow():
    # ax+b=y
    # ax+b-y=0
    matMap = openFile()
    x = matMap[:, 0]
    d = ones((len(x), 1))
    x = np.column_stack((x, d))
    y = matMap[:, 1]
    A = [0, 0]
    gradFollow(A, x, y)


startGradFollow()

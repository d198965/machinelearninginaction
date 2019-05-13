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
from plotdata import *


def calculate():
    # 从数据图标可以看出,数据分割用一条线性直线即可,所以令 hx = ax+by+c,hx是对这个点的描述
    # 令 估计初始值 令 a=1 b=1 c=1 [1,1,1]
    # https://blog.csdn.net/pakko/article/details/37878837
    mat0, mat1 = openTestFile()
    matSum = np.row_stack((mat0, mat1))

    oneMat = ones((len(matSum), 1))
    xMat = np.column_stack((matSum, oneMat))

    yMat = zeros((len(mat0), 1))
    yMat = np.row_stack((yMat, ones((len(mat1), 1))))
    B = ones((1, 3))
    B[0][0] = 1
    B[0][1] = 1
    B[0][2] = 1
    B = maxEvaluate(yMat, xMat, B, 1.0 / len(matSum))
    plotTestDataMapMap(B)
    return


def maxEvaluate(yMat, xMat, B, a, maxCount=100):
    count = 0
    while True:
        A = np.dot(xMat, B.T)
        gA = sigmod(A)
        e = gA - yMat
        dB = a * np.dot(xMat.T, e)
        if count == maxCount:
            print dB.T
            print B
            break
        B = B - dB.T
        count += 1
    return B


def maxEvaluateRandom(yMat, xMat, B, a):
    for k in range(200):
        for i in range(len(yMat)):
            A = np.dot(xMat[i], B.T)
            gA = sigmod(A)
            e = gA - yMat[i]
            dB = a * xMat[i] * e
            B = B - dB
    print B
    return B


def maxEvaluateRandomIndex(yMat, xMat, B, number=150):
    m, n = shape(xMat)
    for k in range(number):
        dataIndex = range(m)
        for i in range(len(yMat)):
            # 每次更新参数时设置动态的步长，且为保证多次迭代后对新数据仍然具有一定影响
            # 添加了固定步长0.1
            a = 0.1 / (1.0 + k + i) + 0.1
            randomIndex = int(random.uniform(0, len(dataIndex)))
            # 计算当前sigmoid函数值
            gA = sigmod(np.dot(xMat[randomIndex], B.T))
            # 计算权值更新
            # ***********************************************
            e = gA - yMat[randomIndex]
            B = B - a * e * xMat[randomIndex]
            # ***********************************************
            # 选取该样本后，将该样本下标删除，确保每次迭代时只使用一次
            del (dataIndex[randomIndex])
    return B


def calculateHorese():
    mat0, mat1 = readHorese("horseColicTraining.txt")
    matSum = np.row_stack((mat0, mat1))
    rowCount, columnCount = shape(matSum)
    oneMat = ones((rowCount, 1))
    xMat = np.column_stack((matSum, oneMat))

    yMat = zeros((len(mat0), 1))
    yMat = np.row_stack((yMat, ones((len(mat1), 1))))
    B = ones((1, columnCount + 1))
    B = maxEvaluateRandomIndex(yMat, xMat, B, 150)
    # B = maxEvaluate(yMat, xMat, B, 1.0 / (len(matSum)), 100)

    mat0, mat1 = readHorese("horseColicTest.txt")
    mat0 = np.column_stack((mat0, ones((len(mat0), 1))))
    mat1 = np.column_stack((mat1, ones((len(mat1), 1))))
    errorCount = 0
    result0 = sigmod(np.dot(mat0, B.T))
    for i in range(len(result0)):
        if result0[i][0] > 0.5:
            errorCount += 1

    print errorCount
    result1 = sigmod(np.dot(mat1, B.T))
    for i in range(len(result1)):
        if result1[i][0] < 0.5:
            errorCount += 1

    print errorCount


def readHorese(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    lineSplit = arrayOLines[0].strip('\n').replace(',', '').replace('\t', ' ').split(' ')
    column = len(lineSplit) - 1
    returnMat0 = zeros((0, column))
    returnMat1 = zeros((0, column))
    for index in range(len(arrayOLines)):
        line = arrayOLines[index]
        datas = line.strip('\n').replace(',', '').replace('\t', ' ').split(' ')
        newRow = []
        for index in range(len(datas) - 1):
            newRow.append(double(datas[index]))
        if double(datas[column]) == 0:
            returnMat0 = np.row_stack((returnMat0, newRow))
        else:
            returnMat1 = np.row_stack((returnMat1, newRow))
    return returnMat0, returnMat1


# calculate()
calculateHorese()
#

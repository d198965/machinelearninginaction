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


def openTestFile():
    # fr = open("testSet.txt")
    fr = open("svmTestData.txt")
    arrayOLines = fr.readlines()
    returnMat0 = zeros((0, 2))
    returnMat1 = zeros((0, 2))
    for index in range(len(arrayOLines)):
        line = arrayOLines[index]
        datas = line.strip('\n').replace(',', '').replace('\t', ' ').split(' ')
        newRow = []
        newRow.append(double(datas[0]))
        newRow.append(double(datas[1]))
        if int(datas[2]) != 1:
            returnMat0 = np.row_stack((returnMat0, newRow))
        else:
            returnMat1 = np.row_stack((returnMat1, newRow))
    return returnMat0, returnMat1


def plotTestDataMap():
    mat0, mat1 = openTestFile()
    print mat0
    plt.figure(1)  # 创建图表1
    plt.plot(mat0[:, 0], mat0[:, 1], 'bo')
    plt.plot(mat1[:, 0], mat1[:, 1], 'ro')
    plt.title("figer")
    plt.show()

def plotTestDataMapMap(B):
    mat0, mat1 = openTestFile()
    plt.figure(1)  # 创建图表1
    plt.plot(mat0[:, 0], mat0[:, 1], 'bo')
    plt.plot(mat1[:, 0], mat1[:, 1], 'ro')
    maxX = max(max(mat0[:, 0]),max(mat1[:, 0]))
    minX = min(min(mat0[:, 0]),min(mat1[:, 0]))
    x = np.linspace(minX, maxX)
    y = -(x * B[0][0] + B[0][2])/B[0][1]
    plt.plot(x, y)
    plt.title("figer")

    oneMat = ones((len(mat0), 1))
    Mat0 = np.column_stack((mat0, oneMat))

    oneMat = ones((len(mat1), 1))
    Mat1 = np.column_stack((mat1, oneMat))

    plt.figure(2)  # 创建图表1
    plt.plot(mat0[:, 0], sigmod(np.dot(Mat0, B.T)), 'bo')
    plt.plot(mat1[:, 0], sigmod(np.dot(Mat1, B.T)), 'ro')
    z = np.linspace(minX, maxX,100)
    plt.plot(z, logistic(z), 'b-')
    plt.title("figer2")

    plt.show()

def sigmod(inX):
    return ones((len(inX), 1.0)) / (ones((len(inX), 1.0)) + exp(-inX))

def logistic(z):
    return 1 / (1 + np.exp(-z))




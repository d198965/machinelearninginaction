#!/usr/bin/python
# -*- coding: UTF-8 -*-
from math import log
import operator
import time
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
from Learn import DataProvider
from Learn.deeplearn.bp import simpleBP as BP


def adbTest():
    labelMat, allMat = DataProvider.readMushRooms()
    halfMat,labelHalfMat,lastMat,lastHalfLabel = getRanomData(labelMat, allMat)
    m, n = shape(lastMat)

    parameters = BP.nn_model(halfMat, labelHalfMat, 40, num_iterations=4000, print_cost=True)

    predictions = BP.predict(parameters, lastMat)
    print (np.dot(1 - lastHalfLabel, predictions.T))
    print (np.dot(lastHalfLabel, 1 - predictions.T))
    print (
        'Accuracy: %f' % float(
            (np.dot(lastHalfLabel, predictions.T) + np.dot(1 - lastHalfLabel, 1 - predictions.T)) / float(
                lastHalfLabel.size) * 100) + '%')

    # errorCount = 0
    # pErrorCount = 0
    # eErrorCount = 0
    #
    # print("LP:", len(lastHalfLabel[lastHalfLabel > 0.5]))
    # print("LE:", len(lastHalfLabel[lastHalfLabel < 0.5]))
    # print (100.0 * errorCount / m, "p:", pErrorCount, "e:", eErrorCount)


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
    halfMat = np.array(halfMat).T
    lastMat = np.array(lastMat).T
    return halfMat, np.array([labelHalfMat]), lastMat, np.array([lastHalfLabel])


adbTest()

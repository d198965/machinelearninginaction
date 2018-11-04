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


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec
    # testEntry = ['love', 'my', 'dalmation']
    # testEntry = ['stupid', 'garbage']


def wordListToVector(wordList):
    wordListVector = []
    index = 0
    for line in wordList:
        for temWord in line:
            wordListVector[index] = temWord
            index += 1


def vectorData(wordList, inputWordList):
    outputVector = [0] * len(wordList)
    for inputWord in inputWordList:
        for i in range(len(wordList)):
            if inputWord == wordList[i]:
                outputVector[i] = 1
                break
    return outputVector


def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def calculatePy(vectorList):
    vectorSum = ones((1, len(vectorList[0])))
    sumValue = 0
    for index in range(len(vectorList)):
        vectorSum += vectorList[index]
        sumValue += sum(vectorList[index])
    return log(vectorSum * 1.0 / sumValue)


def calculateMultiParam():
    return


def classfiyNB(testEntry, wordList, p0, p0Vector, p1Vector):
    testVector = vectorData(wordList, testEntry)
    p0Result = math.exp(np.dot(testVector, p0Vector.T) + log(p0))
    p1Result = math.exp(np.dot(testVector, p1Vector.T) + log(1 - p0))
    if p0Result > p1Result:
        print 0
    else:
        print 1
    print p0Result, p1Result
    return


def testData():
    dataSet, classify = loadDataSet()
    vocabList = createVocabList(dataSet)
    vectorList0 = []
    vectorList1 = []

    sumP0 = 0
    for k in range(len(classify)):
        if classify[k] == 0:
            temVector = vectorData(vocabList, dataSet[k])
            vectorList0.append(temVector)
            sumP0 += 1
        else:
            temVector = vectorData(vocabList, dataSet[k])
            vectorList1.append(temVector)
    p0 = sumP0 * 1.0 / len(classify)
    p0Vector = calculatePy(vectorList0)
    p1Vector = calculatePy(vectorList1)

    testEntry = ['love', 'my', 'dalmation']

    testEntry = ['stupid', 'garbage']

    classfiyNB(testEntry, vocabList, p0, p0Vector, p1Vector)


testData()

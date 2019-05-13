#!/usr/bin/python
# -*- coding: UTF-8 -*-
from numpy import *
import numpy as np

MAX_VALUE = 20.0
MIN_VALUE = 10.0
STEP_COUNT = 50

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





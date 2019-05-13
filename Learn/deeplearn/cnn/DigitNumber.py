#!/usr/bin/python
# -*- coding: UTF-8 -*-
from multiprocessing import Pool, Manager
from math import log
import operator
from os import listdir
import matplotlib.pyplot as plt
import datetime
from numpy import *
import numpy as np
from Learn import DataProvider

# labelMat, digitData = DataProvider.digitData(False)
# 卷积(5*5)\采样(2*2)\卷积(5*5)\采样(2*2)\卷积(5*5)\->隐藏层\输出层
# 图像大小               32->28->14->10->5->(5->1*1)->(隐藏层1)->输出数值
# 节点(特征图\神经元)个数  1->6->6—>16->16->120->(隐藏层)120->10(结果)
from Learn.deeplearn.cnn.ConvObj import ConvObj

KEY_W1 = "w1"
KEY_B1 = "b1"
KEY_W2 = "w2"
KEY_B2 = "b2"
KEY_W3 = "w3"
KEY_B3 = "b3"
KEY_W4 = "w4"
KEY_B4 = "b4"
KEY_W5 = "w5"
KEY_B5 = "b5"


def update_parameters(parameters, grads, learning_rate=1.0):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters[KEY_W1]
    b1 = parameters[KEY_B1]
    W2 = parameters[KEY_W2]
    b2 = parameters[KEY_B2]
    W3 = parameters[KEY_W3]
    b3 = parameters[KEY_B3]
    W4 = parameters[KEY_W4]
    b4 = parameters[KEY_B4]
    W5 = parameters[KEY_W5]
    b5 = parameters[KEY_B5]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    dW3 = grads["dW3"]
    db3 = grads["db3"]
    dW4 = grads["dW4"]
    db4 = grads["db4"]
    dW5 = grads["dW5"]
    db5 = grads["db5"]

    # Update rule for each parameter
    W1 -= np.dot(dW1, learning_rate)
    b1 -= np.dot(db1, learning_rate)
    W2 -= np.dot(dW2, learning_rate)
    b2 -= np.dot(db2, learning_rate)
    W3 -= np.dot(dW3, learning_rate)
    b3 -= np.dot(db3, learning_rate)
    W4 -= np.dot(dW4, learning_rate)
    b4 -= np.dot(db4, learning_rate)
    W5 -= np.dot(dW5, learning_rate)
    b5 -= np.dot(db5, learning_rate)

    parameters = {KEY_W1: W1,
                  KEY_B1: b1,
                  KEY_W2: W2,
                  KEY_B2: b2,
                  KEY_W3: W3,
                  KEY_B3: b3,
                  KEY_W4: W4,
                  KEY_B4: b4,
                  KEY_W5: W5,
                  KEY_B5: b5}

    return parameters


def backward_propagation(parameters, cache, X, Y, YLabel, sumX):
    m = X.shape[0]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters[KEY_W1]
    W2 = parameters[KEY_W2]
    W3 = parameters[KEY_W3]
    W4 = parameters[KEY_W4]
    W5 = parameters[KEY_W5]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    C5 = cache["C5"]
    S4 = cache["S4"]
    C3 = cache["C3"]
    K3 = cache["K3"]
    O3 = cache["O3"]
    S2 = cache["S2"]
    C1 = cache["C1"]
    Z1 = cache["Z1"]

    # Backward propagation: calculate dW1, db1, dW2, db2.

    # 计算输出-隐藏层反向参数
    dZ7 = (A2 - YLabel.T)
    # dGj = dZ7 * A2 * (1 - A2)
    # dGj = dZ7
    dW5 = np.dot(dZ7, A1.T) / m
    db5 = np.sum(dZ7, axis=1, keepdims=True) / m

    # 计算隐藏层-C5反向参数 120*84
    dZ6 = np.dot(W5.T, dZ7) * A1 * (1 - A1)
    dW4 = np.dot(dZ6, C5.T) / m  # 84*120
    db4 = np.sum(dZ6, axis=1, keepdims=True) / m

    # 计算C5-S4的反向参数 dw3:5*5*120  : 当前层为卷积层，与其相连的上一层相关核权重及偏置
    dZ5 = np.dot(W4.T, dZ6)  # 120*n
    dW3 = []  # 120*16*5*5
    dB3 = []  # 120
    for index in range(0, len(dZ5)):
        temDW3 = []
        for k in range(0, len(S4[0])):
            ddDW3 = []
            for j in range(0, len(S4)):
                temSample = S4[j][k]
                if len(ddDW3) == 0:
                    ddDW3 = temSample * dZ5[index][j]
                else:
                    ddDW3 += temSample * dZ5[index][j]
            temDW3.append(ddDW3 / m)
        temDB3 = sum(dZ5[index] / m)
        dW3.append(temDW3)
        dB3.append(np.array([temDB3]))

    # 计算S4-C3的反向参数 dS4:16*5*5  当前层为下采样(pooling)层且下一层为卷积层时反向传播的原理如下
    # 利用卷积核(W3(120*5*5))和卷积层(C5)误差敏感项(dZ5:120*n)计算
    # S4输出为C5(120*1*1)
    # 获取S4(16 * 5 * 5)层卷积层误差敏感项dS4(16*5*5)
    dS4 = []
    for index in range(0, len(S4[0])):  # 一共16层
        temDS4 = np.zeros((5, 5))
        for k in range(0, len(dZ5)):  # 每个元素被120次卷积运算
            temDS4 += sum(dZ5[k] / m) * FZ(W3[k][index])
        dS4.append(temDS4 / len(dZ5))

    # 计算C3-S2的反向参数
    # 计算dC3(16*10*10)
    dC3 = []  # 扩充dS4
    dEC3 = []
    for index in range(0, len(S4[0])):  # 一共16层
        temDC3 = np.zeros((10, 10))
        for i in range(0, 10):
            for j in range(0, 10):
                temDC3[i][j] = dS4[index][i / 2][j / 2] / 4
        dC3.append(temDC3)
        # dC3 ->10*10 每一边补4圈0 -> 18*18
        temEDC3 = np.zeros((18, 18))
        for k in range(0, 10):
            for j in range(0, 10):
                temEDC3[k + 18 - 14][j + 18 - 14] = temDC3[k][j]
        dEC3.append(temEDC3)

    # 求dW2(5*5*16),需要先求S2个层的和
    dW2 = []
    dB2 = []
    for j in range(0, len(K3[0])):
        temOrigin = np.zeros((14, 14))
        for index in range(0, len(K3)):  # 一共16层
            temOrigin += K3[index][j]
        dW2.append(conv(dC3[j], 0, temOrigin / len(K3)))
        dB2.append(np.array([sum(dC3[j])]))

    dS2 = []  # 6*14*14
    for i in range(0, len(O3)):
        # 第i层被用于C2卷积的哪些层
        temC3List = O3[i]
        temDS2 = np.zeros((14, 14))
        for j in range(0, len(temC3List)):
            temC3Index = temC3List[j]
            temDS2 += conv(FZ(W2[temC3Index]), 0, dEC3[temC3Index])
        dS2.append(temDS2 / len(temC3List))

    # 计算S2-C1的反向参数
    dC1 = []
    for i in range(0, len(O3)):
        temOrigin = np.zeros((28, 28))
        for index in range(0, 28):
            for j in range(0, 28):
                temOrigin[index][j] = dS2[i][index / 2][j / 2] / 4
        dC1.append(temOrigin)

    # 计算C1-输入的反向参数
    dW1 = []
    dB1 = []
    for n in range(len(dC1)):
        dW1.append(conv(dC1[n], 0, sumX))
        dB1.append(np.array([sum(dC1[n])]))

    # print "dW1",dW1[2]
    # print "dS4",dS4
    # print "dC3",dC3
    # print "dS2",dS2[0]
    # print "dW3", dW3[3][0]
    # print "dW2", dW2[2]
    # print "dW1", dW1[2]
    # print "dW4", dW4
    # print "W4", W4
    # print "b4",b4
    # print "W5", W5
    # print "Z1", Z1
    # print "C5", C5
    # print "dW5",dW5[3]

    grads = {
        "db1": dB1,
        "dW1": dW1,
        "db2": dB2,
        "dW2": dW2,
        "db3": dB3,
        "dW3": dW3,
        "dW4": dW4,
        "db4": db4,
        "dW5": dW5,
        "db5": db5}

    return grads


def unConv(dz, input, convSize):
    dW = np.zeros((convSize, convSize))
    for i in range(0, len(dz)):
        temDz = dz[i]  # 卷积层的矩阵误差
        sumSample = []
        for j in range(0, len(input[i])):
            if len(sumSample) == 0:
                sumSample = input[i][j]
            else:
                sumSample += input[i][j]
        sumSample /= len(input[i])
        dW += conv(temDz, 0, sumSample)

    dW /= len(dz)
    db = np.sum(dz)

    return dW, db


def FZ(mat):
    return np.array(fz(list(map(fz, mat))))


def fz(a):
    return a[::-1]


def convWork(parameters, queueIndex, queue, X, firtstConvCount, secondConvCount, thirdConvCount, forthConvCount,
             sampleSize):
    W1 = parameters[KEY_W1]
    b1 = parameters[KEY_B1]
    W2 = parameters[KEY_W2]
    b2 = parameters[KEY_B2]
    W3 = parameters[KEY_W3]
    b3 = parameters[KEY_B3]

    temC1 = []
    temS2 = []
    temC3 = []
    temK3 = []  # 16*5*5
    temS4 = []
    temC5 = []
    O3 = {}

    # 卷积(5*5)
    for covIndex in range(0, firtstConvCount):
        temC1.append(conv(W1[covIndex], b1[covIndex], X))

    # 采样(2*2)
    for covIndex in range(0, firtstConvCount):
        temS2.append(aveSample(sampleSize, temC1[covIndex]))

    # 卷积(5*5)
    # 先生成每个卷积所用的采样层 再进行卷积计算
    # (LeNet-5 会对每一层做配置一个卷积核,这里做了简化,每一个组合配一个卷积核,获取均值)
    startCovSize = 3
    for simpleSize in range(startCovSize, firtstConvCount):
        for index in range(0, firtstConvCount):
            startIndex = index
            sumSample = []
            temCount = 0
            for sampleIndex in range(0, simpleSize):
                index = startIndex + 1
                startIndex = index
                temCount += 1
                if index >= firtstConvCount:
                    index = index - firtstConvCount
                if len(sumSample) == 0:
                    sumSample = temS2[index]
                else:
                    sumSample += temS2[index]

                # 记录当前层被哪一层处理了
                temList = []
                if (O3.has_key(index)):
                    temList = O3.get(index)
                else:
                    O3[index] = temList
                temList.append(len(temK3))  # 记录输出层索引

            sumSample /= temCount
            temK3.append(sumSample)
            tem = conv(W2[len(temC3)], b2[len(temC3)], sumSample)
            temC3.append(tem)
            if len(temC3) == secondConvCount:
                break
        if len(temC3) == secondConvCount:
            break

    # 采样(2*2)
    for index in range(0, secondConvCount):
        temS4.append(aveSample(sampleSize, temC3[index]))

    # 卷积(5*5)
    for index in range(0, thirdConvCount):
        oneLeaf = []
        for i in range(0, len(temS4)):
            if len(oneLeaf) == 0:
                oneLeaf = conv(W3[index][i], b3[index], temS4[i])
            else:
                oneLeaf += conv(W3[index][i], b3[index], temS4[i])
        temC5.append(oneLeaf[0][0][0] / len(temS4))

    result = []
    result.append(temC1)
    result.append(temS2)
    result.append(temC3)
    result.append(temK3)
    result.append(temS4)
    result.append(temC5)
    result.append(O3)
    result.append(queueIndex)
    queue.put(result)


constFenM = 10
def forward_propagation(X, firtstConvCount, secondConvCount, thirdConvCount, forthConvCount, sampleSize, parameters):
    # Retrieve each parameter from the dictionary "parameters"
    W4 = parameters[KEY_W4]
    b4 = parameters[KEY_B4]
    W5 = parameters[KEY_W5]
    b5 = parameters[KEY_B5]
    # Implement Forward Propagation to calculate A2 (probabilities)
    C1 = []
    S2 = []
    C3 = []
    S4 = []
    C5 = []
    K3 = []
    O3 = {}

    pool = Pool(4)
    queue = Manager().Queue()
    for xIndex in range(0, len(X)):
        pool.apply_async(convWork, args=(
            parameters, xIndex, queue, X[xIndex], firtstConvCount, secondConvCount, thirdConvCount, forthConvCount,
            sampleSize))
    pool.close()
    pool.join()

    temCount = 0
    while not queue.empty():
        temItem = queue.get()
        C1.insert(temItem[7], temItem[0])
        S2.insert(temItem[7], temItem[1])
        C3.insert(temItem[7], temItem[2])
        K3.insert(temItem[7], temItem[3])
        S4.insert(temItem[7], temItem[4])
        C5.insert(temItem[7], temItem[5])
        O3 = temItem[6]
        temCount += 1

    C5T = array(C5).T
    C5T = C5T / constFenM

    # for index in range(len(C5)):
    #     C5[index] = 10 * C5[index] / maxC5
    # minC5 = abs(min(C5))
    # maxC5 = abs(max(C5))
    # if minC5 > maxC5:
    #     maxC5 = minC5

    # 数值处理
    # print "B3",parameters[KEY_B3]
    # print "W3",parameters[KEY_W3]
    # print "C5",C5[0]
    # 激活
    Z1 = np.dot(W4, C5T) + b4
    A1 = sigmoid(Z1)
    # print "A1", A1[0]
    Z2 = np.dot(W5, A1) + b5
    A2 = softMax(Z2)

    # 输出层 激活
    cache = {
        "C1": C1,
        "S2": S2,
        "C3": C3,
        "K3": K3,
        "O3": O3,
        "S4": S4,
        "C5": C5T,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2}

    return A2, cache


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    # if inX>=0:
    #     return 1.0/(1+exp(-inX))
    # else:
    # s = exp(x)/(1+exp(x))
    return s


def softMax(values):
    temResult = []
    for i in range(0, values.shape[1]):
        scores_shift = values[:, i] - np.max(values[:, i])
        softmax_output = np.exp(scores_shift) / np.sum(np.exp(scores_shift))
        temResult.append(softmax_output)
    return array(temResult).T


def softMaxLoss(values, trueIndex):
    result = np.exp(values[trueIndex])
    return -result + log(np.sum(np.exp(values)))


def compute_cost(A2, Y, parameters):
    m = shape(Y)[0]  # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - np.sum(logprobs) / m

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def conv(W, b, originData):
    column = shape(originData)[1]
    row = shape(originData)[0]
    W = array(W)
    wShape = shape(W)
    if (len(wShape) == 0):
        convSize = 1
    elif (len(wShape) == 1):
        convSize = wShape[0]
    else:
        convSize = wShape[1]
    result = []

    for rowIndex in range(0, row - convSize + 1):
        temRow = []
        for colIndex in range(0, column - convSize + 1):
            temRangeArray = originData[rowIndex:rowIndex + convSize, colIndex:colIndex + convSize]
            temValue = sum(temRangeArray * W) + b
            temRow.append(temValue)
        result.append(temRow)

    # pool = Pool()
    # queue = Manager().Queue()
    # for rowIndex in range(0, row - convSize + 1):
    #     temRow = []
    #     result.append(temRow)
    #     for colIndex in range(0, column - convSize + 1):
    #         temRow.append(0)
    #         temRangeArray = originData[rowIndex:rowIndex + convSize, colIndex:colIndex + convSize]
    #         pool.apply_async(convMulWork, args=(
    #             parameters, queue, rowIndex, colIndex, W, b, temRangeArray))
    # pool.close()
    # pool.join()
    #
    # while not queue.empty():
    #     temItem = queue.get()
    #     result[temItem.i][temItem.j] = temItem.value

    return array(result)


# simpleSize * simpleSize
def maxSample(simpleSize, originData):
    result = []
    column = shape(originData)[1] / simpleSize
    row = shape(originData)[0] / simpleSize
    for rowIndex in range(0, row):
        temRow = []
        for colIndex in range(0, column):
            temRangeArray = originData[rowIndex * simpleSize:rowIndex * simpleSize + simpleSize,
                            colIndex * simpleSize:colIndex * simpleSize + simpleSize]
            temValue = np.max(temRangeArray)
            temRow.append(temValue)
        result.append(temRow)
    return array(result)


def aveSample(simpleSize, originData):
    result = []
    column = shape(originData)[1] / simpleSize
    row = shape(originData)[0] / simpleSize
    for rowIndex in range(0, row):
        temRow = []
        for colIndex in range(0, column):
            temRangeArray = originData[rowIndex * simpleSize:rowIndex * simpleSize + simpleSize,
                            colIndex * simpleSize:colIndex * simpleSize + simpleSize]
            temValue = sum(temRangeArray) / (simpleSize * simpleSize)
            temRow.append(temValue)
        result.append(temRow)
    return array(result)


def nn_model(X, Y, YLabel, convSize, firtstConvCount, secondConvCount, thirdConvCount, forthConvCount, simpleSize,
             num_iterations=10000,
             print_cost=False):
    np.random.seed(3)
    n_y = 10

    parameters = initialize_parameters(convSize, firtstConvCount, secondConvCount, thirdConvCount, forthConvCount,
                                       simpleSize, n_y)
    W1 = parameters[KEY_W1]
    b1 = parameters[KEY_B1]
    W2 = parameters[KEY_W2]
    b2 = parameters[KEY_B2]
    W3 = parameters[KEY_W3]
    b3 = parameters[KEY_B3]
    W4 = parameters[KEY_W4]
    b4 = parameters[KEY_B4]
    W5 = parameters[KEY_W5]
    b5 = parameters[KEY_B5]

    sumX = np.zeros((shape(X)[1], shape(X)[1]))
    for n in range(0, len(X)):
        sumX += X[n]
    sumX /= (len(X) * 1.0)
    for i in range(0, num_iterations):
        startTime = datetime.datetime.now()
        print "count:", i
        A2, cache = forward_propagation(X, firtstConvCount, secondConvCount, thirdConvCount,
                                        forthConvCount,
                                        simpleSize, parameters)
        print "forward_propagation:", datetime.datetime.now() - startTime
        startTime2 = datetime.datetime.now()
        grads = backward_propagation(parameters, cache, X, Y, YLabel, sumX)
        print "backward_propagation:", datetime.datetime.now() - startTime2
        parameters = update_parameters(parameters, grads)

        if i % 10 == 0 or i == 98 or i == 99:
            predictions = predict(X, Y, YLabel, 5, 6, 16, 120, 84, 2, parameters)
            diff = ((Y - predictions) == 0)
            print predictions
            rightCount = np.dot(np.array([1] * len(Y)), diff)
            print rightCount
            print (
                'Accuracy: %d' % (float(rightCount) / float(Y.size) * 100) + '%')

        print "update_parameters:", datetime.datetime.now() - startTime
        # Print the cost every 1000 iterations
        # if print_cost and i % 1000 == 0:
        #     cost = compute_cost(A2, Y, parameters)
        #     print ("Cost after iteration %i: %f" % (i, cost))

    return parameters


def initialize_parameters(convSize, firtstConvCount, secondConvCount, thirdConvCount, forthConvCount, simpleSize, n_y):
    np.random.seed(2)
    # w=5*5*6 b=1*6
    w1 = []
    for i in range(0, firtstConvCount):
        w1.append(np.random.randn(convSize, convSize))
    b1 = np.zeros((firtstConvCount, 1))

    # w=5*5*16 b=1*16
    w2 = []
    for i in range(0, secondConvCount):
        w2.append(np.random.randn(convSize, convSize))
    b2 = np.zeros((secondConvCount, 1))

    # w=5*5*120 b=1*120
    w3 = []
    for i in range(0, thirdConvCount):
        temW3 = []
        for j in range(0, secondConvCount):
            temW3.append(np.random.randn(convSize, convSize))
        w3.append(temW3)
    b3 = np.zeros((thirdConvCount, 1))

    # w=120*84 b=84
    w4 = np.random.randn(forthConvCount, thirdConvCount) * 0.01
    b4 = np.zeros((forthConvCount, 1))

    # w=84*10 b=10
    w5 = np.random.randn(n_y, forthConvCount) * 0.01
    b5 = np.zeros((n_y, 1))
    # --- 激活函数
    parameters = {KEY_W1: w1,
                  KEY_B1: b1,
                  KEY_W2: w2,
                  KEY_B2: b2,
                  KEY_W3: w3,
                  KEY_B3: b3,
                  KEY_W4: w4,
                  KEY_B4: b4,
                  KEY_W5: w5,
                  KEY_B5: b5}

    return parameters


def predict(X, Y, YLabel, convSize, firtstConvCount, secondConvCount, thirdConvCount, forthConvCount, simpleSize,
            parameters):
    A2, cache = forward_propagation(X, firtstConvCount, secondConvCount, thirdConvCount,
                                    forthConvCount,
                                    simpleSize, parameters)
    result = A2.T
    maxList = np.max(result, axis=1)
    predictions = []
    for index in range(0, len(result)):
        predictions.append(np.where(result[index] == maxList[index])[0][0])
    return predictions


def layer_sizes(X, Y):
    n_y = Y.shape[1]  # size of output layer
    return (n_y)


def writeParameters(parameters):
    f = open('parameters.txt', 'w')  # 清空文件内容再写
    keys = parameters.keys()
    for key in keys:
        value = parameters[key]
        f.write(key)
        f.write('\n')
        f.write(value)
        f.write('\n')
    f.close()


def readParameters():
    f = open('parameters.txt', 'r')  # 清空文件内容再写
    parameters = {}
    while True:
        key = f.readline()
        if key == None or key == "":
            break


## 计算入口代码
# Y, X, YLabel = DataProvider.digitData(True)
# parameters = nn_model(X, Y, YLabel, 5, 6, 16, 120, 84, 2, num_iterations=80, print_cost=True)
#
# # 保存 parameters
# # writeParameters(parameters)
#
# Y, X, YLabel = DataProvider.digitData(False)
# predictions = predict(X, Y, YLabel, 5, 6, 16, 120, 84, 2, parameters)
#
# diff = ((Y - predictions) == 0)
# print predictions
# rightCount = np.dot(np.array([1] * len(Y)), diff)
# print rightCount
# print (
#     'Accuracy: %d' % (float(rightCount) / float(Y.size) * 100) + '%')

# 将错误结果文件打印出来
Y, X, YLabel = DataProvider.digitData(False)
DataProvider.printWrongDigitClassFile("/Users/zdh/Develepment/Game/machinelearninginaction/Learn/deeplearn/cnn/result")
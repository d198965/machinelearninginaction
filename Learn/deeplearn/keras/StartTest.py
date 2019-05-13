#!/usr/bin/python
# -*- coding: UTF-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import matplotlib.pyplot as plt
from Learn.deeplearn.bp.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def loadLineData():
    """

    :rtype: object
    """
    fr = open("testSet.txt")
    # fr = open("svmTestData.txt")
    arrayOLines = fr.readlines()
    returnMat = []
    labelMat = []
    for index in range(len(arrayOLines)):
        line = arrayOLines[index]
        datas = line.strip('\n').replace(',', '').replace('\t', ' ').split(' ')
        temData = []
        temData.append(float(datas[0]))
        temData.append(float(datas[1]))
        returnMat.append(temData)
        if int(datas[2]) != 1:
            labelMat.append(0)
        else:
            labelMat.append(1)
    return np.array(returnMat), np.array(labelMat)


model = Sequential()

model.add(Dense(units=4, input_dim=2))
model.add(Activation("relu"))
model.add(Dense(units=1))
model.add(Activation("tanh"))
model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])

X, Y = loadLineData()

# # 创建数据集
# X = np.linspace(-1, 1, 200)
# np.random.shuffle(X)    # 将数据集随机化
# Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, )) # 假设我们真实模型为：Y=0.5X+2
# # 绘制数据集plt.scatter(X, Y)
# plt.show()
#
# X_train, Y_train = X[:160], Y[:160]     # 把前160个数据放到训练集
# X_test, Y_test = X[160:], Y[160:]       # 把后40个点放到测试集

model.fit(X, Y, epochs=500, batch_size=32)
cost = model.evaluate(X, Y, batch_size=128)

print('test cost:', cost)

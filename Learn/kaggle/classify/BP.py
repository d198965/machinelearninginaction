# encoding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Learn.kaggle.classify import DataProvider


def add_layer(input, in_size, out_size, activation_function=None):
    weight = tf.Variable(tf.random_normal([in_size, out_size]) * 0.1)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    W_mul_x_plus_b = tf.matmul(input, weight) + biases
    if activation_function == None:
        return W_mul_x_plus_b
    else:
        return activation_function(W_mul_x_plus_b)


def getYLabel(y_data):
    yLabel = []
    leftCount = 0
    for item in y_data:
        if item == 0:
            yLabel.append([1, 0])
            leftCount += 1
        else:
            yLabel.append([0, 1])
    return yLabel, leftCount


def getYResult(yLabel):
    yResult = []
    for item in yLabel:
        if item[0] >= item[1]:
            yResult.append(0)
        else:
            yResult.append(1)
    return yResult

def getPredictProb(yLabel):
    yResult = []
    for item in yLabel:
        yResult.append(item[1])
    return yResult

def compareToHalfLoop(index, compareValues, input):
    return index < tf.size(compareValues)



# 创建一个具有输入层，隐藏层，输出层的三层神经网络，神经元个数分别为1，10，1
# x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 创建输入数据  np.newaxis分别是在列(第二维)上增加维度，原先是（300，）变为（300，1）
# noise = np.random.normal(0, 0.05, x_data.shape)
# y_data = np.square(x_data) + 1 + noise  # 创建输入数据对应的输出

trainData, testData = DataProvider.readModelTrainData()
target1Count = DataProvider.readOneCount(trainData)
zeroData = DataProvider.readZeroOriginal(trainData)
fields = DataProvider.getEffectField()
x_data, y_data = DataProvider.getArrayData(trainData, fields)
DataProvider.standardValues(x_data)
label_data, leftCount = getYLabel(y_data)
weightRight = leftCount * 1.0 / (len(y_data) - leftCount)

x_test, y_test = DataProvider.getArrayData(testData, fields)
DataProvider.standardValues(x_test)
label_test, d = getYLabel(y_test)

# testData = DataProvider.readTestData()
# testValues = DataProvider.getValuesData(testData,DataProvider.getEffectField(True))

# 定义输入数据
xs = tf.placeholder(tf.float32, [None, len(fields)])
ys = tf.placeholder(tf.float32, [None, 2])

# 定义一个隐藏层
hidden_layer1 = add_layer(xs, len(fields), 20, activation_function=tf.nn.leaky_relu)

hidden_layer2 = add_layer(hidden_layer1, 20, 4, activation_function=tf.nn.leaky_relu)

# 定义一个输出层
prediction = add_layer(hidden_layer2, 4, 2, activation_function=tf.nn.softmax)

# 求解神经网络参数
# 1.定义损失函数
# loss = tf.reduce_mean(tf.reduce_sum(-ys * tf.log(prediction) - (1 - ys) * tf.log(1 - prediction)))

weightRight = 8.0
coe = tf.constant([1.0, weightRight])
y_coe = ys * coe

lossPrediction = tf.abs(ys - prediction)  # ys - prediction

# outputWeight = ((lossPrediction - 0.5) / tf.maximum(tf.abs(lossPrediction - 0.5), 0.00001) + 9) / 8

# extraPrediction = tf.divide(prediction, outputWeight)

loss = -tf.reduce_mean(y_coe * tf.log(prediction))

# loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=prediction, targets=ys, pos_weight=0.11))

# 2.定义训练过程
train_step = tf.train.AdamOptimizer(0.1).minimize(loss)
# train_step = tf.train.AdamOptimizer().minimize(loss)
# AdamOptimizer()
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()
sess.run(init)

# 3.进行训练
maxCount = 1000
for i in range(maxCount):
    sess.run(train_step, feed_dict={xs: x_data, ys: label_data})
    if i % 100 == 0 and i != 0:
        # 计算预测值
        prediction_value = sess.run(prediction, feed_dict={xs: x_test})
        # print(prediction_value)

        # print label_test
        #
        # effect = sess.run(outputWeight, feed_dict={xs: x_test, ys: label_test})
        # print(effect)
        #
        # lossEffect = sess.run(extraPrediction, feed_dict={xs: x_test, ys: label_test})
        # print(lossEffect)

        DataProvider.evaluate(getPredictProb(prediction_value), y_test)

        # if i%100 == 0:
        #     lossValue = sess.run(loss, feed_dict={xs: x_data, ys: label_data})
        #     print("loss:", lossValue)
        #     save_path = saver.save(sess, "models/pretrained.ckpt", global_step=i)
        #     print("saved to %s" % save_path)
        #     if lossValue < 0.1:
        #         break  # 关闭sess

# testData = DataProvider.readTestData()
# testValues = DataProvider.getValuesData(testData, DataProvider.getEffectField())
# DataProvider.standardValues(testValues)
# prediction_value = sess.run(prediction, feed_dict={xs: testValues})
# result = getYResult(prediction_value)
# DataProvider.outputResult2(result)

sess.close()

# encoding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Learn.kaggle.classify import DataProvider


# ----Weight Initialization---#
# One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


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


trainData, testData = DataProvider.readModelTrainData()
target1Count = DataProvider.readOneCount(trainData)
zeroData = DataProvider.readZeroOriginal(trainData)
fields = DataProvider.getCNNField()
x_data, y_data = DataProvider.getArrayData(trainData, fields)
DataProvider.standardValues(x_data)
label_data, leftCount = getYLabel(y_data)
weightRight = leftCount * 1.0 / (len(y_data) - leftCount)

x_test, y_test = DataProvider.getArrayData(testData, fields)
DataProvider.standardValues(x_test)
label_test, d = getYLabel(y_test)

x = tf.placeholder(tf.float32, [None, 196])
y_ = tf.placeholder(tf.float32, [None, 2])
x_image = tf.reshape(x, [-1, 14, 14, 1])

# ----first convolution layer----#
# he convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32].
# The first two dimensions are the patch size,
# the next is the number of input channels, and the last is the number of output channels.
W_conv1 = weight_variable([3, 3, 1, 16])
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 16, 32])
b_conv2 = bias_variable([32])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([2, 2, 32, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

W_fc1 = weight_variable([64, 128])
b_fc1 = bias_variable([128])

h_conv3_flat = tf.reshape(h_conv3, [-1, 64])
h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([128, 2])
b_fc2 = bias_variable([2])
prediction = tf.nn.softmax(tf.matmul(h_fc1_dropout, W_fc2) + b_fc2)

# ------train and evaluate----#
coe = tf.constant([1.0, 5.0])
y_coe = y_ * coe
cross_entropy = -tf.reduce_mean(y_coe * tf.log(prediction))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(300):
        train_step.run(feed_dict={x: x_data, y_: label_data, keep_prob: 0.5})
        if i % 10 == 0:
            prediction_value = sess.run(prediction, feed_dict={x: x_test, keep_prob: 0.5})
            DataProvider.evaluate(getPredictProb(prediction_value), y_test)

sess.close()

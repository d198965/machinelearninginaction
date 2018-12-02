#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import zeros
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


def loadData():
    np.random.seed(1)  # set a seed so that the results are consistent
    X, Y = load_planar_dataset()
    # plt.figure(1)
    # plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    #
    # shape_X = X.shape
    # shape_Y = Y.shape
    # m = shape_X[1]
    # # Train the logistic regression classifier
    #
    # clf = sklearn.linear_model.LogisticRegressionCV();
    # clf.fit(X.T, Y.T);
    #
    # # Plot the decision boundary for logistic regression
    # plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    # plt.title("Logistic Regression")
    #
    # # Print accuracy
    # LR_predictions = clf.predict(X.T)
    # print ('Accuracy of logistic regression: %d ' % float(
    # (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) +
    #    '% ' + "(percentage of correctly labelled datapoints)")
    # plt.show()

    return X, Y


def loadLineData():
    # fr = open("testSet.txt")
    fr = open("svmTestData.txt")
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
            labelMat.append([0])
        else:
            labelMat.append([1])
    return np.array(returnMat).T, np.array(labelMat).T


def readHorese(fileName):
    fr = open(fileName)
    arrayOLines = fr.readlines()
    lineSplit = arrayOLines[0].strip('\n').replace(',', '').replace('\t', ' ').split(' ')
    column = len(lineSplit) - 1
    returnMat = []
    labelMat = []
    for index in range(len(arrayOLines)):
        line = arrayOLines[index]
        datas = line.strip('\n').replace(',', '').replace('\t', ' ').split(' ')
        newRow = []
        for index in range(len(datas) - 1):
            newRow.append(float(datas[index]))
        returnMat.append(newRow)
        if float(datas[column]) == 0:
            labelMat.append([0])
        else:
            labelMat.append([1])

    return np.array(returnMat).T, np.array(labelMat).T


def layer_sizes(X, Y):
    n_x = X.shape[0]  # size of input layer
    n_h = 4
    n_y = Y.shape[0]  # size of output layer

    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # 双曲正切函数 A1的导数=1-np.power(A1,2)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def forward_propagation1(X, parameters):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = - np.sum(logprobs) / m

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost


def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = (A2 - Y)  # * A2 * (1 - A2)
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # 双曲正切函数np.tanh的导数
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def backward_propagation1(parameters, cache, X, Y):
    m = X.shape[1]

    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dGj = dZ2 * A2 * (1 - A2)
    dW2 = np.dot(dGj, A1.T) / m
    db2 = np.sum(dGj, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dGj) * A1 * (1 - A1)
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A2, cache = forward_propagation1(X, parameters)

        grads = backward_propagation1(parameters, cache, X, Y)

        parameters = update_parameters(parameters, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            cost = compute_cost(A2, Y, parameters)
            print ("Cost after iteration %i: %f" % (i, cost))

    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation1(X, parameters)
    predictions = (A2 > 0.5)  # A2矩阵中大于0.5的元素会被转为True,否则转为False

    return predictions


X, Y = loadLineData()
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# X, Y = loadData()
# parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# X, Y = readHorese('horseColicTraining.txt')
# parameters = nn_model(X, Y, n_h=30, num_iterations=10000, print_cost=True)
# X, Y = readHorese('horseColicTest.txt')

# Plot the decision boundary
# loadLineData() 和 loadData()可用
plt.figure(1)
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# Print accuracy
predictions = predict(parameters, X)
print (np.dot(1 - Y, predictions.T))
print (np.dot(Y, 1 - predictions.T))
print (
'Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')

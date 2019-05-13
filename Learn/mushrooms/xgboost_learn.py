import numpy as np
from Learn import DataProvider
import xgboost as xgb
import operator
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, roc_curve, precision_score, roc_auc_score, \
    f1_score


def logRession(trainMat, trainLabelMat, lastMat, lastHalfLabel):
    model_LR = LogisticRegression(solver="lbfgs")
    model_LR.fit(trainMat, trainLabelMat)
    predicted = model_LR.predict(lastMat)
    predicted_proba = model_LR.predict(lastMat)
    model_LR.score(lastMat, lastHalfLabel)
    print("Accuracy is: " + str(model_LR.score(lastMat, lastHalfLabel)))
    print("Precision score is: " + str(round(precision_score(lastHalfLabel, predicted), 3)))
    evaluate(lastHalfLabel, predicted)


def xgboosttest_linear(trainMat, trainLabelMat, lastMat, lastHalfLabel):
    xgbparams = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }
    plst = xgbparams.items()
    dtrain = xgb.DMatrix(trainMat, trainLabelMat)
    model_XGB = xgb.train(plst, dtrain, 20)
    # model_XGB.fit(trainMat,trainLabelMat)
    dtest = xgb.DMatrix(lastMat)
    predicted = model_XGB.predict(dtest)
    # print("Precision score is: " + str(round(precision_score(lastHalfLabel, predicted), 3)))
    evaluate(lastHalfLabel, predicted)

def xgboosttest_softmax(trainMat, trainLabelMat, lastMat, lastHalfLabel):
    xgbparams = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',
        'num_class': 2,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }
    plst = xgbparams.items()
    dtrain = xgb.DMatrix(trainMat, trainLabelMat)
    model_XGB = xgb.train(plst, dtrain, 20)
    # model_XGB.fit(trainMat,trainLabelMat)
    dtest = xgb.DMatrix(lastMat)
    predicted = model_XGB.predict(dtest)
    print("Precision score is: " + str(round(precision_score(lastHalfLabel, predicted), 3)))
    evaluate(lastHalfLabel, predicted)


def xgboosttest_logistic(trainMat, trainLabelMat, lastMat, lastHalfLabel):
    xgbparams = {
        'booster': 'gbtree',
        'objective': 'binary:logistic',
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }
    plst = xgbparams.items()
    dtrain = xgb.DMatrix(trainMat, trainLabelMat)
    model_XGB = xgb.train(plst, dtrain, 20)
    # model_XGB.fit(trainMat,trainLabelMat)
    dtest = xgb.DMatrix(lastMat)
    predicted = model_XGB.predict(dtest)
    # print("Precision score is: " + str(round(precision_score(lastHalfLabel, predicted), 3)))
    evaluate(lastHalfLabel, predicted)

    model = XGBClassifier()
    model.fit(trainMat, trainLabelMat)

    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()



def xgboosttest_softprob(trainMat, trainLabelMat, lastMat, lastHalfLabel):
    xgbparams = {
        'booster': 'gbtree',
        'objective': 'multi:softprob',
        'num_class': 2,
        'gamma': 0.1,
        'max_depth': 8,
        'lambda': 2,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 3,
        'silent': 1,
        'eta': 0.1,
        'seed': 1000,
        'nthread': 4,
    }
    plst = xgbparams.items()
    dtrain = xgb.DMatrix(trainMat, trainLabelMat)
    model_XGB = xgb.train(plst, dtrain, 20)
    # model_XGB.fit(trainMat,trainLabelMat)
    dtest = xgb.DMatrix(lastMat)
    predicted = model_XGB.predict(dtest)
    dealPredict = []
    for item in predicted:
        dealPredict.append(1 - item[0])
    # print("Precision score is: " + str(round(precision_score(lastHalfLabel, dealPredict),3)))
    evaluate(lastHalfLabel, dealPredict)


def evaluate(lastHalfLabel, resultClassify):
    errorCount = 0
    pErrorCount = 0
    eErrorCount = 0
    for index in range(len(lastHalfLabel)):
        if (lastHalfLabel[index] - 0.5) * (
                    resultClassify[index] - 0.5) < 0:
            if (lastHalfLabel[index] == 0):
                eErrorCount += 1
            else:
                pErrorCount += 1
            errorCount += 1
    print("LP:", len(lastHalfLabel[lastHalfLabel > 0.5]))
    print("LE:", len(lastHalfLabel[lastHalfLabel < 0.5]))
    print (100.0 * (1 - float(errorCount) / len(lastHalfLabel)), "p:", pErrorCount, "e:", eErrorCount)


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
    halfMat = np.array(halfMat)
    lastMat = np.array(lastMat)
    return halfMat, np.array(labelHalfMat), lastMat, np.array(lastHalfLabel)


labelMat, allMat = DataProvider.readMushRooms()

trainMat, trainLabelMat, lastMat, lastHalfLabel = getRanomData(labelMat, allMat)
# xgboosttest_softmax(trainMat,trainLabelMat,lastMat,lastHalfLabel)
# xgboosttest_softprob(trainMat,trainLabelMat,lastMat,lastHalfLabel)
xgboosttest_logistic(trainMat, trainLabelMat, lastMat, lastHalfLabel)
# xgboosttest_linear(trainMat, trainLabelMat, lastMat, lastHalfLabel)

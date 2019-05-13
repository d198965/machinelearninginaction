from numpy import *
import numpy as np
from Learn.kaggle.classify import DataProvider
import xgboost as xgb
from xgboost import XGBClassifier

xgbparams = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 30,
    'lambda': 1,
    'alpha':0.5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 100,
    'silent': 1,
    'eta': 0.5,
    'seed': 1000,
    'nthread': 6
}

iteraterCount = 200


def analysis():
    trainData = DataProvider.readTrainData()
    dataFields = []

    for temColumnValue in trainData.columns:
        if temColumnValue != 'ID_code' and temColumnValue != 'target':
            dataFields.append(temColumnValue)
            # print temColumnValue," === max:", max(trainData[temColumnValue]), " min:", min(trainData[temColumnValue])
    for k in range(0, 5):
        print k
        print trainData.shape[0] / 10
        anaysisData = trainData.sample(trainData.shape[0] / 10)
        trainMatrix = anaysisData[dataFields].values
        trainLabel = anaysisData['target'].values
        attrAnalysis(trainMatrix, trainLabel, dataFields)


def attrAnalysis(trainMat, trainLabelMat, columns):
    model = XGBClassifier()
    model.fit(np.array(trainMat), np.array(trainLabelMat))
    outPutValue = '{'
    for index in range(0, len(model.feature_importances_)):
        if index == 0:
            outPutValue += "'" + columns[index] + "':" + str(model.feature_importances_[index])
        else:
            outPutValue += ',' + "'" + columns[index] + "':" + str(model.feature_importances_[index])
    outPutValue += '}'
    print outPutValue
    # pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    # pyplot.show()


def xgboosttest(trainMat, trainLabelMat, lastMat):
    plst = xgbparams.items()
    dtrain = xgb.DMatrix(trainMat, trainLabelMat)
    model_XGB = xgb.train(plst, dtrain, iteraterCount)
    dtest = xgb.DMatrix(lastMat)
    predicted = model_XGB.predict(dtest)
    return predicted


def run():
    trainData, testData = DataProvider.readModelTrainData()

    target1Count = DataProvider.readOneCount(trainData)
    zeroData = DataProvider.readZeroOriginal(trainData)

    oneValue, onLabel = DataProvider.getArrayData(DataProvider.readOneOriginal(trainData),
                                                  DataProvider.getEffectField(True))

    testValues, testLabels = DataProvider.getArrayData(testData, DataProvider.getEffectField(True))

    # testData = DataProvider.readTestData()
    # testValues = DataProvider.getValuesData(testData,DataProvider.getEffectField(True))

    print '1 count = ', target1Count

    arrPredict = []
    iteratorCount = 10
    for index in range(0, iteratorCount):
        trainValues, trainLabels = DataProvider.baggingRead(zeroData, int(target1Count * 1.1),
                                                            DataProvider.getEffectField(True))
        trainValues = np.row_stack((trainValues, oneValue))

        trainLabels = np.append(trainLabels, onLabel)
        predict = xgboosttest(trainValues, trainLabels, testValues)
        arrPredict.append(predict)
        print "iterator:", index

        sumPredict = []
        for temIndex in range(0, len(testValues)):
            minValue = 1
            maxValue = 0
            for temArr in arrPredict:
                minValue = min(temArr[temIndex], minValue)
                maxValue = max(temArr[temIndex], maxValue)
            if minValue < 1 - maxValue:
                sumPredict.append(minValue)
            else:
                sumPredict.append(maxValue)

        DataProvider.evaluate(sumPredict, testLabels)
        # if index == iteratorCount -1:
        #     print "output======",sumPredict
        #     DataProvider.outputResult2(sumPredict)

run()

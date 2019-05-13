#!/usr/bin/python
# -*- coding: UTF-8 -*-
import Learn.plot.BasicMap as map
from numpy import *
import numpy as np
import pandas as pd
from Learn import DataProvider
import xgboost as xgb
import operator
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from xgboost import plot_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, roc_curve, precision_score, roc_auc_score, \
    f1_score

trainAbbr = "happiness_train_abbr.csv"
trainComplete = "happiness_train_complete.csv"
testAbbr = "happiness_test_abbr.csv"
testComplete = "happiness_test_complete.csv"

importance = {'survey_type':0.000293686,'province':0.0135095,'city':0.0152717,'county':0.00881057,'gender':0.00528634,'birth':0.00910426,'nationality':0.00499266,'religion':0.00234949,'religion_freq':0.00352423,'edu':0.00587372,'edu_status':0.000881057,'income':0.0132159,'political':0.00352423,'floor_area':0.0164464,'property_0':0.0,'property_1':0.000293686,'property_2':0.0020558,'property_3':0.0020558,'property_4':0.00117474,'property_5':0.000587372,'property_6':0.0,'property_7':0.000293686,'property_8':0.00264317,'height_cm':0.0114537,'weight_jin':0.0240822,'health':0.0375918,'health_problem':0.00792952,'depression':0.0599119,'hukou':0.00176211,'hukou_loc':0.0,'media_1':0.00323054,'media_2':0.00969163,'media_3':0.00969163,'media_4':0.00528634,
              'media_5':0.000881057,'media_6':0.00234949,'leisure_1':0.0105727,'leisure_2':0.00146843,'leisure_3':0.00646109,'leisure_4':0.00499266,'leisure_5':0.00264317,
              'leisure_6':0.00675477,'leisure_7':0.0041116,'leisure_8':0.00881057,'leisure_9':0.0179148,'leisure_10':0.00293686,'leisure_11':0.000587372,'leisure_12':0.00146843,'socialize':0.00734214,'relax':0.0105727,'learn':0.00176211,'social_neighbor':0.00881057,'social_friend':0.00910426,'socia_outing':0.00323054,'equity':0.0816446,'class':0.0328928,'class_10_before':0.00528634,'class_10_after':0.0237885,'class_14':0.00558003,'work_exper':0.0120411,'insur_1':0.000293686,'insur_2':0.0020558,'insur_3':0.000293686,
              'insur_4':0.000293686,'family_income':0.0140969,'family_m':0.0182085,'family_status':0.0317181,'house':0.00792952,'car':0.00440529,'invest_0':0.000293686,'invest_1':0.000293686,'invest_2':0.0,'invest_3':0.0,'invest_4':0.0,'invest_5':0.0,'invest_6':0.0,'invest_7':0.0,'invest_8':0.0,'son':0.00558003,'daughter':0.00704846,'minor_child':0.000293686,'marital':0.00352423,'marital_1st':0.0264317,'s_birth':0.0126285,'s_edu':0.00499266,'s_political':0.0041116,'s_hukou':0.00176211,'s_work_exper':0.000881057,'f_edu':0.00499266,'f_political':0.000293686,'f_work_14':0.00558003,'m_edu':0.0061674,
              'm_political':0.00176211,'m_work_14':0.00381791,'status_peer':0.0208517,'status_3_before':0.0176211,'view':0.0117474,'inc_ability':0.0173275,'inc_exp':0.00910426,'trust_1':0.00851689,'trust_2':0.00469897,'trust_3':0.00528634,'trust_4':0.00234949,'trust_5':0.0108664,'trust_6':0.00381791,'trust_7':0.00440529,'trust_8':0.00264317,'trust_9':0.00499266,'trust_10':0.00499266,'trust_13':0.00293686,'neighbor_familiarity':0.0108664,'public_service_1':0.00499266,'public_service_2':0.010279,'public_service_3':0.00998532,'public_service_4':0.00763583,'public_service_5':0.0117474,'public_service_6':0.0155653,'public_service_7':0.0190896,'public_service_8':0.0170338,'public_service_9':0.00969163,
              'F':0.00998532,
              'Z':0.00792952,
              'q':0.00646109}




def readData(fileName):
    data = pd.read_csv(fileName)
    # 补全数据
    columns = data.columns.values
    if 'edu_other' in columns:
        data.drop('edu_other', axis=1, inplace=True)
    if 'property_other' in columns:
        data.drop('property_other', axis=1, inplace=True)
    # if 's_hukou' in columns:
    #     data.drop('s_hukou', axis=1, inplace=True)
    if 'invest_other' in columns:
        data.drop('invest_other', axis=1, inplace=True)
    return data


def dealData(data):
    # 异常值处理
    for temColumn in data.columns:
        if temColumn == 'id' or temColumn == 'survey_time' or temColumn == 'happiness':
            continue
        print temColumn
        minu_data = []
        if temColumn.find('income') >= 0 or temColumn == 'inc_exp':
            minu_data = ((data[temColumn].values) < 100)
        else:
            minu_data = ((data[temColumn].values) < 0)
        minuLength = 0
        for temMinus in minu_data:
            if temMinus:
                minuLength += 1
        for temMinuIndex in range(0, len(minu_data)):
            if minu_data[temMinuIndex]:
                data.loc[temMinuIndex, [temColumn]] = NAN

    for temColumn in data.columns:
        if temColumn == 'id' or temColumn == 'survey_time':
            continue
        print temColumn
        nan_data = (np.isnan(data[temColumn])).values
        nanLength = 0
        for temNan in nan_data:
            if temNan:
                nanLength += 1
        if nanLength * 4 > len(nan_data):
            data.drop(temColumn, axis=1, inplace=True)
            continue
        elif nanLength > 0:
            fillValue = -1
            if temColumn == 'edu_status':
                fillValue = min(data[temColumn])
            elif temColumn == 'edu_yr' or temColumn == 'join_party':
                fillValue = 2016
            elif temColumn == 'income' or temColumn == 's_income':
                fillValue = 0
            elif temColumn == 'work_type':
                fillValue = 1
            elif temColumn == 'work_status':
                fillValue = 3
            elif temColumn == 'marital_1st':
                fillValue = 2016
            elif temColumn == 's_birth':
                fillValue = -2
                for temNanIndex in range(0, len(nan_data)):
                    if nan_data[temNanIndex]:
                        data.loc[temNanIndex, [temColumn]] = data["birth"].values[temNanIndex]
            elif temColumn == 'marital_now':
                fillValue = -2
                for temNanIndex in range(0, len(nan_data)):
                    if nan_data[temNanIndex]:
                        data.loc[temNanIndex, [temColumn]] = data["marital_now"].values[temNanIndex]
            elif temColumn == 's_political':
                fillValue = 0
            elif temColumn == 's_work_status':
                fillValue = 3
            elif temColumn == 's_work_status':
                fillValue = 1
            if fillValue != -2:
                if fillValue == -1:
                    data[temColumn] = data[temColumn].fillna(data[temColumn].median())
                else:
                    data[temColumn] = data[temColumn].fillna(fillValue)
        data[temColumn] = data[temColumn].astype('int')

    if 'happiness' in data.columns:
        happyList = data['happiness'].values
        dropIndex = []
        for temIndex in range(0, len(happyList)):
            if happyList[temIndex] <= 0 or happyList[temIndex] > 5:
                print temIndex
                dropIndex.append(temIndex)
        data['happiness'] -= 1
        data.drop(dropIndex, axis=0, inplace=True)
    return data


def createAttr(data):
    for temColumn in data.columns:
        if temColumn == 'id' or temColumn == 'survey_time':
            continue
        if temColumn.find('_yr') >= 0 or temColumn.find('birth') >= 0 or temColumn.find('marital_') >= 0:
            data[temColumn] = 2016 - data[temColumn]

    if 'marital_now' in data.columns and 'marital_1st' in data.columns:
        pd.concat([data, pd.DataFrame(columns=list('p'))])
        data['p'] = data['marital_1st'] - data['marital_now']
        data['p'] = data['p'].astype('int')

    if 'class_10_before' in data.columns and 'class' in data.columns:
        pd.concat([data, pd.DataFrame(columns=list('F'))])
        data['F'] = data['class'] - data['class_10_before']
        data['F'] = data['F'].astype('int')

    if 'class_10_after' in data.columns and 'class' in data.columns:
        pd.concat([data, pd.DataFrame(columns=list('F'))])
        data['Z'] = data['class_10_after'] - data['class']
        data['Z'] = data['Z'].astype('int')

    if 's_birth' in data.columns and 'birth' in data.columns:
        pd.concat([data, pd.DataFrame(columns=list('q'))])
        data['q'] = (data['birth'] - data['s_birth'])*((data['gender'] - 1.5)/abs(data['gender'] - 1.5))
        data['q'] = data['q'].astype('int')

    return data


def dataAnalysis(data):
    print(data.columns)
    # 删除异常数据
    happyList = data['happiness'].values
    print (min(happyList), max(happyList))
    # map.draw_hist(happyList, 'happiness', 'A', 'B', 0, 10, 0, len(happyList))


xgbparams = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'num_class': 5,
    'gamma': 0.1,
    'max_depth': 30,
    'lambda': 2,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

iteraterCount = 30


def xgboosttest_softmax(trainMat, trainLabelMat, lastMat):
    plst = xgbparams.items()
    dtrain = xgb.DMatrix(trainMat, trainLabelMat)
    model_XGB = xgb.train(plst, dtrain, iteraterCount)
    dtest = xgb.DMatrix(lastMat)
    predicted = model_XGB.predict(dtest)
    return predicted


def evaluate(predicted, lastHalfLabel):
    presion = 0
    for index in range(0, len(lastHalfLabel)):
        presion += pow(lastHalfLabel[index] - predicted[index], 2)
    presion /= len(lastHalfLabel)
    print("Precision score is: " + str(round(presion, 3)))

    for value in range(0, 5):
        temPresion = 0.0
        temLength = 0
        values = (lastHalfLabel == value)
        for valueIndex in range(0, len(values)):
            if values[valueIndex]:
                temLength += 1
                temPresion += pow(lastHalfLabel[valueIndex] - predicted[valueIndex], 2)
        temPresion /= temLength
        print("Precision score is: " + str(value) + ":" + str(round(temPresion, 3)))


def attrAnalysis(trainMat, trainLabelMat,columns):
    model = XGBClassifier()
    model.fit(np.array(trainMat), np.array(trainLabelMat))
    outPutValue = '{'
    for index in range(0, len(model.feature_importances_)):
        if index == 0:
            outPutValue += "'"+columns[index]+"':"+str(model.feature_importances_[index])
        else:
            outPutValue += ',' + "'"+columns[index]+"':"+str(model.feature_importances_[index])
    outPutValue += '}'
    print outPutValue
    pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
    pyplot.show()


def preDealData(temData):
    theData = dealData(temData)
    theData = createAttr(theData)
    return theData




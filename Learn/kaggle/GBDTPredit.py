#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import datetime
import time
import numpy as np
import pandas as pd
from numpy import *
from itertools import chain
from numpy.core.umath import NAN

MAX_VALUE = 20.0
MIN_VALUE = 10.0
STEP_COUNT = 50

TargetCols = [
    'assetCodes',
    'urgency',
    'takeSequence',
    'marketCommentary',
    'relevance',
    'sentimentWordCount'
]

TargetColsMean = [
    'assetCodes_sum',
    'urgency_sum',
    'takeSequence_sum',
    'marketCommentary_sum',
    'relevance_sum',
    'sentimentWordCount_sum',
    'v_mean',
    'p_mean',
    'l_mean',
    's_mean'
]


def readMarketTrainData(market_train_df, news_train_df):
    news_train_df_aggregated, tem, tem1 = readStockNewsData(news_train_df, market_train_df)
    del tem, tem1

    # Join with train
    market_train_df = market_train_df.join(news_train_df_aggregated, on=['time', 'assetCode'])

    # Free memory
    del news_train_df_aggregated

    marketDetaData = {}
    newsTimeList = list(set(news_train_df['time']))
    for temStockTime in newsTimeList:
        stockInfoList = market_train_df[market_train_df['time'] == temStockTime]
        marketDetaData[temStockTime] = caculateMarketRate(stockInfoList)

    nan_data = np.isnan(market_train_df['urgency_sum'])
    nan_index = []
    for nanIndex in range(0, len(nan_data)):
        if nan_data[nanIndex]:
            nan_index.append(nanIndex)
    target_market_train_df = market_train_df.drop(nan_index)

    del market_train_df

    marketDeta = []
    # for temIndex in range(0, len(target_market_train_df)):
    #     marketDeta.append(marketDetaData[target_market_train_df['time'].values[temIndex]])

    stockRate = (
        (target_market_train_df['close'] - target_market_train_df['open']) / target_market_train_df['open']).values

    targetValues = target_market_train_df[TargetColsMean]

    stockMarketData = targetValues.apply(
            lambda x: (x) / (abs(x) + 0.001) * (abs(x) - np.min(abs(x)) * 1.0) / (
                np.max(abs(x)) - np.min(abs(x)))).values
    stockMarketData[np.isnan(stockMarketData)] = 1.0

    return np.array(stockMarketData), np.array(stockRate), targetValues


def readStockNewsData(news_train_df, market_train_df):
    news_cols_agg = {
        'urgency': ['sum'],
        'takeSequence': ['sum'],
        'marketCommentary': ['sum'],
        'relevance': ['sum'],
        'sentimentWordCount': ['sum'],
        'sentimentPositive': ['sum'],
        'sentimentNeutral': ['sum']
    }
    market_train_df['time'] = market_train_df['time'].astype(str).str[0:10]
    news_train_df['assetCodes'] = news_train_df['assetCodes'].str.findall("'([\w\./]+)'")
    news_train_df['time'] = news_train_df['time'].astype(str).str[0:10]
    news_train_df['marketCommentary'] = news_train_df['marketCommentary'].astype('int')

    assetCodes_index = news_train_df.index.repeat(news_train_df['assetCodes'].apply(len))
    assetCodes_expanded = list(chain(*news_train_df['assetCodes']))
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

    # Create expandaded news (will repeat every assetCodes' row)
    news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True,
                                      suffixes=(['', '_old']))
    news_cols_agg['assetCodes'] = ['mean']
    news_train_df_expanded['assetCodes'] = news_train_df_expanded['assetCodes'].str.len()

    news_train_df_expanded = pd.concat([news_train_df_expanded, pd.DataFrame(columns=list('vpls'), dtype=float)])

    market_tem_data = market_train_df[['time', 'assetCode', 'open', 'close', 'volume']]

    # x前两日\y前一日\z下一个交易日\time 当前日期
    market_tem_data['time'] = pd.to_datetime(market_tem_data['time']) - datetime.timedelta(days=2)
    market_tem_data['time'] = market_tem_data['time'].apply(lambda x: x.strftime('%Y-%m-%d'))

    XDayMerge = pd.merge(market_tem_data, news_train_df_expanded, on=['time', 'assetCode'])

    market_tem_data['time'] = pd.to_datetime(market_tem_data['time']) + datetime.timedelta(days=1)
    market_tem_data['time'] = market_tem_data['time'].apply(lambda x: x.strftime('%Y-%m-%d'))

    YDayMerge = pd.merge(market_tem_data, news_train_df_expanded, on=['time', 'assetCode'])

    market_tem_data['time'] = pd.to_datetime(market_tem_data['time']) + datetime.timedelta(days=1)
    market_tem_data['time'] = market_tem_data['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    TDayMerge = pd.merge(market_tem_data, news_train_df_expanded, on=['time', 'assetCode'])

    pre_market_news_expanded = pd.concat([XDayMerge, YDayMerge, TDayMerge], axis=0)

    pre_market_news_expanded['v'] = pre_market_news_expanded['volume'] * pre_market_news_expanded['close']
    pre_market_news_expanded['l'] = (pre_market_news_expanded['close'] - pre_market_news_expanded['open']) / \
                                    pre_market_news_expanded['open']
    pre_market_news_expanded['p'] = pre_market_news_expanded['close']
    pre_market_news_expanded['s'] = pre_market_news_expanded.apply(
            lambda x: (x['sentimentPositive'] + x['sentimentNeutral'] / 2 - 0.5) * 2, axis=1)
    nan_data = np.isnan(pre_market_news_expanded['v'])
    nan_index = []
    for nanIndex in range(0, len(nan_data)):
        if nan_data.values[nanIndex]:
            nan_index.append(nanIndex)
    pre_market_news_expanded = pre_market_news_expanded.drop(nan_index)

    pre_market_news_expanded['time'] = pd.to_datetime(pre_market_news_expanded['time']) + datetime.timedelta(days=1)
    pre_market_news_expanded['time'] = pre_market_news_expanded['time'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # Free memory
    del news_train_df, df_assetCodes

    news_cols_agg['v'] = ['mean']
    news_cols_agg['p'] = ['mean']
    news_cols_agg['l'] = ['mean']
    news_cols_agg['s'] = ['mean']
    news_cols_agg['assetCodes'] = ['sum']
    news_cols_agg.pop('sentimentPositive')
    news_cols_agg.pop('sentimentNeutral')

    grouInfo = pre_market_news_expanded.groupby(['time', 'assetCode'])
    stockCodes = []
    for key in grouInfo.groups.keys():
        stockCodes.append(key[1])
    news_train_df_aggregated = grouInfo.agg(news_cols_agg)
    del news_train_df_expanded

    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

    news_train_df_export = news_train_df_aggregated[TargetColsMean]
    stockMarketData = news_train_df_export.apply(
            lambda x: (x) / (abs(x) + 0.0001) * (x - np.min(x) * 1.0) / (np.max(x) - np.min(x))).values
    stockMarketData[np.isnan(stockMarketData)] = 1.0

    return news_train_df_aggregated, stockMarketData, stockCodes


def caculateMarketRate(stockInfoList):
    if len(stockInfoList) == 0:
        return 0
    rate = sum(((stockInfoList['open'] - stockInfoList['close']) / stockInfoList['open']).values) / len(stockInfoList)
    return rate


def caculateNewsRate(oneNewsData):
    if len(oneNewsData) == 0:
        return 0
    rate = sum(((oneNewsData['sentimentPositive'] + oneNewsData['sentimentNeutral'] / 2 - 0.5) * 2).values)
    return rate / np.sqrt(len(oneNewsData))


def readPredictStockNewsData(news_train_df2, market_train_df2, trainValues):
    news_cols_agg = {
        'urgency': ['sum'],
        'takeSequence': ['sum'],
        'marketCommentary': ['sum'],
        'relevance': ['sum'],
        'sentimentWordCount': ['sum'],
        'sentimentPositive': ['sum'],
        'sentimentNeutral': ['sum']
    }
    market_train_df2['time'] = market_train_df2['time'].astype(str).str[0:10]
    news_train_df2['assetCodes'] = news_train_df2['assetCodes'].str.findall("'([\w\./]+)'")
    news_train_df2['time'] = news_train_df2['time'].astype(str).str[0:10]
    news_train_df2['marketCommentary'] = news_train_df2['marketCommentary'].astype('int')

    assetCodes_index = news_train_df2.index.repeat(news_train_df2['assetCodes'].apply(len))
    assetCodes_expanded = list(chain(*news_train_df2['assetCodes']))
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

    # Create expandaded news (will repeat every assetCodes' row)
    news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df2[news_cols], left_on='level_0', right_index=True,
                                      suffixes=(['', '_old']))
    news_cols_agg['assetCodes'] = ['mean']
    news_train_df_expanded['assetCodes'] = news_train_df_expanded['assetCodes'].str.len()

    news_train_df_expanded = pd.concat([news_train_df_expanded, pd.DataFrame(columns=list('vpls'), dtype=float)])

    market_tem_data = market_train_df2[['time', 'assetCode', 'open', 'close', 'volume']]

    # x前两日\y前一日\z下一个交易日\time 当前日期
    market_tem_data['time'] = pd.to_datetime(market_tem_data['time']) - datetime.timedelta(days=2)
    market_tem_data['time'] = market_tem_data['time'].apply(lambda x: x.strftime('%Y-%m-%d'))

    XDayMerge = pd.merge(market_tem_data, news_train_df_expanded, on=['time', 'assetCode'])

    market_tem_data['time'] = pd.to_datetime(market_tem_data['time']) + datetime.timedelta(days=1)
    market_tem_data['time'] = market_tem_data['time'].apply(lambda x: x.strftime('%Y-%m-%d'))

    YDayMerge = pd.merge(market_tem_data, news_train_df_expanded, on=['time', 'assetCode'])

    market_tem_data['time'] = pd.to_datetime(market_tem_data['time']) + datetime.timedelta(days=1)
    market_tem_data['time'] = market_tem_data['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    TDayMerge = pd.merge(market_tem_data, news_train_df_expanded, on=['time', 'assetCode'])

    pre_market_news_expanded = pd.concat([XDayMerge, YDayMerge, TDayMerge], axis=0)

    pre_market_news_expanded['v'] = pre_market_news_expanded['volume'] * pre_market_news_expanded['close']
    pre_market_news_expanded['l'] = (pre_market_news_expanded['close'] - pre_market_news_expanded['open']) / \
                                    pre_market_news_expanded['open']
    pre_market_news_expanded['p'] = pre_market_news_expanded['close']
    pre_market_news_expanded['s'] = pre_market_news_expanded.apply(
            lambda x: (x['sentimentPositive'] + x['sentimentNeutral'] / 2 - 0.5) * 2, axis=1)
    nan_data = np.isnan(pre_market_news_expanded['v'])
    nan_index = []
    for nanIndex in range(0, len(nan_data)):
        if nan_data.values[nanIndex]:
            nan_index.append(nanIndex)
    pre_market_news_expanded = pre_market_news_expanded.drop(nan_index)

    pre_market_news_expanded['time'] = pd.to_datetime(pre_market_news_expanded['time']) + datetime.timedelta(days=1)
    pre_market_news_expanded['time'] = pre_market_news_expanded['time'].apply(lambda x: x.strftime('%Y-%m-%d'))

    # Free memory
    del news_train_df2, df_assetCodes

    news_cols_agg['v'] = ['mean']
    news_cols_agg['p'] = ['mean']
    news_cols_agg['l'] = ['mean']
    news_cols_agg['s'] = ['mean']
    news_cols_agg['assetCodes'] = ['sum']
    news_cols_agg.pop('sentimentPositive')
    news_cols_agg.pop('sentimentNeutral')

    grouInfo = pre_market_news_expanded.groupby(['time', 'assetCode'])
    stockCodes = []
    for key in grouInfo.groups.keys():
        stockCodes.append(key[1])
    news_train_df_aggregated = grouInfo.agg(news_cols_agg)
    del news_train_df_expanded

    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

    news_train_df_export = news_train_df_aggregated[TargetColsMean]
    stockMarketData = news_train_df_export.apply(
            lambda x: (x) / (abs(x) + 0.001) * (x - np.min(trainValues[x.name].values) * 1.0) / (
                np.max(trainValues[x.name].values) - np.min(trainValues[x.name].values))).values
    stockMarketData[np.isnan(stockMarketData)] = 1.0
    stockMarketData[np.isinf(stockMarketData)] = 1.0

    return stockMarketData, stockCodes


def lastIndexOf(result, code):
    for index in range(len(result) - 1, -1, -1):
        if result[index] == code:
            return index
    return -1


def checkParamters(paramster, stockRate, predictStockRate):
    sumValue = stockRate * predictStockRate[0]
    # print(sumValue)
    predictions = sumValue >= 0
    count = 0
    for result in predictions:
        if result:
            count += 1
    print(stockRate)
    print(predictStockRate)
    # print(paramster)
    print(count * 1.0 / len(stockRate))


def dealOutputY(marketY):
    result = []
    for Yindex in range(0, len(marketY)):
        if marketY[Yindex] < 0:
            result.append(MIN_VALUE)
        else:
            result.append(MAX_VALUE)
    return np.array(result)


def predict(resultTrees, marketX):
    m, n = shape(marketX)
    resultClassify = zeros((m, 1))
    for i in range(len(resultTrees)):
        temTree = resultTrees[i][0]
        resultClassify[marketX[:, temTree[1]] < temTree[0]] += temTree[2]
        resultClassify[marketX[:, temTree[1]] >= temTree[0]] += temTree[3]
    return resultClassify


def checkPredict(resultTrees, marketX, stockRate):
    m, n = shape(marketX)
    predictStockRateY = dealOutputY(stockRate)
    resultClassify = predict(resultTrees, marketX)
    errorCount = 0
    for index in range(m):
        if (predictStockRateY[index] - (MAX_VALUE + MIN_VALUE) / 2) * (
                    resultClassify[index][0] - (MAX_VALUE + MIN_VALUE) / 2) < 0:
            errorCount += 1
    print errorCount * 100.0 / m


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


train_market_df = pd.read_csv("marketdata_sample.csv")
train_news_df = pd.read_csv("news_sample.csv")
slice_news = train_news_df[np.round(10).astype(int): train_news_df.shape[0]]
print ("slice begin", time.time())
for index in range(0, slice_news.shape[0]):
    slice_news.index._data[index] = index
slice_market = train_market_df[np.round(10).astype(int): train_market_df.shape[0]]
for index in range(0, slice_market.shape[0]):
    slice_market.index._data[index] = index
print ("read begin", time.time())
stockMarketData, stockRate, trainValues = readMarketTrainData(slice_market, slice_news)
print ("read done", time.time())

for col in trainValues.columns.values:
    print(col, min(trainValues[col].values), max(trainValues[col].values))

# deta=marketDeta, n_k=20,
dealStockRateY = dealOutputY(stockRate)

trainStockRate = stockRate[0:len(dealStockRateY) / 2]
trainStockRateY = dealStockRateY[0:len(dealStockRateY) / 2]
trainMarketData = stockMarketData[0:len(dealStockRateY) / 2]

resultTrees = gbdtClassify(trainMarketData, trainStockRateY, 1, 0, 1, 50)

print ("stock done", time.time())
print (resultTrees)
checkPredict(resultTrees, trainMarketData, trainStockRate)


testStockRate = stockRate[len(dealStockRateY) / 2:len(dealStockRateY)]
testStockRateY = dealStockRateY[len(dealStockRateY) / 2:len(dealStockRateY)]
testMarketData = stockMarketData[len(dealStockRateY) / 2:len(dealStockRateY)]

checkPredict(resultTrees, testMarketData, testStockRate)


# resultTrees = getStockParamters()

# 开始预测
print ("开始预测------------")

# market_train_df2 = pd.read_csv("marketdata_sample.csv")
# news_train_df2 = pd.read_csv("news_sample.csv")
#
# predictStockNewsData, stockCodes = readPredictStockNewsData(news_train_df2, market_train_df2, trainValues)
# result = predict(resultTrees, predictStockNewsData)
# print(result)
# if stockCodes.__contains__('A.N'):
#     print result[lastIndexOf(stockCodes, 'A.N')]

print ("完成预测------------")

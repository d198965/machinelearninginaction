#!/usr/bin/python
# -*- coding: UTF-8 -*-
from multiprocessing import Pool, Manager
from math import exp
import numpy as np
import pandas as pd
import pickle
import time

'''
def LFM(user_items, F, N, alpha, lambdaVallue):
    #初始化P,Q矩阵
    [P, Q] = InitModel(user_items, F)
    #开始迭代
    for step in range(0, N):
        #从数据集中依次取出user以及该user喜欢的iterms集
        for user, items in user_item.iterms():
            #随机抽样，为user抽取与items数量相当的负样本，并将正负样本合并，用于优化计算
            samples = RandSelectNegativeSamples(items)
            #依次获取item和user对该item的兴趣度
            for item, rui in samples.items():
                #根据当前参数计算误差  PS:转载的博客中rui写成了eui
                eui = rui - Predict(user, item)
                #优化参数
                for f in range(0, F):
                    P[user][f] += alpha * (eui * Q[f][item] - lambdaVallue * P[user][f])
                    Q[f][item] += alpha * (eui * P[user][f] - lambdaVallue * Q[f][item])
        #每次迭代完后，都要降低学习速率。一开始的时候由于离最优值相差甚远，因此快速下降；
        #当优化到一定程度后，就需要放慢学习速率，慢慢的接近最优值。
        alpha *= 0.9
'''


def getResource(fileName):
    return pd.read_csv(fileName)


def userItemLinkWork(userID, frameData, movieIds, queue):
    userItemlist = list(frameData[frameData['UserID'] == userID]['MovieID'])
    otherItemList = list(item for item in movieIds if item not in userItemlist)

    negativeCount = len(userItemlist)
    negativeItemList = list()
    if (len(otherItemList) != 0):
        randomIndexList = np.random.randint(0, len(otherItemList), size=negativeCount)
        for index in randomIndexList:
            negativeItemList.append(otherItemList[index])

    itemDict = {}
    for item in userItemlist: itemDict[item] = 1
    for item in negativeItemList: itemDict[item] = 0

    queue.put({userID: itemDict})


def initUserItemPool(userIds, movieIds, frameData):
    pool = Pool()
    queue = Manager().Queue()
    userItems = []
    for itemId in userIds:
        pool.apply_async(userItemLinkWork, args=(itemId, frameData, movieIds, queue))
    pool.close()
    pool.join()
    while not queue.empty(): userItems.append(queue.get())
    return userItems


def initPQMatrix(userIds, movieIds, classCount):
    arrayp = np.random.rand(len(userIds), classCount)
    arrayq = np.random.rand(classCount, len(movieIds))
    p = pd.DataFrame(arrayp, columns=range(0, classCount), index=userIds)
    q = pd.DataFrame(arrayq, columns=movieIds, index=range(0, classCount))
    return p, q


def initModel(frameData, classCount):
    userIds = list(set(frameData['UserID']))
    movieIds = list(set(frameData['MovieID']))
    # 获取用户——>数据对照组
    userItems = initUserItemPool(userIds, movieIds, frameData)
    # 获取 p\q矩阵
    p, q = initPQMatrix(userIds, movieIds, classCount)
    return userItems, p, q


def predixCalculate(p, q, userID, movieID):
    p = np.mat(p.ix[userID].values)
    q = np.mat(q[movieID].values).T
    r = (p * q).sum()
    return sigmod(r)


def sigmod(x):
    '''
    单位阶跃函数,将兴趣度限定在[0,1]范围内
    :param x: 兴趣度
    :return: 兴趣度
    '''
    y = 1.0 / (1 + exp(-x))
    return y


def calculateModel(frameData, classCount, alpha, lamada, literCount):
    userItems, p, q = initModel(frameData, classCount)
    for countIndex in range(0, literCount):
        for userItem in userItems:
            for userId, itemList in userItem.items():
                for item, rui in itemList.items():
                    erui = rui - predixCalculate(p, q, userId, item)
                    for classK in range(0, classCount):
                        p[classK][userId] += alpha * (q[item][classK] * erui - lamada * p[classK][userId])
                        q[item][classK] += alpha * (p[classK][userId] * erui - lamada * q[item][classK])
            print userItem
    return p, q


def calculateModel2(frameData, classCount, alpha, lamada, literCount):
    userItems, p, q = initModel(frameData, classCount)
    pool = Pool()
    for countIndex in range(0, literCount):
        for userItem in userItems:
            pool.apply_async(calculateOneUserItem, args=(userItem, p, q, alpha, lamada, classCount))
    pool.close()
    pool.join()
    return p, q


def calculateOneUserItem(userItem, p, q, alpha, lamada, classCount):
    for userId, itemList in userItem.items():
        for item, rui in itemList.items():
            erui = rui - predixCalculate(p, q, userId, item)
            for classK in range(0, classCount):
                p[classK][userId] += alpha * (q[item][classK] * erui - lamada * p[classK][userId])
                q[item][classK] += alpha * (p[classK][userId] * erui - lamada * q[item][classK])
    print userItem

def predicTopN(userId, topN, p, q):
    userItemlist = list(set(frame[frame['UserID'] == userId]['MovieID']))
    otherItemList = [item for item in set(frame['MovieID'].values) if item not in userItemlist]
    predictResult = list(predixCalculate(p, q, userId - 1, otherItem - 1) for otherItem in otherItemList)
    predictResult.sort()
    print predictResult[:topN]


frame = getResource("ratings.csv")
# frame = getResource("simple_movie_rating.csv")
p, q = calculateModel2(frame, 5, 0.2, 0.1, 10)
predicTopN(2, 2, p, q)

#!/usr/bin/python
# -*- coding: UTF-8 -*-
import collections
from multiprocessing import Pool, Manager
from math import exp
import numpy as np
import pandas as pd
from PersonRankItem import PersonRankItem
import time as time

'''
user -> item表
item -> user表
'''


def getResource(fileName):
    return pd.read_csv(fileName)


def initModel():
    userIds = list(set(frameData['UserID']))
    movieIds = list(set(frameData['MovieID'] * -1))
    # userItems = initItems(userIds, True)
    # movieItems = initItems(movieIds, False)
    pool = Pool(2)
    queue = Manager().Queue()
    pool.apply_async(initUserItem, args=(queue,))
    pool.apply_async(initMovieItem, args=(queue,))
    pool.close()
    pool.join()
    userItems = {}
    movieItems = {}
    if not queue.empty():
        userItems = queue.get()
    if not queue.empty():
        movieItems = queue.get()
    if userItems.keys()[0] > 0:
        return userItems, movieItems
    else:
        return movieItems, userItems


def work(isUser, queue, itemId):
    if isUser:
        relationItems = list(set(frameData[frameData['UserID'] == itemId]['MovieID'] * -1))
    else:
        relationItems = list(set(frameData[frameData['MovieID'] == itemId * -1]['UserID']))
    queue.put(PersonRankItem(itemId, relationItems))
    return


def initUserItem(queue):
    itemList = {}
    temItemId = -1
    relationItems = []
    indexList = frameData.index
    for index in indexList:
        values = frameData.loc[index].values
        if (temItemId != values[0]):
            itemList[temItemId] = (PersonRankItem(temItemId, relationItems))
            relationItems = [values[1] * -1]
        else:
            relationItems.append(values[1] * -1)
        temItemId = values[0]
    itemList[temItemId] = (PersonRankItem(temItemId, relationItems))
    queue.put(itemList)


def initMovieItem(queue):
    temFrameData = frameData.sort_values("MovieID", inplace=False)
    itemList = {}
    temItemId = 1
    relationItems = []
    indexList = temFrameData.index
    for index in indexList:
        values = temFrameData.loc[index].values
        if (temItemId != values[1] * -1 and temItemId != ""):
            itemList[temItemId] = (PersonRankItem(temItemId, relationItems))
            relationItems = [values[0]]
        else:
            relationItems.append(values[0])
        temItemId = values[1] * -1
    itemList[temItemId] = (PersonRankItem(temItemId, relationItems))
    queue.put(itemList)


def initItems(idList, isUser):
    pool = Pool()
    queue = Manager().Queue()
    items = {}
    for itemId in idList:
        pool.apply_async(work, args=(isUser, queue, itemId))
    pool.close()
    pool.join()
    while not queue.empty():
        temItem = queue.get()
        items[temItem.item] = temItem
    return items


def calculateOneTime(allItemSet, userItem, alpha):
    for key, item in allItemSet.items():
        for relationId in item.relationIds:
            allItemSet[relationId].temValue += alpha * (item.value * 1.0) / len(item.relationIds)
    allItemSet[userItem.item].temValue += 1 - alpha

    for key, item in allItemSet.items():
        item.addSum()


def calculate(userId, literCount, alpha):
    userItem = userItemSet[userId]
    userItem.value = 1
    for liter in range(0, literCount):
        calculateOneTime(allItemSet, userItem, alpha)

    sortResult = sorted(movieItemSet.values())
    for v in sortResult:
        print v.item, " ", v.value


# frameData = getResource("simple_movie_rating.csv")
frameData = getResource("ratings.csv")
startTime = time.time()
userItemSet, movieItemSet = initModel()
print "init:", time.time() - startTime
allItemSet = dict(userItemSet.items() + movieItemSet.items())
calculate(2, 10, 0.6)

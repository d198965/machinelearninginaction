#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import xgboost as xg


def loopMatrix(matrixSize):
    matrix = np.zeros((matrixSize, matrixSize))
    print(matrix)

    value = 0
    changeLoop = 1  # 1(左右) 2(上下) 3(右左) 4(下上)
    rowIndex = -1
    while (rowIndex < matrixSize):
        rowIndex += 1
        colmIndex = -1
        if rowIndex == matrixSize:
            break
        while (colmIndex < matrixSize):
            if value == matrixSize * matrixSize:
                print(matrix)
                return
            colmIndex += 1
            if colmIndex == matrixSize:
                break
            value += 1
            if changeLoop == 1:
                matrix[rowIndex][colmIndex] = value
                print(rowIndex, colmIndex)
            elif changeLoop == 2:
                matrix[colmIndex][rowIndex] = value
                print(colmIndex, rowIndex)
            elif changeLoop == 3:
                matrix[rowIndex][rowIndex - colmIndex - 1] = value
                print(rowIndex, rowIndex - colmIndex - 1)
            else:
                matrix[matrixSize - rowIndex - colmIndex - 1][rowIndex - 1] = value
                print(matrixSize - rowIndex - colmIndex - 1, rowIndex - 1)

            if changeLoop == 1 and (colmIndex == matrixSize - 1 or matrix[rowIndex][colmIndex + 1] != 0):
                temValue = colmIndex
                colmIndex = rowIndex
                rowIndex = temValue
                changeLoop = 2
            elif changeLoop == 2 and (colmIndex == matrixSize - 1 or matrix[colmIndex + 1][rowIndex] != 0):
                changeLoop = 3
                rowIndex -= 1
                break
            elif changeLoop == 3 and (colmIndex == matrixSize - 1 or rowIndex - colmIndex - 2 < 0 or matrix[rowIndex][
                        rowIndex - colmIndex - 2] != 0):
                rowIndex = matrixSize - rowIndex - 1
                changeLoop = 4
                break
            elif changeLoop == 4 and (
                    colmIndex == matrixSize - 1 or matrix[matrixSize - rowIndex - colmIndex - 2][rowIndex] != 0):
                changeLoop = 1
                colmIndex = rowIndex - 1

    print(matrix)


matrixM = input()
loopMatrix(matrixM)
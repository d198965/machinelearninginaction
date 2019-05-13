from HappyTrain import *
import tensorflow as tf

trainData = pd.read_csv('Result.csv')
dataAnalysis(trainData)

dataFields = []
for temColumnValue in trainData.columns:
    if temColumnValue != 'id' and temColumnValue != 'survey_time' and temColumnValue != 'happiness':
        dataFields.append(temColumnValue)

dataFinalFields = dataFields
# for fieldIndex in range(0, len(dataFields)):
#     if importance[fieldIndex] >= 0:
#         dataFinalFields.append(dataFields[fieldIndex])
trainMatrix = trainData[dataFinalFields].values
trainLabel = trainData['happiness'].values
# trainData.to_csv('Result.csv', index=False)

# 1500 1000

leftData = []
leftLabel = []

threeCount = 0
twoCount = 0
forthCount = 0
for temIndex in range(0, len(trainLabel)):
    if trainLabel[temIndex] == 3 and threeCount < 2000:
        threeCount += 1
        leftData.append(trainMatrix[temIndex])
        leftLabel.append(trainLabel[temIndex])
    elif trainLabel[temIndex] == 2 and twoCount < 1000:
        twoCount += 1
        leftData.append(trainMatrix[temIndex])
        leftLabel.append(trainLabel[temIndex])
    elif trainLabel[temIndex] == 4 and forthCount < 1000:
        forthCount += 1
        leftData.append(trainMatrix[temIndex])
        leftLabel.append(trainLabel[temIndex])
    elif trainLabel[temIndex] == 0 or trainLabel[temIndex] == 1:
        leftData.append(trainMatrix[temIndex])
        leftLabel.append(trainLabel[temIndex])

parameters = BP.nn_model(leftData, leftLabel, 60, num_iterations=400, print_cost=True)

predictions = BP.predict(parameters, trainMatrix)
print (np.dot(1 - trainLabel, predictions.T))
print (np.dot(trainLabel, 1 - predictions.T))
print (
        'Accuracy: %f' % float(
            (np.dot(lastHalfLabel, predictions.T) + np.dot(1 - lastHalfLabel, 1 - predictions.T)) / float(
                lastHalfLabel.size) * 100) + '%')

predictData = xgboosttest_softmax(leftData, leftLabel,
                                  trainMatrix)
evaluate(predictData, trainLabel)

attrAnalysis(leftData, leftLabel,dataFinalFields)

# testData = readData(testComplete)
# testData = preDealData(testData)
# testMatrix = testData[dataFinalFields].values
#
# predictData = xgboosttest_softmax(leftData, leftLabel, testMatrix)
# outputPd = pd.read_csv('happiness_submit.csv')
# for outputIndex in range(0,len(predictData)):
#     outputPd.loc[outputIndex, ['happiness']] = predictData[outputIndex]+1
# outputPd['happiness'] = outputPd['happiness'].astype('int')
# outputPd.to_csv('submit.csv',index=False)
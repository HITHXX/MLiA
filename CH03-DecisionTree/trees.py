# coding=utf-8
from math import log
import operator
import treePlotter
import pickle


# 计算香农信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

#  创建数据集
def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet,labels

def splitDataSet(dataSet, axis, value):  #将数据集按照某一特征分割
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis] #  左闭右开
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):  # 选择最好的数据集划分方式
    numFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0  #最大信息增益熵
    bestFeature = -1  #最好的特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)  #创建唯一的分类树标签
        newEntropy = 0.0
        for value in uniqueVals:  #计算每种标签的信息增益熵
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)  # -sum(plog(p))
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):  # 多数表决法确定叶子节点分类
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 所有类标签都相同，只有一种分类
        return classList[0] #返回该类标签
    if len(dataSet[0])==1:
        return majorityCnt(classList)  # 用完了所有特征，多类表决
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]  # 取出最优特征
    myTree ={bestFeatLabel:{}}
    del(labels[bestFeat])  # 删除该特征
    featValues = [example[bestFeat] for example in dataSet]  # 取出所有特征值
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree


def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  #将标签字符串转换为索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ =='dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:   classLabel = secondDict[key]
    return classLabel


# 使用pickle模块存储决策树
def storeTree(inputTree,filename):
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    fr = open(filename)
    return pickle.load(fr)


def analyseLenses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['age','prescript','astigmatic','tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    treePlotter.createPlot(lensesTree)

def main():
    #myDat,labels = createDataSet()
    #print(myDat)
    #print(labels)
    #print(calcShannonEnt(myDat))
    #print(splitDataSet(myDat,0,1))
    #print chooseBestFeatureToSplit(myDat)
    #myTree = createTree(myDat,labels)
    #myTree = treePlotter.retrieveTree(0)
    #print(classify(myTree,labels,[1,0]))
    #print(classify(myTree,labels,[1,1]))
    #storeTree(myTree,'classifierStorage.txt')
    #print(grabTree('classifierStorage.txt'))
    #print(myTree)
    analyseLenses()

main()
# -*- coding: utf-8 -*-
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # 训练集数据点个数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # x-xi
    sqDiffMat = diffMat ** 2  # 平方
    sqDistances = sqDiffMat.sum(axis=1)  # 第一轴，即逐行相加
    distances = sqDistances ** 0.5  # 开根号，得到欧氏距离
    sortedDisIndicies = distances.argsort()  # 排序后按从小到大返回索引
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDisIndicies[i]]  # 取出标签
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 对应标签位置+1,若标签不存在，默认值为0
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


#group, labels = createDataSet()
#print(classify0([0, 0], group, labels, 3))


def file2matrix(filename):
    fr = open(filename)  # 打开文件
    arrayOLines = fr.readlines()  # 按行读入
    numberOfLines = len(arrayOLines)  # 行数
    returnMat = zeros((numberOfLines, 3))  # numberOfLines行,3列
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()  #截取回车
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 将数据赋值给returnMat
        classLabelVector.append(int(listFromLine[-1]))  # 将标签赋给classLabelVector
        index += 1
    return returnMat, classLabelVector


def img2vector(filename):  #将二维图片转为一维向量
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        #print "i:%d" % i
        for j in range(32):
            #print "j:%d" %j
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


def autoNorm(dataSet):  #归一化函数 newValue = (oldValue-min)/(max-min)
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    #normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))  #注意这里的/不是矩阵除法,Numpy中矩阵除法使用函数linalg.solve(matA,matB)
    return normDataSet,ranges,minVals


def handwritingClassTest():   # 对手写数字的分类测试
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))  # m个文件，每个文件都是一个1024长度的向量
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 取得文件夹下所有文件
        fileStr = fileNameStr.split('.')[0]  # 取得文件头
        classNumStr = int(fileStr.split('_')[0])  # 取得实际分类
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector("trainingDigits/%s"% fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector("testDigits/%s" % fileNameStr)  # 待分类图片
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with %d, the real answer is: %d" % (classifierResult,classNumStr)
        if(classifierResult!=classNumStr):errorCount +=1.0
    print "\n the total number of errors is:%d " % errorCount
    print "\n the total error rate is %f "% (errorCount/float(mTest))

def datingClassTest():  #约会网站kNN算法测试
    hoRatio = 0.10  #测试率 0.1
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    norMat,ranges,minVals = autoNorm(datingDataMat)
    m = norMat.shape[0]
    numTestVecs = int(m*hoRatio)  #训练数据起始点
    errorCount = 0.0
    for i in range(numTestVecs):
        #对测试数据集进行测试
        classifierResult = classify0(norMat[i,:],norMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with:%d,the real answer is :%d" % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount+=1.0
    print "the total error rate is :%f" % (errorCount/float(numTestVecs))


def classifyPerson():  # 约会网站对输入进行分类
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of the time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles,percentTats,iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ",resultList[classifierResult-1]


def main():
    # datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # #ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.show()
    # norMat,ranges,minVals = autoNorm(datingDataMat)
    # print(norMat,ranges,minVals)
    datingClassTest()
    classifyPerson()
    handwritingClassTest()

main()
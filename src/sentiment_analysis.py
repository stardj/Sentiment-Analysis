import os
import re
import sys
import copy
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

'''
Author: Yinghui Jiang

Date: 2018/03/6
'''


# deal with the document issues like reading the file splitting the sentence etc.
class Utils:
    def __init__(self):
        pass

    # read the file
    def getContext(self, fileName):
        with open(fileName, 'r+') as file:
            text = file.read()
        return text.lower()

    # get the absolute path
    def getPath(self):
        return sys.path[0]

    # get the file name list in the given path
    def getFileNames(self, dirPath):
        fileNames = []
        for root, dirs, files in os.walk(dirPath):
            fileNames = files  # get the name list
        return fileNames

    # to split the whole sentence into the word list
    def segmentDoc(self, text):
        p = re.compile(r"([A-Za-z]+)")  # move the punctuation and numbers, only keep the words
        return list(p.findall(text))  # return the list of words


# for building the feature space
class BuildFreatureSpace:
    def __init__(self):
        self.wordsList = []
        self.target = []
        self.NEGDIR = "/neg/"
        self.POSDIR = "/pos/"
        self.FILETYPE = ["NEG", "POS"]
        self.docT = {}  # for store the each doc's word-list
        self.result = []
        self.words, weight, target = self.buildFeatureSpace()
        self.getResult(weight.tolist())  # return the feature space

    # get every doc's word list and append them to one string
    def getWordsList(self):
        for vals in self.docT.values():
            strT = ""
            self.target.append(vals[0])
            for word in vals[1]:
                strT = strT + word + " "
            self.wordsList.append(strT)

    # build the term vector for each doc with their target
    def buildDocTermVec(self, tools, fileList, fileType):
        for fileName in fileList:
            val = []
            context = ""
            if (fileType == self.FILETYPE[0]):  # negtive text
                # print(fileName, ":", fileType)
                val.append(-1)  # negtive is -1
                context = tools.getContext(self.NEGDIR + fileName)
            elif (fileType == self.FILETYPE[1]):  # positive text
                # print(fileName, ":", fileType)
                val.append(1)  # positive is +1
                context = tools.getContext(self.POSDIR + fileName)
            else:
                raise  # file type error
            val.append(tools.segmentDoc(context))  # for word count
            self.docT[fileName] = val  # store the doc in memory

    # build the standard TF feature space
    def buildTF(self):
        vectorizer = CountVectorizer()
        testX = vectorizer.fit_transform(self.wordsList)
        return vectorizer.get_feature_names(), testX.toarray()

    # this is the main method for this class
    def buildFeatureSpace(self):
        tools = Utils()
        path = tools.getPath()
        self.NEGDIR = path + self.NEGDIR  # negative files' path
        self.POSDIR = path + self.POSDIR  # positive files' path
        negFileList = tools.getFileNames(self.NEGDIR)  # get absolute negative docs' name list
        posFileList = tools.getFileNames(self.POSDIR)  # get absolute positive docs' name list
        try:
            self.buildDocTermVec(tools, negFileList, self.FILETYPE[0])  # store the negative doc wordlist
            self.buildDocTermVec(tools, posFileList, self.FILETYPE[1])  # store the positive doc wordlist
        except Exception as e:
            print(e.message())
        self.getWordsList()
        wrods, weight = self.buildTF()  # get the standard TF feature space
        return wrods, weight, self.target

    # append the each doc's word vector and target
    def getResult(self, weight):
        for index in range(len(self.target)):
            self.result.append([weight[index], self.target[index]])

    # the API for other class to get the feature space
    def traning(self):
        return copy.copy(self.result), copy.copy(self.words)


# perceptron class
class Perceptron:
    def __init__(self, trainingSet, testingSet):
        self.trainingSet = trainingSet  # the training data
        self.testingSet = testingSet  # the testing data
        self.W = np.zeros(len(trainingSet[0][0]), np.float)  # init the matrix W
        self.b = 0  # init b

    # traning perceotron
    def trainPerceptron(self, loopNums):  # loopNums is the cycle index
        for i in range(loopNums):
            random.shuffle(self.trainingSet)
            flag = True  # flag for fitting
            for X, y in self.trainingSet:
                if self.check(X, y):  # check the value of loss function
                    self.update(X, y)  # update parameters
                    flag = False
            if (flag):  # if every sample have been classify correctly, break loop
                break
        return self.W, self.b

    # the loss function
    def check(self, X, y):
        temp = y * (((np.array(self.W) * (np.array(X))).sum()) + self.b)
        if temp <= 0:
            return True
        else:
            return False

    # the update function
    def update(self, X, y):
        self.W += y * np.array(X)
        self.b += y


class Test:
    def __init__(self):
        self.data = BuildFreatureSpace()
        self.dataSet, self.wordsVec = self.data.traning()  # get the standard TF feature space
        self.trainData = self.dataSet[:800]
        self.trainData += self.dataSet[1000:1800]  # the traning data
        self.testData = self.dataSet[800:1000]
        self.testData += self.dataSet[1800:2000]  # the testing data
        self.W = [0 for i in range(len(self.trainData[0][0]))]  # init W
        self.b = 0  # init b

    # method for calculate accuracy
    def accuracy(self):
        acc = 0
        for X, y in self.testData:
            temp = y * (((np.array(self.W) * (np.array(X))).sum()) + self.b)
            if temp >= 0:
                acc += 1
        result = (acc + 0.0) / len(self.testData)
        print("accuracy: ", result)
        return result

    # method for calculate recall
    def recall(self):
        posT = 0
        negF = 0
        for X, y in self.testData:
            temp = (((np.array(self.W) * (np.array(X))).sum()) + self.b)
            if temp > 0:
                if y == 1:
                    posT += 1
                else:
                    negF += 1
        print("recall: ", (posT + 0.0) / (posT + negF))

    # method for calculate top 10 of positive words and negative words
    def topTen(self):
        temp = {}
        for i in range(len(self.W)):
            temp[self.wordsVec[i]] = self.W[i]  # make the W and the corresponded word into a dictionary
        result = sorted(temp.items(), key=lambda x: x[1], reverse=True)  # sort the W
        posTop = result[:10]  # the first 10 words corresponding to the W
        negTop = result[::-10]  # the last 10 words corresponding to the W
        posT = ""
        negT = ""
        for i in range(10):
            posT += posTop[i][0] + " "
            negT += negTop[i][0] + " "
        print("The positive top 10 is: ", posT)
        print("The negative top 10 is: ", negT)


if __name__ == "__main__":
    acc = []
    startTime = time.time()  # start time
    t = Test()
    p = Perceptron(t.trainData, t.testData)  # traning perceptron
    # for i in range(15):
    #     t.W, t.b = p.trainPerceptron(i)  # get the W and b
    #     if (i == 0):
    #         acc.append(0.0)
    #     else:
    #         print("loop {}".format(i))
    #         acc.append(t.accuracy())

    t.W, t.b = p.trainPerceptron(50)
    print("recycle : {} times".format(50))
    print("######## The final result #######")
    t.accuracy()
    t.recall()
    t.topTen()
    aaa = [i for i in range(15)]
    plt.plot(aaa, acc, 'r-')
    endTime = time.time()  # end time
    print(endTime - startTime, "s")  # show the time cost
    plt.show()
    plt.pause(5)
    plt.close()

# coding = utf-8
# author = zhangchenzuo
# data = 2020/05/07
# email = chenzuozhang@buaa.edu.cn

"""
最大熵分类，对数字多分类
"""
import pandas as pd
import numpy as np

import time

from collections import defaultdict
from sklearn.model_selection import train_test_split


class MaxEnt():
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.labels_ = set(labels)  # 判断分类数量
        self.featureNum = len(features[0])  # 特征个数
        self.sampleNum = len(labels)  # 样本数
        self.n = 0  # 不重复的样本个数，在函数calfixy中更新
        self.M = 10000  # 学习率
        self.fixydic = self.calfixy()  # 记录所有不同的键值对
        self.w = [0]*self.n  # 因为一共有n个不同的样本数，因此也需要这么多w
        self.xy2id, self.id2xy = self.builddict()  # 定义了两个字典，方便每次定向的更新w
        self.Ep_xy = self.calcEp_xy()  # p_xy是在给定数据集之后就确定的，不随着w而改变。是数据集的经验分布。
        self.iter = 10  # 迭代次数

    def calfixy(self):
        """
        统计出现的不同的键值对
        :return:
        """
        # 首先按照不同的特征划分，然后每个特征对应一个字典，字典的键是（x,y）
        fixydict = [defaultdict(int) for _ in range(self.featureNum)]
        for i in range(self.sampleNum):
            for j in range(self.featureNum):
                fixydict[j][(self.features[i][j], self.labels[i])] += 1   # 第j个特征，出现键值对（x，y）的次数
        # 更新总共的不同键值对数量，再init中需要利用这个数据构建等多的Wi
        for i in fixydict:
            self.n += len(i)
        return fixydict

    def builddict(self):
        """
        建立字典，每一个不同的特征下的特征值和标签对，对应一个编号
        :return:
        """
        xy2id = [{} for _ in range(self.featureNum)]
        id2xy = {}
        index = 0
        for j in range(self.featureNum):
            for (x,y) in self.fixydic[j]:
                xy2id[j][(x, y)] = index
                id2xy[index] = (x, y)
                index += 1
        return xy2id, id2xy

    def calcEp_xy(self):
        """
        82页最下方的公式
        计算期望，在所有的特征数下，某键值对次数除以总的样本数
        :return:
        """
        Ep_xy = [0]*self.n
        for i in range(self.featureNum):
            for (x,y) in self.fixydic[i]:
                id = self.xy2id[i][(x,y)]
                Ep_xy[id] = self.fixydic[i][(x,y)]/self.sampleNum  # 这里分母是样本数是因为第i个特征，键值对的最大就是样本数
        return Ep_xy

    def calcEpxy(self):
        """
        计算书83页最上面那个期望
        需要调用函数，calcpwy_x 用于计算Pw(y|x)。并且是分别对每个样本都进行一次计算
        :return:
        """
        Epxy = [0]*self.n
        # 对每一个样本进行遍历
        for i in range(self.sampleNum):
            # 计算出该样本的P(y|x)，根据6.22
            Pwy_xs = self.calcpwy_x(self.features[i])
            # 对样本中的每一个特征进行遍历
            for j in range(self.featureNum):
                # 第i个样本的第j个特征的值
                x = self.features[i][j]
                for Pyx, y in Pwy_xs:
                    if (x, y) in self.xy2id[j]:
                        id = self.xy2id[j][(x, y)]
                        Epxy[id] += Pyx * (1.0 / self.sampleNum)
        return Epxy

    def calcpwy_x(self, feature):
        """
        计算书85页公式6.22
        调用函数cal_pxy计算对于不同的y情况下，exp(wifi(x,y))
        :param feature:某一个样本的全部特征
        :return:
        """
        Pyxs = [(self.cal_pyx(feature, y)) for y in self.labels_]
        Z = sum([prob for prob, y in Pyxs])  # 计算Z(w)
        return [(prob / Z, y) for prob, y in Pyxs]

    def cal_pyx(self, feature, y):
        """
        计算w_if_i(x,y),对于每一个y，计算与这个y有关的所有键值对
        :return:
        """
        result = 0
        for i in range(self.featureNum):
            x = feature[i]
            if (x, y) in self.xy2id[i]:
                id = self.xy2id[i][(x, y)]
                result += self.w[id]  # 因为只要出现过，就可以视为1
        return (np.exp(result), y)

    def maxEntropyTrain(self):
        for i in range(self.iter):
            print('iterater times %d' % i)
            Epxy = self.calcEpxy()  # 因为是与w的值有关，因此每次更新完w的值都需要重新进行计算
            simgaList = [0]*self.n # 采用IIS算法，初始化sigma

            for j in range(self.n):
                simgaList[j] = (1/self.M)*np.log(self.Ep_xy[j]/Epxy[j])
            self.w = [self.w[k] + simgaList[k] for k in range(self.n)]

    def predict(self, features):
        """
        根据得到的pw(y|x)，求解概率最大的y
        :param features:  某一个样本的全部特征
        :param labels_test:  全体标签集
        :return:
        """
        result = self.calcpwy_x(features)
        return max(result, key=lambda x: x[0])[1]

    def test(self,features_test, labels_test):
        """
        计算预测准确度
        :param features_test: 全体特征集
        :param labels_test: 全体标签集
        :return: 返回预测准确度
        """
        n = len(labels_test)
        corr = 0
        for i in range(n):
            result = self.predict(features_test[i])
            #print(result, labels_test[i])
            if result == labels_test[i]:
                corr += 1
        return corr/n


if __name__ == "__main__":

    print('Start read data')

    time_1 = time.time()

    #raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    imgs = data[0:100, 1:]
    labels = data[0:100, 0]
    # 对数值进行归一化
    features = np.zeros((np.shape(imgs)[0], np.shape(imgs)[1]))
    for i in range(np.shape(imgs)[0]):  # 这里采用了归一化，防止一些数值太大，尤其是exp(wx)
        features[i] = [float(int(num)/255) for num in imgs[i]]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.33, random_state=1)

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    met = MaxEnt(train_features, train_labels)
    met.maxEntropyTrain()

    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    score = met.test(test_features, test_labels)
    print("The accruacy socre is ", score)

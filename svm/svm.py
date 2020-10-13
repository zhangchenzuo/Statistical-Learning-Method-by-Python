# coding = uft-8
# author = chenzuozhang
# date = 2020/05/21
# email = chenzuozhang@buaa.edu.cn

"""
利用mnist数据集实现svm分类器，实现0与非0的二分类
"""

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
# 以下作为对比
from sklearn import svm
from sklearn.metrics import accuracy_score
import random


class SVM(object):
    def __init__(self, features, labels, sigma=10, C=200, toler=0.001):
        """
        :param features: 训练集的特征，n*m  n个样本，m个特征
        :param labels: 训练集的标签，n*1 n个样本
        :param sigma:  高斯核中的σ
        :param C:  软间隔的惩罚参数
        :param toler:  不是松弛变量，应该是一个容差
        """
        self.features = features
        self.labels = labels
        self.sample_num, self.feature_num = np.shape(features)
        self.sigma = sigma
        self.C = C
        self.toler = toler

        self.Kernel = self.calKernel()
        self.alpha = [0]*self.sample_num
        self.b = 0
        self.E = [0]*self.sample_num
        self.supportVecIndex = []

    def calKernel(self):
        """
        文中公式7.90，采用高斯核
        计算核函数
        :return: 返回得到的核函数，形式是一个矩阵，n*n，n是样本数
        """
        n = self.sample_num
        kernel = np.zeros((n,n))
        for i in range(n):
            x = self.features[i]
            for j in range(i, n):
                z = self.features[j]
                x_z = np.dot((x-z), (x-z).T)
                result = np.exp(-1*x_z/(2*self.sigma**2))
                kernel[i][j] = result
                kernel[j][i] = result
        return kernel


    def isSatisfyKKT(self, i):
        """
        公式（7.111-113）
        判断第i个alpha是否符合KKT
        这里考虑了容差，在一定的范围内就设为0
        :return:
        """
        gxi = self.cal_gxi(i)
        yi = self.labels[i]
        alpha = self.alpha[i]

        if abs(alpha)<self.toler and yi*gxi >= 1:
            return True
        elif -self.toler < alpha < self.C+self.toler and abs(yi*gxi-1) < self.toler:
            return True
        elif abs(alpha-self.C) < self.toler and yi*gxi <= 1:
            return True
        return False

    def cal_gxi(self, i):
        """
        公式(7.104)
        计算第样本的预测值
        :param i:
        :return:
        """
        gxi = 0
        for j in range(self.sample_num):
            if self.alpha[j] != 0:
                alpha = self.alpha[j]
                yi = self.labels[j]
                k = self.Kernel[i][j]
                gxi += alpha*yi*k
        return gxi+self.b

    def cal_Ei(self, i):
        """
        公式(7.105)
        计算标签值与目前预测值之差
        :param i:
        :return:
        """
        gxi = self.cal_gxi(i)
        Ei = gxi - self.labels[i]
        return Ei

    def getAlphaJ(self, i, E1):
        """
        选择alpha2
        :return:
        """
        E2 = 0
        maxE1_E2 = -1
        maxIndex = -1
        # 首先进行查表，记录下来已经有的
        nozeroE = [i for i, Ei in enumerate(self.E) if Ei != 0]
        for j in nozeroE:
            E2_temp = self.cal_Ei(j)
            if abs(E1 - E2_temp) > maxE1_E2:
                # 更新最大值
                maxE1_E2 = abs(E1 - E2_temp)
                E2 = E2_temp
                maxIndex = j
        # 对于初始情况，随便选取一个作为E2
        if maxIndex == -1:
            maxIndex = i
            while maxIndex == i:
                maxIndex = int(random.uniform(0, self.sample_num))
            E2 = self.cal_Ei(maxIndex)
        return E2, maxIndex

    def train(self, iter=200):
        """
        按照SMO算法进行迭代训练
        Smo模型进行训练找到所有的支持向量
        主要涉及公式(7.106,107 108 109)
        :return:
        """
        iterstep = 0
        parachanged = 1

        while iterstep < iter and parachanged > 0:
            if iterstep % 100 == 0:
                print('iter:%d:%d'%(iterstep, iter))
            iterstep += 1
            parachanged = 0

            for i in range(self.sample_num):
                if self.isSatisfyKKT(i):
                    continue
                E1 = self.cal_Ei(i)
                E2, j = self.getAlphaJ(i, E1)

                alpha1_old = self.alpha[i]
                alpha2_old = self.alpha[j]
                y1 = self.labels[i]
                y2 = self.labels[j]

                if y1 != y2:
                    L = max(0, alpha2_old - alpha1_old)
                    H = min(self.C, self.C + alpha2_old - alpha1_old)
                else:
                    L = max(0, alpha2_old + alpha1_old - self.C)
                    H = min(self.C, alpha1_old + alpha2_old)
                if L == H:
                    continue
                K11 = self.Kernel[i][i]
                K12 = self.Kernel[i][j]
                K21 = K12
                K22 = self.Kernel[j][j]

                eta = K11 + K22 - 2 * K12
                alpha2_new_un = alpha2_old + y2 * (E1 - E2) / eta
                if alpha2_new_un > H:
                    alpha2_new = H
                elif alpha2_new_un < L:
                    alpha2_new = L
                else:
                    alpha2_new = alpha2_new_un
                # 公式(7.109)
                alpha1_new = alpha1_old + y1 * y2 * (alpha2_new - alpha2_old)

                b1_new = -1 * E1 - y1 * K11 * (alpha1_new - alpha1_old) - y2 * K21 * (alpha2_new - alpha2_old) + self.b
                b2_new = -1 * E2 - y1 * K12 * (alpha1_new - alpha1_old) - y2 * K22 * (alpha2_new - alpha2_old) + self.b
                if 0 < alpha2_new < self.C:
                    bnew = b2_new
                elif 0 < alpha1_new < self.C:
                    bnew = b1_new
                else:
                    bnew = (b1_new + b2_new) / 2

                self.b = bnew
                self.alpha[i] = alpha1_new
                self.alpha[j] = alpha2_new
                self.E[i] = self.cal_Ei(i)
                self.E[j] = self.cal_Ei(j)
                if abs(alpha2_new-alpha2_old) >= 0.00001:
                    parachanged += 1

        for i in range(self.sample_num):
            if self.alpha[i] > 0:
                self.supportVecIndex.append(i)

    def calsiglekerner(self, x, y):
        result = np.dot((x-y), (x-y).T)
        result = np.exp(-1*result/(2*self.sigma**2))
        return result

    def predict(self, feature):
        result = 0
        for i in self.supportVecIndex:
            temp = self.calsiglekerner(feature, self.features[i])
            result += self.labels[i]*self.alpha[i]*temp
        if result+self.b > 0:
            return 1
        else:
            return 0

    def test(self, features, labels):
        n, m = np.shape(features)
        corr = 0
        for i in range(n):
            result = self.predict(features[i])
            # if labels[i] == 0:
            #     print(result, labels[i])
            if result == labels[i]:
                corr += 1
        return corr/n


if __name__ == "__main__":
    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    # raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    imgs = data[0:1000, 1:]
    labels = data[0:1000, 0]
    # 对数值进行归一化
    features = np.zeros((np.shape(imgs)[0], np.shape(imgs)[1]))
    for i in range(np.shape(imgs)[0]):  # 这里采用了归一化，防止一些数值太大，尤其是exp(wx)
        features[i] = [float(int(num) / 255) for num in imgs[i]]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.33, random_state=1)

    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    met = SVM(train_features, train_labels)
    met.train()

    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')

    print('Start predicting')
    score = met.test(test_features, test_labels)
    print("The accruacy socre is ", score)

    print('Use package')
    time_2 = time.time()
    clf = svm.SVC()  # svm class
    clf.fit(train_features, train_labels)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)

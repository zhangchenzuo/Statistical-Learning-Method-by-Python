# coding = utf-8
# auther = zhangchenzuo
# data = 2020/04/08
# email = chenzuozhang@buaa.edu.cn

'''
利用minist实现数字二分类
多层感知机的练习
'''

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split


class Perceptron(object):

    def __init__(self):
        self.learn_rate = 0.001
        self.max_iter = 50  # 最大迭代次数

    def train(self, feature, label):
        feature = np.array(feature)
        label = np.array(label)
        m, n = np.shape(feature)  # 行列
        self.w = np.zeros((1, n))  # 1行n列
        self.b = 0

        for _ in range(self.max_iter):
            for i in range(m):
                result = np.dot(self.w, feature[i].T)+self.b  # 为了实现w*x这里需要将x转置
                if label[i]*result <= 0:  # 根据书中的算法，如果小于0表示归类错误，需要进行随机梯度下降
                    self.w += self.learn_rate*label[i]*feature[i]
                    self.b += self.learn_rate*label[i]

    def test(self, feature, label):
        feature = np.array(feature)
        label = np.array(label)
        m, n = np.shape(feature)
        correct = 0
        for i in range(m):
            result = label[i]*(np.dot(self.w, feature[i])+self.b)
            if result > 0:
                correct += 1
        return correct/m


if __name__ == '__main__':
    print('Start read data')
    time1 = time.time()
    raw_data = pd.read_csv('../data/train.csv', header=0)
    #raw_data = pd.read_csv('../data/train_binary.csv', header=0)  # 二分类数据集
    data = np.array(raw_data.values)
    # 注意区别矩阵和列表
    features = data[:, 1:]
    labels = data[:, 0]
    # print(label)

    for i in range(len(labels)):
        if labels[i] <= 5:
            labels[i] = -1
        else:
            labels[i] = 1
    # 划分训练集和测试集
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, random_state=1)

    time2 = time.time()
    print('read data cost', time2-time1, 's', '\n')

    print('Start train')
    p = Perceptron()
    p.train(train_x, train_y)
    time3 = time.time()
    print('train cost', time3-time2)

    print('Start test')
    result = p.test(test_x, test_y)
    print('The accrucy socre is', result)
    print('test cost', time.time()-time3)



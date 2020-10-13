# coding = utf-8
# author = zhangchenzuo
# data = 2020/04/29
# email = chenzuozhang@buaa.edu.cn

'''
实现朴素贝叶斯算法，对图片矩阵的数值进行了二值化
'''

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split


class NaiveBayes(object):
    def __init__(self):
        self.classnum = 10  # 分类种类
        self.lamba = 1  # 拉普拉斯平滑
        self.choice_num = 2 # 每个特征可能的取值个数，这里由于采用了二值化，因此都是2

    def train(self, features, labels):
        '''
        :param features: 特征
        :param labels: 标签
        :return: p_yk 是训练集中标签的分布 （1*classnum），p_xy是训练集中，计算出的条件概率分布。（classnum * featurnum * valuenum）
        * 注意： 返回的两个矩阵都取了log，为了防止数值太小*
        相当于利于训练集计算处一个表格，推断新的数据只需要查表
        '''
        n, m = np.shape(features)
        p_yk = np.zeros((1, self.classnum)) # 计算 Y_ck的概率，也就是不同数字的概率
        for i in range(self.classnum):
            p_yk[0,i] = (np.sum(labels == i)+self.lamba)/(n+self.classnum*self.lamba)

        # 统计label=1，第j个特征，为x的个数
        count_x = np.zeros((self.classnum, m, self.choice_num))
        for i in range(n):
            for j in range(m):
                label = labels[i]
                count_x[label][j][features[i, j]] += 1

        # 根据上面的统计计算相应的概率
        p_xy = np.zeros((self.classnum, m, self.choice_num))
        for i in range(self.classnum):
            for j in range(m):
                p_xy[i][j][0] = (count_x[i][j][0]+self.lamba)/(np.sum(labels == i)+self.choice_num*self.lamba)
                p_xy[i][j][1] = (count_x[i][j][1] + self.lamba) / (np.sum(labels == i) + self.choice_num * self.lamba)

        p_xy = np.log(p_xy)
        p_yk = np.log(p_yk)
        return p_yk, p_xy

    def test(self, p_yk, p_xy, features, labels):
        '''
        :return: 返回预测准确率
        '''
        n, m = np.shape(features)
        corr = 0
        for i in range(n):
            predict_label = self.calculate(features[i], p_yk, p_xy)
            if predict_label == labels[i]:
                corr += 1
        return corr/n

    def calculate(self, feature, p_yk, p_xy):
        '''
        计算待求数据的值，注意这里都已经转化为log，因此概率相乘变成了相加
        '''
        ans = [0]*self.classnum
        for i in range(self.classnum):
            # 计算相应的条件概率分布
            for j in range(len(feature)):
                ans[i] += p_xy[i][j][feature[j]]
            # 加上先验分布
            ans[i] += p_yk[0, i]
        return ans.index(max(ans))

if __name__ == '__main__':
    print('Star')
    time1 = time.time()

    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    features = np.array(data[:, 1:])
    labels = np.array(data[:, 0])
    n, m = np.shape(features)

    for i in range(n):
        for j in range(m):
            if features[i, j] >= 128:
                features[i, j] = 1
            else:
                features[i, j] = 0
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=1)

    time2 = time.time()
    print('read data cost', time2-time1, '\n')

    print('Star train ')
    n = NaiveBayes()
    p_yk, p_xy = n.train(train_features, train_labels)
    time3 = time.time()
    print('train cost', time3-time2, '\n')

    print('Star test')
    result = n.test(p_yk, p_xy, test_features, test_labels)
    time4 = time.time()
    print('test cost', time4-time3, '\n')
    print('acc',result)


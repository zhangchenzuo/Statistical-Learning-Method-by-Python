# coding = utf-8
# author = zhangchenzuo
# date = 2020/05/14
# email = chenzuozhang@buaa.edu.cn

"""
利用mnist数据率，实现adaboost算法。
对数据集进行二分类，所有特征数值进行二值化
在训练集样本800，测试集200时，0标签0.7391，1标签0.9944
总正确率为0.965
"""
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split


class AdaBoost(object):
    def __init__(self, features, labels, iternum=15):
        self.sample_num, self.feature_num = np.shape(features)
        self.features = features
        self.labels = labels
        self.iternum = iternum
        self.D = [1/self.sample_num]*self.sample_num

    def cal_em(self, div, rule, features, labels):
        """
        计算基于某个特征进行判断的错误率
        :param div:  分类尺度
        :param rule:  判断类型
        :param features:  全部样本的某一特征数据
        :param labels:  全部样本的标签
        :return:
        """
        predict = []
        if rule == 'Lisone':
            L = 1
            H = -1
        else:
            L = -1
            H = 1
        em = 0
        for i in range(self.sample_num):
            if features[i]<div:
                predict.append(L)
                if labels[i] != L:
                    em += self.D[i]
            else:
                predict.append(H)
                if labels[i] != H:
                    em += self.D[i]
        return np.array(predict), em

    def buildsingletree(self):
        """
        构建单独的树，最简单的树桩
        :return:
        """
        singletree = {}  # 构建一个树，形式是字典
        # 初始化分类误差率
        singletree['em'] = 1
        # 寻找最适合分类的特征
        for i in range(self.feature_num):
            for div in [-0.5, 0.5, 1.5]:
                for rule in ['Lisone', 'Hisone']:
                    feature = [self.features[j][i] for j in range(self.sample_num)]
                    predict, em = self.cal_em(div, rule, feature, self.labels)
                    if em < singletree['em']:
                        singletree['em'] = em
                        singletree['div'] = div
                        singletree['rule'] = rule
                        singletree['feature_index'] = i
                        singletree['Gm'] = predict
        return singletree

    def buildBoosttree(self):
        """
        生成boost树
        :return:
        """
        tree = []
        finallpredict = [0]*self.sample_num
        for i in range(self.iternum):
            curTree = self.buildsingletree()
            em = curTree['em']
            Gm = curTree['Gm']
            alpha = 0.5 * np.log((1-em)/em)
            self.D = np.multiply(self.D, np.exp(-1 * alpha * np.multiply(self.labels, Gm)))
            self.D = self.D/sum(self.D)
            curTree['alpha'] = alpha
            tree.append(curTree)

            # -----以下代码用来辅助，可以去掉---------------
            # 根据8.6式将结果加上当前层乘以α，得到目前的最终输出预测
            finallpredict += alpha * Gm
            # 计算当前最终预测输出与实际标签之间的误差
            error = sum([1 for i in range(self.sample_num) if np.sign(finallpredict[i]) != self.labels[i]])
            # 计算当前最终误差率
            finallError = error / self.sample_num
            # 如果误差为0，提前退出即可，因为没有必要再计算算了
            # 打印一些信息
            print('iter:%d:%d, sigle error:%.4f, finall error:%.4f' % (i, self.iternum, curTree['em'], finallError))
            if finallError == 0:
                return tree

        # 返回整个提升树
        return tree

    def predict(self, tree, feature):
        """
        实现预测，根据adaboost生成的树
        :param tree:
        :param feature:
        :return:
        """
        n = len(tree)
        result = 0
        for i in range(n):
            curtree = tree[i]
            rule = curtree['rule']
            feature_index = curtree['feature_index']
            div = curtree['div']
            alpha = curtree['alpha']
            if rule == 'Lisone':
                L = 1
                H = -1
            else:
                L = -1
                H = 1
            if feature[feature_index] < div:
                result += alpha * L
            else:
                result += alpha * H
        return np.sign(result)

    def test(self, features, labels, tree):
        """
        计算测试集的准确性，分别计算了对于两个不同标签的准确率
        :param features:
        :param labels:
        :param tree:
        :return:
        """
        corr = 0
        corr_0 = 0
        corr_1 = 0
        a0 = 0
        a1 = 0
        n = np.shape(features)[0]
        for i in range(n):
            result = self.predict(tree, features[i])
            if labels[i] == -1:
                a0 += 1
                if result == labels[i]:
                    corr_0 += 1
            elif labels[i] == 1:
                a1 += 1
                if result == labels[i]:
                    corr_1 += 1
            if result == labels[i]:
                corr += 1
        return corr/n, corr_0/a0, corr_1/a1


if __name__ == '__main__':
    print('Start read data')
    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = data[0:1000, 1:]
    labels = data[0:1000, 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    # 对数值进行二值化，大于128是1，小于为0
    features = []
    for i in range(np.shape(imgs)[0]):
        features.append([int(int(num) > 128) for num in imgs[i]])
    features = np.array(features)

    # 对标签值进行修改，0设置为-1，1还是1
    label_ = []
    for i in range(len(labels)):
        if labels[i] == 0:
            label_.append(-1)
        else:
            label_.append(1)
    label_ = np.array(label_)
    #a = [features[i][1] for i in range(np.shape(imgs)[0])]
    train_features, test_features, train_labels, test_labels = train_test_split(features, label_, test_size=0.2, random_state=1)



    time_2 = time.time()
    print('read data cost ', time_2 - time_1, ' second', '\n')

    print('Start training')
    ada = AdaBoost(train_features, train_labels)
    tree = ada.buildBoosttree()

    time_3 = time.time()
    print('training cost ', time_3 - time_2, ' second', '\n')


    print('Start predicting')
    score, acc_0, acc_1 = ada.test(test_features, test_labels, tree)
    time_4 = time.time()
    print('predicting cost ', time_4 - time_3, ' second', '\n')
    print("The accruacy socre is ", score)
    print("for 0 acc is %0.4f, for 1 acc is %0.4f" %(acc_0, acc_1))


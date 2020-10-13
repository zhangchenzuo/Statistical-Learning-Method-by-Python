# coding = utf-8
# author = zhangchenzuo
# data = 2020/04/30
# email = chenzuozhang@buaa.edu.cn

'''
在mnist数据集上，实现决策树ID3算法
未包含剪枝，未包含回归问题
'''

import numpy as np
import pandas as pd
import time
import logging
from sklearn.model_selection import train_test_split


# 采用日志的方式进行计时
def log(func):
    def wrapper(*args, **kwargs):
        star_time = time.time()
        logging.debug('Star %s' % func.__name__)
        ret = func(*args, **kwargs)

        end_time = time.time()
        logging.debug('End %s, cost %s s' % (func.__name__, end_time-star_time))
        return ret
    return wrapper


class Tree(object):
    # 利用树的形式进行存储
    def __init__(self, node, Class=None, feature=None):
        self.node = node  # 每个节点的属性，是否属于叶节点
        self.Class = Class  # 只有在叶节点才有意义，存储具体的分类结果
        self.feature = feature  # 只有在内节点才有意义，当前内节点的分类标准是什么(索引值)
        self.dic = {}  # 存储子树

    def add_tree(self, feature_value, sub_tree):
        '''
        将子树存入主树
        :param feature_value:  该节点对应特征的特征值
        :param sub_tree:  完成构建的子树
        '''
        self.dic[feature_value] = sub_tree  # 按照不同的设置存储不一样的子树

    def predict(self, features):
        '''
        在树的结构上进行迭代查找
        :param features: 某个样本的全部特征 1*n
        :return: 整体时采用的递归的方法，类似并查集，最终返回值是叶节点的分类。也就是该样本的预测分类
        '''
        if self.node == 'leaf':
            return self.Class
        tree = self.dic[features[self.feature]]
        return tree.predict(features)


class DecisionTree(object):
    def __init__(self):
        self.bar = 0.1  # 信息熵的门限

    def major_class(self, labels):
        '''
        得到当前集合中，数量最多的标签作为改分类下的预测值
        :param labels: n*1
        :return: 返回当前样本集下类型最多的类型
        '''
        dic = {}
        for i in labels:
            if i not in dic:
                dic[i] = 1
            else:
                dic[i] += 1
        ans = list(dic.items())
        ans.sort(key=lambda x:-x[1])
        return ans[0][0]


    def cal_H_D(self, labels):
        '''
        计算样本集的信息熵
        :param labels: 整个训练集的样本 n*1
        :return: 返回训练集的信息熵
        '''
        n = len(labels)
        label_set = set(label for label in labels)
        H_D = 0
        for label in label_set:
            p = labels[labels == label].size/n
            H_D -= p*np.log2(p)
        return H_D

    def cal_H_D_A(self, feature, labels):
        '''
        计算基于某给定特征的条件概率的熵
        :param feature: n多样本的某一条特征的取值 n*1
        :param labels: 对应的标签 n*1
        :return: 当前给定特征的条件概率熵
        '''
        n = len(labels)
        feature_set = set(i for i in feature)
        H_D_A = 0
        for i in feature_set:
            temp = labels[feature == i]  # 相当于得到对应索引的标签值
            p = self.cal_H_D(temp)
            H_D_A += (feature[feature == i].size/n) * p
        return H_D_A

    def get_best_feature(self, features, labels):
        '''
        得到最大信息增益的特征，在ID3中是基于信息增益计算的
        :param features: n*m  n个样本每个样本有m个特征
        :param labels: n*1对应的标签
        :return: G_max最大的信息增益，feature_index 对应的特征索引 是[0,m-1]中的哪一个
        '''
        n, m = np.shape(features)
        H_D = self.cal_H_D(labels)
        G_max = -1
        feature_index = -1
        for i in range(m):
            H_D_A = self.cal_H_D_A(np.array(features[:, i].flat), labels)
            G_H_A = H_D - H_D_A
            if G_H_A>G_max:
                G_max = G_H_A
                feature_index = i
        return G_max, feature_index

    def build_tree(self, features, labels, res_feature_index):
        """
        !!核心函数，构建子树，递归的构建
        :param features: n*m 所用的样本的所有特征
        :param labels:  n*1 所有的样本的标签
        :param res_feature_index:  剩余的还没有进行划分的标签位置的索引 最初是[0, m-1]
        :return: 返回是一个树结构
        """
        n, m = np.shape(features)
        # 首先选出来最大信息增益是哪个特征
        G_max, feature_index = self.get_best_feature(features, labels)

        # 1. 如果所有节点属于同一类
        if len(set(labels)) == 1:
            # print('one class',labels[0])
            return Tree(node='leaf', Class=labels[0])

        # 2. 如果只有一个特征
        if len(res_feature_index) == 0:
            # print('one feature', self.major_class(labels))
            return Tree(node='leaf', Class=self.major_class(labels))

        # 3. 信息增益小于门限值
        if G_max<self.bar:
            # print('too small', self.major_class(labels))
            return Tree(node='leaf', Class=self.major_class(labels))

        # 构建子树,循环调用函数
        # 首先需要从可用的特征中，删除该特征列，然后对与特征的可能取值构建相应的子树
        res_feature_index = list(filter(lambda x:x != feature_index, res_feature_index))
        # print('do tree',len(res_feature_index))

        # 注意此时的节点都是内节点，我们需要给出该节点进行分类的特征所以
        tree = Tree(node='inter', feature=feature_index)
        feature_values = set(value for value in features[:, feature_index])

        # 这里是对某一特征的所有特征值都进行操作
        for feature_value in feature_values:
            index = []
            for i in range(n):
                if features[i, feature_index] == feature_value:
                    index.append(i)
            # ！！！！核心，对样本进行分类，符合给定特征的特征值的样本放在一起
            labels_next = labels[index]
            features_next = features[index]
            # 对着分支下数据在有进步构建子树
            sub_tree = self.build_tree(features_next, labels_next, res_feature_index)
            # 将改子树添加到当前树下
            tree.add_tree(feature_value, sub_tree)
        return tree

    @log
    def train(self, fetures_train, labels_train, res_feature_index):
        """
        这个函数是为了方便调用接口，这样可以用日志统计时间
        :param fetures_train:
        :param labels_train:
        :param res_feature_index:
        :return: 只有在整个递归函数完成全部的递归任务时候才返回，返回值是树的结构
        """
        return self.build_tree(fetures_train, labels_train, res_feature_index)

    @log
    def test(self, features, labels, tree):
        """
        统计预测结果
        :param features: 测试集的所有样本和全部特征
        :param labels:  测试集的全部样本的全部标签
        :param tree: 构建完成得到的树
        :return:
        """
        corr = 0
        n = len(labels)
        for i in range(n):
            tmp_predict = tree.predict(features[i, :])
            if tmp_predict == labels[i]:
                corr += 1
        return corr/n


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
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

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=1)
    DT = DecisionTree()
    print('star train')
    tree = DT.train(features_train, labels_train, [i for i in range(m)])
    print('end train')
    acc = DT.test(features_test, labels_test, tree)
    print('accuracy is', acc)

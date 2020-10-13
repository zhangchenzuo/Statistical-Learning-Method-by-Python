# coding = utf-8
# author = zhangchenzuo
# data = 2020/05/04
# email = chenzuozhang@buaa.edu.cn

"""
实现逻辑斯蒂回归，二分类
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time


class LogisticRegression():
    def __init__(self):
        self.iter = 100
        self.learning_rate = 0.01

    def train(self, features_train, labels_train):
        n, m = np.shape(features_train)
        w = np.zeros((1, m+1))
        for _ in range(self.iter):
            for i in range(n):
                # 矩阵是无法进行append的，需要先转换为list
                x = list(features_train[i])
                x.append(1)
                x = np.array(x)
                y = labels_train[i]
                wx = np.exp(np.dot(w, x))  # np.dot是点乘，每个元素相乘并加和
                yx = y*x
                # yx = np.dot(y, x)  # 这里也需要采用点乘，如果是对于列表的形式
                # !!!! 这里需要区分矩阵和列表的一个区别，对于列表2*[1,1]是[1,1,1,1]。而对于矩阵是[2, 2]
                # 对似然函数极大化，求导可以得到 yx-(x*exp(w*x))/1+exp(w*x)
                w += self.learning_rate * (yx - (wx * x)/(1+wx))
        return w

    def predic(self, w, feature):
        feature = list(feature)
        feature.append(1)
        wx = np.dot(w, feature)
        y_predict = np.exp(wx)/(1+np.exp(wx))
        if y_predict > 0.5:
            return 1
        return 0

    def test(self, features_test, labels_test, w):
        corr = 0
        n, m = np.shape(features_test)
        for i in range(n):
            y_label = labels_test[i]
            y_predict = self.predic(w, features_test[i])
            if y_label == y_predict:
                corr += 1
        return corr/n


if __name__ == "__main__":
    print('Start read data')

    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    imgs = np.array(data[0:100, 1:])
    features = np.zeros((np.shape(imgs)[0], np.shape(imgs)[1]))
    for i in range(np.shape(imgs)[0]):  # 这里采用了归一化，防止一些数值太大，尤其是exp(wx)
        features[i] = [float(int(num)/255) for num in imgs[i]]
    labels = np.array(data[0:100, 0])

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=1)
    time_2 = time.time()
    print('read data cost', time_2-time_1)
    print('Star train')
    LR = LogisticRegression()
    w = LR.train(train_features, train_labels)
    time_3 = time.time()
    print('training cost', time_3-time_2)
    print('Star test')
    result = LR.test(test_features, test_labels, w)
    print('predicting cost ', time.time() - time_3)

    print('The accruacy is', result)
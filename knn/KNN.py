# coding = utf-8
# author = Zhangchenzuo
# data = 2020/04/28
# email = chenzuozhang@buaa.edu.cn

'''
实现knn
'''
import numpy as np
import pandas as pd
import time
import heapq


class KNN(object):
    def __init__(self):
        self.k = 10

    def test(self, train_data, train_label, test_data, test_label):
        m, n = np.shape(train_data)
        test_m, test_n = np.shape(test_data)
        acc = 0
        for j in range(test_m):
            dis = []
            for i in range(m):  # 依次计算待求样本点和所有已经标记的样本点的距离 （欧氏距离）
                distance = np.sqrt(np.sum(np.square(train_data[i]-test_data[j])))
                dis.append((distance, train_label[i]))
            result = list(heapq.nsmallest(self.k, dis))  # 利用堆得到前10个
            # 进行表决投票
            count = [0] * self.k
            for i in range(self.k):
                count[result[i][1]] += 1
            predict_result_index = max(count)
            predict_result = count.index(predict_result_index)
            # 进行判断是否准确
            if predict_result == test_label[j]:
                acc += 1
            else:
                print(predict_result, test_label[j])
        return acc/test_m


if __name__ == '__main__':
    print('Star')
    time1 = time.time()
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values

    # 由于计算较慢，这里只计算前200个
    train_features = np.array(data[0:5000, 1:])
    train_labels = np.array(data[0:5000, 0])
    test_features = np.array(data[5001:5100, 1:])
    test_labels = np.array(data[5001:5100, 0])

    time2 = time.time()
    print('dada cost', time2-time1, '\n')

    print('Predict')
    k = KNN()
    result = k.test(train_features, train_labels, test_features, test_labels)
    time3 = time.time()
    print('predict cost', time3-time2, '\n')
    print('accuracy is', result)





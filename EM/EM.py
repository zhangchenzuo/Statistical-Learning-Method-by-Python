# coding = utf-8
# autuor = zhangchenzuo
# date = 2020/05/16
# email = chenzuozhang@buaa.edu.cn

"""
EM算法实现，高斯混合模型
GMM（Gaussian mixed model）
"""

import numpy as np
import random
import time

class EM(object):
    def __init__(self, data_train):
        self.samplenum = len(data_train)
        self.data = np.array(data_train)
        # 对初始值真的很敏感，如果设置为完全一样的数值，就会无法收敛
        self.mu1 = 0
        self.mu2 = 1
        self.sigma1 = np.std(data_train)
        self.sigma2 = np.std(data_train)
        self.alpha1 = 0.3
        self.alpha2 = 0.7

    def calguass(self, mu, sigma):
        """
        计算（9.25），假如是第k模型，在目前的估计参数下，出现相应的y的概率
        :param mu:
        :param sigma:
        :return:
        """
        y = self.data
        result = (1/(np.sqrt(2 * np.pi) * sigma)) * np.exp(-1 * (y - mu)**2 / (2 * sigma ** 2))
        return result

    def E_step(self):
        """
        算法9.2
        依据每次的最新的参数，alpha，mu，sigma，计算新的gamma
        gamma是每个分模型的对观测数据的响应系数，反映了数据来自两个模型的比例
        :return:
        """
        sum1 = self.alpha1 * self.calguass(self.mu1, self.sigma1)
        sum2 = self.alpha2 * self.calguass(self.mu2, self.sigma2)
        self.gamma1 = sum1 / (sum1+sum2)
        self.gamma2 = 1-self.gamma1

        # print(self.gamma1, self.gamma2)

    def M_step(self):
        """
        按照算法9.2更新，统计参数
        :return:
        """
        y = self.data
        self.mu1 = np.dot(self.gamma1, y)/sum(self.gamma1)
        self.mu2 = np.dot(self.gamma2, y)/sum(self.gamma2)

        self.sigma1 = np.sqrt(np.dot(self.gamma1, (y - self.mu1) ** 2) / sum(self.gamma1))
        self.sigma2 = np.sqrt(np.dot(self.gamma2, (y - self.mu2) ** 2) / sum(self.gamma2))

        self.alpha1 = sum(self.gamma1) / self.samplenum
        self.alpha2 = sum(self.gamma2) / self.samplenum

    def EM_train(self, iter=500):
        for _ in range(iter):
            self.E_step()
            self.M_step()
        print('the Parameters predict is:')
        print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f' % (
            self.alpha1, self.mu1, self.sigma1, self.alpha2, self.mu2, self.sigma2
        ))


def loaddata(mu1, mu2, sigma1, sigma2, alpha1, alpha2):
    """
    生成混合数据
    :param mu1:
    :param mu2:
    :param sigma1:
    :param sigma2:
    :param alpha1:
    :param alpha2:
    :return:
    """
    length = 1000

    data0 = np.random.normal(mu1, sigma1, int(length*alpha1))
    data1 = np.random.normal(mu2, sigma2, int(length*alpha2))

    data = []
    data.extend(data1)
    data.extend(data0)
    random.shuffle(data)
    return data


if __name__ == '__main__':
    start = time.time()

    # 设置两个高斯模型进行混合，这里是初始化两个模型各自的参数
    # 见“9.3 EM算法在高斯混合模型学习中的应用”
    # alpha是“9.3.1 高斯混合模型” 定义9.2中的系数α
    # mu0是均值μ
    # sigmod是方差σ
    # 在设置上两个alpha的和必须为1，其他没有什么具体要求，符合高斯定义就可以
    alpha0 = 0.1; mu0 = -2; sigmod0 = 0.5
    alpha1 = 0.9; mu1 = 0.5; sigmod1 = 1

    # 初始化数据集
    dataSetList = loaddata(mu0, sigmod0, mu1, sigmod1, alpha0, alpha1)

    # 打印设置的参数
    print('---------------------------')
    print('the Parameters set is:')
    print('alpha0:%.1f, mu0:%.1f, sigmod0:%.1f, alpha1:%.1f, mu1:%.1f, sigmod1:%.1f'%(
        alpha0, mu0, sigmod0, alpha1, mu1, sigmod1
    ))

    # 开始EM算法，进行参数估计
    Em = EM(dataSetList)
    Em.EM_train()

    # 打印时间
    print('----------------------------')
    print('time span:', time.time() - start)
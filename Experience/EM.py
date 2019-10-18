from numpy import *
import numpy as np
import random
import copy

SIGMA = 6
EPS = 0.0001


# 生成方差相同,均值不同的样本
def generate_data():
    Miu1 = 20
    Miu2 = 40
    N = 1000
    X = mat(zeros((N, 1)))
    for i in range(N):
        temp = random.uniform(0, 1)
        if (temp > 0.5):
            X[i] = temp * SIGMA + Miu1
        else:
            X[i] = temp * SIGMA + Miu2
    return X


# EM算法
def my_EM(X):
    k = 2
    N = len(X)
    Miu = np.random.rand(k, 1)
    Posterior = mat(zeros((N, 2)))
    # 先求后验概率
    for iter in range(1000):
        for i in range(N):
            dominator = 0
            for j in range(k):
                dominator = dominator + np.exp(-1.0 / (2.0 * SIGMA ** 2) * (X[i] - Miu[j]) ** 2)
            for j in range(k):
                numerator = np.exp(-1.0 / (2.0 * SIGMA ** 2) * (X[i] - Miu[j]) ** 2)
                Posterior[i, j] = numerator / dominator
        oldMiu = copy.deepcopy(Miu)
        # 最大化
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i, j] * X[i]
                dominator = dominator + Posterior[i, j]
            Miu[j] = numerator / dominator
        # print((abs(Miu - oldMiu)).sum())
        if (abs(Miu - oldMiu)).sum() < EPS:
            print('=='*20)
            print(Miu)
            print(iter)
            break


if __name__ == '__main__':
    X = generate_data()
    my_EM(X)

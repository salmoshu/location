import random
import numpy as np
import matplotlib.pyplot as plt

'''
一维线性情况：
假设一个人做步长恒定为1m的直线运动
指纹库也集中在一条直线上面
'''

Z = []
noise = []
# 对输入数据加入gauss噪声
# 定义gauss噪声的均值和方差
for v in range (100):
    n = random.gauss(0, 3) # (mu, sigma)
    v += n
    noise.append(n)
    Z.append(v)

Z # WiFi指纹定位值
T = [] # PDR预测位置
S = [] # 融合估计位置
X = np.matrix('10; 1') # 状态
P = np.matrix('0.8, 0; 0, 0.8') # 状态协方差矩阵
F = np.matrix('1, 1; 0, 1') # 状态转移矩阵
Q = np.matrix('0.01, 0; 0, 0.01') # 状态转移协方差矩阵
H = np.matrix('1, 0') # 观测矩阵
R = 3 # 观测噪声方差

# PDR
T.append(X)
X_temp = X
for i in range(99):
    X_temp = F * X_temp
    T.append(X_temp)

# KF fusion
S.append(X)
for i in range(99):
    X_ = F * X
    P_ = F * P * F.T + Q
    K = P_ * H.T / (H * P_ * H.T + R)
    X = X_ + K * (Z[i] - H * X_)
    P = (np.eye(2) - K * H) * P_
    S.append(X)

X0 = []
X1 = []
for i1, i2 in zip(S, T):
    X0.append(i1[0:][0])
    X1.append(i2[0:][0])

index = range(100)
plt.scatter(index, np.array(X1)+80, label='pdr')
plt.plot(index, np.array(index)+80)
plt.scatter(index, np.array(Z)+40, label='wifi')
plt.plot(index, np.array(index)+40)
plt.scatter(index, X0, label='fusion')
plt.plot(index, index)
plt.xlabel('status')
plt.ylabel('distance')
plt.legend()
plt.show()
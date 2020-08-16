import location.pdr as pdr
import location.wifi as wifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import neighbors

'''
    通过KF确定初始位置和初始航向角
'''

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))

fingerprint_path = path + '/fusion/Fingerprint'
target_fingerprint1 = path + '/fusion/Fingerprint/2-1.csv'
target_fingerprint2 = path + '/fusion/Fingerprint/2-2.csv'
target_fingerprint3 = path + '/fusion/Fingerprint/2-3.csv'

df_still1 = pd.read_csv(target_fingerprint1)
df_still2 = pd.read_csv(target_fingerprint2)
df_still3 = pd.read_csv(target_fingerprint3)

rssi1 = df_still1[[col for col in df_still1.columns if 'rssi' in col]].values
rssi1 = np.unique(rssi1, axis=0)
wifi = wifi.Model(rssi1)

rssi2 = df_still2[[col for col in df_still2.columns if 'rssi' in col]].values
rssi2 = np.unique(rssi2, axis=0)

rssi3 = df_still3[[col for col in df_still3.columns if 'rssi' in col]].values
rssi3 = np.unique(rssi3, axis=0)

fingerprint_rssi, fingerprint_position = wifi.create_fingerprint(fingerprint_path)
wknn_reg = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance', metric='euclidean')

predict1 = wknn_reg.fit(fingerprint_rssi, fingerprint_position).predict(rssi1)
predict2 = wknn_reg.fit(fingerprint_rssi, fingerprint_position).predict(rssi2)
predict3 = wknn_reg.fit(fingerprint_rssi, fingerprint_position).predict(rssi3)


def kf(predict):
    Z = predict # 观测值（WiFi指纹定位值）
    T = [] # 状态预测
    S = [] # 融合估计
    X = np.matrix('2; 1') # 初始状态
    P = np.matrix('0.1, 0; 0, 0.1') # 状态协方差矩阵
    F = np.matrix('1, 0; 0, 1') # 状态转移矩阵
    Q = np.matrix('0.01, 0; 0, 0.01') # 状态转移协方差矩阵
    H = np.matrix('1, 0; 0, 1') # 观测矩阵
    R = np.matrix('2, 0; 0, 2') # 观测噪声方差

    # 状态预测
    T.append(X)
    X_temp = X
    for i in range(99):
        X_temp = F * X_temp
        T.append(X_temp)

    # KF fusion
    S.append(X)
    for i in range(len(predict)):
        X_ = F * X
        P_ = F * P * F.T + Q
        K = P_ * H.T * np.linalg.pinv(H * P_ * H.T + R)
        X = X_ + K * (Z[i].reshape(2, 1) - H * X_)
        P = (np.eye(2) - K * H) * P_
        S.append(X)
    
    Sx = []
    Sy = []
    for v in S:
        Sx.append(v[0, 0])
        Sy.append(v[1, 0])
    
    return Sx, Sy

S1x, S1y = kf(predict1)
S2x, S2y = kf(predict2)
S3x, S3y = kf(predict3)

print(S1x[-1], S1y[-1])
print(S2x[-1], S2y[-1])
print(S3x[-1], S3y[-1])

Z1x = predict1[:, 0]
Z1y = predict1[:, 1]
Z2x = predict2[:, 0]
Z2y = predict2[:, 1]
Z3x = predict3[:, 0]
Z3y = predict3[:, 1]

plt.plot(Z1x, Z1y, '*', label='Wi-Fi预测点0')
plt.plot(Z2x, Z2y, '*', label='Wi-Fi预测点1')
# plt.plot(Z3x, Z3y, '*', label='knn3')

plt.plot(S1x, S1y, '+', label='Kalman滤波点0')
plt.plot(S2x, S2y, '+', label='Kalman滤波点1')
# plt.plot(S3x, S3y, '+', label='kf3')

plt.plot(
    # [S1x[-1], S2x[-1], S3x[-1]],
    # [S1y[-1], S2y[-1], S3y[-1]],
    [S1x[-1], S2x[-1]],
    [S1y[-1], S2y[-1]],
    '^-'
    ,color='grey'
    ,label='预测航向'
)
plt.plot([2, 2], [1, 1.8], 'o-', label='真实轨迹')
plt.legend(prop = {'size':16})

# plt.legend()
plt.show()
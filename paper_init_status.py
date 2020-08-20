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

'''
实验设定与结果：
一、LType-01
1.250:686
2.687:1082
3.1083:1455

init_position:( 1.95 , 0.93 )
init_angle: 7.73 
deviation : 0.09

二、LType-02
1.250:657
2.658:1048
3.1049:1427

init_position:( 1.89 , 0.97 )
init_angle: -41.01 
deviation : 0.11

三、LType-03
1.250:724
2.725:1092
3.1093:1481

init_position:( 1.86 , 0.94 )
init_angle: 22.45 
deviation : 0.15

五、LType-05
1.250:613
2.614:952
3.953:1318

init_position:( 2.02 , 0.98 )
init_angle: -22.88 
deviation : 0.03

六、LType-06
1.250:651
2.652:1042
3.1043:1391

init_position:( 1.84 , 0.96 )
init_angle: -22.83 
deviation : 0.16
'''

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))

fingerprint_path = path + '/fusion/Fingerprint'

# target_fingerprint1 = path + '/fusion/Fingerprint/2-1.csv'
# target_fingerprint2 = path + '/fusion/Fingerprint/2-2.csv'
# target_fingerprint3 = path + '/fusion/Fingerprint/2-3.csv'

# df_still1 = pd.read_csv(target_fingerprint1)
# df_still2 = pd.read_csv(target_fingerprint2)
# df_still3 = pd.read_csv(target_fingerprint3)

target_fingerprint = path + '/fusion/LType/LType-06.csv'
df_still = pd.read_csv(target_fingerprint)
df_still1 = df_still.iloc[250:651, 0:9]
df_still2 = df_still.iloc[652:1042, 0:9]
df_still3 = df_still.iloc[1043:1391, 0:9]

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

# print(S1x[-1], S1y[-1])
# print(S2x[-1], S2y[-1])
# print(S3x[-1], S3y[-1])

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

init_angle = np.arctan((S2x[-1]-S1x[-1]) / (S2y[-1]-S1y[-1])) * 180 / np.pi

S1x[-1] = round(S1x[-1], 2)
S1y[-1] = round(S1y[-1], 2)
init_angle = round(init_angle, 2)
deviation = round(np.sqrt((S1x[-1]-2)**2 + (S1y[-1]-1)**2), 2)
print('init_position:(', S1x[-1] ,',', S1y[-1] , ')')
print('init_angle:', init_angle ,'')
print('deviation :', deviation)

# plt.legend()
plt.show()
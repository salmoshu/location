import os
import location.pdr as pdr
import location.wifi as wifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))

real_trace_file = path + '/fusion/LType/RealTrace.csv'
walking_data_file = path + '/fusion/LType/LType-03.csv'
fingerprint_path = path + '/fusion/Fingerprint'

df_walking = pd.read_csv(walking_data_file) # 实验数据
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹

# 主要特征参数
rssi = df_walking[[col for col in df_walking.columns if 'rssi' in col]].values
linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values

pdr = pdr.Model(linear, gravity, rotation)
wifi = wifi.Model(rssi)

# 指纹数据
fingerprint_rssi, fingerprint_position = wifi.create_fingerprint(fingerprint_path)

# 找到峰值处的rssi值
steps = pdr.step_counter(frequency=70, walkType='fusion')
print('steps:', len(steps))
result = fingerprint_rssi[0].reshape(1, rssi.shape[1])
for k, v in enumerate(steps):
    index = v['index']
    value = rssi[index]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)

predict, _ = wifi.wknn_strong_signal_reg(fingerprint_rssi, fingerprint_position, result, real_trace)

def accumlate():
    APx = [-1, -1, -1, 5, 5, 4, 16, 16]
    APy = [-1, 11, 16, -1, 10, 16, 10, 16]
    x = real_trace[:, 0]
    y = real_trace[:, 1]
    last_d = None
    T = []
    for v1, v2 in zip(x, y):
        d = []
        for d1, d2 in zip(APx, APy):
            d.append(
                ((d1-v1)**2 + (d2-v2)**2)**1/2
            )
        
        if last_d is None:
            pass
        else:
            s = 1
            for s1, s2 in zip(last_d, d):
                s = s * (s1/s2)
            T.append(s)
        last_d = d
    return T


def accuracy():
    S = []
    for v1, v2 in zip(predict, real_trace):
        if len(S)==len(predict)-1:
            break

        p_x = v1[0]
        p_y = v1[1]
        r_x = v2[0]
        r_y = v2[1]

        v = ((p_x-r_x)**2 + (p_y-r_y)**2) ** 1/2

        S.append(v)
    return S

rssi = result
distance = []
for i, v in enumerate(rssi):
    value = v.reshape(1, rssi.shape[1])
    if i==0:
        lastValue = value
        continue
    d = euclidean_distances(value, lastValue)
    distance.append(round(d[0, 0], 2))

# T = accumlate()
# T = np.array(T)

# S = accuracy()
# S = np.array(S)

# D = distance
# D = np.array(D)

# # X = S
# Y = D/(10*np.log10(T))

# plt.plot(range(len(Y)), Y, 'o')
# plt.show()

print(np.array(distance)/10)
print((distance))
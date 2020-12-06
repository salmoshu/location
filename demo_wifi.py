import location.pdr as pdr
import location.wifi as wifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))

real_trace_file = path + '/fusion/LType/RealTrace.csv'
walking_data_file = path + '/fusion/LType/LType-02.csv'
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

# 找到峰值出的rssi值
steps = pdr.step_counter(frequency=70, walkType='abnormal')
print('steps:', len(steps))
result = fingerprint_rssi[0].reshape(1, rssi.shape[1])
for k, v in enumerate(steps):
    index = v['index']
    value = rssi[index]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)

# # knn算法
predict, accuracy = wifi.knn_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
print('knn accuracy:', accuracy, 'm')

# 添加区域限制的knn回归
# predict, accuracy = wifi.ml_limited_reg('knn', fingerprint_rssi, fingerprint_position, result, real_trace)
# print('knn_limited accuracy:', accuracy, 'm')

# svm算法
# predict, accuracy = wifi.svm_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('svm accuracy:', accuracy, 'm')

# rf算法
# predict, accuracy = wifi.rf_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('rf accuracy:', accuracy, 'm')

# 添加区域限制rf的rf算法
# predict, accuracy = wifi.ml_limited_reg('rf', fingerprint_rssi, fingerprint_position, result, real_trace)
# print('rf_limited accuracy:', accuracy, 'm')

# gdbt算法
# predict, accuracy = wifi.dbdt(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('gdbt accuracy:', accuracy, 'm')

# 多层感知机
# predict, accuracy = wifi.nn(fingerprint_rssi, fingerprint_position, result, real_trace)
# print('nn accuracy:', accuracy, 'm')

wifi.show_trace(predict, real_trace=real_trace)
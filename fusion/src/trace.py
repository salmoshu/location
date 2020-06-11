from FUSION import FUSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'C:/Users/salmo/Desktop/location/fusion/experiment_data'

walking_data_file = path + '/S-Type/SType-08.csv'
real_trace_file = path + '/S-Type/RealTrace.csv'
df_walking = pd.read_csv(walking_data_file) # 实验数据
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹

fingerprint_rssi_file = path + '/Fingerprint/rssi.csv'
fingerprint_position_file = path + '/Fingerprint/position.csv'
fingerprint_rssi = pd.read_csv(fingerprint_rssi_file).values # 指纹数据-信号强度
fingerprint_position = pd.read_csv(fingerprint_position_file).values # 指纹数据-坐标点

# 主要特征参数
timestamp = df_walking['timestamp'].values
rssi = df_walking[[col for col in df_walking.columns if 'rssi' in col]].values
linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values

fusion = FUSION(timestamp, rssi, linear, gravity, rotation)

# 找到峰值出的rssi值
steps = fusion.step_counter()
result = fingerprint_rssi[0].reshape(1, 4)
for k, v in enumerate(steps):
    index = v['index']
    value = rssi[index]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)

# knn算法
predict, accuracy = fusion.knn_reg(fingerprint_rssi, fingerprint_position, result, real_trace, k=3)
print('knn accuracy:', accuracy)

# fusion.show_trace('pdr', real_trace=real_trace, offset=np.pi/2)
fusion.show_trace('wifi', real_trace=real_trace, predict=predict)
# fusion.show_steps()
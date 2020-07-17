import location.pdr as pdr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
# walking_data_file = path + '/linear_8m/linear01.csv'

walking_data_file = path + '/fusion01/SType/SType-10.csv'
real_trace_file = path + '/fusion01/SType/RealTrace.csv'
real_trace = pd.read_csv(real_trace_file).values # 真实轨迹

df_walking = pd.read_csv(walking_data_file)

# 获得线性加速度、重力加速度、姿态仰角的numpy.ndarray数据
linear = df_walking[[col for col in df_walking.columns if 'linear' in col]].values
gravity = df_walking[[col for col in df_walking.columns if 'gravity' in col]].values
rotation = df_walking[[col for col in df_walking.columns if 'rotation' in col]].values

pdr = pdr.Model(linear, gravity, rotation)

# Demo1：显示垂直方向合加速度与步伐波峰分布
# frequency：数据采集频率
# walkType：行走方式（normal为正常走路模式，fusion为做融合定位实验时走路模式）
# pdr.show_steps(frequency=70, walkType='normal')

# Demo2：显示数据在一定范围内的分布情况，用来判断静止数据呈现高斯分布
# 传入参数为静止状态x（y或z）轴线性加速度
# pdr.show_gaussian(linear[:,2])

# Demo3：显示三轴线性加速度分布情况
# pdr.show_acc()

# Demo4：获取步伐新息
# 返回值steps为字典类型，index为样本序号，acceleration为步伐加速度峰值
# steps = pdr.step_counter(frequency=70, walkType='normal')
# stride = pdr.step_stride # 步长推算函数
# # 计算步长推算的平均误差
# accuracy = []
# for v in steps:
#     a = v['acceleration']
#     print(stride(a))
#     accuracy.append(stride(a)-0.8)
# acc_mean = np.mean(accuracy)
# #求方差
# acc_var = np.var(accuracy)
# #求标准差
# acc_std = np.std(accuracy)
# print("mean: %f" % acc_mean)
# print("var: %f" % acc_var)
# print("std: %f" % acc_std)

# Demo5：显示PDR预测轨迹
# 注意：PDR不清楚初始位置与初始航向角
# pdr.show_trace(frequency=70, walkType='normal')
pdr.show_trace(frequency=70, walkType='fusion', real_trace=real_trace, offset=np.pi/2)

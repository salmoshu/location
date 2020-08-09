import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import location.pdr as pdr
import location.wifi as wifi
import location.fusion as fusion

'''
EKF仿真实验
注意假如有n个状态，那么就有n-1次状态转换
'''

fusion = fusion.Model()

sigma_wifi = 1
sigma_pdr = .1
sigma_yaw = 15/360

L = 0.8

file_path = os.path.abspath(os.path.join(os.getcwd(), "./simulation_trace.npy"))
position = np.load(file_path)
X_real = position[:,0]
Y_real = position[:,1]

# 角度
angle = [0]*10 + [np.pi/2]*5 + [np.pi]*10 + [3*np.pi/2]*6 # 由于最后一个顶点没有转弯，因此多一个
# angle = [np.pi/2]*10 + [np.pi]*5 + [3*np.pi/2]*10 + [0]*5 # 对初始航向角的验证
for k, v in enumerate(angle):
    angle[k] = v + random.gauss(0, sigma_yaw)

# 目前不考虑起始点（设定为0，0），因此wifi数组长度比实际位置长度少1
X_wifi = []
Y_wifi = []
observation_states = []
for i in range(len(angle)):
    x = X_real[i] + random.gauss(0, sigma_wifi) # (mu, sigma)
    y = Y_real[i] + random.gauss(0, sigma_wifi) # (mu, sigma)
    X_wifi.append(x)
    Y_wifi.append(y)
    observation_states.append(np.matrix([
        [x], [y], [L], [angle[i]]
    ]))

# 假定步长L=0.8
# 假定初始位置为(0, 0)
L = 0.8
theta_counter = 0
def state_conv(parameters_arr):
    global theta_counter
    theta_counter = theta_counter+1
    x = parameters_arr[0]
    y = parameters_arr[1]
    theta = parameters_arr[2]
    return x+L*np.sin(theta), y+L*np.cos(theta), angle[theta_counter]

X = np.matrix('0; 0; 0') # 初始状态
# X = np.matrix('2; 2; 0') # 对初始状态进行验证

X_pdr = [X[0, 0]]
Y_pdr = [X[1, 0]]

transition_states = [X]
for k, v in enumerate(angle):
    if k==0: V = X
    if k==len(angle)-1: break
    x, y, theta = state_conv([V[0, 0], V[1, 0], V[2, 0]])
    V = np.matrix([[x],[y],[theta]])
    X_pdr.append(x)
    Y_pdr.append(y)
    transition_states.append(np.matrix([
        [x],[y],[theta]
    ]))
theta_counter = 0

# 状态协方差矩阵（初始状态不是非常重要，经过迭代会逼近真实状态）
P = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
# 观测矩阵
H = np.matrix([[1, 0, 0],
               [0, 1, 0],
               [0, 0, 0],
               [0, 0, 1]])
# 状态转移协方差矩阵
Q = np.matrix([[sigma_pdr**2, 0, 0],
               [0, sigma_pdr**2, 0],
               [0, 0, sigma_yaw**2]])
# 观测噪声方差
R = np.matrix([[sigma_wifi**2, 0, 0, 0],
               [0, sigma_wifi**2, 0, 0],
               [0, 0, 0, 0],
               [0, 0, 0, sigma_yaw**2]])

def jacobF_func(i):
    return np.matrix([[1, 0, L*np.cos(angle[i])],
                      [0, 1, -L*np.sin(angle[i])],
                      [0, 0, 1]])

S = fusion.ekf2d(
    transition_states = transition_states # 状态数组
   ,observation_states = observation_states # 观测数组
   ,transition_func = state_conv # 状态预测函数
   ,jacobF_func = jacobF_func # 一阶线性的状态转换公式
   ,initial_state_covariance = P
   ,observation_matrices = H
   ,transition_covariance = Q
   ,observation_covariance = R
)

X_ekf = []
Y_ekf = []

for v in S:
    X_ekf.append(v[0, 0])
    Y_ekf.append(v[1, 0])

x = X_ekf
y = Y_ekf
for k in range(0, len(x)):
    plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))

plt.grid()
plt.plot(X_real, Y_real, 'o-', label='real tracks')
plt.plot(X_wifi, Y_wifi, 'r.', label='observation points')
plt.plot(X_pdr, Y_pdr, 'o-', label='dead reckoning')
plt.plot(X_ekf, Y_ekf, 'o-', label='ekf positioning')
plt.legend()
plt.show()
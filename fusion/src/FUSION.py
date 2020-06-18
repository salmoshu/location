'''
1.FUSION参数列表（5个参数）：
时间戳;
WIFI信号强度矩阵
线性加速度矩阵（x轴加速度、y轴加速度、z轴加速度）；
重力加速度矩阵（x轴重力加速度、y轴重力加速度、z轴重力加速度）;
四元数矩阵（四元数x、四元数y、四元数z、四元数w）

2.FUSION参数类型：
numpy.ndarray
'''

import types
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

class FUSION(object):
    def __init__(self, timestamp, rssi, linear, gravity, rotation):
        self.timestamp = timestamp
        self.rssi = rssi
        self.linear = linear
        self.gravity = gravity
        self.rotation = rotation

###############################################################
######################   误差计算相关代码   #####################
###############################################################

    # 标准差
    def square_accuracy(self, predictions, labels):
        return np.sqrt(np.mean(np.sum((predictions - labels)**2, 1)))

###############################################################
########################   PDR相关代码   #######################
###############################################################
    
    # 四元数转化为欧拉角
    def quaternion2euler(self):
        rotation = self.rotation
        x = rotation[:, 0]
        y = rotation[:, 1]
        z = rotation[:, 2]
        w = rotation[:, 3]
        pitch = np.arcsin(2*(w*y-z*x))
        roll = np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y))
        yaw = np.arctan2(2*(w*z+x*y),1-2*(z*z+y*y))
        return pitch, roll, yaw
    
    # 获得手机坐标系与地球坐标系之间的角度（theta）
    # 实验前提：手机x轴与重力方向垂直
    def coordinate_conversion(self):
        gravity = self.gravity
        linear = self.linear

        g_x = gravity[:, 0]
        g_y = gravity[:, 1]
        g_z = gravity[:, 2]

        linear_x = linear[:, 0]
        linear_y = linear[:, 1]
        linear_z = linear[:, 2]
        
        theta = np.arctan(np.abs(g_z/g_y))

        # 得到垂直方向加速度（除去g）
        a_vertical = linear_y*np.cos(theta) + linear_z*np.sin(theta)

        return a_vertical
    
    '''
    步数检测函数
    返回值：
    steps
    字典型数组，每个字典保存了峰值位置（index）与该点的合加速度值（acceleration）
    '''
    def step_counter(self):
        offset = 0.7
        g = 9.794

        a_vertical = self.coordinate_conversion()

        slide = 40 * offset # 滑动窗口（100Hz的采样数据）
        frequency = 100 * offset

        # 行人加速度阈值
        min_acceleration = 0.2 * g # 0.2g
        max_acceleration = 2 * g   # 2g

        # 峰值间隔(s)
        min_interval = 0.4
        max_interval = 1

        # 计算步数
        steps = []
        peak = {'index': 0, 'acceleration': 0}

        # 以40*offset为滑动窗检测峰值
        # 条件1：峰值在0.2g~2g之间
        for i, v in enumerate(a_vertical):
            if v >= peak['acceleration'] and v >= min_acceleration and v <= max_acceleration:
                peak['acceleration'] = v
                peak['index'] = i
            if i%slide == 0 and peak['index'] != 0:
                steps.append(peak)
                peak = {'index': 0, 'acceleration': 0}
        
        # 条件2：两个峰值之前间隔至少大于0.4s*offset
        # del使用的时候，一般采用先记录再删除的原则
        temp = steps[0]
        dirty_points = []
        for key, step_dict in enumerate(steps):
            if step_dict['index'] == steps[0]['index']:
                continue
            if step_dict['index']-temp['index'] < min_interval*frequency:
                if step_dict['acceleration'] <= temp['acceleration']:
                    dirty_points.append(key)
                else:
                    temp = step_dict
                    dirty_points.append(key-1)
            else:
                temp = step_dict
        
        counter = 0 # 记录删除数量，作为偏差值
        for key in dirty_points:
            del steps[key-counter]
            counter = counter + 1
        
        return steps
    
    # 步长推算
    # 目前的方法不具备科学性，临时使用
    def step_stride(self, max_acceleration):
        return np.power(max_acceleration, 1/4) * 0.5

    # 航向角
    # 根据姿势直接使用yaw
    def step_heading(self):
        _, _, yaw = self.quaternion2euler()
        return yaw
    
    # 步行轨迹的每一个相对坐标位置
    # 返回的是
    def pdr_position(self, offset):
        init_step = (0, 0)
        yaw = self.step_heading()
        init_theta = yaw[0] # 初始角度

        steps = self.step_counter()
        position_x = []
        position_y = []
        x = init_step[0]
        y = init_step[1]
        position_x.append(x)
        position_y.append(y)
        for v in steps:
            index = v['index']
            length = self.step_stride(v['acceleration'])
            theta = yaw[index] - init_theta + offset
            x = x + length*np.sin(theta)
            y = y + length*np.cos(theta)
            position_x.append(x)
            position_y.append(y)
        return position_x, position_y

###############################################################
#####################   WIFI指纹相关代码   #####################
###############################################################

    def ml_limited_reg(self, type, offline_rss, offline_location, online_rss, online_location):
        if type == 'knn':
            k = 3
            ml_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
        elif type == 'rf':
            ml_reg = RandomForestRegressor(n_estimators=10)

        init_x = 0
        init_y = 0
        predict = np.array([[init_x, init_y]])
        limited_rss = None
        limited_location = None
        offset = 2 # m

        for k, v in enumerate(online_rss):
            if k == 0:
                continue
            for v1, v2 in zip(offline_rss, offline_location):
                if (v2[0] >= init_x-offset and v2[0] <= init_x+offset) and (v2[1] >= init_y-offset and v2[1] <= init_y+offset):
                    v1 = v1.reshape(1, v1.size)
                    v2 = v2.reshape(1, v2.size)
                    if limited_rss is None:
                        limited_rss = v1
                        limited_location = v2
                    else:
                        limited_rss = np.concatenate((limited_rss, v1), axis=0)
                        limited_location = np.concatenate((limited_location, v2), axis=0)
            v = v.reshape(1, v.size)
            predict_point = ml_reg.fit(limited_rss, limited_location).predict(v)
            predict = np.concatenate((predict, predict_point), axis=0)
            init_x = predict_point[0][0]
            init_y = predict_point[0][1]
            limited_rss = None
            limited_location = None
        
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    # knn regression
    def knn_reg(self, offline_rss, offline_location, online_rss, online_location):
        k = 3
        knn_reg = neighbors.KNeighborsRegressor(n_neighbors=k, weights='uniform', metric='euclidean')
        predict = knn_reg.fit(offline_rss, offline_location).predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
    
    # 支持向量机
    def svm_reg(self, offline_rss, offline_location, online_rss, online_location):
        clf_x = svm.SVR(C=1000, gamma=0.01)
        clf_y = svm.SVR(C=1000, gamma=0.01)
        clf_x.fit(offline_rss, offline_location[:, 0])
        clf_y.fit(offline_rss, offline_location[:, 1])
        x = clf_x.predict(online_rss)
        y = clf_y.predict(online_rss)
        predict = np.column_stack((x, y))
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
    
    # 随机森林
    def rf_reg(self, offline_rss, offline_location, online_rss, online_location):
        estimator = RandomForestRegressor(n_estimators=150)
        estimator.fit(offline_rss, offline_location)
        predict = estimator.predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

    # 梯度提升
    def dbdt(self, offline_rss, offline_location, online_rss, online_location):
        clf = MultiOutputRegressor(ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=10))
        clf.fit(offline_rss, offline_location)
        predict = clf.predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy
    
    # 多层感知机
    def nn(self, offline_rss, offline_location, online_rss, online_location):
        clf = MLPRegressor(hidden_layer_sizes=(100, 100))
        clf.fit(offline_rss, offline_location)
        predict = clf.predict(online_rss)
        accuracy = self.square_accuracy(predict, online_location)
        return predict, accuracy

###############################################################
######################   轨迹显示相关代码   #####################
###############################################################
    
    '''
    show_函数为数据可视化部分:
    show_steps显示步伐检测图像
    show_trace显示运动轨迹图，分为rssi、pdr、fusion三种类型
    '''

    # 显示步伐检测图像
    def show_steps(self):
        a_vertical = self.coordinate_conversion()
        steps = self.step_counter()

        index_test = []
        value_test = []
        for v in steps:
            index_test.append(v['index'])
            value_test.append(v['acceleration'])

        plt.plot(a_vertical)
        plt.scatter(index_test, value_test, color='r')
        plt.show()

    # 显示运动轨迹图，分为wifi、pdr、fusion三种类型
    def show_trace(self, type, **kw):
        plt.grid()
        if 'real_trace' in kw:
                real_trace = kw['real_trace'].T
                trace_x = real_trace[0]
                trace_y = real_trace[1]
                l1, = plt.plot(trace_x, trace_y, color='g')
                plt.scatter(trace_x, trace_y, color='orange')
                for k in range(0, len(trace_x)):
                    plt.annotate(k, xy=(trace_x[k], trace_y[k]), xytext=(trace_x[k]+0.1,trace_y[k]+0.1), color='green')
        if type == 'pdr':
            if 'offset' in kw:
                offset = kw['offset']
            if 'predict_trace' in kw:
                predict = kw['predict_trace'].T
                x = predict[0]
                y = predict[1]
            else:
                steps = self.step_counter()
                print("steps: ", len(steps))
                distance = 0
                for v in steps:
                    distance = distance + self.step_stride(v['acceleration'])
                x, y = self.pdr_position(offset)
            for k in range(0, len(x)):
                    plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
            l2, = plt.plot(x, y, color='b')
            plt.scatter(x, y, color='r')

            plt.legend(handles=[l1,l2],labels=['real tracks','pdr predicting'],loc='best')
            plt.show()
        elif type == 'wifi':
            if 'predict_trace' in kw:
                predict = kw['predict_trace'].T
                x = predict[0]
                y = predict[1]

                for k in range(0, len(x)):
                    plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
            
            l2, = plt.plot(x, y, c='blue')
            plt.scatter(x, y, c='red')
            plt.legend(handles=[l1,l2],labels=['real tracks','wifi predicting'],loc='best')
            plt.show()
        else: # type == 'fusion
            pass
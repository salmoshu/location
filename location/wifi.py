'''
1.Model参数列表（1个参数）：
WIFI信号强度矩阵

2.Model参数类型：
numpy.ndarray
'''

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
import matplotlib.ticker as ticker
from scipy.stats import norm
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class Model(object):
    def __init__(self, rssi):
        self.rssi = rssi

    def create_fingerprint(self, path):
        RSSI = None
        X = None
        Y = None
        # path为采集的指纹数据目录，每个坐标为一个文件，文件的命名格式为：x_y
        directory = os.walk(path)  
        for _, _, file_list in directory:
            for file_name in file_list:
                position = file_name.split('.')[0].split('-') # 获取文件记录的坐标
                x = np.array([[int(position[0])]])
                y = np.array([[int(position[1])]])
                df = pd.read_csv(path + "/" + file_name)
                columns = [col for col in df.columns if 'rssi' in col]
                rssi = df[columns].values
                rssi = rssi[300:] # 视前300行数据为无效数据
                rssi_mean = np.mean(rssi, axis=0).reshape(1, rssi.shape[1])
                if RSSI is None:
                    RSSI = rssi_mean
                    X = x
                    Y = y
                else:
                    RSSI = np.concatenate((RSSI,rssi_mean), axis=0)
                    X = np.concatenate((X,x), axis=0)
                    Y = np.concatenate((Y,y), axis=0)
        fingerprint = np.concatenate((RSSI, X, Y), axis=1)
        fingerprint = pd.DataFrame(fingerprint, index=None, columns = columns+['x', 'y'])
        rssi = fingerprint[[col for col in fingerprint.columns if 'rssi' in col]].values
        position = fingerprint[['x', 'y']].values
        return rssi, position

    # 标准差
    def square_accuracy(self, predictions, labels):
        accuracy = np.sqrt(np.mean(np.sum((predictions - labels)**2, 1)))
        return round(accuracy, 3)

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

    '''
        data为np.array类型
    '''
    def determineGaussian(self, data, merge, interval=1, wipeRange=300):
        offset = wipeRange
        data = data[offset:]

        minValue = np.min(data)
        maxValue = np.max(data)
        meanValue = np.mean(data)

        length = math.ceil((maxValue-minValue)/interval)
        counterArr = length * [0]
        valueRange = length * [0]

        textstr = '\n'.join((
                r'$max=%.8f$' % (maxValue, ),
                r'$min=%.8f$' % (minValue, ),
                r'$mean=%.8f$' % (meanValue, )))

        if merge==True:
            # 区间分段样本点
            result = []
            temp_data = data[0]
            for i in range(0, len(data)):
                if temp_data == data[i]:
                    continue
                else:
                    result.append(temp_data)
                    temp_data = data[i]
            data = result

        for index in range(len(counterArr)):
            valueRange[index] = minValue + interval*index

        for value in data:
            key = int((value - minValue) / interval)
            if key >=0 and key <length:
                counterArr[key] += 1
        
        if merge==True:
            print('Wi-Fi Scan Times:', len(data))

        normal_mean = np.mean(data)
        normal_sigma = np.std(data)
        normal_x = np.linspace(minValue, maxValue, 100)
        normal_y = norm.pdf(normal_x, normal_mean, normal_sigma)
        normal_y = normal_y * np.max(counterArr) / np.max(normal_y)

        _, ax = plt.subplots()

        # Be sure to only pick integer tick locations.
        for axis in [ax.xaxis, ax.yaxis]:
            axis.set_major_locator(ticker.MaxNLocator(integer=True))

        ax.bar(valueRange, counterArr, label='distribution')
        ax.plot(normal_x, normal_y, 'r-', label='fitting')
        plt.xlabel('value')
        plt.ylabel('count')
        plt.title('信号强度数据的高斯拟合')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        plt.legend()
        plt.show()
    
    def rssi_fluctuation(self, merge, wipeRange=300):
        # wipeRange=300表示前300行数据中包含了无效数据，可以直接去除
        offset = wipeRange
        rssi = self.rssi[offset:]
        rows = rssi.shape[0]
        columns = rssi.shape[1]
        lines = [0]*(columns+1)
        labels = [0]*(columns+1)

        filename = ''

        if merge == False:
            for i in range(0, columns):
                lines[i], = plt.plot(range(0, rows), rssi[:, i])
                labels[i] = 'rssi' + str(i+1)
            
            plt.title('指纹库文件'+filename+'数据波动情况')
            plt.legend(handles=lines, labels=labels, loc='best')
            plt.xlabel('样本点数目/个')
            plt.ylabel('WiFi信号强度/dBm')
            plt.show()

        elif merge == True:
            # 采集周期（以一定样本点数目为一个周期）
            indexs = []
            results = []
            
            # 区间分段样本点
            for i in range(0, columns):
                counter = 0
                intervals = []
                result = []
                temp_rssi = rssi[0, i]
                for j in range(0, rows):
                    if temp_rssi == rssi[j, i]:
                        counter = counter +1
                    else:
                        intervals.append(counter)
                        result.append(temp_rssi)
                        temp_rssi = rssi[j, i]
                indexs.append(intervals)
                results.append(result)
                intervals = []
            
            # 确定最小长度
            length = 0
            for i in range(0, columns):
                if length==0:
                    length = len(results[i])
                else:
                    if len(results[i]) < length:
                        length = len(results[i])
            
            # 显示图像
            for i in range(0, columns):
                lines[i], = plt.plot(range(0, length), results[i][:length])
                labels[i] = 'rssi' + str(i+1)
            
            plt.title('指纹库文件'+filename+'数据波动情况')
            plt.legend(handles=lines, labels=labels, loc='best')
            plt.xlabel('WiFi扫描次数/次')
            plt.ylabel('WiFi信号强度/dBm')
            plt.xticks(range(0, length, int(length/5))) # 保证刻度为整数
            plt.show()

    # 显示运动轨迹图
    def show_trace(self, predict_trace, **kw):
        plt.grid()
        handles = []
        labels = []
        if 'real_trace' in kw:
            real_trace = kw['real_trace'].T
            trace_x = real_trace[0]
            trace_y = real_trace[1]
            l1, = plt.plot(trace_x, trace_y, 'o-')
            handles.append(l1)
            labels.append('real tracks')
            for k in range(0, len(trace_x)):
                plt.annotate(k, xy=(trace_x[k], trace_y[k]), xytext=(trace_x[k]+0.1,trace_y[k]+0.1), color='green')

        predict = predict_trace.T
        x = predict[0]
        y = predict[1]

        for k in range(0, len(x)):
            plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
        
        l2, = plt.plot(x, y, 'o-')
        handles.append(l2)
        labels.append('wifi predicting')
        plt.scatter(x, y, c='red')
        plt.legend(handles=handles ,labels=labels, loc='best')
        plt.show()
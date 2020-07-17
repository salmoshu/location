'''
1.Model参数列表（1个参数）：
WIFI信号强度矩阵

2.Model参数类型：
numpy.ndarray
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor

class Model(object):
    def __init__(self, rssi):
        self.rssi = rssi

    # 标准差
    def square_accuracy(self, predictions, labels):
        return np.sqrt(np.mean(np.sum((predictions - labels)**2, 1)))

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

    # 显示运动轨迹图，分为wifi、pdr、fusion三种类型
    def show_trace(self, **kw):
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

        if 'predict_trace' in kw:
            predict = kw['predict_trace'].T
            x = predict[0]
            y = predict[1]

            for k in range(0, len(x)):
                plt.annotate(k, xy=(x[k], y[k]), xytext=(x[k]+0.1,y[k]+0.1))
        
        l2, = plt.plot(x, y, 'o')
        handles.append(l2)
        labels.append('wifi predicting')
        plt.scatter(x, y, c='red')
        plt.legend(handles=handles ,labels=labels, loc='best')
        plt.show()
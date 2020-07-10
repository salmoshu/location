import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ANALYSIS(object):
    def __init__(self):
        pass

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
    
    def rssi_fluctuation(self, rssi, merge, wipeRange=300):
        # offset=300表示前300行数据中包含了无效数据，可以直接去除
        offset = wipeRange
        rssi = rssi[offset:]
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
            plt.show()
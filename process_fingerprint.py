'''
以excel形式生成指纹数据库
返回：
rssi文件：指纹数据库rssi值（每一行为一组数据）
match文件：指纹所对应的坐标（每一行为一组坐标）
'''

import os
import pandas as pd
import numpy as np

RSSI = None
POSITION = None

def process(file):
    global RSSI
    df = pd.read_csv(file)
    rssi = df[[col for col in df.columns if 'rssi' in col]].values
    rssi = rssi[300:] # 视前300行数据为无效数据

    values = []
    
    # 遍历rssi每列(遍历numpy数组列最好的方式是遍历其转置)
    for column in rssi.T:
        number, frequency = occurrence_frequency(column)
        remove(number, frequency)
        result = average(number, frequency)
        values.append(result)
    
    values = np.array([values])
    
    if RSSI is None:
        RSSI = values
    else:
        RSSI = np.concatenate((RSSI,values),axis=0)

# 求rssi的平均值
def average(number, frequency):
    number_sum = 0
    frequency_sum = 0
    for v1, v2 in zip(number, frequency):
        number_sum = number_sum + v1*v2
        frequency_sum = frequency_sum + v2
    result = number_sum/frequency_sum
    
    return result

# 去除出现频率最低的数
def remove(number, frequency):
    min_number = min(frequency)
    min_index = frequency.index(min_number)
    del frequency[min_index]
    del number[min_index]

# 找到特定出现的几个rssi数值以及出现次数
def occurrence_frequency(numbers):
    number = []
    frequency = []
    for value in numbers:
        if value not in number:
            number.append(value)
            frequency.append(1)
        else:
            index = number.index(value)
            frequency[index] = frequency[index] + 1
    
    return number, frequency

if __name__ == "__main__":
    m = 5 
    n = 9

    path = os.path.abspath(os.path.join(os.getcwd(), "./data/fusion01/Fingerprint"))
    for i in range(0, m):
        for j in range(0, n):
            file = '/' + str(i) + '_' + str(j) + '.csv'
            process(path + file)
            position = np.array([[i, j]], dtype='int32')
            if i == 0 and j == 0:
                POSITION = position
            else:
                POSITION = np.concatenate((POSITION,position),axis=0)
    
    np.savetxt(path + "/rssi.csv", RSSI, delimiter=',')
    np.savetxt(path + "/position.csv", POSITION, delimiter=',')
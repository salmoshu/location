'''
1.PDR参数列表（八个参数）：
时间戳、x轴加速度、y轴加速度、z轴加速度、
四元数x、四元数y、四元数z、四元数w
2.PDR参数类型：
numpy.ndarray
'''

import numpy as np
import matplotlib.pyplot as plt

class PDR(object):
    def __init__(self, timestamp, linear_x, linear_y, linear_z, 
                rotation_x, rotation_y, rotation_z, rotation_w):
        self.timestamp = timestamp
        self.linear_x = linear_x
        self.linear_y = linear_y
        self.linear_z = linear_z
        self.rotation_x = rotation_x
        self.rotation_y = rotation_y
        self.rotation_z = rotation_z
        self.rotation_w = rotation_w
    
    # 四元数转化为欧拉角
    def quaternion2euler(self, type):
        x = self.rotation_x
        y = self.rotation_y
        z = self.rotation_z
        w = self.rotation_w
        if type == 'pitch':
            p = np.asin(2*(w*y-z*x))
            return p
        elif type == 'roll':
            r = np.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
            return r
        elif type == 'yaw':
            y = np.atan2(2*(w*z+x*y),1-2*(z*z+y*y))
            return y
    
    '''
    步数检测函数
    返回值：
    1.resultant_acceleration
    合加速度ndarray
    2.steps
    字典型数组，每个字典保存了峰值位置（index）与该点的合加速度值（acceleration）
    '''
    def pedometer(self):
        # 三轴加速度
        linear_x = self.linear_x
        linear_y = self.linear_y
        linear_z = self.linear_z

        resultant_acceleration = np.sqrt(np.square(linear_x) + np.square(linear_y) + np.square(linear_z))

        slide = 40 # 滑动窗口（100Hz的采样数据）
        frequency = 100

        # 行人加速度阈值
        min_acceleration = 0.2 * 9.8 # 0.2g
        max_acceleration = 2 * 9.8   # 2g

        # 峰值间隔(s)
        min_interval = 0.4
        max_interval = 1

        # 计算步数
        steps = []
        peak = {'index': 0, 'acceleration': 0}
        for i, v in enumerate(linear_z):
            if v >= peak['acceleration'] and v >= min_acceleration and v <= max_acceleration:
                peak['acceleration'] = v
                peak['index'] = i
            if i%slide == 0 and peak['index'] != 0:
                steps.append(peak)
                peak = {'index': 0, 'acceleration': 0}
        
        temp = steps[0]
        for key, step_dict in enumerate(steps):
            if step_dict['index'] == steps[0]['index']:
                continue
            if step_dict['index']-temp['index'] < min_interval*frequency:
                if step_dict['acceleration'] < temp['acceleration']:
                    del steps[key]
                else:
                    del steps[key-1]
            else:
                temp = step_dict

        return linear_z, steps
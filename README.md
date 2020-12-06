# Location

**Location**是一个多源数据融合定位的科研项目，目前包含了Wi-Fi指纹定位、行人航位推算（PDR）与基于EKF的Wi-Fi指纹与PDR的融合定位，本项目针对的平台为安卓平台，融合使用的数据为Wi-Fi数据、加速度计数据、陀螺仪数据、重力传感器数据、姿态仰角数据。

需要注意的是，安卓平台主要用于采集实验数据，数据分析与定位轨迹生成主要使用Python离线完成。

# 目录

------

* [第三方依赖](#第三方依赖)
* [实验数据](#实验数据)
* [开始使用](#开始使用)
    * [Demo1 PDR定位](#Demo1-PDR定位)
    * [Demo2 Wi-Fi指纹定位](#Demo2-Wi-Fi指纹定位)
    * [Demo3 融合定位](#Demo3-融合定位)
* [数据采集程序](#数据采集程序)
* [历史版本](#历史版本)

# 第三方依赖

- Numpy
- Pandas
- Matplotlib
- Sklearn

# 实验数据

| **data** | **实验数据** |
|:--|---|
| data/count_steps | 正常直线行走数据，包含了安卓或苹果设备采集的数据，用来判断步伐检测的结果是否良好，其中安卓程序为自己开发的程序，苹果数据采集程序为phyphox（使用的时候注意修改属性名，方便后续程序使用）。 |
| data/fusion          | 小范围融合定位实验，具体数据包含了：Fingerprint（指纹数据）、LeftBorder（左边缘测试点）、RightBorder（右边缘测试点）和LType（L型路线行走数据）。 |
| data/linear_08m        | 正常直线行走数据，行走步数为10步，每一步为固定步长0.8m。     |
| data/rssi_fluctuation  | 一组长时间记录Wi-Fi变化的数据，每个文件大约记录2万条样本数据，该数据可以用来分析Wi-Fi数据的波动情况。 |
| data/still             | 静止数据，用来分析惯性传感器数据的特性。                     |

# 开始使用

Location的使用前提：操作的文件格式大概为如下所示，其中rssi根据实验过程中AP数量决定，AP为指定的固定路由器，比较适合小范围实验。

| timestamp     | rssi1 | rssi2 | rssi3 | rssi4 | linear-x | linear-y | linear-z | gravity-x | gravity-y | gravity-z | rotation-x | rotation-y | rotation-z | rotation-w |
| ------------- | ----- | ----- | ----- | ----- | -------- | -------- | -------- | --------- | --------- | --------- | ---------- | ---------- | ---------- | ---------- |
| 1591610015190 | -37   | -45   | -66   | -64   | 0.1322   | -0.0105  | 0.5227   | 0.4463    | 2.8838    | 9.3623    | 0.078042   | -0.12869   | -0.75696   | 0.635895   |
| ...           | ...   | ...   | ...   | ...   | ...      | ...      | ...      | ...       | ...       | ...       | ...        | ...        | ...        | ...        |

可以利用pandas读取数据，并获取为numpy.ndarray格式数据，location模块中类的参数均为numpy.ndarray格式。

```python
df = pd.read_csv(file)
rssi = df[[col for col in df.columns if 'rssi' in col]].values
linear = df[[col for col in df.columns if 'linear' in col]].values
gravity = df[[col for col in df.columns if 'gravity' in col]].values
rotation = df[[col for col in df.columns if 'rotation' in col]].values
```

location模块提供以下功能：

| **location**       | **自定义定位工具包（可以进行数据分析与处理、定位解算与可视化，不同文件可配合使用）** |
| :----------------- | ------------------------------------------------------------ |
| location/fusion.py | 融合定位工具包，包含了EKF融合定位算法。                      |
| locaiton/pdr.py    | PDR定位工具包，包含了步伐检测、航向角推算、步长推算和定位结束输出等常见功能。 |
| location/wifi.py   | wifi指纹定位工具包，包含了常见的在线匹配算法。               |

## Demo1 PDR定位

要进行PDR操作，需要创建一个`pdr.Model`对象：

```python
import location.pdr as pdr
pdr = pdr.Model(linear, gravity, rotation)
```

### Demo1.1 show_steps函数：显示垂直方向合加速度与步伐波峰分布

`show_steps`参数使用说明如下：

- **frequency** - 数据采集频率
- **walkType** - 行走模式

| walkType | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| normal   | 正常行走模式。                                               |
| abnormal | 小范围实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔（大约为2~3秒），所以在实验的过程中，两步之间的控制时间大于3s，算法部分进行了不同处理。 |

示例1，对data/linear_08m中的数据进行分析。

```python
pdr.show_steps(frequency=70, walkType='normal')
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_1.png)

示例2，对fusion01/SType中的数据进行分析。

```pytyon
pdr.show_steps(frequency=70, walkType='abnormal')
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_2.png)

可以发现，对比示例1中的结果，该图峰值之间的间隔比较大。

### Demo1.2 show_gaussian函数：查看数据在一定范围内的分布情况

用来判断数据的分布情况，同时可利用高斯函数拟合（可选），可以用来分析静止惯性数据。

`show_gaussian`参数使用说明如下：

- **data** - 某一轴加速度数据
- **fit** - 布尔值，是否进行高斯拟合

示例1，对data/still中的静止数据进行分析。

```python
acc_z = linear[:,2]
pdr.show_gaussian(acc_z, True)
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_3.png)

示例2，对data/linear_08m中的数据及进行分析：

```python
acc_z = linear[:,2]
pdr.show_gaussian(acc_z, False)
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_4.png)

### Demo1.3 show_data函数：查看三轴加速度的分布情况

`show_data`参数使用说明如下：

| dataType | 描述                         |
| :------- | :--------------------------- |
| linear   | 查看三轴线性加速度的分布情况 |
| gravity  | 查看三轴重力加速度的分布情况 |
| rotation | 查看旋转四元数的数据分布情况 |

示例1，对data/linear_08m中的数据及行分析：

```python
pdr.show_data("linear")
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_5.png)

示例2，对data/linear_08m中的数据及行分析：

```python
pdr.show_data("gravity")
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_6.png)

示例3，对data/linear_08m中的数据及行分析：

```python
pdr.show_data("rotation")
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_7.png)

### Demo1.4 step_stride函数：步长推算函数

`step_stride`传入的是某一时刻的函数值，往往是垂直方向合加速度的峰值点，从而推算出步长值。

```python
stride_length = pdr.step_stride(acc)
```

### Demo1.5 step_counter函数：获取数据中的步伐信息

`step_counter`会返回一个包含每一个步伐信息的字典类型数组，其中index为样本序号，acceleration为步伐加速度峰值，函数参数使用说明如下：

- **frequency** - 数据采集频率
- **walkType** - 行走模式

| walkType | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| normal   | 正常行走模式。                                               |
| abnormal | 实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔，所以在实验的过程中，两步之间的控制时间大于3s，所以算法部分进行了不同处理。 |

示例1，对data/linear_08m中的数据进行分析。

```python
pdr.show_steps(frequency=70, walkType='normal')
print(steps)
```

结果如下：

```python
[
    {
        'index': 298,
        'acceleration': 3.88391844277864
    },
    {
        'index': 354,
        'acceleration': 6.163800299609754
    },
    ......
]
```

示例2，利用data/linear_08m中的数据分析步长推算算法的准确度：

```python
steps = pdr.step_counter(frequency=70, walkType='normal')
print('steps:', len(steps))
stride = pdr.step_stride
accuracy = []
for v in steps:
    a = v['acceleration']
    print(stride(a))
    accuracy.append(
        np.abs(stride(a)-0.8)
    )
square_sum = 0
for v in accuracy:
    square_sum += v*v
acc_mean = (square_sum/len(steps))**(1/2)
print("mean: %f" % acc_mean) # 平均误差
print("min: %f" % np.min(accuracy)) # 最小误差
print("max: %f" % np.max(accuracy)) # 最大误差
print("sum: %f" % np.sum(accuracy)) # 累积误差
```

结果如下：

```shell
steps: 10
0.7019198589090117
0.7878293284920566
0.8188530449804803
0.7910022847592212
0.7700272110821945
0.7766284915404985
0.8636455533175651
0.7555013698705025
0.8041832313898543
0.8379454829417488
mean: 0.043746
min: 0.004183
max: 0.098080
sum: 0.341719
```

### Demo1.6 step_heading函数：步长推算函数

实验过程中手机采用HOLDING模式，即手握手机放在胸前，并且x轴与地面平行，`step_heading`直接返回每一时刻的偏航角yaw值，初始值默认设为0，这里的都是相对值。

```python
yaw = pdr.step_heading()
```

### Demo1.7 pdr_position函数：获取数据中的步伐信息

`pdr_position`会返回每一步的x、y坐标值，以及每一步的步长和航向角：

- **frequency** - 数据采集频率
- **walkType** - 行走模式
- **offset** - 初始航向角大小
- **initPosition** - 初始位置，格式为两个元素的元组形式，分别表示x与y

| walkType | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| normal   | 正常行走模式。                                               |
| fusion   | 实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔，所以在实验的过程中，两步之间的控制时间大于3s，所以算法部分进行了不同处理。 |

示例：

```python
position_x, position_y, strides, angle = pdr.show_steps(frequency=70, walkType='abnormal'， offset=np.pi/2, initPosition=(0,0))
```

### Demo1.8 show_trace函数：输出行走轨迹图

`show_trace`内部使用了`pdr_position`，可以输出轨迹图像。

- **frequency** - 数据采集频率
- **walkType** - 行走模式
- **offset** - 轨迹偏差角度（指上北下南地图中的轨迹旋转到输出轨迹的角度值，实验过程使用了安卓的旋转矢量软件传感器，它集成了加速度计、陀螺仪和磁力计的数据，最终输出了一个以地球坐标系为基础的绝对信息，但实验输出图有时候会分析一个相对定位的情景，所以可以用该偏差值进行修正）
- **initPosition** - 初始位置，格式为两个元素的元组形式，分别表示x与y
- **realTrace** - 两列的numpy.ndarray格式数据，表示真实轨迹坐标，主要是为了方便轨迹的对比（可选）

| walkType | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| normal   | 正常行走模式。                                               |
| abnormal | 小范围实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔，所以在实验的过程中，两步之间的控制时间大于3s，所以算法部分进行了不同处理。 |

示例1，显示data/fusion/LType数据的预测轨迹图：

```python
pdr.show_trace(frequency=70, walkType='abnormal', initPosition=(2,1))
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_8.png)

示例2，显示data/fusion/LType数据的预测轨迹图：

```python
pdr.show_trace(frequency=70, walkType='abnormal', offset=0, real_trace=real_trace, initPosition=(2,1))
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_9.png)

## Demo2 Wi-Fi指纹定位

要进行wifi操作，需要创建一个`wifi.Model`对象，由于实验数据中是以每一步作为一个定位状态，因此这里`pdr`与`wifi`配合使用。从代码层面而言这不是必要的，也可以更换成其他业务。

```python
import location.wifi as wifi
wifi = wifi.Model(rssi)
```

### Demo2.1 rssi_fluctuation函数：查看wifi数据的波动情况

`rssi_fluctuation`需要传入一个布尔值merge作为参数。

| merge | 描述                                                         |
| :---- | :----------------------------------------------------------- |
| True  | 以Wi-Fi扫描次数作为x轴                                       |
| False | 以样本点数目作为x轴，事实上样本点数目合并一个Wi-Fi扫描间隔数据从而得到了合并的效果。 |

示例1，显示data/rssi_fluctuation数据：

```python
wifi.rssi_fluctuation(False)
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo2_1.png)

示例2，显示data/rssi_fluctuation数据：

```python
wifi.rssi_fluctuation(True)
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo2_2.png)

### Demo2.2 determineGaussian函数：查看wifi数据的高斯拟合情况

`determineGaussian`的参数如下所示：

- **data**- numpy.ndarray格式数据，源自一个AP的wifi信号强度数据
- **merge**- 是否合并为每一个扫描次数
- **interval**- 划分间隔，默认值为1
- **wipeRange**- 剔除样本点数目

示例1，显示data/rssi_fluctuation数据：

```python
wifi.determineGaussian(rssi[:, 0], True, wipeRange=170*100)
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo2_3.png)

### Demo2.3 create_fingerpirnt函数：构建一个指纹库

`create_fingerpint`返回一个指纹数据均值与其对应的坐标，传入的参数为所有采集好的指纹数据的文件目录，对文件命名的要求是：需要按照`x-y.csv`的方式对文件命名，比如坐标是`(0,0)`，则文件名为`0-0.csv`。

示例1，对于指纹库data/fusion01/Fingerpint：

```python
rssi, position = wifi.create_fingerprint(fingerprint_path)
```

`rssi`和`position`可以直接带入sklearn的机器学习算法中进行训练。

### Demo2.4 在线匹配算法

这一部分仅作为参考，可以参考https://www.cnblogs.com/rubbninja/p/6186847.html ，建议大家能够用更优的算法到这一部分。为了我自己操作方便，wifi模块在sklearn的基础上对一些常见的在线匹配算法进行了简单的封装，常见的算法与其模如下：

| 方法         | 算法                             |
| ------------ | -------------------------------- |
| wifi.knn_reg | knn                              |
| wifi.svm_reg | support vector machine           |
| wifi.rf_reg  | random forest                    |
| wifi.dbdt    | Gradient Boosting for regression |
| wifi.nn      | Multi-layer Perceptron regressor |

示例1，对于数据data/fusion/LType，如果使用knn：

```python
pdr = pdr.Model(linear, gravity, rotation)
wifi = wifi.Model(rssi)

# 真实轨迹
real_trace = pd.read_csv(real_trace_file).values
# 指纹数据（作为离线数据）
fingerprint_rssi, fingerprint_position = wifi.create_fingerprint(fingerprint_path)

# 找到峰值出的rssi值（作为在线匹配数据）
steps = pdr.step_counter(frequency=70, walkType='fusion')
print('steps:', len(steps))
result = fingerprint_rssi[0].reshape(1, 4)
for k, v in enumerate(steps):
    index = v['index']
    value = rssi[index]
    value = value.reshape(1, len(value))
    result = np.concatenate((result,value),axis=0)

# knn算法
predict, accuracy = wifi.knn_reg(fingerprint_rssi, fingerprint_position, result, real_trace)
print('knn accuracy:', accuracy, 'm')
```

经过上述操作可以得到预测坐标序列和误差均值。

```powershell
knn accuracy: 1.93 m
```

其他算法类似。

### Demo2.5 show_trace函数：生成wifi指纹定位轨迹

`show_trace`传入预测坐标序列和真实轨迹（可选）。

示例1，对于数据data/fusion/LType：

```python
wifi.show_trace(predict, real_trace=real_trace)
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo2_4.png)

由于wifi数据的波动非常大，因此结果会有些难看。

## Demo3 融合定位

要进行fusion操作，需要创建一个`fusion.Model`对象，配合`pdr`与`wifi`一起使用。

```python
import location.fusion as fusion
fusion = fusion.Model()
```

目前fusion中只有ekf算法，为`ekf2d`函数，返回融合的状态列向量的数组，参数列表如下：

| 参数                     | 描述                       |
| ------------------------ | -------------------------- |
| transition_states        | 状态转移序列               |
| observation_states       | 观测值序列                 |
| transition_func          | 状态转移函数               |
| jacobF_func              | 状态转移矩阵的一阶线性函数 |
| initial_state_covariance | 初始状态协方差             |
| observation_matrices     | 观测矩阵                   |
| transition_covariance    | 状态转移协方差             |
| observation_covariance   | 观测协方差                 |

示例1，对于数据data/fusion/LType：

```python
# 已知数据
angle # 航向角序列
L # 步长序列

sigma_wifi = 2
sigma_pdr = .2
sigma_yaw = 10/360
L # 步长序列

# 状态转移函数（参入包含所有状态参数的数组）
def state_conv(parameters_arr):
    x = parameters_arr[0]
    y = parameters_arr[1]
    theta = parameters_arr[2]
    return x+L*np.sin(theta), y+L[i]*np.cos(theta), angle[i] # 伪代码i表示第i个状态
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
    return np.matrix([[1, 0, L[i]*np.cos(angle[i])],
                      [0, 1, -L[i]*np.sin(angle[i])],
                      [0, 0, 1]])

S = fusion.ekf2d(
    transition_states = transition_states
   ,observation_states = observation_states
   ,transition_func = state_conv
   ,jacobF_func = jacobF_func
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
```

对所有轨迹数据进行可视化：

![](https://github.com/salmoshu/location/raw/master/image/Demo3_1.png)



# 数据采集程序

开发安卓数据采集程序这一部分如果想了解更多，请参考我在csdn写的博客：https://blog.csdn.net/qq_28675933/article/details/103668811 。

### Step 1 安装Android Studio

从安卓官网上下载并安装安卓开发工具：https://developer.android.com/studio。

### Step 2 新建项目并替换掉相关文件

使用Android Studio新建一个空项目以后，用android中的文件替换掉空项目中的文件，替换的时候注意修改第一行的包名。

| **android**         | **数据采集程序**                                             |
| :------------------ | ------------------------------------------------------------ |
| activity_main.xml   | 替换位置：项目目录\程序名\app\src\main\res\layout            |
| AndroidManifest.xml | 替换位置：项目目录\程序名\app\src\main                       |
| FileUtil.java       | 替换位置：项目目录\程序名\app\src\main\java\com\example\data |
| LogUtil.java        | 替换位置：项目目录\程序名\app\src\main\java\com\example\data |
| MainActivity.java   | 替换位置：项目目录\程序名\app\src\main\java\com\example\data |
| WifiUtil.java       | 替换位置：项目目录\程序名\\app\src\main\java\com\example\data |

### Step 3 修改文件代码参数

做与WiFi相关的实验时，一般都是使用固定好的路由器，路由器的MAC或者SSID已知，因此需要修改MainActivity.java文件中路由器参数，方便做后续实验。需要修改的参数为`levelArray`、`apNames`、`rssis`，现在假设实验的路由器SSID分别为`router-01`、`router-02`、`router-03`、`router-04`，则需要修改为如下所示：

```java
package com.example.yourappname;

import ...

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    // 配置路由器参数
    private int[] levelArray = new int[4];
    private String[] apNames = new String[]{"router-01","router-02","router-03","router-04"};
    private String rssis = "rssi1,rssi2,rssi3,rssi4";
   
    ...
```

其中`rssis`必须命名为`"rssi"`的形式。

### Step 4 采集数据并导出

使用程序的时候，注意给程序“存储”与“位置”权限，对于采集好的数据，使用adb工具导出，adb工具在Android Studio安装好并且完成初始化之后就捆绑安装。为了方便操作，在环境变量里面配置了adb，这样方便全局操作，adb工具的位置一般在`C:\Users\username\AppData\Local\Android\Sdk\platform-tools`，一次实例操作如下所示。

```powershell
PS C:\user\salmo\Desktop> adb shell
HWFRD:/ $ su
HWFRD:/ # cp -R /data/data/com.example.data2 /sdcard/documents
HWFRD:/ # exit
HWFRD:/ $ exit
PS C:\Users\salmo\Desktop> adb pull /sdcard/documents
/sdcard/documents/: 4 files pulled, 0 skipped. 4.7 MB/s (478123 bytes in 0.096s)
PS C:\Users\salmo\Desktop>
```

由于权限问题，数据先导出到sd卡中，再由sd卡导出到电脑里面，整个过程不算太容易，欢迎大家自寻更好的数据导出方法。

### Step 5 数据文件

一个使用该程序采集的数据文件大概长这个样子：

| timestamp     | rssi1 | rssi2 | rssi3 | rssi4 | linear-x | linear-y | linear-z | gravity-x | gravity-y | gravity-z | rotation-x | rotation-y | rotation-z | rotation-w |
| ------------- | ----- | ----- | ----- | ----- | -------- | -------- | -------- | --------- | --------- | --------- | ---------- | ---------- | ---------- | ---------- |
| 1591610015190 | -37   | -45   | -66   | -64   | 0.1322   | -0.0105  | 0.5227   | 0.4463    | 2.8838    | 9.3623    | 0.078042   | -0.12869   | -0.75696   | 0.635895   |
| ...           | ...   | ...   | ...   | ...   | ...      | ...      | ...      | ...       | ...       | ...       | ...        | ...        | ...        | ...        |



# 历史版本

1. [初始提交版本V1.0](https://github.com/salmoshu/location/tree/207761d6d4e62300dd5a74074e8ace3996b455e9)

   实现了融合定位的基础功能，留作自己备份，建议读者使用最新版本。
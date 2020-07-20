# Location

**Location**是一个多源数据融合定位的项目，目前包含了Wi-Fi指纹定位、行人航位推算（PDR）与基于EKF的Wi-Fi

指纹与PDR的融合定位，本项目针对的平台为安卓平台，融合使用的数据为Wi-Fi数据、加速度计数据、陀螺仪数据、重力传感器数据、姿态仰角数据。

# 目录

------

* [第三方依赖](#第三方依赖)
* [工作目录](#工作目录)
* [开始使用](#开始使用)
    * [Demo1 PDR定位](#Demo1 PDR定位)
    * [Demo2 Wi-Fi指纹定位](#Demo2 Wi-Fi指纹定位)
    * [Demo3 融合定位](#Demo3 融合定位)
* [实验流程](#实验流程)
    * [第一步 开发安卓数据采集程序](#第一步 开发安卓数据采集程序)
    * [第二步 Location库的基本介绍](#第二步 Location库的基本介绍)
    * [Step 3 Create your algorithm](#step-3-create-your-algorithm)
    * [Step 4 Run the simulation](#step-4-run-the-simulation)
    * [Step 5 Show results](#step-5-show-results)
* [Acknowledgement](#Acknowledgement)

# 第三方依赖

- Numpy
- Pandas
- Matplotlib
- Sklearn

# 工作目录

|:--|---|
| **data_collection**    | **数据采集程序**                                             |
| activity_main.xml      | 替换位置：项目目录\程序名\app\src\main\res\layout            |
| AndroidManifest.xml    | 替换位置：项目目录\程序名\app\src\main                       |
| FileUtil.java          | 替换位置：项目目录\程序名\app\src\main\java\com\example\data |
| LogUtil.java           | 替换位置：项目目录\程序名\app\src\main\java\com\example\data |
| MainActivity.java      | 替换位置：项目目录\程序名\app\src\main\java\com\example\data |
| WifiUtil.java          | 替换位置：项目目录\程序名\\app\src\main\java\com\example\data |
| **data** | **实验数据** |
| data/count_steps | 正常直线行走数据，包含了安卓或苹果设备采集的数据，用来判断步伐检测的结果是否良好，其中安卓程序为自己开发的程序，苹果数据采集程序为phyphox（使用的时候注意修改属性名，方便后续程序使用）。 |
| data/fusion01          | 第一次融合定位实验，由于安卓Wi-Fi扫描频率较低，不能满足事实定位，因为在实际实验的时候每一个状态维持的时间大约在3s以上，具体数据包含了：Fingerprint（指纹数据）、Rectangle（矩阵路线行走数据）和SType（S型路线行走数据）。 |
| data/fusion02          |                                                              |
| data/linear_08m        | 正常直线行走数据，行走步数为10步，每一步为固定步长0.8m。     |
| data/rssi_fluctuation  | 一组长时间记录Wi-Fi变化的数据，每个文件大约记录2万条样本数据，该数据可以用来分析Wi-Fi数据的波动情况。 |
| data/still             | 静止数据，用来分析惯性传感器数据的特性。                     |
| **location**           | **自定义定位工具包（可以进行数据分析与处理、定位解算与可视化，不同文件可配合使用）** |
| location/analysis.py   | 分析Wi-Fi数据的特性，可以合并到wifi.py中。                   |
| location/fusion.py     | 融合定位工具包，包含了EKF融合定位算法。                      |
| locaiton/pdr.py        | PDR定位工具包，包含了步伐检测、航向角推算、步长推算和定位结束输出等常见功能。 |
| location/wifi.py       | wifi指纹定位工具包，包含了常见的在线匹配算法。               |



# 开始使用

Location模块的wifi和fusion功能使用前提（pdr由于不使用wifi数据因此不受影响）：操作的文件格式大概为如下所示，其中rssi根据实验过程中AP数量决定。

| timestamp     | rssi1 (dbm) | rssi2 (dbm) | rssi3 (dbm) | rssi4 (dbm) | linear-x(m/s) | linear-y(m/s) | linear-z(m/s) | gravity-x(m/s) | gravity-y(m/s) | gravity-z(m/s) | rotation-x | rotation-y | rotation-z | rotation-w |
| ------------- | ----------- | ----------- | ----------- | ----------- | ------------- | ------------- | ------------- | -------------- | -------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| 1591610015190 | -37         | -45         | -66         | -64         | 0.1322        | -0.0105       | 0.5227        | 0.4463         | 2.8838         | 9.3623         | 0.078042   | -0.12869   | -0.75696   | 0.635895   |
| ...           | ...         | ...         | ...         | ...         | ...           | ...           | ...           | ...            | ...            | ...            | ...        | ...        | ...        | ...        |

可以利用pandas读取数据，并获取为numpy.ndarray格式数据，location模块中类的参数均为numpy.ndarray格式。

```python
df = pd.read_csv(file)
rssi = df[[col for col in df.columns if 'rssi' in col]].values
linear = df[[col for col in df.columns if 'linear' in col]].values
gravity = df[[col for col in df.columns if 'gravity' in col]].values
rotation = df[[col for col in df.columns if 'rotation' in col]].values
```

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
| fusion   | 实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔，所以在实验的过程中，两步之间的控制时间大于3s，所以算法部分进行了不同处理。 |

示例1，对data/linear_08m中的数据进行分析。

```python
pdr.show_steps(frequency=70, walkType='normal')
```

结果如下：


![](https://github.com/salmoshu/location/raw/master/image/Demo1_1.png)

示例2，对fusion01/SType中的数据进行分析。

```pytyon
pdr.show_steps(frequency=70, walkType='fusion')
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
| fusion   | 实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔，所以在实验的过程中，两步之间的控制时间大于3s，所以算法部分进行了不同处理。 |

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

实验过程中手机采用HOLDING模式，即手握手机放在胸前，并且x轴与地面平行，`step_heading`直接返回每一时刻的偏航角yaw值，初始值设为0。

```python
yaw = pdr.step_heading()
```

### Demo1.7 pdr_position函数：获取数据中的步伐信息

`pdr_position`会返回每一步的x、y坐标值，以及每一步的步长和航向角：

- **frequency** - 数据采集频率
- **walkType** - 行走模式
- **offset** - 初始航向角大小

| walkType | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| normal   | 正常行走模式。                                               |
| fusion   | 实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔，所以在实验的过程中，两步之间的控制时间大于3s，所以算法部分进行了不同处理。 |

示例：

```python
position_x, position_y, strides, angle = pdr.show_steps(frequency=70, walkType='fusion'， offset=np.pi/2)
```

### Demo1.8 show_trace函数：获取数据中的步伐信息

`show_trace`内部使用了`pdr_position`，可以输出轨迹图像。

- **frequency** - 数据采集频率
- **walkType** - 行走模式
- **offset** - 初始航向角大小（可选）
- **realTrace** - 真是轨迹坐标（可选）

| walkType | 描述                                                         |
| :------- | :----------------------------------------------------------- |
| normal   | 正常行走模式。                                               |
| fusion   | 实验行走模式。在做Wi-Fi指纹与PDR融合定位的时候，由于安卓设备Wi-Fi扫描有一定的时间间隔，所以在实验的过程中，两步之间的控制时间大于3s，所以算法部分进行了不同处理。 |

示例1，显示data/SType数据的预测轨迹图：

```python
pdr.show_trace(frequency=70, walkType='fusion')
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_8.png)

示例2，显示data/SType数据的预测轨迹图：

```python
pdr.show_trace(frequency=70, walkType='fusion', offset=np.pi/2, real_trace=real_trace)
```

结果如下：

![](https://github.com/salmoshu/location/raw/master/image/Demo1_9.png)

## Demo2 Wi-Fi指纹定位

这一部分如果想了解更多，请参考我在csdn写的博客：https://blog.csdn.net/qq_28675933/article/details/103668811。

### Demo2.1 安装Android Studio

从安卓官网上下载并安装安卓开发工具：https://developer.android.com/studio。

### Demo2.2 新建项目并替换掉相关文件

使用Android Studio新建一个空项目以后，用data_collection中的文件替换掉空项目中的文件，替换的时候注意修改第一行的包名。

### Demo2.3 修改文件代码参数



## Demo3 融合定位

这一部分如果想了解更多，请参考我在csdn写的博客：https://blog.csdn.net/qq_28675933/article/details/103668811。

### Demo3.1 安装Android Studio

从安卓官网上下载安卓开发工具：https://developer.android.com/studio。

### Demo3.2 新建项目并替换掉相关文件

使用Android Studio新建一个空项目以后，用data_collection中的文件替换掉空项目中的文件，替换的时候注意修改第一行的包名。

### Demo3.3 修改文件代码参数























# 实验流程

## 第一步 开发安卓数据采集程序

这一部分如果想了解更多，请参考我在csdn写的博客：https://blog.csdn.net/qq_28675933/article/details/103668811。

### Step 1.1 安装Android Studio

从安卓官网上下载并安装安卓开发工具：https://developer.android.com/studio。

### Step 1.2 新建项目并替换掉相关文件

使用Android Studio新建一个空项目以后，用data_collection中的文件替换掉空项目中的文件，替换的时候注意修改第一行的包名。

### Step 1.3 修改文件代码参数

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

### Step 1.4 采集数据并导出

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

### Step 1.5 数据文件

一个使用该程序采集的数据文件大概长这个样子：

| timestamp     | rssi1 (dbm) | rssi2 (dbm) | rssi3 (dbm) | rssi4 (dbm) | linear-x(m/s) | linear-y(m/s) | linear-z(m/s) | gravity-x(m/s) | gravity-y(m/s) | gravity-z(m/s) | rotation-x | rotation-y | rotation-z | rotation-w |
| ------------- | ----------- | ----------- | ----------- | ----------- | ------------- | ------------- | ------------- | -------------- | -------------- | -------------- | ---------- | ---------- | ---------- | ---------- |
| 1591610015190 | -37         | -45         | -66         | -64         | 0.1322        | -0.0105       | 0.5227        | 0.4463         | 2.8838         | 9.3623         | 0.078042   | -0.12869   | -0.75696   | 0.635895   |
| ...           | ...         | ...         | ...         | ...         | ...           | ...           | ...           | ...            | ...            | ...            | ...        | ...        | ...        | ...        |



## 第二步 Location库的基本介绍

A motion profile specifies the initial states of the vehicle and motion command that drives the vehicle to move, as shown in the following table.

| Ini lat (deg) | ini lon (deg) | ini alt (m) | ini vx_body (m/s) | ini vy_body (m/s) | ini vz_body (m/s) | ini yaw (deg) | ini pitch (deg) | ini roll (deg) |
|---|---|---|---|---|---|---|---|---|
| 32 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| command type | yaw (deg) | pitch (deg) | roll (deg) | vx_body (m/s) | vy_body (m/s) | vz_body (m/s) | command duration (s) |	GPS visibility |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 200 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

The initial position should be given in the LLA (latitude, longitude and altitude) form. The initial velocity is specified in the vehicle body frame. The initial attitude is represented by Euler angles of ZYX rotation sequence.

Motion commands define how the vehicle moves from its initial state. The simulation will generate true angular velocity, acceleration, magnetic field, position, velocity and attitude according to the commands. Combined with sensor error models, these true values are used to generate gyroscope, accelerometer, magnetometer and GPS output.
There is only one motion command in the above table. Indeed, you can add more motion commands to specify the attitude and velocity of the vehicle. You can also define GPS visibility of the vehicle for each command.

Five command types are supported, as listed below.

| Command type | Comment |
|---|---|
| 1 | Directly define the Euler angles change rate and body frame velocity change rate. The change rates are given by column 2~7. The units are deg/s and m/s/s. Column 8 gives how long the command will last. If you want to fully control execution time of each command by your own, you should always choose the motion type to be 1 |
| 2 | Define the absolute attitude and absolute velocity to reach. The target attitude and velocity are given by column 2~7. The units are deg and m/s. Column 8 defines the maximum time to execute the command. If actual executing time is less than max time, the remaining time will not be used and the next command will be executed immediately. If the command cannot be finished within max time, the next command will be executed after max time. |
| 3 | Define attitude change and velocity change. The attitude and velocity changes are given by column 2~7. The units are deg and m/s. Column 8 defines the maximum time to execute the command. |
| 4 | Define absolute attitude and velocity change. The absolute attitude and velocity change are given by column 2~7. The units are deg and m/s. Column 8 defines the maximum time to execute the command. |
| 5 | Define attitude change and absolute velocity. The attitude change and absolute velocity are given by column 2~7. The units are deg and m/s. Column 8 defines the maximum time to execute the command. |

### An example of motion profile

| Ini lat (deg) | ini lon (deg) | ini alt (m) | ini vx_body (m/s) | ini vy_body (m/s) | ini vz_body (m/s) | ini yaw (deg) | ini pitch (deg) | ini roll (deg) |
|---|---|---|---|---|---|---|---|---|
| 32 | 120 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| command type | yaw (deg) | pitch (deg) | roll (deg) | vx_body (m/s) | vy_body (m/s) | vz_body (m/s) | command duration (s) |	GPS visibility |
| 1 | 0| 0 | 0 | 0 | 0 | 0 | 200 | 1 |
| 5 | 0 | 45 | 0 | 10 | 0 | 0 | 250 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 1 |
| 3 | 90 | -45 | 0 | 0 | 0 | 0 | 25 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 |
| 3 | 180 | 0 | 0 | 0 | 0 | 0 | 25 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 |
| 3 | -180 | 0 | 0 | 0 | 0 | 0 | 25 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 |
| 3 | 180 | 0 | 0 | 0 | 0 | 0 | 25 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 |
| 3 | -180 | 0 | 0 | 0 | 0 | 0 | 25 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 |
| 3 | 180 | 0 | 0 | 0 | 0 | 0 | 25 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 |
| 3 | -180 | 0 | 0 | 0 | 0 | 0 | 25 | 1 |
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 50 | 1 |
| 5 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 1 |

The initial latitude, longitude and altitude of the vehicle are 32deg, 120deg and 0 meter, respectively. The initial velocity of the vehicle is 0. The initial Euler angles are 0deg pitch, 0deg roll and 0deg yaw, which means the vehicle is level and its x axis points to the north.

| command type | yaw (deg) | pitch (deg) | roll (deg) | vx_body (m/s) | vy_body (m/s) | vz_body (m/s) | command duration (s) |	GPS visibility |
|---|---|---|---|---|---|---|---|---|
| 1 | 0| 0 | 0 | 0 | 0 | 0 | 200 | 1 |

This command is of type 1. Command type1 directly gives Euler angle change rate and velocity change rate. In this case, they are zeros. That means keep the current state (being static) of the vehicle for 200sec. During this period, GPS is visible.

| command type | yaw (deg) | pitch (deg) | roll (deg) | vx_body (m/s) | vy_body (m/s) | vz_body (m/s) | command duration (s) |	GPS visibility |
|---|---|---|---|---|---|---|---|---|
| 5 | 0 | 45 | 0 | 10 | 0 | 0 | 250 | 1 |

This command is of type 5. Command type 5 defines attitude change and absolute velocity. In this case, the pitch angle will be increased by 45deg, and the velocity along the x axis of the body frame will be accelerated to 10m/s. This command should be executed within 250sec.

| command type | yaw (deg) | pitch (deg) | roll (deg) | vx_body (m/s) | vy_body (m/s) | vz_body (m/s) | command duration (s) |	GPS visibility |
|---|---|---|---|---|---|---|---|---|
| 3 | 90 | -45 | 0 | 0 | 0 | 0 | 25 | 1 |

This command is of type 3. Command type 3 defines attitude change and velocity change. In this case, the yaw angle will be increased by 90deg, which is a right turn. The pitch angle is decreased by 45deg. The velocity of the vehicle does not change. This command should be executed within 25sec.

The following figure shows the trajectory generated from the motion commands in the above table. The trajectory sections corresponding to the above three commands are marked by command types 1, 5 and 3.
<div align=center>
<img width="500"  src="https://github.com/Aceinna/gnss-ins-sim/blob/master/gnss_ins_sim/docs/images/motion_profile_demo.png"/>
</div>

## Step 3 Create your algorithm

```python
algo = allan_analysis.Allan() # an Allan analysis demo algorithm
```

An algorithm is an object of a Python class. It should at least include the following members:

### self.input and self.output

The member variable 'input' tells **gnss-ins-sim** what data the algorithm need. 'input' is a tuple or list of strings.

The member variable 'output' tells **gnss-ins-sim** what data the algorithm returns. 'output' is a tuple or list of strings.

Each string in 'input' and 'output' corresponds to a set of data supported by **gnss-ins-sim**. The following is a list of supported data by **gnss-ins-sim**.

| name | description |
|-|-|
| 'ref_frame' | Reference frame used as the navigation frame and the attitude reference. <br> 0: NED (default), with x axis pointing along geographic north, y axis pointing eastward, z axis pointing downward. Position will be expressed in LLA form, and the velocity of the vehicle relative to the ECEF frame will be expressed in local NED frame. <br> 1: a virtual inertial frame with constant g, x axis pointing along geographic or magnetic north, z axis pointing along g, y axis completing a right-handed coordinate system. Position and velocity will both be in the [x y z] form in this frame. <br> **Notice: For this virtual inertial frame, position is indeed the sum of the initial position in ecef and the relative position in the virutal inertial frame. Indeed, two vectors expressed in different frames should not be added. This is done in this way here just to preserve all useful information to generate .kml files. Keep this in mind if you use this result.|
| 'fs' | Sample frequency of IMU, units: Hz |
| 'fs_gps' | Sample frequency of GNSS, units: Hz |
| 'fs_mag' | Sample frequency of magnetometer, units: Hz |
| 'time' | Time series corresponds to IMU samples, units: sec. |
| 'gps_time' | Time series corresponds to GNSS samples, units: sec. |
| 'algo_time' | Time series corresponding to algorithm output, units: ['s']. If your algorithm output data rate is different from the input data rate, you should include 'algo_time' in the algorithm output. |
| 'gps_visibility' | Indicate if GPS is available. 1 means yes, and 0 means no. |
| 'ref_pos' | True position in the navigation frame. When users choose NED (ref_frame=0) as the navigation frame, positions will be given in the form of [Latitude, Longitude, Altitude], units: ['rad', 'rad', 'm']. When users choose the virtual inertial frame, positions (initial position + positions relative to the  origin of the frame) will be given in the form of [x, y, z], units:  ['m', 'm', 'm']. |
| 'ref_vel' | True velocity w.r.t the navigation/reference frame expressed in the NED frame, units: ['m/s', 'm/s', 'm/s']. |
| 'ref_att_euler' | True attitude (Euler angles, ZYX rotation sequency), units: ['rad', 'rad', 'rad'] |
| 'ref_att_quat' | True attitude (quaternions) |
| 'ref_gyro' | True angular velocity in the body frame, units: ['rad/s', 'rad/s', 'rad/s'] |
| 'ref_accel' | True acceleration in the body frame, units: ['m/s^2', 'm/s^2', 'm/s^2'] |
| 'ref_mag' | True geomagnetic field in the body frame, units: ['uT', 'uT', 'uT'] (only available when axis=9 in IMU object) |
| 'ref_gps' | True GPS position/velocity, ['rad', 'rad', 'm', 'm/s', 'm/s', 'm/s'] for NED (LLA), ['m', 'm', 'm', 'm/s', 'm/s', 'm/s'] for virtual inertial frame (xyz) (only available when gps=True in IMU object) |
| 'gyro' | Gyroscope measurements, 'ref_gyro' with errors |
| 'accel' | Accelerometer measurements, 'ref_accel' with errors |
| 'mag' | Magnetometer measurements, 'ref_mag' with errors |
| 'gps' | GPS measurements, 'ref_gps' with errors |
| 'ad_gyro' | Allan std of gyro, units: ['rad/s', 'rad/s', 'rad/s'] |
| 'ad_accel' | Allan std of accel, units: ['m/s2', 'm/s2', 'm/s2'] |
| 'pos' | Simulation position from algo, units: ['rad', 'rad', 'm'] for NED (LLA), ['m', 'm', 'm'] for virtual inertial frame (xyz). |
| 'vel' | Simulation velocity from algo, units: ['m/s', 'm/s', 'm/s'] |
| 'att_euler' | Simulation attitude (Euler, ZYX)  from algo, units: ['rad', 'rad', 'rad'] |
| 'att_quat' | Simulation attitude (quaternion)  from algo |
| 'wb' | Gyroscope bias estimation, units: ['rad/s', 'rad/s', 'rad/s'] |
| 'ab' | Accelerometer bias estimation, units: ['m/s^2', 'm/s^2', 'm/s^2'] |
| 'gyro_cal' | Calibrated gyro output, units: ['rad/s', 'rad/s', 'rad/s'] |
| 'accel_cal' | Calibrated acceleromter output, units: ['m/s^2', 'm/s^2', 'm/s^2'] |
| 'mag_cal' | Calibrated magnetometer output, units: ['uT', 'uT', 'uT'] |
| 'soft_iron' | 3x3 soft iron calibration matrix |
| 'hard_iron' | Hard iron calibration, units: ['uT', 'uT', 'uT'] |

### self.run(self, set_of_input)

This is the main procedure of the algorithm. **gnss-ins-sim** will call this procedure to run the algorithm.
'set_of_input' is a list of data that is consistent with self.input.
For example, if you set self.input = ['fs', 'accel', 'gyro'], you should get the corresponding data this way:

```python
  def run(self, set_of_input):
      # get input
      fs = set_of_input[0]
      accel = set_of_input[1]
      gyro = set_of_input[2]
```

### self.get_results(self)

**gnss-ins-sim** will call this procedure to get resutls from the algorithm. The return should be consistent with self.output.
For example, if you set self.output = ['allan_t', 'allan_std_accel', 'allan_std_gyro'], you should return the results this way:

```python
  def get_results(self):
      self.results = [tau,
                      np.array([avar_ax, avar_ay, avar_az]).T,
                      np.array([avar_wx, avar_wy, avar_wz]).T]
      return self.results
```

### self.reset(self)

**gnss-ins-sim** will call this procedure after run the algorithm. This is necessary when you want to run the algorithm more than one time and some states of the algorithm should be reinitialized.

## Step 4 Run the simulation

### step 4.1 Create the simulation object

```python
  sim = ins_sim.Sim(
        # sample rate of imu (gyro and accel), GPS and magnetometer
        [fs, fs_gps, fs_mag],
        # initial conditions and motion definition,
        # see IMU in ins_sim.py for details
        data_path+"//motion_def-90deg_turn.csv",
        # reference frame
        ref_frame=1,
        # the imu object created at step 1
        imu,
        # vehicle maneuver capability
        # [max accel, max angular accel, max angular rate]
        mode=np.array([1.0, 0.5, 2.0]),
        # specifies the vibration model for IMU
        env=None,
        #env=np.genfromtxt(data_path+'//vib_psd.csv', delimiter=',', skip_header=1),
        # the algorithm object created at step 2
        algorithm=algo)
```

**gnss-ins-sim** supports running multiple algorithms in one simulation. You can refer to demo_multiple_algorihtms.py for example.

There are three kinds of vibration models:

| vibration model | description |
|-|-|
| 'ng-random' | normal-distribution random vibration, rms is n*9.8 m/s^2 |
| 'n-random' | normal-distribution random vibration, rms is n m/s^2 |
| 'ng-mHz-sinusoidal' | sinusoidal vibration of m Hz, amplitude is n*9.8 m/s^2 |
| 'n-mHz-sinusoidal' | sinusoidal vibration of m Hz, amplitude is n m/s^2 |
| numpy array of size (n,4) | single-sided PSD. [freqency, x, y, z], m^2/s^4/Hz |

### Step 4.2 Run the simulation

```python
sim.run()     # run for 1 time
sim.run(1)    # run for 1 time
sim.run(100)  # run for 100 times
```

## Step 5 Show results

```python
# generate a simulation summary,
# and save the summary and all data in directory './data'.
# You can specify the directory.
sim.results('./data/')

# generate a simulation summary, do not save any file
sim.results()

# plot interested data
sim.plot(['ref_pos', 'gyro'], opt={'ref_pos': '3d'})
```

# Acknowledgement

- Geomagnetic field model [https://github.com/cmweiss/geomag/tree/master/geomag](https://github.com/cmweiss/geomag/tree/master/geomag)
- MRepo [http://www.instk.org](http://www.instk.org/)

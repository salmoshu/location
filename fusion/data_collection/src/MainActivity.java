package com.example.data;

import androidx.appcompat.app.AppCompatActivity;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.net.wifi.ScanResult;
import android.net.wifi.WifiManager;
import android.os.Bundle;
import android.os.Handler;
import android.view.View;
import android.widget.TextView;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

public class MainActivity extends AppCompatActivity implements SensorEventListener {
    private WifiBroadcastReceiver wifiReceiver;
    private List<ScanResult> wifiResultList = new ArrayList<>();
    private StringBuilder dataStringBuilder = new StringBuilder("timestamp,rssi1,rssi2,rssi3,rssi4," +
            "linear-x,linear-y,linear-z,gravity-x,gravity-y,gravity-z,rotation-x,rotation-y,rotation-z,rotation-w\n");
    private int[] levelArray = new int[4];

    private SensorManager sensorManager;
    private Sensor linear;
    private Sensor rotation;
    private Sensor gravity;
    private float[] linearArray= new float[3];
    private float[] rotationArray= new float[4];
    private float[] gravityArray = new float[3];

    private Handler handler;
    private Runnable runnable;

    private int counter = 0; // 记录秒数

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        registerWifiReceiver();

        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        linear = sensorManager.getDefaultSensor(Sensor.TYPE_LINEAR_ACCELERATION);
        rotation = sensorManager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR);
        gravity = sensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        unregisterReceiver(wifiReceiver);
    }

    /** Called when the user touches the "开始" button */
    public void collectRSSI(View view) {
        final Context context = this;
        final TextView textView = (TextView)findViewById(R.id.status_label);
        // 读取wifi数据并保存
        LogUtil.i("START", "RSSI DATA COLLECTING");

        handler = new Handler();

        runnable = new Runnable() {
            @Override
            public void run() {
                ++counter;
                if(counter%100 == 0) {
                    textView.setText("数据记录中: " + counter + "条数据");
                }

                WifiUtil.scanStart(context);
                // 没有实时数据的时候，保存老数据的值
                dataStringBuilder.append(System.currentTimeMillis());

                // 默认四个AP都存在，且SSID分别为
                // EXPERIMENT-01 -02 -03 -04
                for(ScanResult scanResult:wifiResultList) {
                    if(scanResult.SSID.equals("EXPERIMENT-01")) {
                        levelArray[0] = scanResult.level;
                    }else if(scanResult.SSID.equals("EXPERIMENT-02")) {
                        levelArray[1] = scanResult.level;
                    }else if(scanResult.SSID.equals("EXPERIMENT-03")) {
                        levelArray[2] = scanResult.level;
                    }else if(scanResult.SSID.equals("EXPERIMENT-04")) {
                        levelArray[3] = scanResult.level;
                    }else {
                        continue;
                    }
                }

                for(int value:levelArray) {
                    dataStringBuilder.append(",");
                    dataStringBuilder.append(value);
                }

                for(float value:linearArray) {
                    dataStringBuilder.append(",");
                    dataStringBuilder.append(value);
                }

                for(float value:gravityArray) {
                    dataStringBuilder.append(",");
                    dataStringBuilder.append(value);
                }

                for(float value:rotationArray) {
                    dataStringBuilder.append(",");
                    dataStringBuilder.append(value);
                }

                dataStringBuilder.append("\n");

                handler.postDelayed(this, 10); // 100Hz
            }
        };

        handler.postDelayed(runnable, 0);
    }

    /** Called when the user touches the "结束" button */
    public void saveFile(View view) throws IOException {
        // 将数据保存到本地文件
        Date date = new Date();
        SimpleDateFormat ft = new SimpleDateFormat("yyyy-MM-dd-hh-mm-ss");
        String fileName = ft.format(date) + ".csv";

        handler.removeCallbacks(runnable);
        TextView textView = (TextView)findViewById(R.id.status_label);
        textView.setText("数据已保存。");
        FileUtil.writeFile(fileName, dataStringBuilder.toString(), this);
//        LogUtil.i("FILE", FileUtil.readFile(fileName, this));

        dataStringBuilder = new StringBuilder("timestamp,rssi1,rssi2,rssi3,rssi4," +
                "linear-x,linear-y,linear-z,gravity-x,gravity-y,gravity-z,rotation-x,rotation-y,rotation-z,rotation-w\n");
        levelArray = new int[4];
        counter = 0;
    }

    // 注册广播
    private void registerWifiReceiver() {
        wifiReceiver  = new WifiBroadcastReceiver();
        IntentFilter filter = new IntentFilter();
        filter.addAction(WifiManager.WIFI_STATE_CHANGED_ACTION);//监听wifi是开关变化的状态
        filter.addAction(WifiManager.NETWORK_STATE_CHANGED_ACTION);//监听wifi连接状态广播,是否连接了一个有效路由
        filter.addAction(WifiManager.SCAN_RESULTS_AVAILABLE_ACTION);//监听wifi列表变化（开启一个热点或者关闭一个热点）
        this.registerReceiver(wifiReceiver, filter);
    }

    // 监听wifi状态
    public class WifiBroadcastReceiver extends BroadcastReceiver {
        @Override
        public void onReceive(Context context, Intent intent) {
            if (WifiManager.SCAN_RESULTS_AVAILABLE_ACTION.equals(intent.getAction())) {
                // 网络发生改变
                scanWifi();
            }
        }
    }

    private void scanWifi() {
        wifiResultList = WifiUtil.scanWifiInfo(this);
    }

    /*
    *  以下部分为传感器数据api中的相关函数
    * */

    @Override
    public final void onAccuracyChanged(Sensor sensor, int accuracy) {
        // Do something here if sensor accuracy changes.
    }

    @Override
    public final void onSensorChanged(SensorEvent event) {
        if (event.sensor.getType() == Sensor.TYPE_LINEAR_ACCELERATION) {
            // 获取加速度传感器的三个参数
            linearArray[0] = event.values[0];
            linearArray[1] = event.values[1];
            linearArray[2] = event.values[2];
        }
        if (event.sensor.getType() == Sensor.TYPE_ROTATION_VECTOR) {
            rotationArray[0] = event.values[0];
            rotationArray[1] = event.values[1];
            rotationArray[2] = event.values[2];
            rotationArray[3] = event.values[3];
        }
        if (event.sensor.getType() == Sensor.TYPE_GRAVITY) {
            // 获取加速度传感器的三个参数
            gravityArray[0] = event.values[0];
            gravityArray[1] = event.values[1];
            gravityArray[2] = event.values[2];
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        // 注册传感器监听函数
        sensorManager.registerListener(this, linear, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this, rotation, SensorManager.SENSOR_DELAY_NORMAL);
        sensorManager.registerListener(this,gravity, SensorManager.SENSOR_DELAY_NORMAL);
    }

    @Override
    protected void onPause() {
        super.onPause();
        // 注销监听函数
        sensorManager.unregisterListener(this);
    }
}

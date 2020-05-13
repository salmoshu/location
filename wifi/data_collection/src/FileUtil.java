package com.example.data;

import android.content.Context;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

import static android.content.Context.MODE_PRIVATE;

public class FileUtil {
    //写数据
    public static void writeFile(String fileName, String writeData, Context context) throws IOException {
        try{
            FileOutputStream fout = context.openFileOutput(fileName, MODE_PRIVATE);
            byte [] bytes = writeData.getBytes();
            fout.write(bytes);
            fout.close();
            LogUtil.i("END", "FILE WRITTEN");
        }

        catch(Exception e){
            e.printStackTrace();
        }
    }

    /*
     * 这里定义的是文件读取的方法
     * */
    public static String readFile(String filename, Context context) throws IOException {
        //打开文件输入流
        FileInputStream input = context.openFileInput(filename);
        byte[] temp = new byte[1024];
        StringBuilder sb = new StringBuilder("");
        int len = 0;
        //读取文件内容:
        while ((len = input.read(temp)) > 0) {
            sb.append(new String(temp, 0, len));
        }
        //关闭输入流
        input.close();
        return sb.toString();
    }
}

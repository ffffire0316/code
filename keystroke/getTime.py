# zju生仪105
# Time: 2022/10/10 21:23

from pynput import keyboard
import time
import  matplotlib.pyplot as plt
import  os,sys
import pandas
import csv
from numpy import array

#全局变量声明，随意初始化一下
press_begin = time.time()
release_begin = time.time()
#特征数据
press_time = []
release_time = []
period_time = []

#按键顺序
label = 0

def on_press(key):
    #避免机器周期影响结果，先用一个形参存当前时间
    now_p = time.time()
    global press_begin
    global label
    print('Key:',label)
    global release_time
    global period_time
    if label == 0:
        press_begin = now_p
        label += 1
        return False
    else:
        release_time.append(now_p-release_begin)
        period_time.append(now_p-press_begin)
        press_begin = now_p
        label += 1
        return False


def on_release(key):
    # 避免机器周期影响结果，先用一个形参存当前时间
    print(key)
    now_r = time.time()
    global label
    global release_begin
    global press_time
    if label >= 1:
        release_begin = now_r
        press_time.append(now_r - press_begin)
    return False

def getKeyData(num:int):
    global press_time
    global release_time
    global period_time
    #press_time, release_time, period_time = [], [], []
    while label<num:
        with keyboard.Listener(
                on_press=on_press) as listener:
            listener.join()
        with keyboard.Listener(
                on_release=on_release) as listener:
            listener.join()
    #时间单位转为毫秒

    press_time = [x*1000 for x in press_time]
    release_time = [x*1000 for x in release_time]
    period_time = [x*1000 for x in period_time]

    return press_time,release_time,period_time


# write_data_using_pandas(data,path)
def getKeyDatas(n:int):
    all_data=[]
    for i in range (n):
        print("It is ",i)
        timeOfPress, timeOfRelease, timeOfPeriod = getKeyData(10)
        print("please input once again")
        global label,press_time,release_time,period_time
        label=0
        press_time,release_time,period_time=[],[],[]
        current_data=timeOfPress+timeOfRelease+timeOfPeriod
        all_data.append(current_data)
    print(all_data)
    return all_data
# write_data_using_pandas(all_data,path)

if __name__=="__main__":


    #测试getKeyDatas函数
    getKeyDatas(2)



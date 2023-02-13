import  pandas as pd
import os,sys
from write_data import set_file_name
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft


def read_data_using_pandas(path):
    df=pd.read_csv(filepath_or_buffer=path,header=0)
    return df

def process_data_fft(data):
    #数据预处理
    complex_array=fft.fft(data)


if __name__=="__main__":
    path=set_file_name('huangyi')
    data_frame=read_data_using_pandas(path)
    print(data_frame)
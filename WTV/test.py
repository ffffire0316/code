import numpy as np
import pywt
import matplotlib.pyplot as plt
from itertools import chain

def list2epoch(data,epoch):
    # 将输入的一维列表按照epoch化成二维列表
    list = [data[i:i + epoch] for i in range(0, len(data), epoch)]
    return list

def data2epoch(data,epoch):
    # 将输入的二维数据合并并按照epoch划分成新的二维数组
    data_epoch=[]
    for i in range(len(data)):
        data_epoch.append(list2epoch(data[i],epoch))
    data_process = list(chain.from_iterable(data_epoch))
    data_process = np.array(data_process)
    return data_process


annotation=[6,0,1,2,1]
interval=[(0,30630),(30630,30750),(30750,31140),(31140,31170),(31170,31260)]
epoch=30
data=np.zeros(792000)
list_process=list2epoch(data,epoch)

# data1=data[]
assert len(annotation)==len(interval)
# print(len(interval))
label=[]
x_data=[]
for i in range(len(annotation)):
    index_start=interval[i][0]
    index_end=interval[i][1]
    data1 = data[index_start:index_end]
    index_len=int((index_end-index_start)/30)
    x_data.append(data1)
    label.append(np.full((index_len,1),annotation[i]))

data_process=data2epoch(x_data,epoch)


label=list(chain.from_iterable(label))
label=np.array(label)
label=label.flatten()

for i in range(len(annotation)-1):
    print(i)
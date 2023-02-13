import numpy as np
from itertools import chain

class Psg_Ann:
    def __init__(self,signal,annotation,interval):
        self.signal=signal
        self.preprocess()
        self.annotation=annotation
        self.interval=interval
        assert len(annotation) == len(interval)
        self.label=[]
        self._data=[]

        for i in range(len(annotation)):
            # 对标签进行扩容以匹配data
            # 每个duration的开始结束点
            index_start = int(interval[i][0])
            index_end = int(interval[i][1])
            index_len = int((index_end - index_start) / 30)
            # 清洗数据 除去annation为5的片段
            if annotation[i]==5:
                print("剔除一次")
                # self.annotation.pop(i)
                # self.interval.pop(i)
            else:
                self.label.append(np.full((index_len, 1), annotation[i]))
                data1=self.signal[index_start:index_end]
                self._data.append(data1)

        self.data_process = self.data2epoch(self._data, 30)
        self.label = list(chain.from_iterable(self.label))
        self.label = np.array(self.label)
        self.label = self.label.flatten()


    #对数据进行预处理
    def preprocess(self):
        # 归一化
        maximums, minimums, avgs = self.signal.max(axis=0), self.signal.min(axis=0),self.signal.sum(axis=0) / self.signal.shape[0]
        self.signal=(self.signal - minimums) / (maximums - minimums)




    def list2epoch(self,data,epoch):
        # 将输入的一维列表按照epoch化成二维列表
        list = [data[i:i + epoch] for i in range(0, len(data), epoch)]
        return list

    def data2epoch(self,data,epoch):
        # 将输入的二维数据合并并按照epoch划分成新的二维数组
        data_epoch=[]
        for i in range(len(data)):
            data_epoch.append(self.list2epoch(data[i],epoch))
        data_process = list(chain.from_iterable(data_epoch))
        data_process = np.array(data_process)
        return data_process


    # def
    #保存至npy数据类型
    # def


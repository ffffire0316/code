import numpy as np
from itertools import chain

class Psg_Ann:
    def __init__(self,signal,annotation,interval):
        self.signal=signal
        if self.signal.shape[0] > self.signal.shape[1]:
            self.signal = self.signal.T
        self.annotation=annotation
        self.interval=interval
        assert len(annotation) == len(interval)

        if annotation[0] == 0:
            self.annotation.pop(0)
            self.interval.pop(0)
        index_start = int(interval[0][0])
        index_end = int(interval[0][1])
        index_len = int((index_end - index_start) / 30)
        self.label =[self.annotation[0]]*index_len
        self.data = self.signal[:,index_start * 100:index_end * 100]

        for i in range(1,len(annotation)):
            # 对标签进行扩容以匹配data
            # 每个duration的开始结束点
            index_start = int(interval[i][0])
            index_end = int(interval[i][1])
            index_len = int((index_end - index_start) / 30)
            # 清洗数据 除去annation为5的片段
            if annotation[i]==5:
                print("剔除一次")
            else:
                self.label+=[self.annotation[i]]* index_len
                data = self.signal[:,index_start * 100:index_end * 100]
                self.data =np.hstack((self.data,data))
        self.label=np.array(self.label)
        self.data_process=self.data.reshape((7,-1,3000))
        self.show_labels_num()
        # print(self.label)


    #对数据进行预处理
    def preprocess(self):
        # 归一化
        maximums, minimums, avgs = self.signal.max(axis=0), self.signal.min(axis=0),self.signal.sum(axis=0) / self.signal.shape[0]
        self.signal=(self.signal - minimums) / (maximums - minimums)

    def show_labels_num(self):
        n_0 = np.sum(self.label == 0)
        n_1 = np.sum(self.label == 1)
        n_2 = np.sum(self.label== 2)
        n_3 = np.sum(self.label == 3)
        n_4 = np.sum(self.label == 4)
        print(n_0, n_1, n_2, n_3, n_4)



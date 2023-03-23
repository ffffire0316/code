import glob
import os
from edf_reader import edf_read
import mne
import numpy as np
from mne.io import read_raw_edf
from mne import read_annotations
# from psg_ann import Psg_Ann
from itertools import chain

class Psg_Ann:
    def __init__(self,signal,annotation,interval):
        self.signal=signal
        if self.signal.shape[0] > self.signal.shape[1]:
            self.signal = self.signal.T
        # self.preprocess()
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
        # self.label.append(label)

        for i in range(1,len(annotation)):
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
                self.label+=[self.annotation[i]]* index_len
                data = self.signal[:,index_start * 100:index_end * 100]
                self.data =np.hstack((self.data,data))
                # data1=self.signal[index_start*100:index_end*100]
                # self._data.append(data1)
        self.label=np.array(self.label)
        self.data_process=self.data.reshape((7,-1,3000))
        self.show_labels_num()

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

    def show_labels_num(self):
        n_0 = np.sum(self.label == 0)
        n_1 = np.sum(self.label == 1)
        n_2 = np.sum(self.label== 2)
        n_3 = np.sum(self.label == 3)
        n_4 = np.sum(self.label == 4)
        print(n_0, n_1, n_2, n_3, n_4)

ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

data_dir=r"E:\workstation\reference\xai-sleep\data\sleepedf"
# output_dir=r"E:\workstation\reference\xai-sleep\data\edf20npz"
psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))

raw = read_raw_edf(psg_fnames[1], preload=False, verbose=False)
sampling_rate = raw.info['sfreq']
signal_dict = {}
digital_signals=[]

if sampling_rate not in signal_dict: signal_dict[sampling_rate] = []
signal_dict[sampling_rate].append((raw.ch_names, raw.get_data()))

# Wrap data into DigitalSignals
for sfreq, signal_lists in signal_dict.items():
    data = np.concatenate([x for _, x in signal_lists], axis=0)
    channel_names = [name for names, _ in signal_lists for name in names]
mne_anno:mne.Annotations=mne.read_annotations(ann_fnames[1])
labels=list(sorted(set(mne_anno.description)))
interval,annotation=[],[]
# label2int={lb:i for i,lb in enumerate(labels)}
# print(mne_anno)
for onset,duration,label in zip(mne_anno.onset,mne_anno.duration,mne_anno.description):
    interval.append((onset,onset+duration))
    annotation.append(ann2label[label])

digital_signals.append(Psg_Ann(data,annotation,interval))
process_signal = Psg_Ann(data, annotation, interval)
print(process_signal)
#




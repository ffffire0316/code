import glob
import os
from edf_reader import edf_read
import mne
import numpy as np
from mne.io import read_raw_edf
from mne import read_annotations
from psg_ann import Psg_Ann
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

data_dir=r"E:\xai-sleep1\data\sleepedf"
output_dir=r"E:\workstation\reference\xai-sleep\data\edf20npz"
psg_fnames = glob.glob(os.path.join(data_dir, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(data_dir, "*Hypnogram.edf"))

raw = read_raw_edf(psg_fnames[1], preload=False, verbose=False)
sampling_rate = raw.info['sfreq']
signal_dict = {}

if sampling_rate not in signal_dict: signal_dict[sampling_rate] = []
signal_dict[sampling_rate].append((raw.ch_names, raw.get_data()))
# Wrap data into DigitalSignals
for sfreq, signal_lists in signal_dict.items():
    data = np.concatenate([x for _, x in signal_lists], axis=0)
    channel_names = [name for names, _ in signal_lists for name in names]
data=data[0]
mne_anno:mne.Annotations=mne.read_annotations(ann_fnames[1])
labels=list(sorted(set(mne_anno.description)))
interval,annotation=[],[]
# label2int={lb:i for i,lb in enumerate(labels)}
# print(mne_anno)
for onset,duration,label in zip(mne_anno.onset,mne_anno.duration,mne_anno.description):
    interval.append((onset,onset+duration))

    annotation.append(ann2label[label])
    #
    # digital_signals.append(Psg_Ann(data,annotation,interval))
process_signal = Psg_Ann(data, annotation, interval)
print(process_signal)
#
# digital_signals.append(Psg_Ann(data,annotation,interval))
# process_signal=Psg_Ann(data,annotation,interval)
# # print(psg_fnames)
# data=np.arange(1,100)
# print(data)
# for i in range(len(psg_fnames)):
#     filename = os.path.basename(psg_fnames[i]).replace(".edf", ".npz")
#     print(filename)
#     save_dict = {
#         "x": data,
#         # "y": annotation,
#         # "fs": sampling_rate
#     }
#     print(os.path.join(output_dir, filename))
#     np.savez(os.path.join(output_dir, filename), **save_dict)
import torch
import torch.utils.data as Data
# BATCH_SIZE=5
# x = torch.linspace(1, 10, 10)   # 训练数据
# print(x)
# y = torch.linspace(10, 1, 10)   # 标签
# print(y)
# # 把数据放在数据库中
# torch_dataset = Data.TensorDataset(x, y)  # 对给定的 tensor 数据，将他们包装成 datas
# print(torch_dataset)
#
# loader = Data.DataLoader(
#     # 从数据库中每次抽出batch size个样本
#     dataset=torch_dataset,       # torch TensorDataset format
#     batch_size=BATCH_SIZE,       # mini batch size
#     shuffle=True,                # 要不要打乱数据 (打乱比较好)
#     num_workers=0,               # 多线程来读数据
# )
#
# def show_batch():
#     for epoch in range(3):
#         for step, (batch_x, batch_y) in enumerate(loader):
#             # training
#             print("steop:{}, batch_x:{}, batch_y:{}".format(step, batch_x, batch_y))
#
# train,test=Data.random_split(dataset=torch_dataset,lengths=[7,3])
import torch
from torch.utils.data import Dataset



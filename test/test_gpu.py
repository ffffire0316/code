import torch
import tensorboard
import time
# import py
#
# print(torch.__version__)  # 查看torch当前版本号
#
# print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
#
# print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True
import numpy as np
import os
import glob
from mne.io import read_raw_edf
import mne
# import pyedflib
data_path=r"E:\xai-sleep\data\sleepedf"
psg_fnames = glob.glob(os.path.join(data_path, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))
# print(psg_fnames)

# 写入数据
raw = read_raw_edf(psg_fnames[0], preload=False, verbose=False)
sampling_rate = raw.info['sfreq']
signal_dict = {}

if sampling_rate not in signal_dict: signal_dict[sampling_rate] = []
signal_dict[sampling_rate].append((raw.ch_names, raw.get_data()))
# Wrap data into DigitalSignals
for sfreq, signal_lists in signal_dict.items():
    data = np.concatenate([x for _, x in signal_lists], axis=0)
    # data = np.transpose(data)
    channel_names = [name for names, _ in signal_lists for name in names]

data = data[0:2][:]  # 取第一er个通道数据 即EEG Fpz-Cz
info = mne.create_info(ch_names=channel_names[0:2], sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data, info)

# # 定义EDF文件名
edf_filename = 'test.edf'
data2 = read_raw_edf(edf_filename, preload=False, verbose=False).get_data()
data_=max((data-data2)[1])

print("")
# raw.save("test.fif",overwrite=True)
#
# edf_path = r'E:\eason\project\deep learning\test\test.edf'
# fif_path = r'E:\eason\project\deep learning\test\test.fif'
# raw2=mne.io.read_raw_fif(fif_path)
# mne.export.export_raw(edf_filename,raw2,fmt="edf",overwrite=True)

# raw_edf=mne.io.Raw(edf_filename,verbose="error")
# raw_edf.save(edf_filename)
# print("1")



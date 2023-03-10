import mne
import numpy as np
import datetime
import os
from mne.io import read_raw_edf

# 定义频率和通道名称
sfreq = 100
ch_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

# 创建信息对象
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

# 创建原始数组

data = np.random.rand(len(ch_names), sfreq * 10) * 100/1000000  # 10秒钟的数据
raw = mne.io.RawArray(data, info)

# 定义EDF文件名
edf_filename = 'test.fif'

# 保存EDF文件
raw.save(edf_filename,overwrite=True)

# 更改文件名
file_path = r'E:\eason\project\deep learning\test\test.fif'

raw=mne.io.read_raw_fif(file_path)
data123=raw.get_data()
edf_file="data.edf"
# raw_edf = mne.io.RawEDF(raw.info, edf_file, verbose='error')
# raw_edf.write('data.edf')
import os
import sys

########################################################################
# set working path
########################################################################
working_dir = os.getcwd()
sys.path.append(working_dir)

from model.model import DeepSleepNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model.loss import *
from model.metric import *
import time
########################################################################
# check file is existing
########################################################################
dataset_file_path = os.path.join(working_dir, "data_edf_20_npz\dataset.npz")
if os.path.isfile(dataset_file_path):
    is_dataset_file_existing = True
else:
    from edf_reader import edf_read

    dataset_file_path = r"E:\xai-sleep\data\sleepedf"
    is_dataset_file_existing = False

# class_dict = { "W", "N1", "N2", "N3", "REM"}
class SleepData(Dataset):
    def __init__(self, data_path, flag):
        if flag:
            dataset = np.load(data_path)
            self.x, self.y = dataset["x"], dataset["y"]
        else:
            self.x, self.y = edf_read(data_path,rewrite=False)

        # self.proprocessed()
        self.x_trans = self.x.reshape(len(self.x), 1, 3000)
        self.x_data = torch.from_numpy(self.x_trans).float()
        # self.y_data = torch.from_numpy(self.y).long()
        self.fft(self.x_data)
    def __getitem__(self, index):
        # pass
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.size(0)

    def proprocessed(self):
        maximums, minimums, avgs = self.x.max(axis=0), self.x.min(axis=0), self.x.sum(axis=0) / \
                                   self.x.shape[0]
        self.x= (self.x - minimums) / (maximums - minimums)

    def fft(self,signal):
        sr=100
        frame_length = int(sr * 0.025)  # 25ms
        hop_length = int(sr * 0.01)  # 10ms
        window = torch.hann_window(frame_length)
        # signal= torch.from_numpy(self.x)
        signal = signal.squeeze()
        stft = torch.stft(signal , frame_length, hop_length, window=window)
        # 计算幅度谱，取对数

        spectrogram = torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2)
        log_spectrogram = torch.log10(spectrogram + 1e-9)  # 加一个小常数避免对数计算中的除零错误

        # 可视化对数谱图
        plt.imshow(log_spectrogram.T, aspect='auto', origin='lower', cmap='jet')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

    def show_raw_data(self):
        n_0 = np.sum(self.y == 0)
        n_1 = np.sum(self.y == 1)
        n_2 = np.sum(self.y == 2)
        n_3 = np.sum(self.y == 3)
        n_4 = np.sum(self.y == 4)
        print(n_0, n_1, n_2, n_3, n_4)

sleep_dataset = SleepData(dataset_file_path, is_dataset_file_existing)
print(sleep_dataset)
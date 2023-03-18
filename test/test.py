import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import mne
import glob,os
from mne.export import export_raw
from mne.io import read_raw_edf

CHANNEL = {'0': 'EEG Fpz-Cz',
           '1': 'EEG Pz-Oz',
           '2': 'EOG horizontal',
           '3': 'Resp oro-nasal',
           '4': 'EMG submental',
           '5': 'Temp rectal',
           '6': 'Event marker'}
ch_names=['EEG Fpz-Cz','EOG horizontal']
ch_types=['eeg','eog']
length=7818000
data_path=r"E:\workstation\reference\xai-sleep\data\sleepedf"
psg_fnames = glob.glob(os.path.join(data_path, "*PSG.edf"))
ann_fnames = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))

get_id=0
def preduce_eeg_eog():
    edf_data = read_raw_edf(psg_fnames[get_id], preload=False, verbose=False).get_data()
    eog0=edf_data[2,:length]
    data_path1=r"E:\workstation\project\code\data\Sleep_100hz_Novel_CNN_eog_denoise.npy"
    data_path2=r"E:\workstation\project\code\data\Sleep_100hz_Simple_CNN_eog_denoise.npy"
    data1=np.load(data_path1)[get_id,:].reshape(1,length)
    data2=np.load(data_path2)
    people1=np.row_stack((data1,eog0))
    info = mne.create_info(ch_names=ch_names, sfreq=100., ch_types=ch_types)
    # # 创建Raw对象
    raw = mne.io.RawArray(data1, info)
    edf_name="people1_eeg.edf"
    # raw.save(edf_name)
    export_raw(edf_name,raw,'edf',overwrite=True)

def preduce_eeg_edf():
    data_path1 = r"E:\workstation\project\code\data\Sleep_100hz_Novel_CNN_eog_denoise.npy"
    data_path2 = r"E:\workstation\project\code\data\Sleep_100hz_Simple_CNN_eog_denoise.npy"
    data1 = np.load(data_path1)[get_id, :].reshape(1, length)
    data2 = np.load(data_path2)
    info = mne.create_info(ch_names=['EEG Fpz-Cz'], sfreq=100., ch_types=['eeg'])
    raw = mne.io.RawArray(data1, info)
    edf_name = "people1_eeg.edf"

    # raw.save(edf_name)
    export_raw(edf_name, raw, 'edf', overwrite=True)

def preduce_eeg_edf_origin():
    # data_path1 = r"E:\workstation\project\code\data\Sleep_100hz_Novel_CNN_eog_denoise.npy"
    # data_path2 = r"E:\workstation\project\code\data\Sleep_100hz_Simple_CNN_eog_denoise.npy"
    # data1 = np.load(data_path1)[get_id, :].reshape(1, length)
    # data2 = np.load(data_path2)
    edf_data = read_raw_edf(psg_fnames[get_id], preload=False, verbose=False).get_data()
    eeg0 = edf_data[0, :length].reshape(1,length)
    info = mne.create_info(ch_names=['EEG Fpz-Cz'], sfreq=100., ch_types=['eeg'])
    raw = mne.io.RawArray(eeg0, info)
    edf_name = "people1_eeg_origin.edf"
    export_raw(edf_name, raw, 'edf', overwrite=True)

preduce_eeg_edf_origin()
edf_name = "people1_eeg_origin.edf"
edf_data = read_raw_edf(psg_fnames[get_id], preload=False, verbose=False).get_data()
pass
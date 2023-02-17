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




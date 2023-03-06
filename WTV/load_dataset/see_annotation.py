# import mne
# import numpy as np
# from psg_ann import Psg_Ann
# from mne.io import read_raw_edf
# import os
# import glob
from edf_reader import *
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
data_path = r"E:\workstation\reference\xai-sleep\data\sleepedf"
x,y=edf_read(data_path,rewrite=True)
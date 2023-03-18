import mne
import numpy as np
import datetime
import os
from mne.io import read_raw_edf

# a=np.arange(30).reshape(3,10)
# a=a[:,None,:]
# b=np.ones(30).reshape(3,10)
# b=b[:,None,:]
# c=np.concatenate((a,b),axis=1)
# print(c.shape)
edf_name="people1_eeg_origin.edf"
raw=read_raw_edf(edf_name)
data=raw.get_data()
pass
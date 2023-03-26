import numpy as np
import pywt
import matplotlib.pyplot as plt
from itertools import chain
#
a=np.arange(20).reshape((4,5))
print(a)
b=a.reshape(-1,1,2)
print(b)
# import torch
# a=torch.ones((128,3000))
# b=a.unsqueeze(1)
# print(b.shape)
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
class WaveletTransNet(nn.Module):
  def __init__(self):
    super(WaveletTransNet, self).__init__()

  def forwward(self):
    print("123")

  def waveletfunction(self,input):

    wavename = 'db5'
    cA, cD = pywt.dwt(input, wavename)
    ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component dipin
    yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component gaopin
    return ya
    # input_len=len(input)

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

class TFLayer(nn.Module):
  """
  Time Frequency conversion
  """

  def __init__(self):
    super(TFLayer, self).__init__()

  def forward(self, x):
    # 傅里叶变换
    outputs = torch.abs(torch.fft.fft(x, dim=-1, norm="forward"))

    outputs = outputs.view(outputs.size()[0], 3000, -1)
    outputs = outputs.unsqueeze(1)

    return outputs
if __name__=="__main__":
  model=TFLayer()
  input=torch.randn((5,3000))
  outp=model(input)

import torch
import torch.nn as nn
import numpy as np
class CapsuleConv(nn.Module):
    def __init__(self,input_size):
        super(CapsuleConv, self).__init__()
        self.conv0=nn.Conv2d(in_channels=1,out_channels=256,kernel_size=(9,9),stride=(1,1))
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        y=self.conv0(x)
        y=self.relu(y)
        return y

class PrimaryCaps(nn.Module):
    def __init__(self):
        super(PrimaryCaps, self).__init__()
        self.conv=nn.Conv2d(256,32,kernel_size=(9,9),stride=(1,1))

    def forward(self):
        # input (128,256,20,20)->(128,32,6,6)



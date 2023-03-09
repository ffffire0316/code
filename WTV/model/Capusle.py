import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs

class CapsuleConv(nn.Module):
    def __init__(self,in_channles,out_channels,kernel_size=9,stride=1):
        super(CapsuleConv, self).__init__()
        self.in_channles=in_channles
        self.out_channles=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.conv0=nn.Conv2d(in_channels=self.in_channles,out_channels=self.out_channles,kernel_size=self.kernel_size,stride=self.stride)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
      # in (batch,1,28,28) -> out (batch,256,20,20)
        y=self.conv0(x)
        y=self.relu(y)
        return y

class PrimaryCaps(nn.Module):
  def __init__(self,in_channels,out_channels,dim_caps=8):
    super(PrimaryCaps, self).__init__()
        # in (batch,256,20,20) -> out (batch,32,6,6)
    self.conv=nn.Conv2d(256,256,kernel_size=9,stride=2)
    self.dim_caps=dim_caps

  def forward(self,x):
        # input 8*(batch,32,6,6)->(batch,8,32,6,6)->(batch,8,1152)
    outputs=self.conv(x)
    outputs=outputs.view(x.size(0), -1, self.dim_caps)
    return squash(outputs)

class DigitsCaps(nn.Module):
  def __init__(
    self,in_num_caps,in_dim_caps,out_num_caps,out_dim_caps,routings=3):
    super(DigitsCaps,self).__init__()
    self.in_num_caps = in_num_caps
    self.in_dim_caps = in_dim_caps
    self.out_num_caps = out_num_caps
    self.out_dim_caps = out_dim_caps
    self.routings = routings
    self.weight = nn.Parameter(
      0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps)
    )

  def forward(self,x):
    x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)
    x_hat_detached = x_hat.detach()
    # print(x_hat_detached.size())
    b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda()
    assert self.routings > 0, "The 'routings' should be > 0."
    for i in range(self.routings):
      c = F.softmax(b, dim=1)
      # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
      if i == self.routings - 1:
        # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
        # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
        # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
        outputs = squash(
          torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True)
        )
        # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
      else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
        outputs = squash(
          torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True)
        )
        # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

        # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
        # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
        # => b.size          =[batch, out_num_caps, in_num_caps]
        b = b + torch.sum(outputs * x_hat_detached, dim=-1)

    return torch.squeeze(outputs, dim=-2)

class CapsuleNet(nn.Module):
  def __init__(self,input_size,classes,routings):
    super(CapsuleNet, self).__init__()
    # input_size (batch,1,28,28)
    self.input_size=input_size
    self.classes=classes
    self.routings=routings
    # Layer 1 Conv Capsule
    self.convcaps=CapsuleConv(input_size,256)

    #Layer 2 Primer Capsule
    self.primaryCaps=PrimaryCaps(256,256,8)
    # Layer 3 DigitCaps
    self.digitcaps=DigitsCaps(
      in_num_caps=6*6*32,
      in_dim_caps=8,
      out_num_caps=classes,
      out_dim_caps=1,
      routings=routings
    )
    self.rule=nn.ReLU()
    self.classify=nn.Sequential(
      nn.Linear(5,5),
      nn.Linear(5,5)
    )

  def forward(self,x):
    print("输入数据形状",x.size())
    x=self.convcaps(x)
    print("输入数据形状", x.size())
    x=self.primaryCaps(x)
    print("输入数据形状", x.size())
    x=self.digitcaps(x)
    print("输入数据形状", x.size())
    length=x.norm(dim=-1)

    return length
if __name__=="__main__":
  # x=torch.tensor()
  x=torch.rand((10,1,28,28)).cuda().float()
  net=CapsuleNet(1,5,3)
  net=net.cuda()
  y=net(x)
  print(y)
  # print(x.size(1))
  # # capusle=CapsuleConv(1).cuda()
  # # y1=capusle(x)
  # # print("y1",y1.size())
  # # primaryCap=PrimaryCaps().cuda()
  # # y2=primaryCap(y1)
  # # print(y2.size())
  # # digcaps=DigitsCaps(1152,8,10,16).cuda()
  # # y3=digcaps(y2)
  # # print("y3",y3.size())




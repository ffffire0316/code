import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # (N,in_channel,x)->(N,out_channels,x_)
        # 卷积核大小为kernel_size*in_channels
        self.conv1=nn.Conv1d(in_channels=1,out_channels=10,kernel_size=1)
        self.flatten=nn.Flatten()
        # 共['Sleep stage 1','Sleep stage 2','Sleep stage 3','Sleep stage 4','Sleep stage ?','Sleep stage R','Sleep stage W']
        self.liner=nn.Linear(10, 5)
        self.liner2 = nn.Linear(5, 5)
        self.sigmod=nn.Sigmoid()
        # self.function = torch.nn.funcitonal.sigmoid
        self.feature1=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=50,stride=6),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8,stride=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),

        )
        self.squential2=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400,stride=50),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=4),
            nn.Dropout(0.5),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=2),
        )



    def forward(self,x):
        # y=self.conv1(x.float())
        # y=self.flatten(y)
        # y=self.liner(y)
        # y = self.liner2(y)
        # y=self.sigmod(y)
        y=self.feature1(x)
        return y


if __name__ == "__main__":
    #  2880为epoch数量 30为epoch长度 1为epoch通道数
    model=Model()
    # print(model)
    # x1=np.ones((2880,30,1))
    # # 优先使用torch.tensor 转换np类型
    # x_3=torch.tensor(x1,dtype=torch.float32)
    # print(x_3)
    # for data in x_3:
    #     print(data.shape)
    #     output=model(data)
    # y = model(x_3)
    # print(y)
    input=np.ones((1,30,1))
    input=torch.tensor(input,dtype=torch.float32)
    print(model(input))
    # writer =SummaryWriter("../log")
    # writer.add_graph(model,x_3)
    # writer.close()




import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # (N,in_channel,x)->(N,out_channels,x_)
        # 卷积核大小为kernel_size*in_channels
        # 共['Sleep stage 1','Sleep stage 2','Sleep stage 3','Sleep stage 4','Sleep stage ?','Sleep stage R','Sleep stage W']
        # self.function = torch.nn.funcitonal.sigmoid
        self.feature1=nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=50,stride=6),
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8,stride=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),


        )
        self.feature2 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400, stride=50),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.reclassify=nn.Sequential(
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(128*61,64),
            nn.Sigmoid(),
            nn.Linear(64, 20),
            nn.Sigmoid(),
            nn.Linear(20,5),
            nn.Sigmoid()

        )



    def forward(self,x):
        x1=self.feature1(x)
        x2=self.feature2(x)
        y=torch.cat((x1,x2),dim=2)
        y=self.reclassify(y)

        return y


if __name__ == "__main__":
    #  2880为epoch数量 30为epoch长度 1为epoch通道数
    model=Model()
    input=np.ones((10,1,3000))
    input=torch.tensor(input,dtype=torch.float32)

    # print(model(input))
    # writer =SummaryWriter("../log")
    # writer.add_graph(model,x_3)
    # writer.close()




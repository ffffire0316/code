import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# # 创建一个SummaryWriter对象，用于记录日志
#
# writer = SummaryWriter("./log")
#
# # 创建一些虚构数据并进行训练
# x = torch.linspace(-5, 5, 100)
# y1 = F.relu(x)
# y2 = F.relu(-x)
# y=torch.stack([y1,y2],dim=1)
# a=y[1]
# # print(y[:][1])
# # print(y.shape)
# for i in range(100):
#     # writer.add_scalar('relu', y1[i], i)
#     # writer.add_scalar('relu', y2[i], i)
#     writer.add_scalar('relu', y[i], i)
#
# # 关闭SummaryWriter对象
# writer.close()

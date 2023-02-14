import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# print(b)
# def cross_entropy(y_hat, y):
#     return - torch.log(y_hat[range(len(y_hat)), y])
#
# loss=cross_entropy(input,output)
# # loss_fn = nn.CrossEntropyLoss
# # loss=loss_fn(input,output)
# print(loss)
# x_input=torch.randn(100,30,1)
# x_input=torch.randn(100,30)
# x_np=np.ones(30).reshape(1,30,1)
# x_input=torch.from_numpy(x_np).to
# print('x_input:\n',x_input).
# x_np=np.arange(30)
# print(x_np)
# x=x_np.reshape((1,30,1))
# print(x)
# net=nn.Conv1d(in_channels=30,out_channels=10,kernel_size=1)
# m = nn.Conv1d(16, 33, 3, stride=2)
# input = torch.randn(20, 16, 50)
# output = m(input)
# print(output)
# y=net(x_input)
# print(y)

# data=[1,1,1,1]
# print(data[0])
# print(data[1:])
# print(data[1::])
# save_dict={
#     "x":data
# }
# print(dataset)
# np.savez(r"./123", **save_dict)
# file=np.load(r"./123.npz")
# print(file["x"])
# x=np.arange(60).reshape(2,30)
x=np.arange(6000).reshape(2,1,3000)
# print(x)
x_tensor=torch.from_numpy(x).to(torch.float32)

feature1 = nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=50, stride=6),
    nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
    nn.Dropout(0.5),
    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1),
    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8),
    nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
    # nn.Flatten(),
    # nn.Linear(128*57,64),
    # nn.Linear(64,5),
    # nn.Sigmoid()

)

feature2= nn.Sequential(
    nn.Conv1d(in_channels=1, out_channels=64, kernel_size=400, stride=50),
    nn.MaxPool1d(kernel_size=4, stride=4),
    nn.Dropout(0.5),
    nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6),
    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
    nn.Conv1d(in_channels=128, out_channels=128, kernel_size=1),
    nn.MaxPool1d(kernel_size=2, stride=2),
)

# (2,1,3000)->(2,128,57)
out1=feature1(x_tensor)
# (2,1,3000)->(2,128,4)
out2=feature2(x_tensor)
# print(out,out.shape)
# input=[out1,out2]
# input2=(out1,out2)
x_concat = torch.cat((out1, out2), dim=2)
x=np.arange(10)
print(x)
x.pop([0,5])
print(x)
for i in range(10):
    print(i)
#
# accuary=(output.argmax(1)==target).sum()
# print(accuary)
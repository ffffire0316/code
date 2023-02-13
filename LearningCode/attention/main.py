import torch
import math
# import softmax
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
Wq=[]
Wk=[]
Wv=[]
input_size=10

A=torch.matmul(Wq,Wk.transpose())
A=A/math.sqrt(input_size)
A_put=torch.softmax(A)
b=torch.matmul(Wk,A_put)

# nn.model
# 1.加载cifar10数据集，返回的是train_loader,test_loader
def get_loader(args):
    # 设置数据加载时的变换形式，包括撞转成tensor,裁剪，归一化
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 默认使用cifar10数据集
    if args.dataset == "cifar10":
        trainset = datasets.CIFAR10(root=r'../data',
                                    train=True,
                                    download=False,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root=r'../data',
                                   train=False,
                                   download=False,
                                   transform=transform_train)
    else:
        trainset = datasets.CIFAR100(root='./data',
                                     train=True,
                                     download=True,
                                     transform=transform_train)
        testset = datasets.CIFAR100(root='./data',
                                    train=False, download=True,
                                    transform=transform_train)

    print("train number:", len(trainset))
    print("test number:", len(testset))

    train_loader = DataLoader(trainset, batch_size=args.train_batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=args.eval_batch_size, shuffle=False)
    print("train_loader:", len(train_loader))
    print("test_loader:", len(test_loader))

    return train_loader, test_loader


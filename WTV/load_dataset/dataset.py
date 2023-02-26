import os
import sys

########################################################################
# set working path
########################################################################
working_dir = os.getcwd()
sys.path.append(working_dir)

from model.model import DeepSleepNet
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model.loss import *
from model.metric import *
########################################################################
# check file is existing
########################################################################
dataset_file_path = os.path.join(working_dir, "load_dataset/.dataset/dataset.npz")
if os.path.isfile(dataset_file_path):
    is_dataset_file_existing = True
else:
    from edf_reader import edf_read

    dataset_file_path = r"E:\xai-sleep\data\sleepedf"
    is_dataset_file_existing = False

# class_dict = { "W", "N1", "N2", "N3", "REM"}
class SleepData(Dataset):
    def __init__(self, data_path, flag):
        if flag:
            dataset = np.load(data_path)
            self.x, self.y = dataset["x"], dataset["y"]
        else:
            psg_fnames = glob.glob(os.path.join(data_path, "*PSG.edf"))
            ann_fnames = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))
            self.x, self.y = edf_read(psg_fnames, ann_fnames)
            n_0 = np.sum(self.y == 0)
            n_1 = np.sum(self.y == 1)
            n_2 = np.sum(self.y == 2)
            n_3 = np.sum(self.y == 3)
            n_4 = np.sum(self.y == 4)
            print(n_0, n_1, n_2, n_3,n_4)

        self.x_trans = self.x.reshape(len(self.x), 1, 3000)
        self.x_data = torch.from_numpy(self.x_trans).float()
        self.y_data = torch.from_numpy(self.y).long()


    def __getitem__(self, index):
        # pass
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.size(0)


if __name__ == "__main__":
    # 读取数据
    sleep_dataset = SleepData(dataset_file_path, is_dataset_file_existing)
    train, test = torch.utils.data.random_split(
        dataset=sleep_dataset, lengths=[0.7, 0.3]
    )
    test_num=len(test)
    BATCH_SIZE = 256
    train_loader = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=train,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    test_loader = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=test,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    # 创建网络模型
    model = DeepSleepNet()
    if torch.cuda.is_available():
        model = model.cuda()
    # 损失函数
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn=FocalLoss()
    loss_fn = loss_fn.cuda()
    # 优化器
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练轮数
    epochs = 100
    total_train_step = 0
    # 添加tensorboard
    writer = SummaryWriter("./log")
    # # 添加 混淆矩阵
    # conf_matrix=torch.zeros(5,5)

    for i in range(epochs):
        print("-------第{}代训练开始--------".format(i + 1))
        train_loss = 0
        train_acc = 0
        # 添加 混淆矩阵
        conf_matrix = torch.zeros(5, 5)
        # # 训练开始 遍历每个人的数据
        model.train()
        for idx, (data, label) in enumerate(train_loader):
            input = data
            target = label
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            loss = loss_fn(output, target)
            # 优化器优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.clone().mean()
            train_acc += (output.argmax(1) == target).sum() / BATCH_SIZE

        train_loss /= idx + 1
        train_acc /= idx + 1
        # print("训练正确率Acc：{}".format(total_test_acc/len(train)))
        writer.add_scalar("train_loss", train_loss, i)
        writer.add_scalar("train_acc", train_acc, i)

        # 测试开始
        test_loss = 0
        test_acc = 0
        model.eval()
        with torch.no_grad():
            for idx, (data, label) in enumerate(test_loader):
                test_input = data
                test_target = label
                if torch.cuda.is_available():
                    test_input = test_input.cuda()
                    test_target = test_target.cuda()
                test_output = model(test_input)
                loss = loss_fn(test_output, test_target)

                test_loss += loss.clone().mean()
                test_acc += (test_output.argmax(1) == test_target).sum() / BATCH_SIZE
                conf_matrix=Confusion_Matrix(preds=test_output,target=test_target,conf_matrix=conf_matrix)
                # conf_matrix=conf_matrix.cpu()

        test_loss /= idx + 1
        test_acc /= idx + 1
        print("test loss:{}".format(test_loss))
        print("test accuracy{}".format(test_acc))
        writer.add_scalar("test_loss", test_loss, i)
        writer.add_scalar("test_acc", test_acc, i)
    # 显示混淆矩阵
    show_conf_mat(conf_matrix, class_dict)
    writer.close()

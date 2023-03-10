import os
import sys

########################################################################
# set working path
########################################################################
working_dir = os.getcwd()
sys.path.append(working_dir)

from model.model import DeepSleepNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from model.loss import *
from model.metric import *
import time
########################################################################
# check file is existing
########################################################################
dataset_file_path = os.path.join(working_dir, "data_edf_20_npz")
a=os.path.isfile(r"E:\workstation\project\code\WTV\load_dataset\data_edf_20_npz\dataset.npz")
if os.path.isdir(dataset_file_path):
    is_dataset_file_existing = True
else:
    from edf_reader import edf_read
    dataset_file_path = r"E:\xai-sleep\data\sleepedf"
    if not os.path.isfile(dataset_file_path):
        dataset_file_path=r"E:\workstation\reference\xai-sleep\data\sleepedf"
    is_dataset_file_existing = False

# class_dict = { "W", "N1", "N2", "N3", "REM"}
class SleepData(Dataset):
    def __init__(self, data_path, flag):
        if flag:
            dataset = np.load(os.path.join(data_path,"dataset.npz"))
            self.x, self.y = dataset["x"], dataset["y"]
        else:
            self.x, self.y = edf_read(data_path,rewrite=False)

        self.proprocessed()
        self.x_trans = self.x.reshape(len(self.x), 1, 3000)
        self.x_data = torch.from_numpy(self.x_trans).float()
        self.y_data = torch.from_numpy(self.y).long()

    def __getitem__(self, index):
        # pass
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.size(0)

    def proprocessed(self):
        maximums, minimums, avgs = self.x.max(axis=0), self.x.min(axis=0), self.x.sum(axis=0) / \
                                   self.x.shape[0]
        self.x= (self.x - minimums) / (maximums - minimums)

    def show_raw_data(self):
        n_0 = np.sum(self.y == 0)
        n_1 = np.sum(self.y == 1)
        n_2 = np.sum(self.y == 2)
        n_3 = np.sum(self.y == 3)
        n_4 = np.sum(self.y == 4)
        print(n_0, n_1, n_2, n_3, n_4)


if __name__ == "__main__":
    # ????????????
    sleep_dataset = SleepData(dataset_file_path, is_dataset_file_existing)
    train, test = torch.utils.data.random_split(
        dataset=sleep_dataset, lengths=[0.7, 0.3]
    )
    test_num=len(test)
    BATCH_SIZE = 128
    train_loader = torch.utils.data.DataLoader(
        # ???????????????????????????batch size?????????
        dataset=train,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # ????????????????????? (???????????????)
        num_workers=0,  # ?????????????????????
    )
    test_loader = torch.utils.data.DataLoader(
        # ???????????????????????????batch size?????????
        dataset=test,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # ????????????????????? (???????????????)
        num_workers=0,  # ?????????????????????
    )
    # ??????????????????
    model = DeepSleepNet()
    if torch.cuda.is_available():
        model = model.cuda()
    # ????????????
    # loss_fn = nn.CrossEntropyLoss()
    alpha = Variable(torch.tensor([0, 1, 0, 0, 0]))
    loss_fn=FocalLoss(alpha=alpha)
    loss_fn = loss_fn.cuda()
    # ?????????
    learning_rate = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # ????????????
    epochs = 20
    total_train_step = 0
    # ??????tensorboard
    writer = SummaryWriter("./log")
    # # ?????? ????????????
    # conf_matrix=torch.zeros(5,5)
    total_time=0
    for i in range(epochs):
        print("-------???{}???????????????--------".format(i + 1))
        train_loss = 0
        train_acc = 0
        # ?????? ????????????
        conf_matrix = torch.zeros(5, 5)
        # # ???????????? ????????????????????????
        train_start_time=time.time()

        model.train()
        for idx, (data, label) in enumerate(train_loader):
            input = data
            target = label
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            output = model(input)
            loss = loss_fn(output, target)
            # ???????????????
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.clone().mean()
            train_acc += (output.argmax(1) == target).sum() / BATCH_SIZE

        train_loss /= idx + 1
        train_acc /= idx + 1
        # print("???????????????Acc???{}".format(total_test_acc/len(train)))
        writer.add_scalar("train_loss", train_loss, i)
        writer.add_scalar("train_acc", train_acc, i)
        train_end_time=time.time()
        train_time=train_end_time-train_start_time
        # ????????????
        test_loss = 0
        test_acc = 0
        test_start_time=time.time()
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

        corrects = conf_matrix.diagonal(offset=0)
        true_kinds = conf_matrix.sum(axis=1)
        precison = corrects / true_kinds
        print("??????????????????????????????????????????{0}".format([i for i in precison]))

        test_loss /= idx + 1
        test_acc /= idx + 1
        print("test loss:{}".format(test_loss))
        print("test accuracy{}".format(test_acc))
        writer.add_scalar("test_loss", test_loss, i)
        writer.add_scalar("test_acc", test_acc, i)
        writer.add_scalar("class 1 acc", precison[0],i)
        writer.add_scalar("class 2 acc", precison[1],i)
        writer.add_scalar("class 3 acc", precison[2],i)
        writer.add_scalar("class 4 acc", precison[3],i)
        writer.add_scalar("class 5 acc", precison[4],i)
        test_end_time=time.time()
        test_time=test_end_time-test_start_time
        print(f"train cost time {train_time:.3f}s,test cost time {test_time:.3f}s")

        total_time=total_time+train_time+test_time
    # ??????????????????
    show_conf_mat(conf_matrix, class_dict)
    print(f"it cost total time {total_time}")
    writer.close()

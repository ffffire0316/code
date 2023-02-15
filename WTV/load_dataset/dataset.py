from edf_reader import edf_read
from model import Model
import glob,os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
class SleepData(Dataset):
    def __init__(self,data_path):
        psg_fnames = glob.glob(os.path.join(data_path, "*PSG.edf"))
        ann_fnames = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))

        self.x,self.y=edf_read(psg_fnames,ann_fnames)
        n_0=np.sum(self.y==0)
        n_1=np.sum(self.y==1)
        n_2=np.sum(self.y==2)
        n_3=np.sum(self.y==3)
        print(n_0,n_1,n_2,n_3)

        self.x_trans=self.x.reshape(len(self.x),1,3000)
        # x1=self.x[0]
        # x2=self.x_trans[0]
        self.x_data= torch.from_numpy(self.x_trans).float()
        self.y_data = torch.from_numpy(self.y).long()
        self.y_data=F.one_hot(self.y_data).float()
        # print(self)

    def __getitem__(self, index):
        # pass
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.size(0)

rewrite=True

if __name__ == "__main__":
    # 读取数据
    data_dir = r"E:\workstation\reference\xai-sleep\data\sleepedf"
    data_dir = r"E:\xai-sleep1\data\sleepedf"

    sleep_dataset=SleepData(data_dir)
    train, test = torch.utils.data.random_split(dataset=sleep_dataset, lengths=[0.7,0.3])
    print(sleep_dataset)

    BATCH_SIZE=128
    train_loader=torch.utils.data.DataLoader(
    # 从数据库中每次抽出batch size个样本
    dataset=train,       # torch TensorDataset format
    batch_size=BATCH_SIZE,       # mini batch size
    shuffle=True,                # 要不要打乱数据 (打乱比较好)
    num_workers=0,               # 多线程来读数据
    )
    test_loader = torch.utils.data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=test,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    # 创建网络模型
    model = Model()
    if torch.cuda.is_available():
        model=model.cuda()
    # 损失函数
    loss_fn = nn.CrossEntropyLoss()
    loss_fn=loss_fn.cuda()
    # loss_fn = nn.MSELoss()
    # 优化器
    learning_rate = 0.0012
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 训练轮数
    epoch = 100
    total_train_step = 0
    total_test_step = 0
    #添加tensorboard
    writer=SummaryWriter("../logtrain")

    for i in range(epoch):
        print("-------第{}论训练开始--------".format(i + 1))
        total_test_acc=0
        # # 训练开始 遍历每个人的数据
        model.train()
        for data,label in train_loader:
            input=data
            target=label
            if torch.cuda.is_available():
                input=input.cuda()
                target=target.cuda()

            output=model(input)
            loss=loss_fn(output,target)
            test_accuracy = (output.argmax(1) == target.argmax(1)).sum()
            total_test_acc=total_test_acc+test_accuracy
            # 优化器优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step=total_train_step+1
            if total_train_step%100==0:
                print("训练次数：{}，Loss：{}".format(total_train_step,loss.item()))
                print("训练正确率Acc：{}".format(total_test_acc/len(train)))
                writer.add_scalar("train_loss",loss.item(),total_train_step)
                writer.add_scalar("train_acc", total_test_acc/len(train), total_train_step)

        # 测试开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data,label in test_loader:
                test_input=data
                test_target=label
                if torch.cuda.is_available():
                    test_input=test_input.cuda()
                    test_target=test_target.cuda()

                test_output=model(test_input)
                test_loss = loss_fn(test_output, test_target)
                total_test_loss=total_test_loss+test_loss.item()

                test_accuracy=(test_output.argmax(1)==test_target.argmax(1)).sum()
                total_accuracy=total_accuracy+test_accuracy
            print("total test loss:{}".format(total_test_loss))
            print("total test accuracy{}".format(total_accuracy/len(test)))
            writer.add_scalar("test_loss",total_test_loss,total_test_step)
            writer.add_scalar("test_acc",total_accuracy/len(test),total_test_step)
            total_test_step+=1
    writer.close()
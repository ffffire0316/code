from edf_reader import edf_read
from model import Model
import glob,os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
class SleepData(Dataset):
    def __init__(self,data_path):
        psg_fnames = glob.glob(os.path.join(data_path, "*PSG.edf"))
        ann_fnames = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))

        self.x,self.y=edf_read(psg_fnames,ann_fnames)
        self.x_trans=self.x.reshape(len(self.x),1,30)
        self.x_data= torch.from_numpy(self.x_trans)
        # self.x_data = torch.from_numpy(self.x)
        self.y_data = torch.from_numpy(self.y).long()
        self.y_data=F.one_hot(self.y_data)

    def __getitem__(self, index):
        # pass
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.x_data.size(0)

if __name__ == "__main__":
    # 读取数据
    data_dir = r"E:\workstation\reference\xai-sleep\data\sleepedf"
    sleep_dataset=SleepData(data_dir)
    train, test = torch.utils.data.random_split(dataset=sleep_dataset, lengths=[0.7,0.3])
    print(sleep_dataset)

    BATCH_SIZE=3
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
    # for x,y in test_loader:
    #     print(y)
    # 创建网络模型
    model = Model()
    # 损失函数
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()
    # 优化器
    learning_rate = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # 训练轮数
    epoch = 20
    total_train_step = 0
    total_test_step = 0
    #添加tensorboard
    writer=SummaryWriter("../logtrain")

    for i in range(epoch):
        print("-------第{}论训练开始--------".format(i + 1))

        # # 训练开始 遍历每个人的数据
        model.train()
        for data,label in train_loader:
            input=data.float()
            # target=F.one_hot(label)
            target=label.to(torch.float32)
            output=model(input).to(torch.float32)
            loss=loss_fn(output,target)

            # 优化器优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step=total_train_step+1
            if total_train_step%100==0:
                print("训练次数：{}，Loss：{}".format(total_train_step,loss.item()))
                writer.add_scalar("train_loss",loss.item(),total_train_step)

        # 测试开始
        model.eval()
        total_test_loss = 0
        total_accuracy = 0
        with torch.no_grad():
            for data,label in test_loader:
                input=data.float()
                target=label.to(torch.float32)
                output=model(input).to(torch.float32)
                sigel_acc=output.argmax(1)
                target_acc=target.argmax(1)
                # xxx_acc=output.argmax(0)
                loss = loss_fn(output, target)
                total_test_loss=total_test_loss+loss.item()
                accuracy=(output.argmax(1)==target.argmax(1)).sum()
                total_accuracy=total_accuracy+accuracy
            print("total test loss:{}".format(total_test_loss))
            print("total test accuracy{}".format(total_accuracy/len(test)))
            writer.add_scalar("test_loss",total_test_loss,total_test_step)
            writer.add_scalar("test_acc",total_accuracy/len(test),total_test_step)
            total_test_step+=1
    writer.close()
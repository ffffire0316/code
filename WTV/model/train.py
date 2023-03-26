import os
import sys

########################################################################
# set working path
########################################################################
working_dir = os.getcwd()
sys.path.append(working_dir)

from model import DeepSleepNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from loss import *
from metric import *
import glob,mne
import time
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}
class Ann:
    def __init__(self,annotation, interval):

      self.annotation = annotation
      self.interval = interval
      assert len(annotation) == len(interval)
      index_start = int(interval[0][0])
      index_end = int(interval[0][1])
      index_len = int((index_end - index_start) / 30)
      self.label = [self.annotation[0]] * index_len

      for i in range(1, len(annotation)):
        # 对标签进行扩容以匹配data
        # 每个duration的开始结束点
        index_start = int(interval[i][0])
        index_end = int(interval[i][1])
        index_len = int((index_end - index_start) / 30)
        # 清洗数据 除去annation为5的片段
        self.label += [self.annotation[i]] * index_len

      self.label = np.array(self.label)
      self.label=self.label[:2606]
      pass


def anno_read(data_path, rewrite=False):

  ann_fnames = glob.glob(os.path.join(data_path, "*Hypnogram.edf"))
  print("读取edf文件,共{}".format(len(ann_fnames)))
  anno_list=[]
  for i in range(20):

    mne_anno: mne.Annotations = mne.read_annotations(ann_fnames[i])
    labels = list(sorted(set(mne_anno.description)))
    interval, annotation = [], []
    label2int = {lb: i for i, lb in enumerate(labels)}
    print(mne_anno)
    for onset, duration, label in zip(mne_anno.onset, mne_anno.duration, mne_anno.description):
      interval.append((onset, onset + duration))
      annotation.append(ann2label[label])
    single_ann = Ann(annotation, interval)
    anno_list.append(single_ann.label)
    # Saving as numpy files
    filename = os.path.basename(ann_fnames[i]).replace(".edf", "(raw).npz")
    save_dict = {
      "y": single_ann.label,
    }

    np.savez(os.path.join(data_path, filename), **save_dict)
    print(" ---------- have saved the {} file in {} ---------".format(filename, data_path))
  anno_dataset=anno_list[0]
  for i in range(1,20):
    anno_dataset=np.hstack((anno_dataset,anno_list[i]))
  pass
  np.save(os.path.join(data_path, "anno_dataset.npy"), anno_dataset)



class SleepData(Dataset):
  def __init__(self, data_path, flag):
    data = np.load(os.path.join(data_path,"Sleep_100hz_Novel_CNN_eog_denoise.npy"))
    # data=self.proprocessed(data)
    label=np.load(os.path.join(data_path,"anno_dataset.npy"))
    data=data.reshape(-1,1,3000)
    y=data[1,:,:].reshape(-1)
    import matplotlib.pyplot as plt
    x=np.arange(3000)
    plt.figure()
    plt.plot(x,y)
    plt.show()
    pass

    self.data,self.label=[],[]
    for i in range(len(label)):
      # a=data[i,:,:]
      # b=label[i]
      # if label[i]==
      if label[i]!=5:
        self.data.append(data[i,:,:])
        self.label.append(label[i])
    self.show_raw_data()
    self.data=torch.from_numpy(np.array(self.data)).float()
    self.label=torch.from_numpy(np.array(self.label)).long()

    pass

  def __getitem__(self, index):
    # pass
    return self.data[index], self.label[index]

  def __len__(self):
    return self.data.shape[0]
    # pass
  def proprocessed(self,x):
    maximums, minimums, avgs = x.max(axis=0),x.min(axis=0),x.sum(axis=0) / \
                               x.shape[0]
    x = (x - minimums) / (maximums - minimums)

  def show_raw_data(self):
      n_0 = np.sum(self.label == 0)
      n_1 = np.sum(self.label == 1)
      n_2 = np.sum(self.label == 2)
      n_3 = np.sum(self.label == 3)
      n_4 = np.sum(self.label == 4)
      print(n_0, n_1, n_2, n_3, n_4)

if __name__ == "__main__":
    dataset_file_path=r"E:\Data\peiyan"
    # anno_path=r"E:\Data\sleepedf"
    # 读取数据
    sleep_dataset = SleepData(dataset_file_path,1)
    # sleep_dataset = SleepData(dataset_file_path, False)
    train, test = torch.utils.data.random_split(
        dataset=sleep_dataset, lengths=[0.7, 0.3]
    )
    test_num=len(test)
    BATCH_SIZE = 128
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
    model = DeepSleepNet()
    if torch.cuda.is_available():
      model = model.cuda()
    # 损失函数
    # loss_fn = nn.CrossEntropyLoss()
    alpha = Variable(torch.tensor([0.1, 1, 1, 1, 1]))
    loss_fn = FocalLoss(alpha=alpha)
    loss_fn = loss_fn.cuda()
    # 优化器
    learning_rate = 1e-5
    # learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # 训练轮数
    epochs = 200
    total_train_step = 0
    # 添加tensorboard
    writer = SummaryWriter("./log")
    # # 添加 混淆矩阵
    # conf_matrix=torch.zeros(5,5)
    total_time = 0
    for i in range(epochs):
      print("-------第{}代训练开始--------".format(i + 1))
      train_loss = 0
      train_acc = 0
      # 添加 混淆矩阵
      conf_matrix = torch.zeros(5, 5)
      # # 训练开始 遍历每个人的数据
      train_start_time = time.time()

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
      train_end_time = time.time()
      train_time = train_end_time - train_start_time
      # 测试开始
      test_loss = 0
      test_acc = 0
      test_start_time = time.time()
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
          conf_matrix = Confusion_Matrix(preds=test_output, target=test_target, conf_matrix=conf_matrix)

      corrects = conf_matrix.diagonal(offset=0)
      true_kinds = conf_matrix.sum(axis=1)
      precison = corrects / true_kinds
      print("每种睡眠阶段的识别准确率为：{0}".format([i for i in precison]))

      test_loss /= idx + 1
      test_acc /= idx + 1
      print("test loss:{}".format(test_loss))
      print("test accuracy{}".format(test_acc))
      writer.add_scalar("test_loss", test_loss, i)
      writer.add_scalar("test_acc", test_acc, i)
      writer.add_scalar("class 1 acc", precison[0], i)
      writer.add_scalar("class 2 acc", precison[1], i)
      writer.add_scalar("class 3 acc", precison[2], i)
      writer.add_scalar("class 4 acc", precison[3], i)
      writer.add_scalar("class 5 acc", precison[4], i)
      test_end_time = time.time()
      test_time = test_end_time - test_start_time
      print(f"train cost time {train_time:.3f}s,test cost time {test_time:.3f}s")

      total_time = total_time + train_time + test_time
    # 显示混淆矩阵
    # show_conf_mat(conf_matrix, class_dict)
    print(f"it cost total time {total_time}")
    writer.close()
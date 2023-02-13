# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
from torch.utils.data import Dataset, DataLoader
import tushare as ts
import pandas as pd
import os
import time
import glob
import matplotlib.pyplot as plt
import numpy as np

pro = ts.pro_api('b5bd31ae5ace0249ae1601e33c17f8a7d13ae7109f2b65295b111368')


# ----------------------下载某只股票数据------------------- #
# code:股票编码 日期格式：2019-05-21 filename：写到要存放数据的根目录即可如D:\data\
# length是筛选股票长度，默认值为False，既不做筛选，可人为指定长度，如200，既少于200天的股票不保存
def get_stock_data(code, date1, date2, filename, length=-1):
    df = pro.daily(ts_code=code, start_date=date1, end_date=date2)
    df1 = pd.DataFrame(df)
    df1 = df1[['trade_date', 'open', 'high', 'close', 'low', 'vol', 'pct_chg']]
    df1 = df1.sort_values(by='trade_date')
    print('共有%s天数据' % len(df1))
    if len(df1) > length:
        path = code + '.csv'
        df1.to_csv(os.path.join(filename, path))


# ------------------------更新股票数据------------------------ #
# 将股票数据从本地文件的最后日期更新至当日
# filename:具体到文件名如d:\data\000001.csv
def update_stock_data(filename):
    (filepath, tempfilename) = os.path.split(filename)
    (stock_code, extension) = os.path.splitext(tempfilename)
    f = open(filename, 'r')
    df = pd.read_csv(f)
    print('股票{}文件中的最新日期为:{}'.format(stock_code, df.iloc[-1, 1]))
    data_now = time.strftime('%Y%m%d', time.localtime(time.time()))
    print('更新日期至：%s' % data_now)
    nf = pro.daily(ts_code=stock_code, start_date=str(df.iloc[-1, 1]), end_date=data_now)
    nf = nf.sort_values(by='trade_date')
    nf = nf.iloc[1:]
    print('共有%s天数据' % len(nf))
    nf = pd.DataFrame(nf)
    nf = nf[['trade_date', 'open', 'high', 'close', 'low', 'vol', 'pct_chg']]
    nf.to_csv(filename, mode='a', header=False)
    f.close()


# ------------------------获取股票长度----------------------- #
# 辅助函数
def get_data_len(file_path):
    with open(file_path) as f:
        df = pd.read_csv(f)
        return len(df)


# --------------------------文件合并------------------------- #
# 将多个文件合并为一个文件，在文件末尾添加
# filename是需要合并的文件夹，tfile是存放合并后文件的文件夹
def merge_stock_data(filename, tfile):
    csv_list = glob.glob(filename + '*.csv')
    print(u'共发现%s个CSV文件' % len(csv_list))
    f = open(csv_list[0])
    df = pd.read_csv(f)
    for i in range(1, len(csv_list)):
        f1 = open(csv_list[i], 'rb')
        df1 = pd.read_csv(f1)
        df = pd.concat([df, df1])
    df.to_csv(tfile + 'train_mix.csv', index=None)



for i in range(1, 2):
    name = '{:0>6d}'.format(i) + '.SZ'
    get_stock_data(name, '20200101', '20220913',
                   'C:/Users/Dark Screen Shield/Desktop/deep learning/SharesProject/SharesData/', 200)


name = '000001.SZ'
PATH = 'C:/Users/Dark Screen Shield/Desktop/deep learning/SharesProject/SharesData/' + name + '.csv'
STEP = 60


# merge_stock_data('C:/Users/Dark Screen Shield/Desktop/deep learning/SharesProject/SharesData', PATH)


class StockData(Dataset):
    def __init__(self, path: str, step: int = 30):
        self.path = path
        self.step = step
        data = pd.read_csv(path).values[1:, 2:]  #取第二行 第三列之后的数据（包括）
        print(len(data))

        self.len = len(data)

        self.y_max = data[:, 3].max()
        self.y_min = data[:, 3].min()

        # visualise the data
        data = self.normalise(data) + 1e-5

        # print(data[:-1,:])
        self.X = torch.tensor(data[:-1, :].astype(np.float32))      #六个参数作为输入
        self.y = torch.tensor(data[1:, 3].astype(np.float32))     #收盘价

        plt.plot(self.y)
        plt.show()

    def __getitem__(self, index):
        return self.X[index:index + self.step], self.y[index + self.step]

    def __len__(self):
        return self.len - 1 - self.step

    def normalise(self, data): #归一化函数
        data = data.T
        for i in range(len(data)):
            data_min = data[i].min()
            data_max = data[i].max()
            data[i] = (data[i] - data_min) / (data_max - data_min)
        return data.T


stock_data = StockData(path=PATH, step=STEP)
data = DataLoader(dataset=stock_data, batch_size=5, shuffle=False)
print(len(data))

# now let us write a LSTM model
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_size=6, hidden_size=20, output_size=1, layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.function = torch.sigmoid

    def forward(self, X):
        X, hidden = self.lstm(X, None)
        X = X[:, -1, :]
        X = self.linear(X)
        X = self.function(X)
        return X


net = Net()
# now Train the model
import torch.optim as op

# LR = 0.01
criteria = nn.MSELoss()
optimiser = op.Adam(net.parameters())
EPCHO = 10

for epcho in range(EPCHO):
    for i, Xy in enumerate(data):

        if i == len(data) * 0.8: break

        X = Xy[0]
        y = Xy[1]

        predict = net(X)
       # predict=reshape(X.shape)
        optimiser.zero_grad()

        loss = criteria(predict, y)
        loss.backward()
        optimiser.step()

    print('Epcho: {}.......... loss is {}'.format(epcho, loss))

torch.save(net, 'C:/Users/Dark Screen Shield/Desktop/deep learning/SharesProject/model/' + name + '.pkl')
net = torch.load('C:/Users/Dark Screen Shield/Desktop/deep learning/SharesProject/model/' + name + '.pkl')
predict = np.array([])
actual = np.array([])

torch.no_grad()

for X, y in data:
    predict = np.append(predict, net(X).data[0, 0])
    actual = np.append(actual, y.data[0])

plt.plot(predict, label='prediction')
plt.plot(actual, label='actual')
plt.vlines(len(data) * 0.8, 0, 1, color="red")  # 竖线
plt.title('stock_predict step = ' + str(STEP))
save_name = 'C:/Users/Dark Screen Shield/Desktop/deep learning/SharesProject/picture/' + name + '_prediction.png'
plt.legend()
plt.savefig(save_name, dpi=600)
plt.rcParams['figure.figsize'] = 20, 10
plt.show()

result_pre=np.array(predict).reshape(-1,1)
result_act=np.array(actual).reshape(-1,1)
result=np.concatenate((predict,actual),axis=1)
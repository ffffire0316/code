import torch
import tensorboard
import time
#
# print(torch.__version__)  # 查看torch当前版本号
#
# print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
#
# print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
npz_path=r"E:\eason\project\deep learning\WTV\utils"
import openai

def get_npz(npz_path):
    file_name_list = []
    file_list = os.listdir(npz_path)
    for file_name in file_list:
        if file_name == "dataset.npz":
            dataset = np.load(os.path.join(npz_path, file_name))
            dataset_x = dataset["x"]
            dataset_y = dataset["y"]
            return dataset_x, dataset_y

        if file_name[-4:] == ".npz" and file_name != "dataset.npz":
            file_name_list.append(os.path.join(npz_path, file_name))
            # 融合数组
    dataset = np.load(file_name_list[0])
    dataset_x = dataset["x"]
    dataset_y = dataset["y"]
    for file_name in file_name_list[1::]:
        data = np.load(file_name)
        data_x = data["x"]
        data_y = data["y"]
        dataset_x = np.r_[dataset_x, data_x]
        dataset_y = np.r_[dataset_y, data_y]
    np.savez(os.path.join(npz_path, 'dataset.npz'), x=dataset_x, y=dataset_y)
    return dataset_x, dataset_y
data_x,data_y=get_npz(npz_path)
data_test=data_x[0]


aa = []
for i in range(200):
    aa.append(np.sin(0.3*np.pi*i))
for i in range(200):
    aa.append(np.sin(0.13*np.pi*i))
for i in range(200):
    aa.append(np.sin(0.05*np.pi*i))
y = aa
y=data_test
wavename = 'db5'
cA, cD = pywt.dwt(y, wavename)

ya = pywt.idwt(cA, None, wavename, 'smooth')  # approximated component
yd = pywt.idwt(None, cD, wavename, 'smooth')  # detailed component

cA2, cD2 = pywt.dwt(yd, wavename)
ya2 = pywt.idwt(cA2, None, wavename, 'smooth')  # approximated component
yd2 = pywt.idwt(None, cD2, wavename, 'smooth')  # detailed component

x = range(len(y))
plt.figure(figsize=(12, 9))
plt.subplot(411)
plt.plot(x, y)
plt.title('original signal')

plt.subplot(412)
plt.plot(x, ya)
plt.title('approximated component')

plt.subplot(413)
plt.plot(x, yd)
plt.title('detailed component')

plt.subplot(414)
plt.plot(x, y)
plt.title('detailed component 2 ')

plt.tight_layout()

plt.show()
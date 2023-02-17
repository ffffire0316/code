import numpy as np
import os
label2int={'Sleep stage 1': 0, 'Sleep stage 2': 1, 'Sleep stage 3': 2, 'Sleep stage 4': 3, 'Sleep stage ?': 4, 'Sleep stage R': 5, 'Sleep stage W': 6}
# merge 2 3 delete 4 6
annotation=[1,0,1,2,5]
# annotation1={1,0,1,2,5}
print(annotation)
duration=[(1,2),(2,3),(21,3),(22,3),(24,3)]
for i in range(len(annotation)):
    if annotation[i]==5:
        print(i)
        annotation.pop(i)
        duration.pop(i)

print(annotation,duration)

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


# ar1 = np.random.rand(2,3)
# ar2 = np.arange(4)
# np.savez(r"E:\workstation\reference\xai-sleep\data\edf20npz",ar1,ar2)
# datas = np.load(r"E:\workstation\reference\xai-sleep\data\edf20npz.npz")
# print(os.absabspa)
# for key, arr in datas.items():
#   print(key, ": ", arr)
# print(datas)
# path = os.path.abspath(__file__)
# print(path)
# a,b=os.path.split(path)
# print(b)
# file_list = os.listdir("./")
# print(file_list)
# for file_name in file_list:
#     print(file_name[-3:])
# path = os.path.dirname(path)
# print(path)
# x=np.ones((2880,30))
# y=np.zeros((1000,30))
# z=[]
# print(x)
# print(x.shape)
# print("________________________")
# # z=np.append(x,y)
# # print(z)
# # print(z.shape)
# print("________________________")
# x=np.concatenate((x,y))
# print(x,x.shape)
# np.c

import numpy as np
import torch

output=torch.tensor([[1,0,0,0,0],[0,1,0,0,0]])
target=torch.tensor([[0,0,0,0,1],[1,0,0,0,0]])
a=np.array([1,2,3])
b=1
print(b-a)
ann2label = {
    "Sleep stage W": 0,
   "Sleep stage 1":  1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
    "Sleep stage ?": 5,
    "Movement time": 5
}

# conf_matrix=torch.zeros(5,5)
# print(conf_matrix)
# def Confusion_Matrix(preds,target,conf_matrix):
#   preds=preds.argmax(1)
#   for i,j in zip(target,preds):
#     conf_matrix[i,j]+=1
#
#   return conf_matrix
# target=[1,2]
# conf_matrix=Confusion_Matrix(output,target,conf_matrix)
# print(conf_matrix)

  # noo=(output=="0").sum()
  # if output
# d=torch.max(output,dim=1)

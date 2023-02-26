import numpy as np
import torch
import matplotlib.pylab as plt
import itertools
class_dict = { "W", "N1", "N2", "N3", "REM"}

# 混淆矩阵
def Confusion_Matrix(preds,target,conf_matrix):
  preds=preds.argmax(1)
  for i,j in zip(target,preds):
    conf_matrix[i,j]+=1
  return conf_matrix

def show_conf_mat(cm,classes,normalize=False,title="Confusion matrix",cmap=plt.cm.Blues):
  '''
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  Input
  - cm : 计算出的混淆矩阵的值
  - classes : 混淆矩阵中每一行每一列对应的列
  - normalize : True:显示百分比, False:显示个数
  '''
  # import matplotlib as plt
  try:
    conf_matrix = np.array(cm.cpu())
  except:
    pass
  corrects = conf_matrix.diagonal(offset=0)
  # 各列相加得到每类样本的数量
  true_kinds = conf_matrix.sum(axis=1)
  # 各行相加得到预测结果中每类数量
  pred_kinds = conf_matrix.sum(axis=0)
  # 总数
  sum_num=np.sum(conf_matrix)
  # 真阴性数量
  TN = sum_num-true_kinds-pred_kinds+corrects
  FP=pred_kinds-corrects


  # 准确率和召回率
  try:
    precison=corrects/true_kinds
    recall=corrects/pred_kinds
    specificity=TN/(TN+FP)
    F1= (precison+recall) / 2
    MF1=(2*precison*recall/(precison+recall)).mean()
    MGm=(np.power((specificity*recall),0.5)).mean()
  except:
    precison=0
    recall=0

  # test_num=int(np.sum(conf_matrix))
  # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_num))
  print(conf_matrix)
  print("每种睡眠阶段总个数：", true_kinds)
  print("每种睡眠阶段预测正确的个数：", corrects)
  print("每种睡眠阶段的识别准确率为：{0}".format([rate * 100 for rate in corrects / true_kinds]))
  print("每一类的F1得分：{0}".format(F1))
  print("每一类的MF1得分：{0}".format(MF1))
  print("每一类的MGm得分：{0}".format(MGm))
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')
  print(cm)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=90)
  plt.yticks(tick_marks, classes)
# 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
  # x,y轴长度一致(问题1解决办法）
  plt.axis("equal")
  # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
  ax = plt.gca()  # 获得当前axis
  left, right = plt.xlim()  # 获得x轴最大最小值
  ax.spines['left'].set_position(('data', left))
  ax.spines['right'].set_position(('data', right))
  for edge_i in ['top', 'bottom', 'right', 'left']:
      ax.spines[edge_i].set_edgecolor("white")
  # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
      plt.text(i, j, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()


def accuracy(output,target):
  with torch.no_grad():
    preds=output.argmax(1)
    assert len(preds)==len(target)
    accu=(preds ==target).sum()
    return accu

def F1(output,target):
  with torch.no_grad():
    preds = output.argmax(1)


if __name__ == "__main__":
  conf_matrix=torch.zeros(5,5)
  output = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0],[0, 0, 0, 1, 0]])
  target = [1, 2,1]
  conf_matrix = Confusion_Matrix(output, target, conf_matrix)
  conf_matrix=conf_matrix.cpu().numpy()
  show_conf_mat(conf_matrix,class_dict)
import torch
import tensorboard
import time
#
# print(torch.__version__)  # 查看torch当前版本号
#
# print(torch.version.cuda)  # 编译当前版本的torch使用的cuda版本号
#
# print(torch.cuda.is_available())  # 查看当前cuda是否可用于当前版本的Torch，如果输出True

start =time.time()
for i in range(1):
  time.sleep(10)
end = time.time()

print(end-start)

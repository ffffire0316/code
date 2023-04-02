import numpy as np

# arr = np.array([2, 3, 4, 3, 5])
# target = 3
#
# indices = np.where(arr == target)[0]
# arr = np.delete(arr, indices)
#

# print(arr)
arr=np.arange(300).reshape((10,30,1))
tagert=np.ones((10,5))
# print(len(arr))
for i in range(1,len(arr)-1):
  a=arr[i-1:i+2,:,:]
  b=a.reshape(-1,arr.shape[2])
  # arr.shape[2]
  print(i)
tagert=tagert[1:-1]
pass
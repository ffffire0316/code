import numpy as np

arr = np.array([2, 3, 4, 3, 5])
target = 3

indices = np.where(arr == target)[0]
arr = np.delete(arr, indices)

print(arr)
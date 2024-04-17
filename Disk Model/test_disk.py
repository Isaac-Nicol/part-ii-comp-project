import numpy as np


element = np.array([1, 2])
array = np.array([[1, 2], [3, 4], [5, 6]])
print(array)
array = np.append(array, [element], axis=0)
print(array)
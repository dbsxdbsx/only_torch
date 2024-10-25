# import numpy as np
# arr = np.array([[1, 2, 3],
#                 [4, 5, 6],
#                ])

# print(np.diag(arr))     # 输出: [1, 5, 9]
# print(np.diag(arr, 1))  # 输出: [2, 6]
# print(np.diag(arr, -1)) # 输出: [4, 8]


import numpy as np
arr = np.array([[1, 2, 3]])
print(np.diag(arr))
# 输出:
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
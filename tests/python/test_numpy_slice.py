import numpy as np

# 创建一个4维张量 [2, 3, 1, 4] 形状
data = np.array(
    [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
    ]
).reshape(2, 3, 1, 4)

# 测试混合切片: [1, 0:2, :, 1:3]
result = data[:, 0:2, :, 1:3]
print(f"切片结果(形状：{result.shape}):\n", result)
# 预期输出类似:
#  [[[ 2.  3.]
#   [ 6.  7.]]

#  [[14. 15.]
#   [18. 19.]]]

# 测试完整切片
full_slice = data[:, :, :, :]
print("\n完整切片形状:", full_slice.shape)  # 预期输出: (2, 3, 1, 4)
print("完整切片与原数组是否相等:", np.array_equal(full_slice, data))  # 预期输出: True

# 测试边界情况 - 空范围
empty_slice = data[0:1, 0:0, :, 0:1]
print("\n空范围切片形状:", empty_slice.shape)  # 预期输出: (1, 0, 1, 1)

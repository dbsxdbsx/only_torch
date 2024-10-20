import torch


def mat_mul(a, b):
    return torch.matmul(a, b)

# 定义输入矩阵，a为2x3矩阵，b为3x2矩阵
a = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]], requires_grad=True)
b = torch.tensor([[7.0, 8.0],
                  [9.0, 10.0],
                  [11.0, 12.0]], requires_grad=True)

print("a的维度:", a.shape)
print("b的维度:", b.shape)

# 计算结果
c = mat_mul(a, b)
print("c的值:")
print(c)
print("c的维度:", c.shape)

# 计算雅可比矩阵
jacobian_a = torch.func.jacrev(mat_mul, argnums=0)(a, b)
jacobian_b = torch.func.jacrev(mat_mul, argnums=1)(a, b)

# 重塑雅可比矩阵
jacobian_a_reshaped = jacobian_a.reshape(c.numel(), a.numel())
jacobian_b_reshaped = jacobian_b.reshape(c.numel(), b.numel())

print("\n对a的雅可比矩阵（重塑后）:")
print(jacobian_a_reshaped)
print("形状（重塑后）:", jacobian_a_reshaped.shape)

print("\n对b的雅可比矩阵（重塑后）:")
print(jacobian_b_reshaped)
print("形状（重塑后）:", jacobian_b_reshaped.shape)

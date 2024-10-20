import torch


def add_node(a, b):
    return a + b

# 定义输入矩阵
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

print("a的维度:", a.shape)
print("b的维度:", b.shape)

# 计算结果
c = add_node(a, b)
print("c的值:", c)

# 计算雅可比矩阵
jacobian_a = torch.func.jacrev(add_node, argnums=0)(a, b)
jacobian_b = torch.func.jacrev(add_node, argnums=1)(a, b)

# 重塑雅可比矩阵为更直观的形式
jacobian_a_reshaped = jacobian_a.reshape(4, 4)
jacobian_b_reshaped = jacobian_b.reshape(4, 4)

print("\n对a的雅可比矩阵（重塑后）:")
print(jacobian_a_reshaped)

print("\n对b的雅可比矩阵（重塑后）:")
print(jacobian_b_reshaped)
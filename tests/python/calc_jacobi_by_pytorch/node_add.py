import torch


def add_node(a, b):
    return a + b


# 定义输入矩阵
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# 计算结果
y = add_node(a, b)

# 计算雅可比矩阵
jacobian_a = torch.func.jacrev(add_node, argnums=0)(a, b)
jacobian_b = torch.func.jacrev(add_node, argnums=1)(a, b)

# 重塑雅可比矩阵为更直观的形式
jacobian_a_reshaped = jacobian_a.reshape(4, 4)
jacobian_b_reshaped = jacobian_b.reshape(4, 4)

# 验证结果
print("\n验证结果:")
print(f"输入a【{a.shape}】:\n{a}")
print(f"输入b【{b.shape}】:\n{b}")
print(f"输出y【{y.shape}】:\n{y}")
print(
    f"\n对a的雅可比矩阵（重塑后）【{jacobian_a_reshaped.shape}】:\n{jacobian_a_reshaped}"
)
print(
    f"\n对b的雅可比矩阵（重塑后）【{jacobian_b_reshaped.shape}】:\n{jacobian_b_reshaped}"
)

import torch


def mat_mul(a, b):
    return torch.matmul(a, b)


# 定义输入矩阵，a为2x3矩阵，b为3x4矩阵
a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
b = torch.tensor(
    [[7.0, 8.0, 9.0, 10.0], [11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]],
    requires_grad=True,
)

# 计算结果
y = mat_mul(a, b)

# 计算雅可比矩阵
jacobian_a = torch.func.jacrev(mat_mul, argnums=0)(a, b)
jacobian_b = torch.func.jacrev(mat_mul, argnums=1)(a, b)

# 重塑雅可比矩阵
jacobian_a_reshaped = jacobian_a.reshape(y.numel(), a.numel())
jacobian_b_reshaped = jacobian_b.reshape(y.numel(), b.numel())

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

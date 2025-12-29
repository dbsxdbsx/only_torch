import torch


def step(x):
    # 当x >= 0时输出1，当x < 0时输出0
    return torch.where(x >= 0, torch.ones_like(x), torch.zeros_like(x))


# 定义输入矩阵
x = torch.tensor([[0.5, -1.0], [0.0, 2.0]], requires_grad=True)
y = step(x)

# 计算雅可比矩阵
jacobian = torch.func.jacrev(step)(x)

# 重塑雅可比矩阵为更直观的形式
jacobian_reshaped = jacobian.reshape(4, 4)

# 验证结果的正确性
print("\n验证结果:")
print(f"输入x【{x.shape}】:\n{x}")
print(f"输出y【{y.shape}】:\n{y}")
print(f"\n雅可比矩阵（重塑后）【{jacobian_reshaped.shape}】:\n{jacobian_reshaped}")

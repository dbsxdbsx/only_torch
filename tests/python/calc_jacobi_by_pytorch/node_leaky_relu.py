#!/usr/bin/env python3
"""
用 PyTorch 生成 LeakyReLU 节点的测试预期值
包括标准 ReLU (negative_slope=0) 和 Leaky ReLU (negative_slope=0.1)
"""

import torch
import torch.nn.functional as F

# 设置打印精度
torch.set_printoptions(precision=16)


def test_leaky_relu(x_data, negative_slope, name):
    """测试 LeakyReLU 的 forward 和 backward"""
    print(f"\n{'=' * 60}")
    print(f"测试: {name}")
    print(f"negative_slope = {negative_slope}")
    print(f"{'=' * 60}")

    # 创建需要梯度的张量
    x = torch.tensor(x_data, dtype=torch.float64, requires_grad=True)
    print(f"\n输入 x:\n{x}")

    # Forward
    y = F.leaky_relu(x, negative_slope=negative_slope)
    print(f"\nLeakyReLU(x) 输出:\n{y}")

    # 计算 Jacobian 矩阵（对角矩阵）
    # 对于逐元素操作，Jacobi 是对角矩阵
    # d(leaky_relu(x))/dx = 1 if x > 0, else negative_slope
    x_flat = x.flatten()
    jacobi_diag = torch.where(
        x_flat > 0, torch.ones_like(x_flat), torch.full_like(x_flat, negative_slope)
    )
    jacobi = torch.diag(jacobi_diag)
    print(f"\nJacobian 对角线元素:\n{jacobi_diag}")
    print(f"\nJacobian 矩阵 ({jacobi.shape[0]}x{jacobi.shape[1]}):\n{jacobi}")

    # 验证：用 autograd 计算
    print("\n--- 验证 (使用 autograd) ---")
    for i in range(y.numel()):
        if x.grad is not None:
            x.grad.zero_()

        # 对输出的第 i 个元素求导
        grad_output = torch.zeros_like(y)
        grad_output.flatten()[i] = 1.0
        y.backward(grad_output, retain_graph=True)

        grad_row = x.grad.flatten()
        print(f"d(y[{i}])/dx = {grad_row.tolist()}")

    return y, jacobi


# ============================================================
# 测试用例 1: 2x2 矩阵，包含正负值
# ============================================================
x1 = [[0.5, -1.0], [0.0, 2.0]]

# 标准 ReLU (slope = 0)
test_leaky_relu(x1, 0.0, "标准 ReLU (2x2)")

# Leaky ReLU (slope = 0.1，与 MatrixSlow 一致)
test_leaky_relu(x1, 0.1, "Leaky ReLU slope=0.1 (2x2)")

# ============================================================
# 测试用例 2: 3x2 矩阵
# ============================================================
x2 = [[1.0, -2.0], [-0.5, 0.5], [3.0, -1.5]]

test_leaky_relu(x2, 0.0, "标准 ReLU (3x2)")
test_leaky_relu(x2, 0.1, "Leaky ReLU slope=0.1 (3x2)")

# ============================================================
# 测试用例 3: 边界值测试（全正、全负、含零）
# ============================================================
print("\n" + "=" * 60)
print("边界值测试")
print("=" * 60)

# 全正值
x_positive = [[1.0, 2.0], [3.0, 4.0]]
test_leaky_relu(x_positive, 0.1, "全正值 (2x2)")

# 全负值
x_negative = [[-1.0, -2.0], [-3.0, -4.0]]
test_leaky_relu(x_negative, 0.1, "全负值 (2x2)")

# 含零
x_with_zero = [[0.0, 1.0], [-1.0, 0.0]]
test_leaky_relu(x_with_zero, 0.1, "含零 (2x2)")

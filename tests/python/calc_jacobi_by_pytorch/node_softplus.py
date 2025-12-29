"""
SoftPlus 激活函数的 PyTorch 参考值计算

SoftPlus: f(x) = ln(1 + e^x)
导数:     f'(x) = sigmoid(x) = 1 / (1 + e^(-x))

用于 Rust 单元测试的预期值验证
"""

import torch
import torch.nn.functional as F


def print_tensor(name: str, t: torch.Tensor):
    """打印张量，便于复制到 Rust 测试"""
    print(f"\n{name}:")
    print(f"  shape: {list(t.shape)}")
    if t.numel() <= 20:
        flat = t.flatten().tolist()
        print(f"  data: {flat}")
    else:
        print(f"  data (first 10): {t.flatten()[:10].tolist()}")


def test_forward_1d():
    """1D 向量前向传播测试"""
    print("\n" + "=" * 60)
    print("1D 向量前向传播测试")
    print("=" * 60)

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    y = F.softplus(x)

    print_tensor("输入 x", x)
    print_tensor("SoftPlus(x)", y)


def test_forward_2d():
    """2D 矩阵前向传播测试"""
    print("\n" + "=" * 60)
    print("2D 矩阵前向传播测试")
    print("=" * 60)

    x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]])
    y = F.softplus(x)

    print_tensor("输入 x", x)
    print_tensor("SoftPlus(x)", y)


def test_backward_1d():
    """1D 向量反向传播测试"""
    print("\n" + "=" * 60)
    print("1D 向量反向传播测试")
    print("=" * 60)

    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = F.softplus(x)

    # 反向传播
    y.backward(torch.ones_like(y))

    print_tensor("输入 x", x)
    print_tensor("SoftPlus(x)", y)
    print_tensor("梯度 d(SoftPlus)/dx = sigmoid(x)", x.grad)

    # 验证梯度等于 sigmoid
    sigmoid_x = torch.sigmoid(x.detach())
    print_tensor("sigmoid(x) (验证)", sigmoid_x)


def test_backward_2d():
    """2D 矩阵反向传播测试"""
    print("\n" + "=" * 60)
    print("2D 矩阵反向传播测试")
    print("=" * 60)

    x = torch.tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], requires_grad=True)
    y = F.softplus(x)

    # 反向传播
    y.backward(torch.ones_like(y))

    print_tensor("输入 x", x)
    print_tensor("SoftPlus(x)", y)
    print_tensor("梯度 d(SoftPlus)/dx", x.grad)


def test_jacobi_diagonal():
    """验证 Jacobian 矩阵是对角矩阵"""
    print("\n" + "=" * 60)
    print("Jacobian 对角矩阵验证")
    print("=" * 60)

    x = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)

    # 计算完整 Jacobian
    y = F.softplus(x)
    jacobian = torch.zeros(3, 3)

    for i in range(3):
        x.grad = None
        y = F.softplus(x)
        y[i].backward(retain_graph=True)
        jacobian[i] = x.grad

    print_tensor("输入 x", x)
    print(f"\nJacobian 矩阵 (应为对角矩阵):\n{jacobian}")
    print(f"\n对角元素: {jacobian.diag().tolist()}")
    print(f"sigmoid(x): {torch.sigmoid(x.detach()).tolist()}")


def test_numerical_stability():
    """数值稳定性测试"""
    print("\n" + "=" * 60)
    print("数值稳定性测试")
    print("=" * 60)

    # 测试极端值
    x = torch.tensor([-50.0, -20.0, 0.0, 20.0, 50.0])
    y = F.softplus(x)

    print_tensor("极端值输入 x", x)
    print_tensor("SoftPlus(x)", y)

    # 验证：大正数时 softplus(x) ≈ x
    print(f"\n大正数时 softplus(50) ≈ 50: {y[-1].item():.6f}")
    # 验证：大负数时 softplus(x) ≈ 0
    print(f"大负数时 softplus(-50) ≈ 0: {y[0].item():.10f}")


def test_chain_with_linear():
    """线性层后接 SoftPlus 的链式传播测试"""
    print("\n" + "=" * 60)
    print("线性层 + SoftPlus 链式传播测试")
    print("=" * 60)

    # 输入
    x = torch.tensor([[1.0, 2.0]], requires_grad=True)
    # 权重
    w = torch.tensor([[0.5, -0.5], [0.3, 0.7]], requires_grad=True)

    # 前向传播: z = x @ w, output = softplus(z)
    z = x @ w
    output = F.softplus(z)

    print_tensor("输入 x", x)
    print_tensor("权重 w", w)
    print_tensor("z = x @ w", z)
    print_tensor("output = softplus(z)", output)

    # 反向传播
    output.backward(torch.ones_like(output))

    print_tensor("x.grad", x.grad)
    print_tensor("w.grad", w.grad)


if __name__ == "__main__":
    test_forward_1d()
    test_forward_2d()
    test_backward_1d()
    test_backward_2d()
    test_jacobi_diagonal()
    test_numerical_stability()
    test_chain_with_linear()

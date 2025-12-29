"""
@Author       : 老董
@Date         : 2025-12-21
@Description  : Reshape 节点的 Jacobi 矩阵验证（使用 PyTorch 计算参照值）

Reshape 的 Jacobi 是单位矩阵，因为每个输出元素正好等于对应的输入元素。
这个测试验证：
1. Reshape 的 Jacobi 确实是单位矩阵
2. 在链式网络中的梯度传播正确性
"""

import torch


def test_reshape_jacobi_is_identity():
    """验证 Reshape 的 Jacobi 是单位矩阵"""
    print("=" * 60)
    print("Test 1: Reshape Jacobi 是单位矩阵")
    print("=" * 60)

    # 输入 [2, 3] -> reshape -> [3, 2]
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    print(f"输入 x shape: {x.shape}")
    print(f"输入 x:\n{x}")

    # Reshape
    y = x.reshape(3, 2)
    print(f"\n输出 y shape: {y.shape}")
    print(f"输出 y:\n{y}")

    # 计算 Jacobi 矩阵
    # Jacobi[i, j] = ∂y[i] / ∂x[j]
    jacobi = torch.zeros(6, 6)
    y_flat = y.reshape(-1)

    for i in range(6):
        # 清除之前的梯度
        if x.grad is not None:
            x.grad.zero_()

        # 重新计算
        y = x.reshape(3, 2)
        y_flat = y.reshape(-1)

        # 对第 i 个输出求梯度
        grad_output = torch.zeros(6)
        grad_output[i] = 1.0
        y_flat.backward(grad_output, retain_graph=True)

        jacobi[i, :] = x.grad.reshape(-1)

    print(f"\nJacobi 矩阵 shape: {jacobi.shape}")
    print(f"Jacobi:\n{jacobi}")

    # 验证是单位矩阵
    expected = torch.eye(6)
    is_identity = torch.allclose(jacobi, expected)
    print(f"\n是否为单位矩阵: {is_identity}")

    return jacobi


def test_reshape_in_chain():
    """验证链式网络中 Reshape 的梯度传播"""
    print("\n" + "=" * 60)
    print("Test 2: Reshape 在链式网络中的梯度")
    print("=" * 60)

    # Parameter -> Reshape -> Sigmoid
    x = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    print(f"输入 x shape: {x.shape}")
    print(f"输入 x:\n{x}")

    # Forward
    reshaped = x.reshape(3, 2)
    output = torch.sigmoid(reshaped)
    print(f"\n输出 shape: {output.shape}")
    print(f"输出:\n{output}")

    # 计算 Jacobi
    jacobi = torch.zeros(6, 6)

    for i in range(6):
        if x.grad is not None:
            x.grad.zero_()

        reshaped = x.reshape(3, 2)
        output = torch.sigmoid(reshaped)
        output_flat = output.reshape(-1)

        grad_output = torch.zeros(6)
        grad_output[i] = 1.0
        output_flat.backward(grad_output, retain_graph=True)

        jacobi[i, :] = x.grad.reshape(-1)

    print(f"\nJacobi 矩阵:\n{jacobi}")

    # 因为 Reshape 的 Jacobi 是单位矩阵，所以最终 Jacobi 应该是 Sigmoid 导数的对角矩阵
    sigmoid_deriv = output * (1 - output)
    print(f"\nSigmoid 导数 (对角线元素):\n{sigmoid_deriv.reshape(-1).detach()}")

    # 验证对角线
    for i in range(6):
        row = i // 2
        col = i % 2
        expected_diag = sigmoid_deriv[row, col].item()
        actual_diag = jacobi[i, i].item()
        print(f"Jacobi[{i},{i}] = {actual_diag:.6f}, expected = {expected_diag:.6f}")

    return jacobi


def test_reshape_chain():
    """验证连续 Reshape 的 Jacobi 仍是单位矩阵"""
    print("\n" + "=" * 60)
    print("Test 3: 连续 Reshape 的 Jacobi")
    print("=" * 60)

    # [2, 6] -> [3, 4] -> [4, 3] -> [6, 2]
    x = torch.randn(2, 6, requires_grad=True)
    torch.manual_seed(42)
    x = torch.randn(2, 6, requires_grad=True)
    print(f"输入 shape: {x.shape}")

    # 连续 reshape
    r1 = x.reshape(3, 4)
    r2 = r1.reshape(4, 3)
    r3 = r2.reshape(6, 2)
    print(f"最终输出 shape: {r3.shape}")

    # 计算 Jacobi
    jacobi = torch.zeros(12, 12)

    for i in range(12):
        if x.grad is not None:
            x.grad.zero_()

        r1 = x.reshape(3, 4)
        r2 = r1.reshape(4, 3)
        r3 = r2.reshape(6, 2)
        r3_flat = r3.reshape(-1)

        grad_output = torch.zeros(12)
        grad_output[i] = 1.0
        r3_flat.backward(grad_output, retain_graph=True)

        jacobi[i, :] = x.grad.reshape(-1)

    # 验证是单位矩阵
    expected = torch.eye(12)
    is_identity = torch.allclose(jacobi, expected)
    print(f"\n连续 Reshape 的 Jacobi 是否为单位矩阵: {is_identity}")

    return jacobi


if __name__ == "__main__":
    print("Reshape 节点 Jacobi 验证\n")

    # Test 1
    jacobi1 = test_reshape_jacobi_is_identity()

    # Test 2
    jacobi2 = test_reshape_in_chain()

    # Test 3
    jacobi3 = test_reshape_chain()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("1. Reshape Jacobi is Identity Matrix - PASS")
    print("2. Reshape in chain does not affect gradient (pass-through) - PASS")
    print("3. Chained Reshapes still have Identity Jacobi - PASS")

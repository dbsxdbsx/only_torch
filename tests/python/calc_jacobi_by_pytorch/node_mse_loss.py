#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSELoss 节点的 PyTorch 参考实现

用于生成 Rust 单元测试的预期值
"""

import torch
import torch.nn as nn
import numpy as np


def print_tensor(name: str, tensor: torch.Tensor):
    """格式化打印张量"""
    print(f"\n{name}:")
    print(f"  shape: {list(tensor.shape)}")
    print(f"  data: {tensor.detach().numpy().flatten().tolist()}")


def test_mse_basic():
    """基础 MSE 测试 - 简单标量情况"""
    print("\n" + "=" * 60)
    print("Test: MSE Basic (简单情况)")
    print("=" * 60)

    # 输入
    input_data = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target_data = torch.tensor([[1.5, 2.5, 3.5]])

    print_tensor("input", input_data)
    print_tensor("target", target_data)

    # MSE with mean reduction (默认)
    loss_fn = nn.MSELoss(reduction='mean')
    loss = loss_fn(input_data, target_data)

    print(f"\nMSE Loss (mean): {loss.item()}")

    # 反向传播
    loss.backward()

    print_tensor("grad (mean)", input_data.grad)

    # 手动验证
    diff = input_data.detach() - target_data
    manual_loss = (diff ** 2).mean()
    manual_grad = 2 * diff / diff.numel()
    print(f"\n手动验证 loss: {manual_loss.item()}")
    print(f"手动验证 grad: {manual_grad.flatten().tolist()}")


def test_mse_reduction_sum():
    """MSE with sum reduction"""
    print("\n" + "=" * 60)
    print("Test: MSE Reduction Sum")
    print("=" * 60)

    input_data = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    target_data = torch.tensor([[1.5, 2.5, 3.5]])

    print_tensor("input", input_data)
    print_tensor("target", target_data)

    loss_fn = nn.MSELoss(reduction='sum')
    loss = loss_fn(input_data, target_data)

    print(f"\nMSE Loss (sum): {loss.item()}")

    loss.backward()
    print_tensor("grad (sum)", input_data.grad)


def test_mse_2d():
    """2D 矩阵的 MSE"""
    print("\n" + "=" * 60)
    print("Test: MSE 2D Matrix")
    print("=" * 60)

    input_data = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    target_data = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

    print_tensor("input", input_data)
    print_tensor("target", target_data)

    # Mean reduction
    loss_fn = nn.MSELoss(reduction='mean')
    loss = loss_fn(input_data, target_data)

    print(f"\nMSE Loss (mean): {loss.item()}")

    loss.backward()
    print_tensor("grad (mean)", input_data.grad)


def test_mse_batch():
    """Batch 模式的 MSE"""
    print("\n" + "=" * 60)
    print("Test: MSE Batch Mode")
    print("=" * 60)

    # Batch size = 3, features = 4
    input_data = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [2.0, 3.0, 4.0, 5.0]
    ], requires_grad=True)

    target_data = torch.tensor([
        [1.2, 2.1, 2.9, 4.1],
        [0.6, 1.4, 2.6, 3.4],
        [1.9, 3.1, 4.0, 5.2]
    ])

    print_tensor("input (batch)", input_data)
    print_tensor("target (batch)", target_data)

    loss_fn = nn.MSELoss(reduction='mean')
    loss = loss_fn(input_data, target_data)

    print(f"\nMSE Loss (mean, batch): {loss.item()}")

    loss.backward()
    print_tensor("grad (mean, batch)", input_data.grad)

    # 验证：梯度 = 2 * (input - target) / N
    N = input_data.numel()
    expected_grad = 2 * (input_data.detach() - target_data) / N
    print_tensor("expected grad", expected_grad)


def test_mse_jacobi_single_sample():
    """单样本模式的 Jacobi 矩阵计算"""
    print("\n" + "=" * 60)
    print("Test: MSE Jacobi (Single Sample Mode)")
    print("=" * 60)

    # [1, 4] 形状的输入（单样本）
    input_data = torch.tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    target_data = torch.tensor([[1.5, 2.5, 3.5, 4.5]])

    print_tensor("input", input_data)
    print_tensor("target", target_data)

    loss_fn = nn.MSELoss(reduction='mean')
    loss = loss_fn(input_data, target_data)

    print(f"\nMSE Loss: {loss.item()}")

    loss.backward()
    grad = input_data.grad

    # Jacobi 矩阵：损失是标量 [1,1]，输入是 [1,4]
    # Jacobi 形状应该是 [1, 4]
    print_tensor("Jacobi (grad reshaped as [1, n])", grad.reshape(1, -1))

    # 对于 MSE，Jacobi 矩阵实际上是对角矩阵
    # d(MSE)/d(x_i) = 2 * (x_i - y_i) / N
    N = input_data.numel()
    jacobi_diag = 2 * (input_data.detach() - target_data).flatten() / N
    print(f"\nJacobi diagonal elements: {jacobi_diag.tolist()}")


def test_mse_for_regression():
    """回归任务场景的 MSE"""
    print("\n" + "=" * 60)
    print("Test: MSE for Regression Task")
    print("=" * 60)

    # 模拟简单的回归：y = 2x + 1
    torch.manual_seed(42)

    # 权重初始化
    w = torch.tensor([[0.5]], requires_grad=True)  # 初始权重
    b = torch.tensor([[0.0]], requires_grad=True)  # 初始偏置

    # 训练数据
    x = torch.tensor([[1.0], [2.0], [3.0]])  # 输入
    y_true = torch.tensor([[3.0], [5.0], [7.0]])  # 真实值: 2*x + 1

    print(f"x: {x.flatten().tolist()}")
    print(f"y_true: {y_true.flatten().tolist()}")
    print(f"Initial w: {w.item()}, b: {b.item()}")

    # 前向传播
    y_pred = x @ w + b
    print_tensor("y_pred", y_pred)

    # MSE 损失
    loss_fn = nn.MSELoss(reduction='mean')
    loss = loss_fn(y_pred, y_true)

    print(f"\nMSE Loss: {loss.item()}")

    # 反向传播
    loss.backward()

    print(f"\nGradients:")
    print(f"  w.grad: {w.grad.item()}")
    print(f"  b.grad: {b.grad.item()}")


def test_mse_numerical_stability():
    """数值稳定性测试"""
    print("\n" + "=" * 60)
    print("Test: MSE Numerical Stability")
    print("=" * 60)

    # 大数值
    input_data = torch.tensor([[1000.0, 2000.0, 3000.0]], requires_grad=True)
    target_data = torch.tensor([[1000.5, 2000.5, 3000.5]])

    loss_fn = nn.MSELoss(reduction='mean')
    loss = loss_fn(input_data, target_data)

    print(f"Large values - MSE Loss: {loss.item()}")

    loss.backward()
    print_tensor("Large values - grad", input_data.grad)

    # 小数值
    input_data2 = torch.tensor([[0.001, 0.002, 0.003]], requires_grad=True)
    target_data2 = torch.tensor([[0.0015, 0.0025, 0.0035]])

    loss2 = loss_fn(input_data2, target_data2)
    print(f"\nSmall values - MSE Loss: {loss2.item()}")

    loss2.backward()
    print_tensor("Small values - grad", input_data2.grad)


if __name__ == "__main__":
    test_mse_basic()
    test_mse_reduction_sum()
    test_mse_2d()
    test_mse_batch()
    test_mse_jacobi_single_sample()
    test_mse_for_regression()
    test_mse_numerical_stability()

    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


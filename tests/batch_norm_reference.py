"""
BatchNorm Python 参考实现（PyTorch 对照值）

用于验证 only_torch 的 BatchNorm 实现正确性。
"""

import torch
import torch.nn as nn
import numpy as np

def batch_norm_1d_reference():
    """BatchNorm1d 训练模式对照值"""
    torch.manual_seed(42)

    # 输入: [batch=4, features=3]
    x = torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
    ], requires_grad=True)

    bn = nn.BatchNorm1d(3, eps=1e-5, momentum=0.1)
    # 初始化 gamma=1, beta=0
    nn.init.ones_(bn.weight)
    nn.init.zeros_(bn.bias)

    y = bn(x)
    print("=== BatchNorm1d 训练模式 ===")
    print(f"输入 shape: {x.shape}")
    print(f"输出:\n{y.detach().numpy()}")
    print(f"running_mean: {bn.running_mean.numpy()}")
    print(f"running_var: {bn.running_var.numpy()}")

    # 反向传播
    loss = y.sum()
    loss.backward()
    print(f"输入梯度:\n{x.grad.numpy()}")

def batch_norm_2d_reference():
    """BatchNorm2d 训练模式对照值"""
    torch.manual_seed(42)

    # 输入: [batch=2, channels=2, H=2, W=2]
    x = torch.arange(1, 17, dtype=torch.float32).reshape(2, 2, 2, 2)
    x.requires_grad_(True)

    bn = nn.BatchNorm2d(2, eps=1e-5, momentum=0.1)
    nn.init.ones_(bn.weight)
    nn.init.zeros_(bn.bias)

    y = bn(x)
    print("\n=== BatchNorm2d 训练模式 ===")
    print(f"输入 shape: {x.shape}")
    print(f"输出:\n{y.detach().numpy()}")
    print(f"running_mean: {bn.running_mean.numpy()}")
    print(f"running_var: {bn.running_var.numpy()}")

if __name__ == "__main__":
    batch_norm_1d_reference()
    batch_norm_2d_reference()

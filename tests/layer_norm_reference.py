"""LayerNorm 参考实现 - 生成 Rust 测试对照值

使用 PyTorch nn.LayerNorm 计算前向传播和反向传播结果，
供 Rust 测试用例验证正确性。
"""

import torch
import torch.nn as nn

def test_layer_norm_basic():
    """基本 LayerNorm: [2, 3] 输入, normalized_shape=[3]"""
    torch.manual_seed(42)
    
    x = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)
    
    ln = nn.LayerNorm(3, eps=1e-5, elementwise_affine=False)
    y = ln(x)
    
    print("=== basic: [2, 3], normalized_shape=[3] ===")
    print(f"input:\n{x}")
    print(f"output:\n{y}")
    
    # 反向传播
    loss = y.sum()
    loss.backward()
    print(f"grad:\n{x.grad}")
    
    return y.detach(), x.grad.clone()


def test_layer_norm_3d():
    """3D LayerNorm: [2, 2, 4] 输入, normalized_shape=[4]"""
    x = torch.tensor([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0]],
        [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8]]
    ], requires_grad=True)
    
    ln = nn.LayerNorm(4, eps=1e-5, elementwise_affine=False)
    y = ln(x)
    
    print("\n=== 3d: [2, 2, 4], normalized_shape=[4] ===")
    print(f"output:\n{y}")
    
    loss = y.sum()
    loss.backward()
    print(f"grad:\n{x.grad}")
    
    return y.detach(), x.grad.clone()


def test_layer_norm_with_affine():
    """带 gamma/beta 的 LayerNorm"""
    x = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], requires_grad=True)
    
    ln = nn.LayerNorm(3, eps=1e-5)
    # gamma=1, beta=0 初始值
    print(f"\n=== affine: gamma={ln.weight.data}, beta={ln.bias.data} ===")
    
    y = ln(x)
    print(f"output (gamma=1, beta=0):\n{y}")
    
    # 自定义 gamma 和 beta
    with torch.no_grad():
        ln.weight.copy_(torch.tensor([2.0, 0.5, 1.0]))
        ln.bias.copy_(torch.tensor([0.1, 0.2, 0.3]))
    
    y2 = ln(x)
    print(f"output (gamma=[2,0.5,1], beta=[0.1,0.2,0.3]):\n{y2}")


if __name__ == "__main__":
    test_layer_norm_basic()
    test_layer_norm_3d()
    test_layer_norm_with_affine()

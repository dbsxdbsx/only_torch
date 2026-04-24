"""
Upsample2d Python 参考实现（PyTorch 对照值）

用途：
- 验证 only_torch 的 Upsample2d 算子（nearest mode）实现正确性
- 单测预期值的真理来源（src/nn/tests/node_upsample2d.rs）

关键点：
- nearest upsample 反向 = sum_pool（对 (s_h × s_w) 块求和）
  ≠ avg_pool（avg_pool 会多除一个 s_h × s_w）
- PyTorch 的 nn.Upsample(mode='nearest') 反向就是 sum 语义

运行：
    python tests/upsample2d_reference.py
"""

import torch
import torch.nn.functional as F


def header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def case_forward_simple() -> None:
    """test_upsample2d_forward_simple 对照值

    输入 [[1, 2], [3, 4]] (1x1x2x2), scale=(2, 2)
    """
    header("Case 1: forward simple, scale=(2,2)")
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    y = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
    print(f"input shape: {tuple(x.shape)}")
    print(f"output shape: {tuple(y.shape)}")
    print(f"output:\n{y.detach().numpy().reshape(4, 4)}")
    expected = [
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [3, 3, 4, 4],
        [3, 3, 4, 4],
    ]
    print(f"Rust 单测预期: {expected}")


def case_vjp_basic() -> None:
    """test_upsample2d_vjp_basic 对照值

    upstream = ones [1,1,4,4] → grad 全 4
    """
    header("Case 2: vjp basic, scale=(2,2), upstream=ones")
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    y = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
    upstream = torch.ones_like(y)
    grad = torch.autograd.grad(y, x, grad_outputs=upstream)[0]
    print(f"grad shape: {tuple(grad.shape)}")
    print(f"grad:\n{grad.numpy().reshape(2, 2)}")
    print("Rust 单测预期: 全部 = 4.0")


def case_vjp_non_unit_upstream() -> None:
    """test_upsample2d_vjp_non_unit_upstream 对照值

    upstream = 1..16 reshape (1,1,4,4)
    grad[i,j] = sum of 2x2 block in upstream
    """
    header("Case 3: vjp non-unit upstream, scale=(2,2)")
    x = torch.zeros(1, 1, 2, 2, requires_grad=True)
    y = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
    upstream = torch.arange(1.0, 17.0).reshape(1, 1, 4, 4)
    grad = torch.autograd.grad(y, x, grad_outputs=upstream)[0]
    print(f"upstream:\n{upstream.numpy().reshape(4, 4)}")
    print(f"grad:\n{grad.numpy().reshape(2, 2)}")
    print("Rust 单测预期: [[14, 22], [46, 54]]")


def case_vjp_asymmetric_scale() -> None:
    """test_upsample2d_vjp_asymmetric_scale 对照值

    scale=(2, 3), upstream=ones [1,1,4,6] → 每个输入位置 grad = 2*3 = 6
    """
    header("Case 4: vjp asymmetric scale=(2,3), upstream=ones")
    x = torch.zeros(1, 1, 2, 2, requires_grad=True)
    y = F.interpolate(x, scale_factor=(2, 3), mode="nearest")
    upstream = torch.ones_like(y)
    grad = torch.autograd.grad(y, x, grad_outputs=upstream)[0]
    print(f"output shape: {tuple(y.shape)} (expected [1,1,4,6])")
    print(f"grad:\n{grad.numpy().reshape(2, 2)}")
    print("Rust 单测预期: 全部 = 6.0")


def case_e2e_backward() -> None:
    """test_upsample2d_e2e_backward 对照值

    param = [[1,2],[3,4]] → upsample(2x2) → flatten → mse_mean(target=0)
    loss = mean(y^2)，dL/dx[i,j] = x[i,j] / 2 → grad = [[0.5, 1.0], [1.5, 2.0]]
    """
    header("Case 5: e2e backward, mse_mean(target=0)")
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    y = F.interpolate(x, scale_factor=(2, 2), mode="nearest")
    flat = y.reshape(1, -1)
    target = torch.zeros_like(flat)
    # mse_mean = mean((flat - target)^2)
    loss = ((flat - target) ** 2).mean()
    loss.backward()
    print(f"loss = {loss.item()} (expected 7.5)")
    print(f"grad:\n{x.grad.numpy().reshape(2, 2)}")
    print("Rust 单测预期: [[0.5, 1.0], [1.5, 2.0]]")


def case_scale3() -> None:
    """test_upsample2d_forward_scale3 对照值

    输入 [[1, 2], [3, 4]] (1x1x2x2), scale=(3, 3) → 输出 (1x1x6x6)
    """
    header("Case 6: forward scale=(3,3)")
    x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True)
    y = F.interpolate(x, scale_factor=(3, 3), mode="nearest")
    print(f"output shape: {tuple(y.shape)} (expected [1,1,6,6])")
    print(f"output:\n{y.detach().numpy().reshape(6, 6)}")
    print("Rust 单测预期: 4 个 3x3 块各自 = 1, 2, 3, 4")


if __name__ == "__main__":
    case_forward_simple()
    case_vjp_basic()
    case_vjp_non_unit_upstream()
    case_vjp_asymmetric_scale()
    case_e2e_backward()
    case_scale3()
    print("\n" + "=" * 60)
    print("[OK] 所有 case 完成。如所有 'Rust 单测预期' 与上方 PyTorch 输出一致，")
    print("    说明 only_torch 的 Upsample2d 与 PyTorch nn.Upsample(mode='nearest')")
    print("    在数值上完全等价（前向 + 反向 sum_pool 语义）。")
    print("=" * 60)

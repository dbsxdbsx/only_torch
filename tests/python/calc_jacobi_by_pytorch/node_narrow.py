# -*- coding: utf-8 -*-
"""
Calculate expected values for Narrow node forward and backward
For Rust unit test verification
narrow(input, dim, start, length) — 沿单轴取连续子范围
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

print("=== Narrow Node Test ===")

# --- Test 1: 2D, axis=1, start=1, length=2 ---
x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
y1 = torch.narrow(x1, 1, 1, 2)
print(f"\nTest 1: x shape={list(x1.shape)}, narrow(axis=1, start=1, len=2)")
print(f"Input:\n{x1.detach().numpy()}")
print(f"Output:\n{y1.detach().numpy()}")

loss1 = y1.sum()
loss1.backward()
print(f"Grad (sum backward):\n{x1.grad.numpy()}")

# --- Test 2: 2D, axis=0, start=0, length=1 ---
x2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
y2 = torch.narrow(x2, 0, 0, 1)
print(f"\nTest 2: x shape={list(x2.shape)}, narrow(axis=0, start=0, len=1)")
print(f"Output:\n{y2.detach().numpy()}")

loss2 = y2.sum()
loss2.backward()
print(f"Grad:\n{x2.grad.numpy()}")

# --- Test 3: Jacobian for VJP test (2x3 -> narrow axis=1, start=0, len=2) ---
x3 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
upstream = torch.ones(2, 2)
y3 = torch.narrow(x3, 1, 0, 2)
y3.backward(upstream)
print(f"\nTest 3: VJP with unit upstream, narrow(axis=1, start=0, len=2)")
print(f"Grad:\n{x3.grad.numpy()}")

print("\n=== Rust Test Data ===")
print(f"// Test 1 forward: {y1.detach().numpy().flatten().tolist()}")
print(f"// Test 1 grad:    {x1.grad.numpy().flatten().tolist()}")
print(f"// Test 2 forward: {y2.detach().numpy().flatten().tolist()}")
print(f"// Test 3 grad:    {x3.grad.numpy().flatten().tolist()}")

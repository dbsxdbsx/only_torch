# -*- coding: utf-8 -*-
"""
Calculate expected values for Permute/Transpose node forward and backward
For Rust unit test verification
permute(dims) — 按指定轴顺序重排维度
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

print("=== Permute Node Test ===")

# --- Test 1: 2D transpose [2,3] -> [3,2] ---
x1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
y1 = x1.permute(1, 0)
print(f"\nTest 1: permute(1,0) on {list(x1.shape)}")
print(f"Input:\n{x1.detach().numpy()}")
print(f"Output shape: {list(y1.shape)}")
print(f"Output:\n{y1.detach().numpy()}")

upstream1 = torch.ones_like(y1)
y1.backward(upstream1)
print(f"Grad:\n{x1.grad.numpy()}")

# --- Test 2: 3D permute [2,3,4] -> [0,2,1] ---
x2 = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_(True)
y2 = x2.permute(0, 2, 1)
print(f"\nTest 2: permute(0,2,1) on {list(x2.shape)}")
print(f"Output shape: {list(y2.shape)}")
print(f"Output:\n{y2.detach().numpy()}")

upstream2 = torch.arange(1, 25, dtype=torch.float32).reshape(2, 4, 3)
y2.backward(upstream2)
print(f"Grad shape: {list(x2.grad.shape)}")
print(f"Grad:\n{x2.grad.numpy()}")

print("\n=== Rust Test Data ===")
print(f"// Test 1 forward: {y1.detach().numpy().flatten().tolist()}")
print(f"// Test 2 output shape: {list(y2.shape)}")
print(f"// Test 2 forward: {y2.detach().numpy().flatten().tolist()}")
print(f"// Test 2 grad: {x2.grad.numpy().flatten().tolist()}")

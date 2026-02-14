# -*- coding: utf-8 -*-
"""
Calculate expected values for TopK node forward and backward
For Rust unit test verification
torch.topk(input, k, dim, sorted) — 取前 K 个最大值
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

print("=== TopK Node Test ===")

# --- Test 1: 1D, k=3 ---
x1 = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], requires_grad=True)
values1, indices1 = torch.topk(x1, k=3, dim=0, sorted=True)
print(f"Test 1: topk(k=3, axis=0) on {list(x1.shape)}")
print(f"Values:  {values1.detach().numpy()}")
print(f"Indices: {indices1.numpy()}")

upstream1 = torch.ones_like(values1)
values1.backward(upstream1)
print(f"Grad: {x1.grad.numpy()}")

# --- Test 2: 2D, axis=1, k=2 ---
x2 = torch.tensor([[3.0, 1.0, 4.0], [5.0, 9.0, 2.0]], requires_grad=True)
values2, indices2 = torch.topk(x2, k=2, dim=1, sorted=True)
print(f"\nTest 2: topk(k=2, axis=1) on {list(x2.shape)}")
print(f"Values:\n{values2.detach().numpy()}")
print(f"Indices:\n{indices2.numpy()}")

upstream2 = torch.ones_like(values2)
values2.backward(upstream2)
print(f"Grad:\n{x2.grad.numpy()}")

# --- Test 3: k=1 (argmax equivalent) ---
x3 = torch.tensor([2.0, 5.0, 1.0, 3.0], requires_grad=True)
values3, indices3 = torch.topk(x3, k=1, dim=0, sorted=True)
print(f"\nTest 3: topk(k=1) on {list(x3.shape)}")
print(f"Values: {values3.detach().numpy()}")
print(f"Indices: {indices3.numpy()}")

values3.backward(torch.ones_like(values3))
print(f"Grad: {x3.grad.numpy()}")

print("\n=== Rust Test Data ===")
print(f"// Test 1 values:  {values1.detach().numpy().tolist()}")
print(f"// Test 1 indices: {indices1.numpy().tolist()}")
print(f"// Test 1 grad:    {x1.grad.numpy().tolist()}")
print(f"// Test 2 values:  {values2.detach().numpy().flatten().tolist()}")
print(f"// Test 2 indices: {indices2.numpy().flatten().tolist()}")
print(f"// Test 2 grad:    {x2.grad.numpy().flatten().tolist()}")

# -*- coding: utf-8 -*-
"""
Calculate expected values for Sort node forward and backward
For Rust unit test verification
torch.sort(input, dim, descending) — 沿轴排序
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

print("=== Sort Node Test ===")

# --- Test 1: 1D ascending ---
x1 = torch.tensor([3.0, 1.0, 4.0, 1.0, 5.0], requires_grad=True)
sorted1, indices1 = torch.sort(x1, dim=0, descending=False)
print(f"Test 1: sort(axis=0, asc) on {list(x1.shape)}")
print(f"Sorted:  {sorted1.detach().numpy()}")
print(f"Indices: {indices1.numpy()}")

upstream1 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
sorted1.backward(upstream1)
print(f"Grad (upstream=[10..50]): {x1.grad.numpy()}")

# --- Test 2: 2D, axis=1, descending ---
x2 = torch.tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], requires_grad=True)
sorted2, indices2 = torch.sort(x2, dim=1, descending=True)
print(f"\nTest 2: sort(axis=1, desc) on {list(x2.shape)}")
print(f"Sorted:\n{sorted2.detach().numpy()}")
print(f"Indices:\n{indices2.numpy()}")

upstream2 = torch.ones_like(sorted2)
sorted2.backward(upstream2)
print(f"Grad:\n{x2.grad.numpy()}")

# --- Test 3: 2D, axis=0 ---
x3 = torch.tensor([[5.0, 2.0], [1.0, 4.0], [3.0, 6.0]], requires_grad=True)
sorted3, indices3 = torch.sort(x3, dim=0, descending=False)
print(f"\nTest 3: sort(axis=0, asc) on {list(x3.shape)}")
print(f"Sorted:\n{sorted3.detach().numpy()}")
print(f"Indices:\n{indices3.numpy()}")

upstream3 = torch.ones_like(sorted3)
sorted3.backward(upstream3)
print(f"Grad:\n{x3.grad.numpy()}")

print("\n=== Rust Test Data ===")
print(f"// Test 1 sorted:  {sorted1.detach().numpy().tolist()}")
print(f"// Test 1 indices: {indices1.numpy().tolist()}")
print(f"// Test 1 grad:    {x1.grad.numpy().tolist()}")
print(f"// Test 2 sorted:  {sorted2.detach().numpy().flatten().tolist()}")
print(f"// Test 2 indices: {indices2.numpy().flatten().tolist()}")
print(f"// Test 2 grad:    {x2.grad.numpy().flatten().tolist()}")

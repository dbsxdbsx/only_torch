# -*- coding: utf-8 -*-
"""
Calculate expected values for Where/Cond node forward and backward
For Rust unit test verification
torch.where(condition, x, y) — 按掩码逐元素选择
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

print("=== Where/Cond Node Test ===")

# --- Test 1: mixed condition ---
cond = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # 不参与梯度
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)

# torch.where 使用 bool condition
result = torch.where(cond.bool(), x, y)
print(f"Condition:\n{cond.numpy()}")
print(f"x:\n{x.detach().numpy()}")
print(f"y:\n{y.detach().numpy()}")
print(f"Result:\n{result.detach().numpy()}")

upstream = torch.ones(2, 2)
result.backward(upstream)
print(f"\nGrad x (unit upstream):\n{x.grad.numpy()}")
print(f"Grad y (unit upstream):\n{y.grad.numpy()}")

# --- Test 2: all true ---
x2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y2 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
cond2 = torch.ones(2, 2)
result2 = torch.where(cond2.bool(), x2, y2)
result2.backward(torch.ones(2, 2))
print(f"\nAll-true grad x: {x2.grad.numpy().flatten().tolist()}")
print(f"All-true grad y: {y2.grad.numpy().flatten().tolist()}")

# --- Test 3: all false ---
x3 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y3 = torch.tensor([[10.0, 20.0], [30.0, 40.0]], requires_grad=True)
cond3 = torch.zeros(2, 2)
result3 = torch.where(cond3.bool(), x3, y3)
result3.backward(torch.ones(2, 2))
print(f"All-false grad x: {x3.grad.numpy().flatten().tolist()}")
print(f"All-false grad y: {y3.grad.numpy().flatten().tolist()}")

print("\n=== Rust Test Data ===")
print(f"// Mixed forward: {result.detach().numpy().flatten().tolist()}")
print(f"// Mixed grad_x:  {x.grad.numpy().flatten().tolist()}")
print(f"// Mixed grad_y:  {y.grad.numpy().flatten().tolist()}")

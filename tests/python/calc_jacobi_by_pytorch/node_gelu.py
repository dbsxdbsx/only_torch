# -*- coding: utf-8 -*-
"""
Calculate expected values for GELU node forward and Jacobian
For Rust unit test verification
Uses tanh approximation (GPT-2 style)
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

# Test data (2x2 matrix)
x = torch.tensor([[0.5, -1.0], [0.0, 2.0]], requires_grad=True)

print("=== GELU Node Test ===")
print(f"Input x:\n{x.detach().numpy()}")

# Forward pass (approximate version)
y = torch.nn.functional.gelu(x, approximate='tanh')
print(f"\nForward gelu(x):\n{y.detach().numpy()}")

# Compute Jacobian via autograd
n = x.numel()
jacobi = np.zeros((n, n))
for i in range(n):
    x_clone = x.clone().detach().requires_grad_(True)
    y_clone = torch.nn.functional.gelu(x_clone, approximate='tanh')
    y_flat = y_clone.flatten()
    y_flat[i].backward()
    jacobi[i, :] = x_clone.grad.flatten().numpy()

print(f"\nFull Jacobian ({n}x{n}):\n{jacobi}")

# Output Rust format data
print("\n=== Rust Test Data ===")
print("// Input value")
print(f"let input_data = &{x.detach().numpy().flatten().tolist()};")
print("\n// Expected forward output")
print(f"let expected_forward = &{y.detach().numpy().flatten().tolist()};")
print("\n// Expected Jacobian diagonal")
diag = np.diag(jacobi)
print(f"let expected_jacobi_diag = &{diag.tolist()};")

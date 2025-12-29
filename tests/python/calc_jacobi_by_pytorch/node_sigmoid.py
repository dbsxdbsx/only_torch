# -*- coding: utf-8 -*-
"""
Calculate expected values for Sigmoid node forward and Jacobian
For Rust unit test verification
"""

import numpy as np
import torch

# Set print precision
torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

# Test data (2x2 matrix)
x = torch.tensor([[0.5, -1.0], [0.0, 2.0]], requires_grad=True)

print("=== Sigmoid Node Test ===")
print(f"Input x:\n{x.detach().numpy()}")

# Forward pass
y = torch.sigmoid(x)
print(f"\nForward sigmoid(x):\n{y.detach().numpy()}")

# Compute Jacobian
# For element-wise ops, Jacobian is a diagonal matrix
# d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
sigmoid_val = y.detach().numpy()
jacobi_diag = sigmoid_val * (1 - sigmoid_val)
print(f"\nJacobian diagonal (sigmoid(x) * (1 - sigmoid(x))):\n{jacobi_diag.flatten()}")

# Build full Jacobian matrix (4x4 diagonal)
n = x.numel()
jacobi = np.zeros((n, n))
np.fill_diagonal(jacobi, jacobi_diag.flatten())
print(f"\nFull Jacobian ({n}x{n}):\n{jacobi}")

# Output Rust format data
print("\n=== Rust Test Data ===")
print("// Input value")
print(f"let input_data = &{x.detach().numpy().flatten().tolist()};")
print("\n// Expected forward output")
print(f"let expected_forward = &{y.detach().numpy().flatten().tolist()};")
print("\n// Expected Jacobian")
print(f"let expected_jacobi = &{jacobi.flatten().tolist()};")

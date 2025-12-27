# -*- coding: utf-8 -*-
"""
Calculate expected values for Tanh node forward and Jacobian
For Rust unit test verification
"""
import torch
import numpy as np

# Set print precision
torch.set_printoptions(precision=8)
np.set_printoptions(precision=8)

# Test data (2x2 matrix)
x = torch.tensor([[0.5, -1.0], [0.0, 2.0]], requires_grad=True)

print("=== Tanh Node Test ===")
print(f"Input x:\n{x.detach().numpy()}")

# Forward pass
y = torch.tanh(x)
print(f"\nForward tanh(x):\n{y.detach().numpy()}")

# Compute Jacobian
# For element-wise ops, Jacobian is a diagonal matrix
# d(tanh(x))/dx = 1 - tanh^2(x)
jacobi_diag = 1 - y.detach().numpy() ** 2
print(f"\nJacobian diagonal (1 - tanh^2(x)):\n{jacobi_diag.flatten()}")

# Build full Jacobian matrix (4x4 diagonal)
n = x.numel()
jacobi = np.zeros((n, n))
np.fill_diagonal(jacobi, jacobi_diag.flatten())
print(f"\nFull Jacobian ({n}x{n}):\n{jacobi}")

# Output Rust format data
print("\n=== Rust Test Data ===")
print(f"// Input value")
print(f"let input_data = &{x.detach().numpy().flatten().tolist()};")
print(f"\n// Expected forward output")
print(f"let expected_forward = &{y.detach().numpy().flatten().tolist()};")
print(f"\n// Expected Jacobian")
print(f"let expected_jacobi = &{jacobi.flatten().tolist()};")


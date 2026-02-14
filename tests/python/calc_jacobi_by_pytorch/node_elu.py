# -*- coding: utf-8 -*-
"""
Calculate expected values for ELU node forward and Jacobian
For Rust unit test verification
elu(x, alpha) = x if x>0, alpha*(exp(x)-1) if x<=0
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

x = torch.tensor([[0.5, -1.0], [0.0, 2.0]], requires_grad=True)
alpha = 1.0

print("=== ELU Node Test ===")
print(f"Input x:\n{x.detach().numpy()}")
print(f"Alpha: {alpha}")

y = torch.nn.functional.elu(x, alpha=alpha)
print(f"\nForward elu(x, alpha={alpha}):\n{y.detach().numpy()}")

n = x.numel()
jacobi = np.zeros((n, n))
for i in range(n):
    x_clone = x.clone().detach().requires_grad_(True)
    y_clone = torch.nn.functional.elu(x_clone, alpha=alpha)
    y_flat = y_clone.flatten()
    y_flat[i].backward()
    jacobi[i, :] = x_clone.grad.flatten().numpy()

print(f"\nFull Jacobian ({n}x{n}):\n{jacobi}")

print("\n=== Rust Test Data ===")
print("// Input value")
print(f"let input_data = &{x.detach().numpy().flatten().tolist()};")
print("\n// Expected forward output")
print(f"let expected_forward = &{y.detach().numpy().flatten().tolist()};")
print("\n// Expected Jacobian diagonal (= gradient with unit upstream)")
diag = np.diag(jacobi)
print(f"let expected_jacobi_diag = &{diag.tolist()};")

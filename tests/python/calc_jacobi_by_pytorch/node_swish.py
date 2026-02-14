# -*- coding: utf-8 -*-
"""
Calculate expected values for Swish/SiLU node forward and Jacobian
For Rust unit test verification
swish(x) = x * sigmoid(x), PyTorch: torch.nn.functional.silu
"""

import numpy as np
import torch

torch.set_printoptions(precision=16)
np.set_printoptions(precision=16)

x = torch.tensor([[0.5, -1.0], [0.0, 2.0]], requires_grad=True)

print("=== Swish/SiLU Node Test ===")
print(f"Input x:\n{x.detach().numpy()}")

y = torch.nn.functional.silu(x)
print(f"\nForward swish(x):\n{y.detach().numpy()}")

n = x.numel()
jacobi = np.zeros((n, n))
for i in range(n):
    x_clone = x.clone().detach().requires_grad_(True)
    y_clone = torch.nn.functional.silu(x_clone)
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

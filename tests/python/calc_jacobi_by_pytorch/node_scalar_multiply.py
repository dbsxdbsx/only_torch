"""
计算ScalarMultiply节点的雅可比矩阵
ScalarMultiply: C = s * M，其中s是标量(1x1)，M是矩阵(m,n)

对于标量s：∂C/∂s = M.flatten().T → shape: [m*n, 1]
对于矩阵M：∂C/∂M = s * I_{m*n} → shape: [m*n, m*n]
"""

import numpy as np
import torch

# 设置测试数据
scalar = torch.tensor([[2.0]], requires_grad=True)  # [1, 1]
matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)  # [2, 3]

print("=== 输入数据 ===")
print(f"标量 s: {scalar.detach().numpy().flatten()}")
print(f"矩阵 M: {matrix.detach().numpy().flatten()}")

# 前向传播: C = s * M
result = scalar * matrix
print("\n=== 前向传播结果 ===")
print(f"C = s * M: {result.detach().numpy().flatten()}")

# 计算对标量的雅可比矩阵
print("\n=== 对标量s的雅可比矩阵 ===")
# ∂C/∂s = M.flatten().T
jacobi_to_scalar = matrix.detach().numpy().flatten().reshape(-1, 1)
print(f"形状: {jacobi_to_scalar.shape}")
print(f"值:\n{jacobi_to_scalar.flatten()}")

# 计算对矩阵的雅可比矩阵
print("\n=== 对矩阵M的雅可比矩阵 ===")
# ∂C/∂M = s * I_{m*n}
m, n = matrix.shape
size = m * n
jacobi_to_matrix = scalar.item() * np.eye(size)
print(f"形状: {jacobi_to_matrix.shape}")
print(f"值:\n{jacobi_to_matrix.flatten()}")

# 验证：使用PyTorch的autograd
print("\n=== PyTorch autograd 验证 ===")


# 对每个输出元素求梯度来构建雅可比矩阵
def compute_jacobi_pytorch(output, input_var):
    """计算output相对于input_var的雅可比矩阵"""
    output_flat = output.flatten()
    input_flat = input_var.flatten()

    jacobi = torch.zeros(len(output_flat), len(input_flat))
    for i in range(len(output_flat)):
        if input_var.grad is not None:
            input_var.grad.zero_()
        output_flat[i].backward(retain_graph=True)
        jacobi[i] = input_var.grad.flatten()

    return jacobi


# 重新创建带梯度的张量
scalar = torch.tensor([[2.0]], requires_grad=True)
matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
result = scalar * matrix

# 计算对标量的雅可比
jacobi_scalar_pytorch = compute_jacobi_pytorch(result, scalar)
print(f"对标量的雅可比 (PyTorch): {jacobi_scalar_pytorch.numpy().flatten()}")

# 重新创建
scalar = torch.tensor([[2.0]], requires_grad=True)
matrix = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
result = scalar * matrix

# 计算对矩阵的雅可比
jacobi_matrix_pytorch = compute_jacobi_pytorch(result, matrix)
print(f"对矩阵的雅可比 (PyTorch):\n{jacobi_matrix_pytorch.numpy().flatten()}")

print("\n=== Rust测试用数据 ===")
print("// 标量值")
print("let scalar_data = &[2.0];")
print("// 矩阵值")
print(f"let matrix_data = &[{', '.join(map(str, matrix.detach().numpy().flatten()))}];")
print("// 预期输出")
print(
    f"let expected_output = &[{', '.join(map(str, result.detach().numpy().flatten()))}];"
)
print("// 对标量的雅可比矩阵")
print(
    f"let expected_jacobi_scalar = &[{', '.join(map(str, jacobi_to_scalar.flatten()))}];"
)
print("// 对矩阵的雅可比矩阵")
print(
    f"let expected_jacobi_matrix = &[{', '.join(map(str, jacobi_to_matrix.flatten()))}];"
)

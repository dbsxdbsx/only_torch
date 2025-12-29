"""
Vanilla RNN Layer PyTorch 参考实现

用于生成 Rust 单元测试的对照数值

运行: python tests/pytorch_reference/rnn_layer_reference.py
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("Vanilla RNN Layer 参考值")
print("=" * 70)


def print_tensor_as_rust(name: str, tensor: torch.Tensor, precision: int = 8):
    """输出 Rust 数组格式"""
    flat = tensor.detach().numpy().flatten()
    values = ", ".join([f"{v:.{precision}f}" for v in flat])
    print(f"const {name}: &[f32] = &[{values}];")


# ==================== 测试 1: 简单前向传播 ====================
print("\n" + "=" * 70)
print("测试 1: 简单前向传播 (batch=2, input=3, hidden=4)")
print("=" * 70)

batch_size = 2
input_size = 3
hidden_size = 4
seq_len = 1  # 单时间步

# 固定权重（与 Rust 测试对齐）
# 注意: PyTorch RNNCell 的权重是 [hidden, input] 格式
# 我们的实现是 [input, hidden] 格式，所以需要转置

# W_ih: [input_size, hidden_size] = [3, 4]
w_ih_data = [
    0.1, 0.2, 0.3, 0.4,  # input[0] -> hidden[0,1,2,3]
    0.5, 0.6, 0.7, 0.8,  # input[1] -> hidden[0,1,2,3]
    0.9, 1.0, 1.1, 1.2,  # input[2] -> hidden[0,1,2,3]
]
w_ih = torch.tensor(w_ih_data, dtype=torch.float32).reshape(input_size, hidden_size)

# W_hh: [hidden_size, hidden_size] = [4, 4]
w_hh_data = [
    0.1, 0.0, 0.0, 0.0,
    0.0, 0.2, 0.0, 0.0,
    0.0, 0.0, 0.3, 0.0,
    0.0, 0.0, 0.0, 0.4,
]
w_hh = torch.tensor(w_hh_data, dtype=torch.float32).reshape(hidden_size, hidden_size)

# b_h: [1, hidden_size] = [1, 4]
b_h_data = [0.1, 0.2, 0.3, 0.4]
b_h = torch.tensor(b_h_data, dtype=torch.float32).reshape(1, hidden_size)

# 输入: [batch, input_size] = [2, 3]
x_data = [
    1.0, 2.0, 3.0,  # batch 0
    0.5, 1.0, 1.5,  # batch 1
]
x = torch.tensor(x_data, dtype=torch.float32).reshape(batch_size, input_size)

# 初始隐藏状态: [batch, hidden] = [2, 4]，全 0
h_prev = torch.zeros(batch_size, hidden_size)

print(f"输入 x: shape={list(x.shape)}")
print(x)
print(f"\nW_ih: shape={list(w_ih.shape)}")
print(w_ih)
print(f"\nW_hh: shape={list(w_hh.shape)}")
print(w_hh)
print(f"\nb_h: shape={list(b_h.shape)}")
print(b_h)

# 手动计算 RNN 前向传播
# h = tanh(x @ W_ih + h_prev @ W_hh + b_h)
input_contrib = x @ w_ih  # [2, 3] @ [3, 4] = [2, 4]
hidden_contrib = h_prev @ w_hh  # [2, 4] @ [4, 4] = [2, 4]
pre_h = input_contrib + hidden_contrib + b_h  # [2, 4]
h = torch.tanh(pre_h)

print(f"\ninput_contrib = x @ W_ih:")
print(input_contrib)
print(f"\nhidden_contrib = h_prev @ W_hh:")
print(hidden_contrib)
print(f"\npre_h = input_contrib + hidden_contrib + b_h:")
print(pre_h)
print(f"\nhidden = tanh(pre_h):")
print(h)

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST1_X", x)
print_tensor_as_rust("TEST1_W_IH", w_ih)
print_tensor_as_rust("TEST1_W_HH", w_hh)
print_tensor_as_rust("TEST1_B_H", b_h)
print_tensor_as_rust("TEST1_HIDDEN", h)


# ==================== 测试 2: 多时间步前向传播 ====================
print("\n" + "=" * 70)
print("测试 2: 多时间步前向传播 (batch=1, input=2, hidden=3, seq_len=3)")
print("=" * 70)

batch_size = 1
input_size = 2
hidden_size = 3
seq_len = 3

# 权重
w_ih = torch.tensor([
    [0.5, 0.3, 0.1],
    [0.2, 0.4, 0.6],
], dtype=torch.float32)  # [2, 3]

w_hh = torch.tensor([
    [0.1, 0.0, 0.0],
    [0.0, 0.2, 0.0],
    [0.0, 0.0, 0.3],
], dtype=torch.float32)  # [3, 3]

b_h = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)  # [1, 3]

# 输入序列: [seq_len, batch, input_size]
sequence = torch.tensor([
    [[1.0, 0.0]],   # t=0
    [[0.0, 1.0]],   # t=1
    [[1.0, 1.0]],   # t=2
], dtype=torch.float32)

h = torch.zeros(batch_size, hidden_size)
hidden_states = []

print("时间步迭代:")
for t in range(seq_len):
    x_t = sequence[t]  # [1, 2]
    input_contrib = x_t @ w_ih
    hidden_contrib = h @ w_hh
    pre_h = input_contrib + hidden_contrib + b_h
    h = torch.tanh(pre_h)
    hidden_states.append(h.clone())
    print(f"  t={t}: x={x_t.tolist()}, h={h.tolist()}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST2_W_IH", w_ih)
print_tensor_as_rust("TEST2_W_HH", w_hh)
print(f"const TEST2_SEQ: &[&[f32]] = &[&[1.0, 0.0], &[0.0, 1.0], &[1.0, 1.0]];")
for t, h_t in enumerate(hidden_states):
    print_tensor_as_rust(f"TEST2_H_T{t}", h_t)


# ==================== 测试 3: 反向传播梯度 ====================
print("\n" + "=" * 70)
print("测试 3: 反向传播梯度 (batch=1, input=2, hidden=2, seq_len=2)")
print("=" * 70)

batch_size = 1
input_size = 2
hidden_size = 2
seq_len = 2

# 权重（需要梯度）
w_ih = torch.tensor([
    [0.5, 0.3],
    [0.2, 0.4],
], dtype=torch.float32, requires_grad=True)  # [2, 2]

w_hh = torch.tensor([
    [0.1, 0.0],
    [0.0, 0.2],
], dtype=torch.float32, requires_grad=True)  # [2, 2]

b_h = torch.tensor([[0.1, 0.2]], dtype=torch.float32, requires_grad=True)  # [1, 2]

# 输出层权重
w_out = torch.tensor([[1.0], [1.0]], dtype=torch.float32, requires_grad=True)  # [2, 1]

# 输入序列
sequence = [
    torch.tensor([[1.0, 0.5]], dtype=torch.float32),  # t=0
    torch.tensor([[0.5, 1.0]], dtype=torch.float32),  # t=1
]

target = torch.tensor([[0.8]], dtype=torch.float32)

h = torch.zeros(batch_size, hidden_size)

print("前向传播:")
for t, x_t in enumerate(sequence):
    input_contrib = x_t @ w_ih
    hidden_contrib = h @ w_hh
    pre_h = input_contrib + hidden_contrib + b_h
    h = torch.tanh(pre_h)
    print(f"  t={t}: h={h.tolist()}")

# 输出和损失
output = h @ w_out  # [1, 1]
loss = ((output - target) ** 2).mean()

print(f"\n输出: {output.item():.8f}")
print(f"损失: {loss.item():.8f}")

# 反向传播
loss.backward()

print(f"\n梯度:")
print(f"  dL/d(W_ih) = {w_ih.grad.tolist()}")
print(f"  dL/d(W_hh) = {w_hh.grad.tolist()}")
print(f"  dL/d(b_h) = {b_h.grad.tolist()}")
print(f"  dL/d(W_out) = {w_out.grad.tolist()}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST3_W_IH", torch.tensor([[0.5, 0.3], [0.2, 0.4]]))
print_tensor_as_rust("TEST3_W_HH", torch.tensor([[0.1, 0.0], [0.0, 0.2]]))
print_tensor_as_rust("TEST3_B_H", torch.tensor([[0.1, 0.2]]))
print_tensor_as_rust("TEST3_W_OUT", torch.tensor([[1.0], [1.0]]))
print(f"const TEST3_SEQ_0: &[f32] = &[1.0, 0.5];")
print(f"const TEST3_SEQ_1: &[f32] = &[0.5, 1.0];")
print(f"const TEST3_TARGET: f32 = 0.8;")
print(f"const TEST3_OUTPUT: f32 = {output.item():.8f};")
print(f"const TEST3_LOSS: f32 = {loss.item():.8f};")
print_tensor_as_rust("TEST3_GRAD_W_IH", w_ih.grad)
print_tensor_as_rust("TEST3_GRAD_W_HH", w_hh.grad)
print_tensor_as_rust("TEST3_GRAD_B_H", b_h.grad)
print_tensor_as_rust("TEST3_GRAD_W_OUT", w_out.grad)

print("\n" + "=" * 70)
print("完成！将上述常量复制到 Rust 测试文件中")
print("=" * 70)


"""
LSTM Layer PyTorch 参考实现

用于生成 Rust 单元测试的对照数值

运行: python tests/pytorch_reference/lstm_layer_reference.py
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("LSTM Layer 参考值")
print("=" * 70)


def print_tensor_as_rust(name: str, tensor: torch.Tensor, precision: int = 8):
    """输出 Rust 数组格式"""
    flat = tensor.detach().numpy().flatten()
    values = ", ".join([f"{v:.{precision}f}" for v in flat])
    print(f"const {name}: &[f32] = &[{values}];")


# ==================== 测试 1: 简单前向传播 ====================
print("\n" + "=" * 70)
print("测试 1: 简单前向传播 (batch=2, input=3, hidden=2)")
print("=" * 70)

batch_size = 2
input_size = 3
hidden_size = 2

# 固定权重（简单值便于手动验证）
# 输入门
w_ii = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)  # [3, 2]
w_hi = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)  # [2, 2]
b_i = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # [1, 2]

# 遗忘门
w_if = torch.tensor([[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]], dtype=torch.float32)
w_hf = torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=torch.float32)
b_f = torch.tensor([[1.0, 1.0]], dtype=torch.float32)  # 初始化为 1

# 候选细胞
w_ig = torch.tensor([[0.3, 0.2], [0.5, 0.4], [0.7, 0.6]], dtype=torch.float32)
w_hg = torch.tensor([[0.3, 0.0], [0.0, 0.3]], dtype=torch.float32)
b_g = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

# 输出门
w_io = torch.tensor([[0.4, 0.3], [0.6, 0.5], [0.8, 0.7]], dtype=torch.float32)
w_ho = torch.tensor([[0.4, 0.0], [0.0, 0.4]], dtype=torch.float32)
b_o = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

# 输入
x = torch.tensor([[1.0, 0.5, 0.2], [0.3, 0.8, 0.1]], dtype=torch.float32)  # [2, 3]
h_prev = torch.zeros(batch_size, hidden_size)
c_prev = torch.zeros(batch_size, hidden_size)

print(f"输入 x: {x.tolist()}")
print(f"h_prev: {h_prev.tolist()}")
print(f"c_prev: {c_prev.tolist()}")

# 手动计算 LSTM 前向传播
# 输入门
pre_i = x @ w_ii + h_prev @ w_hi + b_i
i_gate = torch.sigmoid(pre_i)
print(f"\n输入门 i: {i_gate.tolist()}")

# 遗忘门
pre_f = x @ w_if + h_prev @ w_hf + b_f
f_gate = torch.sigmoid(pre_f)
print(f"遗忘门 f: {f_gate.tolist()}")

# 候选细胞
pre_g = x @ w_ig + h_prev @ w_hg + b_g
g_gate = torch.tanh(pre_g)
print(f"候选细胞 g: {g_gate.tolist()}")

# 输出门
pre_o = x @ w_io + h_prev @ w_ho + b_o
o_gate = torch.sigmoid(pre_o)
print(f"输出门 o: {o_gate.tolist()}")

# 细胞状态
c = f_gate * c_prev + i_gate * g_gate
print(f"\n细胞状态 c: {c.tolist()}")

# 隐藏状态
h = o_gate * torch.tanh(c)
print(f"隐藏状态 h: {h.tolist()}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST1_X", x)
print_tensor_as_rust("TEST1_W_II", w_ii)
print_tensor_as_rust("TEST1_W_HI", w_hi)
print_tensor_as_rust("TEST1_B_I", b_i)
print_tensor_as_rust("TEST1_W_IF", w_if)
print_tensor_as_rust("TEST1_W_HF", w_hf)
print_tensor_as_rust("TEST1_B_F", b_f)
print_tensor_as_rust("TEST1_W_IG", w_ig)
print_tensor_as_rust("TEST1_W_HG", w_hg)
print_tensor_as_rust("TEST1_B_G", b_g)
print_tensor_as_rust("TEST1_W_IO", w_io)
print_tensor_as_rust("TEST1_W_HO", w_ho)
print_tensor_as_rust("TEST1_B_O", b_o)
print_tensor_as_rust("TEST1_HIDDEN", h)
print_tensor_as_rust("TEST1_CELL", c)


# ==================== 测试 2: 多时间步 ====================
print("\n" + "=" * 70)
print("测试 2: 多时间步前向传播 (batch=1, input=2, hidden=2, seq_len=3)")
print("=" * 70)

batch_size = 1
input_size = 2
hidden_size = 2

# 简化权重
w_ii = torch.tensor([[0.5, 0.3], [0.2, 0.4]], dtype=torch.float32)
w_hi = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
b_i = torch.zeros(1, hidden_size)

w_if = torch.tensor([[0.3, 0.5], [0.4, 0.2]], dtype=torch.float32)
w_hf = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
b_f = torch.ones(1, hidden_size)

w_ig = torch.tensor([[0.4, 0.2], [0.3, 0.5]], dtype=torch.float32)
w_hg = torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=torch.float32)
b_g = torch.zeros(1, hidden_size)

w_io = torch.tensor([[0.2, 0.4], [0.5, 0.3]], dtype=torch.float32)
w_ho = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
b_o = torch.zeros(1, hidden_size)

sequence = [
    torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    torch.tensor([[1.0, 1.0]], dtype=torch.float32),
]

h = torch.zeros(batch_size, hidden_size)
c = torch.zeros(batch_size, hidden_size)

print("时间步迭代:")
hidden_states = []
cell_states = []

for t, x_t in enumerate(sequence):
    i_gate = torch.sigmoid(x_t @ w_ii + h @ w_hi + b_i)
    f_gate = torch.sigmoid(x_t @ w_if + h @ w_hf + b_f)
    g_gate = torch.tanh(x_t @ w_ig + h @ w_hg + b_g)
    o_gate = torch.sigmoid(x_t @ w_io + h @ w_ho + b_o)

    c = f_gate * c + i_gate * g_gate
    h = o_gate * torch.tanh(c)

    hidden_states.append(h.clone())
    cell_states.append(c.clone())
    print(f"  t={t}: h={h.tolist()}, c={c.tolist()}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST2_W_II", w_ii)
print_tensor_as_rust("TEST2_W_HI", w_hi)
print_tensor_as_rust("TEST2_W_IF", w_if)
print_tensor_as_rust("TEST2_W_HF", w_hf)
print_tensor_as_rust("TEST2_W_IG", w_ig)
print_tensor_as_rust("TEST2_W_HG", w_hg)
print_tensor_as_rust("TEST2_W_IO", w_io)
print_tensor_as_rust("TEST2_W_HO", w_ho)
for t, (h_t, c_t) in enumerate(zip(hidden_states, cell_states)):
    print_tensor_as_rust(f"TEST2_H_T{t}", h_t)
    print_tensor_as_rust(f"TEST2_C_T{t}", c_t)


# ==================== 测试 3: 反向传播梯度 ====================
print("\n" + "=" * 70)
print("测试 3: BPTT 反向传播 (batch=1, input=2, hidden=2, seq_len=2)")
print("=" * 70)

batch_size = 1
input_size = 2
hidden_size = 2

# 需要梯度的权重
w_ii = torch.tensor([[0.5, 0.3], [0.2, 0.4]], dtype=torch.float32, requires_grad=True)
w_hi = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32, requires_grad=True)
b_i = torch.zeros(1, hidden_size, requires_grad=True)

w_if = torch.tensor([[0.3, 0.5], [0.4, 0.2]], dtype=torch.float32, requires_grad=True)
w_hf = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32, requires_grad=True)
b_f = torch.ones(1, hidden_size, requires_grad=True)

w_ig = torch.tensor([[0.4, 0.2], [0.3, 0.5]], dtype=torch.float32, requires_grad=True)
w_hg = torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=torch.float32, requires_grad=True)
b_g = torch.zeros(1, hidden_size, requires_grad=True)

w_io = torch.tensor([[0.2, 0.4], [0.5, 0.3]], dtype=torch.float32, requires_grad=True)
w_ho = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32, requires_grad=True)
b_o = torch.zeros(1, hidden_size, requires_grad=True)

w_out = torch.tensor([[1.0], [1.0]], dtype=torch.float32, requires_grad=True)

sequence = [
    torch.tensor([[1.0, 0.5]], dtype=torch.float32),
    torch.tensor([[0.5, 1.0]], dtype=torch.float32),
]
target = torch.tensor([[0.8]], dtype=torch.float32)

h = torch.zeros(batch_size, hidden_size)
c = torch.zeros(batch_size, hidden_size)

print("前向传播:")
for t, x_t in enumerate(sequence):
    i_gate = torch.sigmoid(x_t @ w_ii + h @ w_hi + b_i)
    f_gate = torch.sigmoid(x_t @ w_if + h @ w_hf + b_f)
    g_gate = torch.tanh(x_t @ w_ig + h @ w_hg + b_g)
    o_gate = torch.sigmoid(x_t @ w_io + h @ w_ho + b_o)

    c = f_gate * c + i_gate * g_gate
    h = o_gate * torch.tanh(c)
    print(f"  t={t}: h={h.tolist()}, c={c.tolist()}")

output = h @ w_out
loss = ((output - target) ** 2).mean()

print(f"\n输出: {output.item():.8f}")
print(f"损失: {loss.item():.8f}")

loss.backward()

print(f"\n主要梯度:")
print(f"  dL/d(w_ii) sum = {w_ii.grad.sum().item():.8f}")
print(f"  dL/d(w_if) sum = {w_if.grad.sum().item():.8f}")
print(f"  dL/d(w_ig) sum = {w_ig.grad.sum().item():.8f}")
print(f"  dL/d(w_io) sum = {w_io.grad.sum().item():.8f}")
print(f"  dL/d(w_out) = {w_out.grad.tolist()}")

print("\n// Rust 测试常量:")
print(f"const TEST3_OUTPUT: f32 = {output.item():.8f};")
print(f"const TEST3_LOSS: f32 = {loss.item():.8f};")
print_tensor_as_rust("TEST3_GRAD_W_II", w_ii.grad)
print_tensor_as_rust("TEST3_GRAD_W_OUT", w_out.grad)

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)


"""
GRU Layer PyTorch 参考实现

用于生成 Rust 单元测试的对照数值

运行: python tests/python/layer_reference/gru_layer_reference.py
"""

import torch

print("=" * 70)
print("GRU Layer 参考值")
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
# 重置门
w_ir = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32)  # [3, 2]
w_hr = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)  # [2, 2]
b_r = torch.tensor([[0.0, 0.0]], dtype=torch.float32)  # [1, 2]

# 更新门
w_iz = torch.tensor([[0.2, 0.1], [0.4, 0.3], [0.6, 0.5]], dtype=torch.float32)
w_hz = torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=torch.float32)
b_z = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

# 候选状态
w_in = torch.tensor([[0.3, 0.2], [0.5, 0.4], [0.7, 0.6]], dtype=torch.float32)
w_hn = torch.tensor([[0.3, 0.0], [0.0, 0.3]], dtype=torch.float32)
b_n = torch.tensor([[0.0, 0.0]], dtype=torch.float32)

# 输入
x = torch.tensor([[1.0, 0.5, 0.2], [0.3, 0.8, 0.1]], dtype=torch.float32)  # [2, 3]
h_prev = torch.zeros(batch_size, hidden_size)

print(f"输入 x: {x.tolist()}")
print(f"h_prev: {h_prev.tolist()}")

# 手动计算 GRU 前向传播
# 重置门
pre_r = x @ w_ir + h_prev @ w_hr + b_r
r_gate = torch.sigmoid(pre_r)
print(f"\n重置门 r: {r_gate.tolist()}")

# 更新门
pre_z = x @ w_iz + h_prev @ w_hz + b_z
z_gate = torch.sigmoid(pre_z)
print(f"更新门 z: {z_gate.tolist()}")

# 候选状态
pre_n = x @ w_in + r_gate * (h_prev @ w_hn) + b_n
n_gate = torch.tanh(pre_n)
print(f"候选状态 n: {n_gate.tolist()}")

# 隐藏状态
h = (1 - z_gate) * n_gate + z_gate * h_prev
print(f"隐藏状态 h: {h.tolist()}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST1_X", x)
print_tensor_as_rust("TEST1_W_IR", w_ir)
print_tensor_as_rust("TEST1_W_HR", w_hr)
print_tensor_as_rust("TEST1_B_R", b_r)
print_tensor_as_rust("TEST1_W_IZ", w_iz)
print_tensor_as_rust("TEST1_W_HZ", w_hz)
print_tensor_as_rust("TEST1_B_Z", b_z)
print_tensor_as_rust("TEST1_W_IN", w_in)
print_tensor_as_rust("TEST1_W_HN", w_hn)
print_tensor_as_rust("TEST1_B_N", b_n)
print_tensor_as_rust("TEST1_HIDDEN", h)


# ==================== 测试 2: 多时间步 ====================
print("\n" + "=" * 70)
print("测试 2: 多时间步前向传播 (batch=1, input=2, hidden=2, seq_len=3)")
print("=" * 70)

batch_size = 1
input_size = 2
hidden_size = 2

# 简化权重
w_ir = torch.tensor([[0.5, 0.3], [0.2, 0.4]], dtype=torch.float32)
w_hr = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
b_r = torch.zeros(1, hidden_size)

w_iz = torch.tensor([[0.3, 0.5], [0.4, 0.2]], dtype=torch.float32)
w_hz = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32)
b_z = torch.zeros(1, hidden_size)

w_in = torch.tensor([[0.4, 0.2], [0.3, 0.5]], dtype=torch.float32)
w_hn = torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=torch.float32)
b_n = torch.zeros(1, hidden_size)

sequence = [
    torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    torch.tensor([[1.0, 1.0]], dtype=torch.float32),
]

h = torch.zeros(batch_size, hidden_size)

print("时间步迭代:")
hidden_states = []

for t, x_t in enumerate(sequence):
    r_gate = torch.sigmoid(x_t @ w_ir + h @ w_hr + b_r)
    z_gate = torch.sigmoid(x_t @ w_iz + h @ w_hz + b_z)
    n_gate = torch.tanh(x_t @ w_in + r_gate * (h @ w_hn) + b_n)
    h = (1 - z_gate) * n_gate + z_gate * h
    hidden_states.append(h.clone())
    print(f"  t={t}: h={h.tolist()}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST2_W_IR", w_ir)
print_tensor_as_rust("TEST2_W_HR", w_hr)
print_tensor_as_rust("TEST2_W_IZ", w_iz)
print_tensor_as_rust("TEST2_W_HZ", w_hz)
print_tensor_as_rust("TEST2_W_IN", w_in)
print_tensor_as_rust("TEST2_W_HN", w_hn)
for t, h_t in enumerate(hidden_states):
    print_tensor_as_rust(f"TEST2_H_T{t}", h_t)


# ==================== 测试 3: 反向传播梯度 ====================
print("\n" + "=" * 70)
print("测试 3: BPTT 反向传播 (batch=1, input=2, hidden=2, seq_len=2)")
print("=" * 70)

batch_size = 1
input_size = 2
hidden_size = 2

# 需要梯度的权重
w_ir = torch.tensor([[0.5, 0.3], [0.2, 0.4]], dtype=torch.float32, requires_grad=True)
w_hr = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32, requires_grad=True)
b_r = torch.zeros(1, hidden_size, requires_grad=True)

w_iz = torch.tensor([[0.3, 0.5], [0.4, 0.2]], dtype=torch.float32, requires_grad=True)
w_hz = torch.tensor([[0.1, 0.0], [0.0, 0.1]], dtype=torch.float32, requires_grad=True)
b_z = torch.zeros(1, hidden_size, requires_grad=True)

w_in = torch.tensor([[0.4, 0.2], [0.3, 0.5]], dtype=torch.float32, requires_grad=True)
w_hn = torch.tensor([[0.2, 0.0], [0.0, 0.2]], dtype=torch.float32, requires_grad=True)
b_n = torch.zeros(1, hidden_size, requires_grad=True)

w_out = torch.tensor([[1.0], [1.0]], dtype=torch.float32, requires_grad=True)

sequence = [
    torch.tensor([[1.0, 0.5]], dtype=torch.float32),
    torch.tensor([[0.5, 1.0]], dtype=torch.float32),
]
target = torch.tensor([[0.8]], dtype=torch.float32)

h = torch.zeros(batch_size, hidden_size)

print("前向传播:")
for t, x_t in enumerate(sequence):
    r_gate = torch.sigmoid(x_t @ w_ir + h @ w_hr + b_r)
    z_gate = torch.sigmoid(x_t @ w_iz + h @ w_hz + b_z)
    n_gate = torch.tanh(x_t @ w_in + r_gate * (h @ w_hn) + b_n)
    h = (1 - z_gate) * n_gate + z_gate * h
    print(f"  t={t}: h={h.tolist()}")

output = h @ w_out
loss = ((output - target) ** 2).mean()

print(f"\n输出: {output.item():.8f}")
print(f"损失: {loss.item():.8f}")

loss.backward()

print("\n主要梯度:")
print(f"  dL/d(w_ir) sum = {w_ir.grad.sum().item():.8f}")
print(f"  dL/d(w_iz) sum = {w_iz.grad.sum().item():.8f}")
print(f"  dL/d(w_in) sum = {w_in.grad.sum().item():.8f}")
print(f"  dL/d(w_out) = {w_out.grad.tolist()}")

print("\n// Rust 测试常量:")
print(f"const TEST3_OUTPUT: f32 = {output.item():.8f};")
print(f"const TEST3_LOSS: f32 = {loss.item():.8f};")
print_tensor_as_rust("TEST3_GRAD_W_IR", w_ir.grad)
print_tensor_as_rust("TEST3_GRAD_W_OUT", w_out.grad)

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)

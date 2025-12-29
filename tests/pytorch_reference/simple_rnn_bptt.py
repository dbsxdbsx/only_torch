"""
简单 RNN BPTT 参考实现

这个脚本计算一个最简单的 RNN 的前向和反向传播，
用于验证 only_torch 的 BPTT 实现是否正确。

网络结构:
    hidden[t] = tanh(h_prev[t] + input[t] * w_scale)
    h_prev[t] = hidden[t-1]  (循环连接)
    output = hidden[T] * w_out
    loss = MSE(output, target)

运行: python tests/pytorch_reference/simple_rnn_bptt.py
"""

import torch
import torch.nn.functional as F

print("=" * 60)
print("PyTorch BPTT 参考值")
print("=" * 60)

# 固定输入
sequence = [1.0, 0.0, 1.0]  # 3 步序列
target = 1.0  # 目标值

# 参数 (与 only_torch 测试一致)
w_scale = torch.tensor([[1.0]], requires_grad=True)
w_out = torch.tensor([[1.0]], requires_grad=True)

print(f"\n输入序列: {sequence}")
print(f"目标值: {target}")
print(f"w_scale 初值: {w_scale.item()}")
print(f"w_out 初值: {w_out.item()}")

# ========== 前向传播 ==========
print("\n" + "=" * 60)
print("前向传播")
print("=" * 60)

hidden = torch.zeros(1, 1)  # 初始隐藏状态
hidden_history = [hidden.clone()]

for t, x in enumerate(sequence):
    x_tensor = torch.tensor([[x]])
    # hidden = tanh(h_prev + input * w_scale)
    pre_hidden = hidden + x_tensor * w_scale
    hidden = torch.tanh(pre_hidden)
    hidden_history.append(hidden.clone())
    print(f"t={t}: input={x}, pre_hidden={pre_hidden.item():.6f}, hidden={hidden.item():.6f}")

# 输出层
output = hidden * w_out
print(f"\noutput = hidden * w_out = {output.item():.6f}")

# 损失
target_tensor = torch.tensor([[target]])
loss = F.mse_loss(output, target_tensor)
print(f"loss = MSE({output.item():.6f}, {target}) = {loss.item():.6f}")

# ========== 反向传播 ==========
print("\n" + "=" * 60)
print("反向传播 (BPTT)")
print("=" * 60)

loss.backward()

print(f"\ndL/d(w_scale) = {w_scale.grad.item():.6f}")
print(f"dL/d(w_out) = {w_out.grad.item():.6f}")

# ========== 手动验证 ==========
print("\n" + "=" * 60)
print("手动验证梯度计算")
print("=" * 60)

print("""
链式法则展开：

设 h[t] = tanh(h[t-1] + x[t] * w_scale)
设 h[0] = 0

h[1] = tanh(0 + 1.0 * w_scale) = tanh(w_scale)
h[2] = tanh(h[1] + 0 * w_scale) = tanh(h[1]) = tanh(tanh(w_scale))
h[3] = tanh(h[2] + 1.0 * w_scale) = tanh(tanh(tanh(w_scale)) + w_scale)

output = h[3] * w_out
loss = (output - target)^2

dL/d(w_out) = 2 * (output - target) * h[3]
dL/d(w_scale) = 2 * (output - target) * w_out * (dh[3]/d(w_scale))

其中 dh[3]/d(w_scale) 需要通过链式法则从 h[3] 追溯到所有 w_scale 的使用处
""")

# 使用 w_scale=1.0, w_out=1.0 的具体数值验证
w_s = 1.0
w_o = 1.0

h1 = torch.tanh(torch.tensor(w_s)).item()
h2 = torch.tanh(torch.tensor(h1)).item()
h3 = torch.tanh(torch.tensor(h2 + w_s)).item()

print(f"h[1] = tanh({w_s}) = {h1:.6f}")
print(f"h[2] = tanh({h1:.6f}) = {h2:.6f}")
print(f"h[3] = tanh({h2:.6f} + {w_s}) = {h3:.6f}")
print(f"output = {h3:.6f} * {w_o} = {h3 * w_o:.6f}")
print(f"loss = ({h3 * w_o:.6f} - {target})^2 = {(h3 * w_o - target)**2:.6f}")

# ========== 用于 Rust 测试的精确值 ==========
print("\n" + "=" * 60)
print("用于 Rust 测试的精确值")
print("=" * 60)

print(f"""
// 序列: [1.0, 0.0, 1.0], 目标: 1.0
// w_scale = 1.0, w_out = 1.0

// 前向传播结果
const HIDDEN_T1: f32 = {hidden_history[1].item():.8f};  // tanh(1.0)
const HIDDEN_T2: f32 = {hidden_history[2].item():.8f};  // tanh(tanh(1.0))
const HIDDEN_T3: f32 = {hidden_history[3].item():.8f};  // tanh(tanh(tanh(1.0)) + 1.0)
const OUTPUT: f32 = {output.item():.8f};
const LOSS: f32 = {loss.item():.8f};

// BPTT 梯度
const GRAD_W_SCALE: f32 = {w_scale.grad.item():.8f};
const GRAD_W_OUT: f32 = {w_out.grad.item():.8f};
""")

print("\n如果 only_torch 的 BPTT 实现正确，应该得到相同的梯度值！")


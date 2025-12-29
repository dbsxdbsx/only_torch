"""
多种 RNN 结构的 PyTorch 参考实现

提供 2-3 种不同的 RNN 结构用于数值对照测试

运行: python tests/python/layer_reference/rnn_multi_structure.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("PyTorch RNN 多结构参考值")
print("=" * 70)


def print_rust_consts(name: str, values: dict):
    """输出 Rust 常量格式"""
    print(f"\n// === {name} ===")
    for k, v in values.items():
        if isinstance(v, float):
            print(f"const {k}: f32 = {v:.8f};")
        elif isinstance(v, list):
            print(f"const {k}: [f32; {len(v)}] = {v};")


# ==================== 结构 1: 双层隐藏 ====================
print("\n" + "=" * 70)
print("结构 1: 双层 RNN (两个连续的循环层)")
print("=" * 70)
print("""
网络结构:
    h1[t] = tanh(x[t] * w_ih1 + h1[t-1] * w_hh1)
    h2[t] = tanh(h1[t] * w_h12 + h2[t-1] * w_hh2)
    output = h2[T] * w_out
    loss = MSE(output, target)
""")

# 参数
w_ih1 = torch.tensor([[0.5]], requires_grad=True)
w_hh1 = torch.tensor([[0.8]], requires_grad=True)
w_h12 = torch.tensor([[0.6]], requires_grad=True)
w_hh2 = torch.tensor([[0.7]], requires_grad=True)
w_out = torch.tensor([[1.0]], requires_grad=True)

sequence = [1.0, 0.5, -0.5, 1.0]
target = 0.5

h1 = torch.zeros(1, 1)
h2 = torch.zeros(1, 1)

print(f"序列: {sequence}")
print(f"目标: {target}")

for t, x in enumerate(sequence):
    x_t = torch.tensor([[x]])
    h1 = torch.tanh(x_t * w_ih1 + h1 * w_hh1)
    h2 = torch.tanh(h1 * w_h12 + h2 * w_hh2)
    print(f"t={t}: x={x:.1f}, h1={h1.item():.6f}, h2={h2.item():.6f}")

output = h2 * w_out
loss = F.mse_loss(output, torch.tensor([[target]]))
loss.backward()

print(f"\noutput = {output.item():.6f}")
print(f"loss = {loss.item():.6f}")
print("\n梯度:")
assert w_ih1.grad is not None and w_hh1.grad is not None
assert w_h12.grad is not None and w_hh2.grad is not None and w_out.grad is not None
print(f"  dL/d(w_ih1) = {w_ih1.grad.item():.8f}")
print(f"  dL/d(w_hh1) = {w_hh1.grad.item():.8f}")
print(f"  dL/d(w_h12) = {w_h12.grad.item():.8f}")
print(f"  dL/d(w_hh2) = {w_hh2.grad.item():.8f}")
print(f"  dL/d(w_out) = {w_out.grad.item():.8f}")

print_rust_consts("结构 1: 双层 RNN", {
    "STRUCT1_H1_FINAL": h1.item(),
    "STRUCT1_H2_FINAL": h2.item(),
    "STRUCT1_OUTPUT": output.item(),
    "STRUCT1_LOSS": loss.item(),
    "STRUCT1_GRAD_W_IH1": w_ih1.grad.item(),
    "STRUCT1_GRAD_W_HH1": w_hh1.grad.item(),
    "STRUCT1_GRAD_W_H12": w_h12.grad.item(),
    "STRUCT1_GRAD_W_HH2": w_hh2.grad.item(),
    "STRUCT1_GRAD_W_OUT": w_out.grad.item(),
})


# ==================== 结构 2: 带 Sigmoid 激活 ====================
print("\n" + "=" * 70)
print("结构 2: Sigmoid RNN (非 tanh 激活)")
print("=" * 70)
print("""
网络结构:
    h[t] = sigmoid(x[t] * w_ih + h[t-1] * w_hh)
    output = h[T] * w_out
    loss = MSE(output, target)
""")

# 重置梯度
w_ih = torch.tensor([[0.5]], requires_grad=True)
w_hh = torch.tensor([[0.9]], requires_grad=True)
w_out2 = torch.tensor([[2.0]], requires_grad=True)

sequence2 = [1.0, -1.0, 0.5]
target2 = 0.8

h = torch.zeros(1, 1)

print(f"序列: {sequence2}")
print(f"目标: {target2}")

h_values = []
for t, x in enumerate(sequence2):
    x_t = torch.tensor([[x]])
    h = torch.sigmoid(x_t * w_ih + h * w_hh)
    h_values.append(h.item())
    print(f"t={t}: x={x:.1f}, h={h.item():.6f}")

output2 = h * w_out2
loss2 = F.mse_loss(output2, torch.tensor([[target2]]))
loss2.backward()

print(f"\noutput = {output2.item():.6f}")
print(f"loss = {loss2.item():.6f}")
print("\n梯度:")
assert w_ih.grad is not None and w_hh.grad is not None and w_out2.grad is not None
print(f"  dL/d(w_ih) = {w_ih.grad.item():.8f}")
print(f"  dL/d(w_hh) = {w_hh.grad.item():.8f}")
print(f"  dL/d(w_out) = {w_out2.grad.item():.8f}")

print_rust_consts("结构 2: Sigmoid RNN", {
    "STRUCT2_H_T1": h_values[0],
    "STRUCT2_H_T2": h_values[1],
    "STRUCT2_H_T3": h_values[2],
    "STRUCT2_OUTPUT": output2.item(),
    "STRUCT2_LOSS": loss2.item(),
    "STRUCT2_GRAD_W_IH": w_ih.grad.item(),
    "STRUCT2_GRAD_W_HH": w_hh.grad.item(),
    "STRUCT2_GRAD_W_OUT": w_out2.grad.item(),
})


# ==================== 结构 3: 带偏置的 RNN ====================
print("\n" + "=" * 70)
print("结构 3: 带偏置的 RNN")
print("=" * 70)
print("""
网络结构:
    h[t] = tanh(x[t] * w_ih + h[t-1] * w_hh + b_h)
    output = h[T] * w_out + b_o
    loss = MSE(output, target)
""")

# 参数
w_ih3 = torch.tensor([[0.3]], requires_grad=True)
w_hh3 = torch.tensor([[0.5]], requires_grad=True)
b_h3 = torch.tensor([[0.1]], requires_grad=True)
w_out3 = torch.tensor([[1.5]], requires_grad=True)
b_o3 = torch.tensor([[-0.2]], requires_grad=True)

sequence3 = [0.5, 1.0, -0.5, 0.0, 1.0]  # 5 步序列
target3 = 0.3

h = torch.zeros(1, 1)

print(f"序列: {sequence3}")
print(f"目标: {target3}")

h_values3 = []
for t, x in enumerate(sequence3):
    x_t = torch.tensor([[x]])
    h = torch.tanh(x_t * w_ih3 + h * w_hh3 + b_h3)
    h_values3.append(h.item())
    print(f"t={t}: x={x:.1f}, h={h.item():.6f}")

output3 = h * w_out3 + b_o3
loss3 = F.mse_loss(output3, torch.tensor([[target3]]))
loss3.backward()

print(f"\noutput = {output3.item():.6f}")
print(f"loss = {loss3.item():.6f}")
print("\n梯度:")
assert w_ih3.grad is not None and w_hh3.grad is not None and b_h3.grad is not None
assert w_out3.grad is not None and b_o3.grad is not None
print(f"  dL/d(w_ih) = {w_ih3.grad.item():.8f}")
print(f"  dL/d(w_hh) = {w_hh3.grad.item():.8f}")
print(f"  dL/d(b_h) = {b_h3.grad.item():.8f}")
print(f"  dL/d(w_out) = {w_out3.grad.item():.8f}")
print(f"  dL/d(b_o) = {b_o3.grad.item():.8f}")

print_rust_consts("结构 3: 带偏置的 RNN", {
    "STRUCT3_H_FINAL": h.item(),
    "STRUCT3_OUTPUT": output3.item(),
    "STRUCT3_LOSS": loss3.item(),
    "STRUCT3_GRAD_W_IH": w_ih3.grad.item(),
    "STRUCT3_GRAD_W_HH": w_hh3.grad.item(),
    "STRUCT3_GRAD_B_H": b_h3.grad.item(),
    "STRUCT3_GRAD_W_OUT": w_out3.grad.item(),
    "STRUCT3_GRAD_B_O": b_o3.grad.item(),
})


# ==================== 结构 4: LeakyReLU RNN ====================
print("\n" + "=" * 70)
print("结构 4: LeakyReLU RNN (分段线性激活)")
print("=" * 70)
print("""
网络结构:
    h[t] = LeakyReLU(x[t] * w_ih + h[t-1] * w_hh, negative_slope=0.1)
    output = h[T] * w_out
    loss = MSE(output, target)

特点：
    - 分段线性激活（x>0 时 slope=1, x<0 时 slope=0.1）
    - 与 tanh/sigmoid 不同，梯度在 x=0 处不连续
    - 测试 BPTT 对非平滑激活的支持
""")

# 参数
w_ih4 = torch.tensor([[0.5]], requires_grad=True)
w_hh4 = torch.tensor([[0.8]], requires_grad=True)
w_out4 = torch.tensor([[1.5]], requires_grad=True)
negative_slope = 0.1

leaky_relu = nn.LeakyReLU(negative_slope=negative_slope)

# 序列包含正负值，确保测试两个区域
sequence4 = [1.0, -0.5, 0.3, -0.8, 0.6]
target4 = 0.7

h = torch.zeros(1, 1)

print(f"序列: {sequence4}")
print(f"目标: {target4}")
print(f"negative_slope: {negative_slope}")

h_values4 = []
for t, x in enumerate(sequence4):
    x_t = torch.tensor([[x]])
    pre_h = x_t * w_ih4 + h * w_hh4
    h = leaky_relu(pre_h)
    h_values4.append(h.item())
    region = "positive" if pre_h.item() > 0 else "negative"
    print(f"t={t}: x={x:5.1f}, pre_h={pre_h.item():8.4f} ({region}), h={h.item():.6f}")

output4 = h * w_out4
loss4 = F.mse_loss(output4, torch.tensor([[target4]]))
loss4.backward()

print(f"\noutput = {output4.item():.6f}")
print(f"loss = {loss4.item():.6f}")
print("\n梯度:")
assert w_ih4.grad is not None and w_hh4.grad is not None and w_out4.grad is not None
print(f"  dL/d(w_ih) = {w_ih4.grad.item():.8f}")
print(f"  dL/d(w_hh) = {w_hh4.grad.item():.8f}")
print(f"  dL/d(w_out) = {w_out4.grad.item():.8f}")

print_rust_consts("结构 4: LeakyReLU RNN", {
    "STRUCT4_H_T1": h_values4[0],
    "STRUCT4_H_T2": h_values4[1],
    "STRUCT4_H_T3": h_values4[2],
    "STRUCT4_H_T4": h_values4[3],
    "STRUCT4_H_T5": h_values4[4],
    "STRUCT4_OUTPUT": output4.item(),
    "STRUCT4_LOSS": loss4.item(),
    "STRUCT4_GRAD_W_IH": w_ih4.grad.item(),
    "STRUCT4_GRAD_W_HH": w_hh4.grad.item(),
    "STRUCT4_GRAD_W_OUT": w_out4.grad.item(),
})


# ==================== 结构 5: SoftPlus RNN ====================
print("\n" + "=" * 70)
print("结构 5: SoftPlus RNN (平滑激活)")
print("=" * 70)
print("""
网络结构:
    h[t] = SoftPlus(x[t] * w_ih + h[t-1] * w_hh)
    output = h[T] * w_out
    loss = MSE(output, target)

特点：
    - 平滑激活函数，是 ReLU 的平滑近似
    - 导数为 sigmoid: f'(x) = 1 / (1 + e^(-x)) = 1 - exp(-softplus(x))
    - 测试 BPTT 对 SoftPlus 的支持（验证从输出计算梯度的正确性）
""")

# 参数
w_ih5 = torch.tensor([[0.3]], requires_grad=True)
w_hh5 = torch.tensor([[0.5]], requires_grad=True)
w_out5 = torch.tensor([[1.0]], requires_grad=True)

softplus = nn.Softplus()

# 序列
sequence5 = [0.5, -0.3, 0.8, -0.5, 0.2]
target5 = 1.0

h = torch.zeros(1, 1)

print(f"序列: {sequence5}")
print(f"目标: {target5}")

h_values5 = []
for t, x in enumerate(sequence5):
    x_t = torch.tensor([[x]])
    pre_h = x_t * w_ih5 + h * w_hh5
    h = softplus(pre_h)
    h_values5.append(h.item())
    print(f"t={t}: x={x:5.1f}, pre_h={pre_h.item():8.4f}, h={h.item():.6f}")

output5 = h * w_out5
loss5 = F.mse_loss(output5, torch.tensor([[target5]]))
loss5.backward()

print(f"\noutput = {output5.item():.6f}")
print(f"loss = {loss5.item():.6f}")
print("\n梯度:")
assert w_ih5.grad is not None and w_hh5.grad is not None and w_out5.grad is not None
print(f"  dL/d(w_ih) = {w_ih5.grad.item():.8f}")
print(f"  dL/d(w_hh) = {w_hh5.grad.item():.8f}")
print(f"  dL/d(w_out) = {w_out5.grad.item():.8f}")

print_rust_consts("结构 5: SoftPlus RNN", {
    "STRUCT5_H_T1": h_values5[0],
    "STRUCT5_H_T2": h_values5[1],
    "STRUCT5_H_T3": h_values5[2],
    "STRUCT5_H_T4": h_values5[3],
    "STRUCT5_H_T5": h_values5[4],
    "STRUCT5_OUTPUT": output5.item(),
    "STRUCT5_LOSS": loss5.item(),
    "STRUCT5_GRAD_W_IH": w_ih5.grad.item(),
    "STRUCT5_GRAD_W_HH": w_hh5.grad.item(),
    "STRUCT5_GRAD_W_OUT": w_out5.grad.item(),
})


print("\n" + "=" * 70)
print("测试完成！将上述常量复制到 Rust 测试文件中使用。")
print("=" * 70)

"""
Linear Layer PyTorch 参考实现

用于生成 Rust 单元测试的对照数值

运行: python tests/python/layer_reference/linear_layer_reference.py
"""

import torch
import torch.nn as nn

print("=" * 70)
print("Linear Layer 参考值")
print("=" * 70)


def print_tensor_as_rust(name: str, tensor: torch.Tensor, precision: int = 8):
    """输出 Rust 数组格式"""
    flat = tensor.detach().numpy().flatten()
    values = ", ".join([f"{v:.{precision}f}" for v in flat])
    print(f"const {name}: &[f32] = &[{values}];")


def print_f32_as_rust(name: str, value: float, precision: int = 8):
    """输出 Rust f32 常量"""
    print(f"const {name}: f32 = {value:.{precision}f};")


# ==================== 测试 1: 简单前向传播 ====================
print("\n" + "=" * 70)
print("测试 1: 简单前向传播 (batch=2, in_features=3, out_features=4)")
print("=" * 70)

batch_size = 2
in_features = 3
out_features = 4

# 输入: [batch, in_features] = [2, 3]
x = torch.tensor([
    [1.0, 2.0, 3.0],   # batch 0
    [0.5, 1.5, 2.5],   # batch 1
], dtype=torch.float32)

# 权重: [in_features, out_features] = [3, 4]
# 注意: PyTorch nn.Linear 的 weight 是 [out, in]，但我们的实现是 [in, out]
# 所以这里直接用 [in, out] 格式，PyTorch 需要转置
w = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],   # in[0] -> out
    [0.5, 0.6, 0.7, 0.8],   # in[1] -> out
    [0.9, 1.0, 1.1, 1.2],   # in[2] -> out
], dtype=torch.float32)

# 偏置: [1, out_features] = [1, 4]
b = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")
print(x)
print(f"\n权重 W: shape={list(w.shape)}")
print(w)
print(f"\n偏置 b: shape={list(b.shape)}")
print(b)

# 手动计算: output = x @ W + b
output = x @ w + b

print(f"\n输出 = x @ W + b: shape={list(output.shape)}")
print(output)

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST1_X", x)
print_tensor_as_rust("TEST1_W", w)
print_tensor_as_rust("TEST1_B", b)
print_tensor_as_rust("TEST1_OUTPUT", output)


# ==================== 测试 2: 反向传播梯度 ====================
print("\n" + "=" * 70)
print("测试 2: 反向传播梯度 (batch=2, in=3, out=2) + MSE Loss")
print("=" * 70)

batch_size = 2
in_features = 3
out_features = 2

# 输入
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 1.5],
], dtype=torch.float32)

# 权重 [in, out]
w = torch.tensor([
    [0.1, 0.2],
    [0.3, 0.4],
    [0.5, 0.6],
], dtype=torch.float32, requires_grad=True)

# 偏置 [1, out]
b = torch.tensor([[0.1, 0.2]], dtype=torch.float32, requires_grad=True)

# 目标
target = torch.tensor([
    [1.0, 0.5],
    [0.5, 1.0],
], dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")
print(x)
print(f"\n权重 W: shape={list(w.shape)}")
print(w)
print(f"\n偏置 b: shape={list(b.shape)}")
print(b)
print(f"\n目标 target: shape={list(target.shape)}")
print(target)

# 前向传播
output = x @ w + b
print(f"\n输出: shape={list(output.shape)}")
print(output)

# MSE Loss
loss = ((output - target) ** 2).mean()
print(f"\nMSE Loss: {loss.item():.8f}")

# 反向传播
loss.backward()

print("\n梯度:")
print(f"  dL/dW shape={list(w.grad.shape)}")
print(w.grad)
print(f"\n  dL/db shape={list(b.grad.shape)}")
print(b.grad)

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST2_X", x)
print_tensor_as_rust("TEST2_W", torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
print_tensor_as_rust("TEST2_B", torch.tensor([[0.1, 0.2]]))
print_tensor_as_rust("TEST2_TARGET", target)
print_tensor_as_rust("TEST2_OUTPUT", output)
print_f32_as_rust("TEST2_LOSS", loss.item())
print_tensor_as_rust("TEST2_GRAD_W", w.grad)
print_tensor_as_rust("TEST2_GRAD_B", b.grad)


# ==================== 测试 3: 多层网络反向传播 ====================
print("\n" + "=" * 70)
print("测试 3: 两层 Linear + ReLU + SoftmaxCrossEntropy")
print("=" * 70)

batch_size = 2
in_features = 4
hidden_features = 3
out_features = 2

# 输入
x = torch.tensor([
    [1.0, 0.5, -0.5, 0.2],
    [0.3, -0.2, 0.8, -0.1],
], dtype=torch.float32)

# 第一层权重 [in=4, hidden=3]
w1 = torch.tensor([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
    [0.2, 0.3, 0.4],
], dtype=torch.float32, requires_grad=True)
b1 = torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32, requires_grad=True)

# 第二层权重 [hidden=3, out=2]
w2 = torch.tensor([
    [0.5, 0.6],
    [0.7, 0.8],
    [0.9, 1.0],
], dtype=torch.float32, requires_grad=True)
b2 = torch.tensor([[0.1, 0.2]], dtype=torch.float32, requires_grad=True)

# 目标 (one-hot)
target = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
], dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")
print(x)

# 前向传播
h1 = x @ w1 + b1
print(f"\n隐藏层 pre-activation: shape={list(h1.shape)}")
print(h1)

# ReLU
h1_relu = torch.relu(h1)
print(f"\n隐藏层 post-ReLU: shape={list(h1_relu.shape)}")
print(h1_relu)

# 第二层
logits = h1_relu @ w2 + b2
print(f"\nlogits: shape={list(logits.shape)}")
print(logits)

# Softmax + CrossEntropy
# 使用 PyTorch 的 CrossEntropyLoss（内含 softmax）
# 注意: CrossEntropyLoss 期望 target 是类索引，不是 one-hot
# 为了手动计算，我们用 softmax + 负对数似然
softmax = torch.softmax(logits, dim=1)
print(f"\nsoftmax: shape={list(softmax.shape)}")
print(softmax)

# Cross entropy: -sum(target * log(softmax)) / batch_size
eps = 1e-10
log_softmax = torch.log(softmax + eps)
cross_entropy_per_sample = -(target * log_softmax).sum(dim=1)
loss = cross_entropy_per_sample.mean()
print(f"\n每样本 cross entropy: {cross_entropy_per_sample.tolist()}")
print(f"平均 loss: {loss.item():.8f}")

# 反向传播
loss.backward()

print("\n梯度:")
print(f"  dL/dW1 shape={list(w1.grad.shape)}")
print(w1.grad)
print(f"\n  dL/db1 shape={list(b1.grad.shape)}")
print(b1.grad)
print(f"\n  dL/dW2 shape={list(w2.grad.shape)}")
print(w2.grad)
print(f"\n  dL/db2 shape={list(b2.grad.shape)}")
print(b2.grad)

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST3_X", x)
print_tensor_as_rust("TEST3_W1", torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4]]))
print_tensor_as_rust("TEST3_B1", torch.tensor([[0.1, 0.1, 0.1]]))
print_tensor_as_rust("TEST3_W2", torch.tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]))
print_tensor_as_rust("TEST3_B2", torch.tensor([[0.1, 0.2]]))
print_tensor_as_rust("TEST3_TARGET", target)
print_tensor_as_rust("TEST3_H1_RELU", h1_relu)
print_tensor_as_rust("TEST3_LOGITS", logits)
print_tensor_as_rust("TEST3_SOFTMAX", softmax)
print_f32_as_rust("TEST3_LOSS", loss.item())
print_tensor_as_rust("TEST3_GRAD_W1", w1.grad)
print_tensor_as_rust("TEST3_GRAD_B1", b1.grad)
print_tensor_as_rust("TEST3_GRAD_W2", w2.grad)
print_tensor_as_rust("TEST3_GRAD_B2", b2.grad)


# ==================== 测试 4: 使用 nn.Linear 验证 ====================
print("\n" + "=" * 70)
print("测试 4: 使用 nn.Linear 验证（确认权重转置正确）")
print("=" * 70)

batch_size = 2
in_features = 3
out_features = 2

# 使用 nn.Linear
linear = nn.Linear(in_features, out_features, bias=True)

# 设置与测试 2 相同的权重（注意 nn.Linear 的 weight 是 [out, in]）
with torch.no_grad():
    # 我们的 W 是 [in, out]，nn.Linear 的是 [out, in]，所以转置
    linear.weight.copy_(torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).T)
    linear.bias.copy_(torch.tensor([0.1, 0.2]))

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [0.5, 1.0, 1.5],
], dtype=torch.float32)

output_nn = linear(x)
print(f"nn.Linear 输出:")
print(output_nn)

# 手动计算验证
w_manual = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
b_manual = torch.tensor([[0.1, 0.2]])
output_manual = x @ w_manual + b_manual
print(f"\n手动计算输出 (x @ W + b):")
print(output_manual)

# 验证一致
assert torch.allclose(output_nn, output_manual), "nn.Linear 和手动计算不一致！"
print("\n✅ nn.Linear 输出与手动计算一致")


print("\n" + "=" * 70)
print("完成！将上述常量复制到 Rust 测试文件中")
print("=" * 70)


"""
Conv2d Layer PyTorch 参考实现

用于生成 Rust 单元测试的对照数值

运行: python tests/python/layer_reference/conv2d_layer_reference.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 70)
print("Conv2d Layer 参考值")
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
print("测试 1: 简单前向传播 (batch=1, C_in=1, H=4, W=4, C_out=2, kernel=2x2)")
print("=" * 70)

batch_size = 1
in_channels = 1
out_channels = 2
H, W = 4, 4
kH, kW = 2, 2
stride = (1, 1)
padding = (0, 0)

# 输入: [batch, C_in, H, W] = [1, 1, 4, 4]
x = torch.tensor([
    [[[1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]]]
], dtype=torch.float32)

# 卷积核: [C_out, C_in, kH, kW] = [2, 1, 2, 2]
kernel = torch.tensor([
    [[[1.0, 0.0],
      [0.0, 1.0]]],   # filter 0: 对角线检测
    [[[0.0, 1.0],
      [1.0, 0.0]]]    # filter 1: 反对角线检测
], dtype=torch.float32)

# 偏置: [C_out] = [2]
bias = torch.tensor([0.5, -0.5], dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")
print(x)
print(f"\n卷积核 kernel: shape={list(kernel.shape)}")
print(kernel)
print(f"\n偏置 bias: shape={list(bias.shape)}")
print(bias)

# 使用 F.conv2d
output = F.conv2d(x, kernel, bias=bias, stride=stride, padding=padding)

print(f"\n输出: shape={list(output.shape)}")
print(output)

# 手动验证第一个输出位置
# filter 0 @ (0,0): 1*1 + 2*0 + 5*0 + 6*1 + 0.5 = 7.5
# filter 1 @ (0,0): 1*0 + 2*1 + 5*1 + 6*0 - 0.5 = 6.5
print(f"\n手动验证 [0,0,0,0]: 1*1 + 6*1 + 0.5 = {1*1 + 6*1 + 0.5}")
print(f"手动验证 [0,1,0,0]: 2*1 + 5*1 - 0.5 = {2*1 + 5*1 - 0.5}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST1_X", x)
print_tensor_as_rust("TEST1_KERNEL", kernel)
print_tensor_as_rust("TEST1_BIAS", bias)
print_tensor_as_rust("TEST1_OUTPUT", output)


# ==================== 测试 2: 带 padding 和 stride ====================
print("\n" + "=" * 70)
print("测试 2: 带 padding=1, stride=2 (batch=1, C_in=1, H=5, W=5, C_out=1, kernel=3x3)")
print("=" * 70)

batch_size = 1
in_channels = 1
out_channels = 1
H, W = 5, 5
kH, kW = 3, 3
stride = (2, 2)
padding = (1, 1)

# 输入: [1, 1, 5, 5]
x = torch.arange(1, 26, dtype=torch.float32).reshape(1, 1, 5, 5)

# 卷积核: [1, 1, 3, 3] - 均值滤波器
kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32) / 9.0

# 无偏置
bias = torch.zeros(1, dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")
print(x)
print(f"\n卷积核 kernel: shape={list(kernel.shape)}")
print(kernel)

output = F.conv2d(x, kernel, bias=bias, stride=stride, padding=padding)

print(f"\n输出: shape={list(output.shape)}")
print(output)

# 输出尺寸: (5 + 2*1 - 3) / 2 + 1 = 3
print(f"预期输出尺寸: {(H + 2*padding[0] - kH) // stride[0] + 1}")

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST2_X", x)
print_tensor_as_rust("TEST2_KERNEL", kernel)
print_tensor_as_rust("TEST2_BIAS", bias)
print_tensor_as_rust("TEST2_OUTPUT", output)


# ==================== 测试 3: 多通道输入输出 ====================
print("\n" + "=" * 70)
print("测试 3: 多通道 (batch=2, C_in=3, H=4, W=4, C_out=2, kernel=2x2)")
print("=" * 70)

batch_size = 2
in_channels = 3
out_channels = 2
H, W = 4, 4
kH, kW = 2, 2
stride = (1, 1)
padding = (0, 0)

# 输入: [2, 3, 4, 4]
torch.manual_seed(42)
x = torch.randn(batch_size, in_channels, H, W)

# 卷积核: [2, 3, 2, 2] - 固定值便于验证
kernel = torch.tensor([
    # filter 0: [3, 2, 2]
    [[[0.1, 0.2], [0.3, 0.4]],   # C_in=0
     [[0.5, 0.6], [0.7, 0.8]],   # C_in=1
     [[0.9, 1.0], [1.1, 1.2]]],  # C_in=2
    # filter 1: [3, 2, 2]
    [[[-0.1, -0.2], [-0.3, -0.4]],
     [[-0.5, -0.6], [-0.7, -0.8]],
     [[-0.9, -1.0], [-1.1, -1.2]]]
], dtype=torch.float32)

bias = torch.tensor([0.1, -0.1], dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")
print(f"卷积核 kernel: shape={list(kernel.shape)}")
print(f"偏置 bias: shape={list(bias.shape)}")

output = F.conv2d(x, kernel, bias=bias, stride=stride, padding=padding)

print(f"\n输出: shape={list(output.shape)}")
print(output)

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST3_X", x)
print_tensor_as_rust("TEST3_KERNEL", kernel)
print_tensor_as_rust("TEST3_BIAS", bias)
print_tensor_as_rust("TEST3_OUTPUT", output)


# ==================== 测试 4: 反向传播梯度 ====================
print("\n" + "=" * 70)
print("测试 4: 反向传播梯度 (batch=1, C_in=1, H=3, W=3, C_out=1, kernel=2x2) + MSE")
print("=" * 70)

batch_size = 1
in_channels = 1
out_channels = 1
H, W = 3, 3
kH, kW = 2, 2

# 输入
x = torch.tensor([
    [[[1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]]]
], dtype=torch.float32)

# 卷积核 (需要梯度)
kernel = torch.tensor([
    [[[0.1, 0.2],
      [0.3, 0.4]]]
], dtype=torch.float32, requires_grad=True)

# 偏置 (需要梯度)
bias = torch.tensor([0.5], dtype=torch.float32, requires_grad=True)

# 目标: conv 输出形状 [1, 1, 2, 2]
target = torch.tensor([
    [[[5.0, 6.0],
      [8.0, 9.0]]]
], dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")
print(x)
print(f"\n卷积核 kernel: shape={list(kernel.shape)}")
print(kernel)
print(f"\n偏置 bias: shape={list(bias.shape)}")
print(bias)
print(f"\n目标 target: shape={list(target.shape)}")
print(target)

# 前向传播
output = F.conv2d(x, kernel, bias=bias, stride=(1, 1), padding=(0, 0))
print(f"\n输出: shape={list(output.shape)}")
print(output)

# MSE Loss
loss = ((output - target) ** 2).mean()
print(f"\nMSE Loss: {loss.item():.8f}")

# 反向传播
loss.backward()

print("\n梯度:")
print(f"  dL/d(kernel) shape={list(kernel.grad.shape)}")
print(kernel.grad)
print(f"\n  dL/d(bias) shape={list(bias.grad.shape)}")
print(bias.grad)

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST4_X", x)
print_tensor_as_rust("TEST4_KERNEL", torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]]))
print_tensor_as_rust("TEST4_BIAS", torch.tensor([0.5]))
print_tensor_as_rust("TEST4_TARGET", target)
print_tensor_as_rust("TEST4_OUTPUT", output)
print_f32_as_rust("TEST4_LOSS", loss.item())
print_tensor_as_rust("TEST4_GRAD_KERNEL", kernel.grad)
print_tensor_as_rust("TEST4_GRAD_BIAS", bias.grad)


# ==================== 测试 5: 多层 CNN + Linear 反向传播 ====================
print("\n" + "=" * 70)
print("测试 5: Conv -> ReLU -> Flatten -> Linear -> SoftmaxCE")
print("=" * 70)

batch_size = 2
in_channels = 1
H, W = 4, 4
out_channels = 2
kH, kW = 2, 2
num_classes = 3

# 输入
x = torch.tensor([
    # batch 0
    [[[1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]]],
    # batch 1
    [[[0.5, 1.0, 1.5, 2.0],
      [2.5, 3.0, 3.5, 4.0],
      [4.5, 5.0, 5.5, 6.0],
      [6.5, 7.0, 7.5, 8.0]]]
], dtype=torch.float32)

# 卷积核 [2, 1, 2, 2]
conv_kernel = torch.tensor([
    [[[0.1, 0.2], [0.3, 0.4]]],
    [[[0.5, 0.6], [0.7, 0.8]]]
], dtype=torch.float32, requires_grad=True)

conv_bias = torch.tensor([0.1, -0.1], dtype=torch.float32, requires_grad=True)

# conv 输出: [2, 2, 3, 3]
# flatten 后: [2, 18]
# linear 权重: [18, 3]
fc_weight = torch.tensor([
    [0.01, 0.02, 0.03],
    [0.04, 0.05, 0.06],
    [0.07, 0.08, 0.09],
    [0.10, 0.11, 0.12],
    [0.13, 0.14, 0.15],
    [0.16, 0.17, 0.18],
    [0.19, 0.20, 0.21],
    [0.22, 0.23, 0.24],
    [0.25, 0.26, 0.27],
    [0.28, 0.29, 0.30],
    [0.31, 0.32, 0.33],
    [0.34, 0.35, 0.36],
    [0.37, 0.38, 0.39],
    [0.40, 0.41, 0.42],
    [0.43, 0.44, 0.45],
    [0.46, 0.47, 0.48],
    [0.49, 0.50, 0.51],
    [0.52, 0.53, 0.54],
], dtype=torch.float32, requires_grad=True)

fc_bias = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32, requires_grad=True)

# 目标 (one-hot)
target = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=torch.float32)

print(f"输入 x: shape={list(x.shape)}")

# 前向传播
conv_out = F.conv2d(x, conv_kernel, bias=conv_bias, stride=(1, 1), padding=(0, 0))
print(f"\nConv 输出: shape={list(conv_out.shape)}")
print(conv_out[0])

relu_out = F.relu(conv_out)
print(f"\nReLU 输出: shape={list(relu_out.shape)}")

flat = relu_out.flatten(start_dim=1)
print(f"\nFlatten 输出: shape={list(flat.shape)}")

logits = flat @ fc_weight + fc_bias
print(f"\nLogits: shape={list(logits.shape)}")
print(logits)

# Softmax + CrossEntropy
softmax = torch.softmax(logits, dim=1)
print(f"\nSoftmax: shape={list(softmax.shape)}")
print(softmax)

eps = 1e-10
log_softmax = torch.log(softmax + eps)
cross_entropy_per_sample = -(target * log_softmax).sum(dim=1)
loss = cross_entropy_per_sample.mean()
print(f"\n每样本 cross entropy: {cross_entropy_per_sample.tolist()}")
print(f"平均 loss: {loss.item():.8f}")

# 反向传播
loss.backward()

print("\n梯度:")
print(f"  dL/d(conv_kernel) shape={list(conv_kernel.grad.shape)}")
print(conv_kernel.grad)
print(f"\n  dL/d(conv_bias) shape={list(conv_bias.grad.shape)}")
print(conv_bias.grad)
print(f"\n  dL/d(fc_weight) shape={list(fc_weight.grad.shape)}")
# 只打印前几行
print(fc_weight.grad[:3])
print("  ...")
print(f"\n  dL/d(fc_bias) shape={list(fc_bias.grad.shape)}")
print(fc_bias.grad)

print("\n// Rust 测试常量:")
print_tensor_as_rust("TEST5_X", x)
print_tensor_as_rust("TEST5_CONV_KERNEL", torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]], [[[0.5, 0.6], [0.7, 0.8]]]]))
print_tensor_as_rust("TEST5_CONV_BIAS", torch.tensor([0.1, -0.1]))
print_tensor_as_rust("TEST5_FC_WEIGHT", fc_weight.detach())
print_tensor_as_rust("TEST5_FC_BIAS", torch.tensor([0.1, 0.2, 0.3]))
print_tensor_as_rust("TEST5_TARGET", target)
print_tensor_as_rust("TEST5_CONV_OUT", conv_out)
print_tensor_as_rust("TEST5_RELU_OUT", relu_out)
print_tensor_as_rust("TEST5_FLAT", flat)
print_tensor_as_rust("TEST5_LOGITS", logits)
print_tensor_as_rust("TEST5_SOFTMAX", softmax)
print_f32_as_rust("TEST5_LOSS", loss.item())
print_tensor_as_rust("TEST5_GRAD_CONV_KERNEL", conv_kernel.grad)
print_tensor_as_rust("TEST5_GRAD_CONV_BIAS", conv_bias.grad)
print_tensor_as_rust("TEST5_GRAD_FC_WEIGHT", fc_weight.grad)
print_tensor_as_rust("TEST5_GRAD_FC_BIAS", fc_bias.grad)


# ==================== 测试 6: 使用 nn.Conv2d 验证 ====================
print("\n" + "=" * 70)
print("测试 6: 使用 nn.Conv2d 验证（确认权重格式正确）")
print("=" * 70)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, bias=True)

# 设置与测试 4 相同的权重
with torch.no_grad():
    conv.weight.copy_(torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]]))
    conv.bias.copy_(torch.tensor([0.5]))

x = torch.tensor([
    [[[1.0, 2.0, 3.0],
      [4.0, 5.0, 6.0],
      [7.0, 8.0, 9.0]]]
], dtype=torch.float32)

output_nn = conv(x)
print(f"nn.Conv2d 输出:")
print(output_nn)

# 使用 F.conv2d 验证
kernel = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
bias = torch.tensor([0.5])
output_f = F.conv2d(x, kernel, bias=bias)
print(f"\nF.conv2d 输出:")
print(output_f)

assert torch.allclose(output_nn, output_f), "nn.Conv2d 和 F.conv2d 不一致！"
print("\n✅ nn.Conv2d 输出与 F.conv2d 一致")


print("\n" + "=" * 70)
print("完成！将上述常量复制到 Rust 测试文件中")
print("=" * 70)


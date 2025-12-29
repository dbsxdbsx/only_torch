#!/usr/bin/env python3
"""
AvgPool2d / MaxPool2d 层的 PyTorch 参考值生成脚本

用于生成 Rust 测试中的预期数值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def print_tensor_as_rust(name: str, tensor: torch.Tensor, precision: int = 8):
    """将 PyTorch tensor 格式化为 Rust 常量"""
    flat = tensor.flatten().tolist()
    formatted = ", ".join([f"{v:.{precision}f}" for v in flat])
    print(f"const {name}: &[f32] = &[{formatted}];")
    print(f"// shape: {list(tensor.shape)}")
    print()


print("=" * 60)
print("测试 1: AvgPool2d 简单前向传播")
print("=" * 60)

# 输入: [batch=1, C=1, H=4, W=4]
x1 = torch.tensor([
    [[[1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]]]
], dtype=torch.float32)

avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
out1 = avg_pool(x1)

print("输入形状:", x1.shape)
print("输出形状:", out1.shape)
print_tensor_as_rust("TEST1_AVG_X", x1)
print_tensor_as_rust("TEST1_AVG_OUTPUT", out1)

print("=" * 60)
print("测试 2: MaxPool2d 简单前向传播")
print("=" * 60)

max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
out2 = max_pool(x1)

print("输入形状:", x1.shape)
print("输出形状:", out2.shape)
print_tensor_as_rust("TEST2_MAX_X", x1)
print_tensor_as_rust("TEST2_MAX_OUTPUT", out2)

print("=" * 60)
print("测试 3: AvgPool2d 多通道多批次")
print("=" * 60)

torch.manual_seed(42)
x3 = torch.randn(2, 3, 4, 4)
avg_pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
out3 = avg_pool3(x3)

print("输入形状:", x3.shape)
print("输出形状:", out3.shape)
print_tensor_as_rust("TEST3_AVG_X", x3)
print_tensor_as_rust("TEST3_AVG_OUTPUT", out3)

print("=" * 60)
print("测试 4: MaxPool2d 多通道多批次")
print("=" * 60)

out4 = max_pool(x3)

print("输入形状:", x3.shape)
print("输出形状:", out4.shape)
print_tensor_as_rust("TEST4_MAX_X", x3)
print_tensor_as_rust("TEST4_MAX_OUTPUT", out4)

print("=" * 60)
print("测试 5: AvgPool2d 反向传播（单层 + MSE Loss）")
print("=" * 60)

torch.manual_seed(42)
x5 = torch.tensor([
    [[[1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]]]
], dtype=torch.float32, requires_grad=True)

avg_pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
out5 = avg_pool5(x5)

# MSE 目标
target5 = torch.tensor([[[[5.0, 7.0], [11.0, 15.0]]]], dtype=torch.float32)
loss5 = F.mse_loss(out5, target5)

loss5.backward()

print("输入形状:", x5.shape)
print("输出形状:", out5.shape)
print_tensor_as_rust("TEST5_AVG_X", x5.detach())
print_tensor_as_rust("TEST5_AVG_TARGET", target5)
print_tensor_as_rust("TEST5_AVG_OUTPUT", out5.detach())
print(f"const TEST5_AVG_LOSS: f32 = {loss5.item():.8f};")
print()
print_tensor_as_rust("TEST5_AVG_GRAD_X", x5.grad)

print("=" * 60)
print("测试 6: MaxPool2d 反向传播（单层 + MSE Loss）")
print("=" * 60)

torch.manual_seed(42)
x6 = torch.tensor([
    [[[1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
      [13.0, 14.0, 15.0, 16.0]]]
], dtype=torch.float32, requires_grad=True)

max_pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
out6 = max_pool6(x6)

# MSE 目标
target6 = torch.tensor([[[[5.0, 7.0], [13.0, 15.0]]]], dtype=torch.float32)
loss6 = F.mse_loss(out6, target6)

loss6.backward()

print("输入形状:", x6.shape)
print("输出形状:", out6.shape)
print_tensor_as_rust("TEST6_MAX_X", x6.detach())
print_tensor_as_rust("TEST6_MAX_TARGET", target6)
print_tensor_as_rust("TEST6_MAX_OUTPUT", out6.detach())
print(f"const TEST6_MAX_LOSS: f32 = {loss6.item():.8f};")
print()
print_tensor_as_rust("TEST6_MAX_GRAD_X", x6.grad)

print("=" * 60)
print("测试 7: Conv2d -> AvgPool2d -> Linear 完整网络反向传播")
print("=" * 60)

torch.manual_seed(42)

# 输入: [batch=2, C=1, H=4, W=4]
x7 = torch.randn(2, 1, 4, 4, requires_grad=True)

# 网络参数
conv7 = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, bias=True)
# 固定权重
torch.manual_seed(123)
conv7.weight.data = torch.randn_like(conv7.weight.data)
conv7.bias.data = torch.randn_like(conv7.bias.data)

avg_pool7 = nn.AvgPool2d(kernel_size=2, stride=2)

# conv输出: [2, 2, 3, 3], pool输出: [2, 2, 1, 1]
# flatten: [2, 2]
fc7 = nn.Linear(2, 3, bias=True)
torch.manual_seed(456)
fc7.weight.data = torch.randn_like(fc7.weight.data)
fc7.bias.data = torch.randn_like(fc7.bias.data)

# 前向
conv_out7 = conv7(x7)
relu_out7 = F.relu(conv_out7)
pool_out7 = avg_pool7(relu_out7)
flat7 = pool_out7.view(2, -1)  # [2, 2]
logits7 = fc7(flat7)
softmax7 = F.softmax(logits7, dim=1)

# 目标 (one-hot)
target7 = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
loss7 = F.cross_entropy(logits7, target7.argmax(dim=1))

loss7.backward()

print("输入形状:", x7.shape)
print("Conv输出形状:", conv_out7.shape)
print("Pool输出形状:", pool_out7.shape)
print("Logits形状:", logits7.shape)
print()
print_tensor_as_rust("TEST7_X", x7.detach())
print_tensor_as_rust("TEST7_CONV_WEIGHT", conv7.weight.detach())
print_tensor_as_rust("TEST7_CONV_BIAS", conv7.bias.detach())
print_tensor_as_rust("TEST7_FC_WEIGHT", fc7.weight.detach().T)  # 转置以匹配 Rust
print_tensor_as_rust("TEST7_FC_BIAS", fc7.bias.detach())
print_tensor_as_rust("TEST7_TARGET", target7)
print()
print_tensor_as_rust("TEST7_CONV_OUT", conv_out7.detach())
print_tensor_as_rust("TEST7_RELU_OUT", relu_out7.detach())
print_tensor_as_rust("TEST7_POOL_OUT", pool_out7.detach())
print_tensor_as_rust("TEST7_FLAT", flat7.detach())
print_tensor_as_rust("TEST7_LOGITS", logits7.detach())
print_tensor_as_rust("TEST7_SOFTMAX", softmax7.detach())
print(f"const TEST7_LOSS: f32 = {loss7.item():.8f};")
print()
print_tensor_as_rust("TEST7_GRAD_CONV_WEIGHT", conv7.weight.grad)
print_tensor_as_rust("TEST7_GRAD_CONV_BIAS", conv7.bias.grad)
print_tensor_as_rust("TEST7_GRAD_FC_WEIGHT", fc7.weight.grad.T)

print("=" * 60)
print("测试 8: Conv2d -> MaxPool2d -> Linear 完整网络反向传播")
print("=" * 60)

torch.manual_seed(42)

# 重置梯度和输入
x8 = torch.randn(2, 1, 4, 4, requires_grad=True)

# 重新初始化网络
conv8 = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0, bias=True)
torch.manual_seed(123)
conv8.weight.data = torch.randn_like(conv8.weight.data)
conv8.bias.data = torch.randn_like(conv8.bias.data)

max_pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

fc8 = nn.Linear(2, 3, bias=True)
torch.manual_seed(456)
fc8.weight.data = torch.randn_like(fc8.weight.data)
fc8.bias.data = torch.randn_like(fc8.bias.data)

# 前向
conv_out8 = conv8(x8)
relu_out8 = F.relu(conv_out8)
pool_out8 = max_pool8(relu_out8)
flat8 = pool_out8.view(2, -1)
logits8 = fc8(flat8)
softmax8 = F.softmax(logits8, dim=1)

target8 = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.float32)
loss8 = F.cross_entropy(logits8, target8.argmax(dim=1))

loss8.backward()

print("输入形状:", x8.shape)
print("Pool输出形状:", pool_out8.shape)
print()
print_tensor_as_rust("TEST8_X", x8.detach())
print_tensor_as_rust("TEST8_CONV_WEIGHT", conv8.weight.detach())
print_tensor_as_rust("TEST8_CONV_BIAS", conv8.bias.detach())
print_tensor_as_rust("TEST8_FC_WEIGHT", fc8.weight.detach().T)
print_tensor_as_rust("TEST8_FC_BIAS", fc8.bias.detach())
print_tensor_as_rust("TEST8_TARGET", target8)
print()
print_tensor_as_rust("TEST8_CONV_OUT", conv_out8.detach())
print_tensor_as_rust("TEST8_RELU_OUT", relu_out8.detach())
print_tensor_as_rust("TEST8_POOL_OUT", pool_out8.detach())
print_tensor_as_rust("TEST8_FLAT", flat8.detach())
print_tensor_as_rust("TEST8_LOGITS", logits8.detach())
print_tensor_as_rust("TEST8_SOFTMAX", softmax8.detach())
print(f"const TEST8_LOSS: f32 = {loss8.item():.8f};")
print()
print_tensor_as_rust("TEST8_GRAD_CONV_WEIGHT", conv8.weight.grad)
print_tensor_as_rust("TEST8_GRAD_CONV_BIAS", conv8.bias.grad)
print_tensor_as_rust("TEST8_GRAD_FC_WEIGHT", fc8.weight.grad.T)

print("=" * 60)
print("脚本执行完成！")
print("=" * 60)


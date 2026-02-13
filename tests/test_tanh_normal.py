"""
TanhNormal 分布（Squashed Gaussian）Python 参考实现
用于生成 Rust 测试的预期值（PyTorch 对照）

TanhNormal: a = tanh(u), u ~ Normal(mean, std)
log_prob(u) = Normal.log_prob(u) - log(1 - tanh²(u) + eps)

运行: python tests/test_tanh_normal.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.distributions import Normal

EPS = 1e-6

print("=" * 60)
print("TanhNormal Distribution Reference Values")
print("=" * 60)

# ==================== Test 1: 基本 log_prob ====================
print("\n--- Test 1: Basic log_prob ---")
mean1 = torch.tensor([[0.0, 1.0]])
std1 = torch.tensor([[1.0, 0.5]])
u1 = torch.tensor([[0.5, -0.3]])  # raw action

base_lp1 = Normal(mean1, std1).log_prob(u1)
tanh_u1 = torch.tanh(u1)
correction1 = torch.log(1 - tanh_u1.pow(2) + EPS)
log_prob1 = base_lp1 - correction1

print(f"  u        = [{u1[0,0].item():.6f}, {u1[0,1].item():.6f}]")
print(f"  tanh(u)  = [{tanh_u1[0,0].item():.8f}, {tanh_u1[0,1].item():.8f}]")
print(f"  base_lp  = [{base_lp1[0,0].item():.8f}, {base_lp1[0,1].item():.8f}]")
print(f"  correct  = [{correction1[0,0].item():.8f}, {correction1[0,1].item():.8f}]")
print(f"  log_prob = [{log_prob1[0,0].item():.8f}, {log_prob1[0,1].item():.8f}]")

# ==================== Test 2: 标准正态 TanhNormal ====================
print("\n--- Test 2: Standard Normal TanhNormal ---")
mean2 = torch.tensor([[0.0]])
std2 = torch.tensor([[1.0]])
u2 = torch.tensor([[0.0]])

base_lp2 = Normal(mean2, std2).log_prob(u2)
tanh_u2 = torch.tanh(u2)
correction2 = torch.log(1 - tanh_u2.pow(2) + EPS)
log_prob2 = base_lp2 - correction2

print(f"  log_prob(u=0) = {log_prob2.item():.8f}")

# ==================== Test 3: log_prob 梯度 ====================
print("\n--- Test 3: log_prob Gradients ---")
mean3 = torch.tensor([[0.0, 1.0]], requires_grad=True)
std3 = torch.tensor([[1.0, 0.5]], requires_grad=True)
u3 = torch.tensor([[0.5, -0.3]])

base_lp3 = Normal(mean3, std3).log_prob(u3)
tanh_u3 = torch.tanh(u3)
correction3 = torch.log(1 - tanh_u3.pow(2) + EPS)
log_prob3 = base_lp3 - correction3

loss3 = log_prob3.sum()
loss3.backward()

print(f"  loss      = {loss3.item():.8f}")
print(f"  mean.grad = [{mean3.grad[0,0].item():.8f}, {mean3.grad[0,1].item():.8f}]")
print(f"  std.grad  = [{std3.grad[0,0].item():.8f}, {std3.grad[0,1].item():.8f}]")

# ==================== Test 4: Batch 测试 ====================
print("\n--- Test 4: Batch (batch=2, dim=2) ---")
mean4 = torch.tensor([[0.0, 1.0], [-1.0, 0.5]])
std4 = torch.tensor([[1.0, 0.5], [0.3, 2.0]])
u4 = torch.tensor([[0.5, -0.3], [0.8, -1.5]])

base_lp4 = Normal(mean4, std4).log_prob(u4)
tanh_u4 = torch.tanh(u4)
correction4 = torch.log(1 - tanh_u4.pow(2) + EPS)
log_prob4 = base_lp4 - correction4

print(f"  log_prob:")
for i in range(2):
    print(f"    [{log_prob4[i,0].item():.8f}, {log_prob4[i,1].item():.8f}]")

# ==================== Test 5: rsample + log_prob 端到端 ====================
print("\n--- Test 5: SAC-style rsample + log_prob ---")
torch.manual_seed(42)
mean5 = torch.tensor([[0.0, 0.5]], requires_grad=True)
std5 = torch.tensor([[1.0, 0.3]], requires_grad=True)

dist5 = Normal(mean5, std5)
u5 = dist5.rsample()
a5 = torch.tanh(u5)

base_lp5 = dist5.log_prob(u5)
correction5 = torch.log(1 - a5.pow(2) + EPS)
log_prob5 = (base_lp5 - correction5).sum(dim=-1, keepdim=True)

print(f"  u       = [{u5[0,0].item():.8f}, {u5[0,1].item():.8f}]")
print(f"  action  = [{a5[0,0].item():.8f}, {a5[0,1].item():.8f}]")
print(f"  log_prob_sum = {log_prob5.item():.8f}")

# 梯度（用于验证链路可通）
loss5 = -log_prob5.mean()
loss5.backward()
print(f"  loss = {loss5.item():.8f}")
print(f"  mean.grad = [{mean5.grad[0,0].item():.8f}, {mean5.grad[0,1].item():.8f}]")
print(f"  std.grad  = [{std5.grad[0,0].item():.8f}, {std5.grad[0,1].item():.8f}]")

print("\n" + "=" * 60)
print("All reference values generated successfully.")

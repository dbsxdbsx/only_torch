"""
Normal 分布 Python 参考实现
用于生成 Rust 测试的预期值（PyTorch 对照）

运行: python tests/test_normal.py
"""

import torch
from torch.distributions import Normal

print("=" * 60)
print("Normal Distribution Reference Values")
print("=" * 60)

# ==================== Test 1: 标准正态分布 ====================
print("\n--- Test 1: Standard Normal (μ=0, σ=1) ---")
dist1 = Normal(torch.tensor([[0.0]]), torch.tensor([[1.0]]))
value1 = torch.tensor([[0.0]])

lp1 = dist1.log_prob(value1)
ent1 = dist1.entropy()
print(f"  log_prob(0) = {lp1.item():.8f}")
print(f"  entropy     = {ent1.item():.8f}")

# ==================== Test 2: 一般正态分布 ====================
print("\n--- Test 2: General Normal (μ=[1,2], σ=[0.5,1]) ---")
mean2 = torch.tensor([[1.0, 2.0]], requires_grad=True)
std2 = torch.tensor([[0.5, 1.0]], requires_grad=True)
dist2 = Normal(mean2, std2)
value2 = torch.tensor([[1.2, 2.5]])

lp2 = dist2.log_prob(value2)
ent2 = dist2.entropy()
print(f"  log_prob = [{lp2[0,0].item():.8f}, {lp2[0,1].item():.8f}]")
print(f"  entropy  = [{ent2[0,0].item():.8f}, {ent2[0,1].item():.8f}]")

# ==================== Test 3: log_prob 梯度 ====================
print("\n--- Test 3: log_prob Gradients ---")
mean3 = torch.tensor([[1.0, 2.0]], requires_grad=True)
std3 = torch.tensor([[0.5, 1.0]], requires_grad=True)
dist3 = Normal(mean3, std3)
value3 = torch.tensor([[1.2, 2.5]])

lp3 = dist3.log_prob(value3)
loss3 = lp3.sum()
loss3.backward()
print(f"  loss     = {loss3.item():.8f}")
print(f"  mean.grad = [{mean3.grad[0,0].item():.8f}, {mean3.grad[0,1].item():.8f}]")
print(f"  std.grad  = [{std3.grad[0,0].item():.8f}, {std3.grad[0,1].item():.8f}]")

# ==================== Test 4: entropy 梯度 ====================
print("\n--- Test 4: entropy Gradients ---")
mean4 = torch.tensor([[1.0, 2.0]], requires_grad=True)
std4 = torch.tensor([[0.5, 1.0]], requires_grad=True)
dist4 = Normal(mean4, std4)

ent4 = dist4.entropy()
ent_loss4 = ent4.sum()
ent_loss4.backward()
print(f"  entropy_sum  = {ent_loss4.item():.8f}")
print(f"  mean.grad    = {mean4.grad}")  # entropy 不依赖 mean，应为 None 或 zero
print(f"  std.grad     = [{std4.grad[0,0].item():.8f}, {std4.grad[0,1].item():.8f}]")

# ==================== Test 5: batch 测试 ====================
print("\n--- Test 5: Batch (batch=3, dim=2) ---")
mean5 = torch.tensor([[0.0, 1.0], [2.0, 3.0], [-1.0, 0.5]])
std5 = torch.tensor([[1.0, 0.5], [2.0, 0.1], [0.3, 1.5]])
dist5 = Normal(mean5, std5)
value5 = torch.tensor([[0.5, 1.2], [1.0, 3.1], [-0.5, 0.0]])

lp5 = dist5.log_prob(value5)
ent5 = dist5.entropy()
print(f"  log_prob:")
for i in range(3):
    print(f"    [{lp5[i,0].item():.8f}, {lp5[i,1].item():.8f}]")
print(f"  entropy:")
for i in range(3):
    print(f"    [{ent5[i,0].item():.8f}, {ent5[i,1].item():.8f}]")

print("\n" + "=" * 60)
print("All reference values generated successfully.")

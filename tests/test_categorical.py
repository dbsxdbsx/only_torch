"""
Categorical 分布 Python 参考实现
用于生成 Rust 测试的预期值（PyTorch 对照）

运行: python tests/test_categorical.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.distributions import Categorical

print("=" * 60)
print("Categorical Distribution Reference Values")
print("=" * 60)

# ==================== Test 1: 基本 log_prob ====================
print("\n--- Test 1: Basic log_prob ---")
logits1 = torch.tensor([[1.0, 2.0, 0.5]])
dist1 = Categorical(logits=logits1)

# log_softmax of logits
log_probs1 = torch.nn.functional.log_softmax(logits1, dim=-1)
probs1 = torch.nn.functional.softmax(logits1, dim=-1)
print(f"  logits    = {logits1[0].tolist()}")
print(f"  probs     = [{probs1[0,0].item():.8f}, {probs1[0,1].item():.8f}, {probs1[0,2].item():.8f}]")
print(f"  log_probs = [{log_probs1[0,0].item():.8f}, {log_probs1[0,1].item():.8f}, {log_probs1[0,2].item():.8f}]")

for a in range(3):
    lp = dist1.log_prob(torch.tensor([a]))
    print(f"  log_prob(a={a}) = {lp.item():.8f}")

entropy1 = dist1.entropy()
print(f"  entropy = {entropy1.item():.8f}")

# ==================== Test 2: 均匀分布 ====================
print("\n--- Test 2: Uniform (equal logits) ---")
logits2 = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
dist2 = Categorical(logits=logits2)
print(f"  log_prob(a=0) = {dist2.log_prob(torch.tensor([0])).item():.8f}")
print(f"  entropy = {dist2.entropy().item():.8f}")

# ==================== Test 3: log_prob 梯度 ====================
print("\n--- Test 3: log_prob Gradients ---")
logits3 = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
dist3 = Categorical(logits=logits3)
action3 = torch.tensor([1])  # 选择 action=1

lp3 = dist3.log_prob(action3)
lp3.backward()
print(f"  log_prob(a=1) = {lp3.item():.8f}")
print(f"  logits.grad = [{logits3.grad[0,0].item():.8f}, {logits3.grad[0,1].item():.8f}, {logits3.grad[0,2].item():.8f}]")

# ==================== Test 4: entropy 梯度 ====================
print("\n--- Test 4: entropy Gradients ---")
logits4 = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
dist4 = Categorical(logits=logits4)

ent4 = dist4.entropy()
ent4.backward()
print(f"  entropy = {ent4.item():.8f}")
print(f"  logits.grad = [{logits4.grad[0,0].item():.8f}, {logits4.grad[0,1].item():.8f}, {logits4.grad[0,2].item():.8f}]")

# ==================== Test 5: Batch 测试 ====================
print("\n--- Test 5: Batch (batch=3) ---")
logits5 = torch.tensor([[1.0, 2.0, 0.5], [0.0, 0.0, 0.0], [-1.0, 3.0, 0.0]])
dist5 = Categorical(logits=logits5)
actions5 = torch.tensor([0, 2, 1])

lp5 = dist5.log_prob(actions5)
ent5 = dist5.entropy()
print(f"  log_prob = [{lp5[0].item():.8f}, {lp5[1].item():.8f}, {lp5[2].item():.8f}]")
print(f"  entropy  = [{ent5[0].item():.8f}, {ent5[1].item():.8f}, {ent5[2].item():.8f}]")

print("\n" + "=" * 60)
print("All reference values generated successfully.")

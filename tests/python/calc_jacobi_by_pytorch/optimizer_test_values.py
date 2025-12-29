"""
验证 optimizer 单元测试的预期值
=================================
此脚本用于生成 Rust 单元测试 `src/nn/tests/optimizer.rs` 中的预期值。

计算图结构：
  output = w @ x
  loss_input = label @ output
  loss = perception_loss(loss_input)  # if x < 0: -x, else: 0

验证的测试:
  - test_sgd_update_formula
  - test_sgd_gradient_accumulation
  - test_adam_update

运行方式: python tests/calc_jacobi_by_pytorch/optimizer_test_values.py
"""

import torch
import torch.optim as optim


def perception_loss(x):
    """PerceptionLoss: x >= 0 时为 0，x < 0 时为 -x (与 Rust 实现一致)"""
    return torch.where(x >= 0, torch.zeros_like(x), -x)


def compute_loss(w, x, label):
    """计算 loss = perception_loss(label * (w * x))"""
    output = w @ x
    loss_input = label @ output
    loss = perception_loss(loss_input)
    return loss


print("=" * 60)
print("Test 1: test_sgd_update_formula")
print("=" * 60)
# 初始值：w=1, x=1, label=-1
w = torch.tensor([[1.0]], requires_grad=True)
x = torch.tensor([[1.0]])
label = torch.tensor([[-1.0]])
lr = 0.1

# 前向传播
output = w @ x  # = 1.0
loss_input = label @ output  # = -1.0
loss = perception_loss(loss_input)  # = 1.0 (因为 -1 < 0)

print(f"w_init = {w.item()}")
print(f"x = {x.item()}")
print(f"label = {label.item()}")
print(f"output = w @ x = {output.item()}")
print(f"loss_input = label @ output = {loss_input.item()}")
print(f"loss = perception_loss(loss_input) = {loss.item()}")

# 反向传播
loss.backward()
print(f"\n梯度 d(loss)/d(w) = {w.grad.item()}")

# SGD 更新: w_new = w_old - lr * grad
w_new = w.item() - lr * w.grad.item()
print(f"\nSGD 更新 (lr={lr}):")
print(f"w_new = w_old - lr * grad = {w.item()} - {lr} * {w.grad.item()} = {w_new}")


print("\n" + "=" * 60)
print("Test 2: test_sgd_gradient_accumulation (3次累积)")
print("=" * 60)
# 初始值：w=1, x=2, label=-1
w = torch.tensor([[1.0]], requires_grad=True)
x = torch.tensor([[2.0]])
label = torch.tensor([[-1.0]])
lr = 0.1

# 模拟 3 次 one_step（梯度累积）
total_grad = 0.0
for i in range(3):
    # 每次迭代需要清零梯度
    if w.grad is not None:
        w.grad.zero_()

    output = w @ x
    loss_input = label @ output
    loss = perception_loss(loss_input)
    loss.backward()

    print(f"Iteration {i + 1}: loss_input={loss_input.item()}, grad={w.grad.item()}")
    total_grad += w.grad.item()

avg_grad = total_grad / 3
print(f"\n累积梯度总和 = {total_grad}")
print(f"平均梯度 = {avg_grad}")

# SGD 更新使用平均梯度
w_new = 1.0 - lr * avg_grad
print(f"\nSGD 更新 (lr={lr}):")
print(f"w_new = w_old - lr * avg_grad = 1.0 - {lr} * {avg_grad} = {w_new}")


print("\n" + "=" * 60)
print("Test 3: test_adam_update")
print("=" * 60)
# 初始值：w=2, x=3, label=-1
w = torch.tensor([[2.0]], requires_grad=True)
x = torch.tensor([[3.0]])
label = torch.tensor([[-1.0]])
lr = 0.1
beta1, beta2, eps = 0.9, 0.999, 1e-8

# 创建 Adam 优化器
optimizer = optim.Adam([w], lr=lr, betas=(beta1, beta2), eps=eps)

print(f"w_init = {w.item()}")
print(f"x = {x.item()}")
print(f"label = {label.item()}")

# 前向传播
output = w @ x  # = 6.0
loss_input = label @ output  # = -6.0
loss = perception_loss(loss_input)  # = 6.0

print(f"output = w @ x = {output.item()}")
print(f"loss_input = label @ output = {loss_input.item()}")
print(f"loss = perception_loss(loss_input) = {loss.item()}")

# 反向传播
loss.backward()
print(f"\n梯度 d(loss)/d(w) = {w.grad.item()}")

# Adam 更新
optimizer.step()
print(f"\nAdam 更新后 w_new = {w.item()}")


print("\n" + "=" * 60)
print("总结：用于 Rust 单元测试的预期值")
print("=" * 60)
print("""
test_sgd_update_formula:
  - w_init = 1.0, x = 1.0, label = -1.0, lr = 0.1
  - expected w_new = 0.9

test_sgd_gradient_accumulation:
  - w_init = 1.0, x = 2.0, label = -1.0, lr = 0.1, iterations = 3
  - expected w_new = 0.8

test_adam_update:
  - w_init = 2.0, x = 3.0, label = -1.0, lr = 0.1
  - expected w_new ≈ 1.9 (Adam 第一步更新约等于 lr)
""")

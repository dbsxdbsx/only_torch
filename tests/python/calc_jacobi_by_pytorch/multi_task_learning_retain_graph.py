"""
多任务学习场景测试 - 两个 loss 共享 backbone，使用 retain_graph

对应 Rust 测试: test_retain_graph_multi_task_learning

网络结构:
    x [4,1] -> w_shared [2,4] -> features [2,1]
                                    |
                    +---------------+---------------+
                    |                               |
             w1 [1,2] -> out1 [1,1]          w2 [1,2] -> out2 [1,1]
             (task 1)                        (task 2)
"""

import torch


def main():
    torch.manual_seed(42)

    # 创建与 Rust 测试相同的输入
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)  # [4, 1]

    # 参数使用默认初始化（Rust 中默认是全 1）
    w_shared = torch.ones(2, 4, requires_grad=True)  # [2, 4]
    w1 = torch.ones(1, 2, requires_grad=True)  # [1, 2]
    w2 = torch.ones(1, 2, requires_grad=True)  # [1, 2]

    print("=" * 60)
    print("输入数据:")
    print(f"x [{x.shape}]:\n{x}")
    print(f"\nw_shared [{w_shared.shape}]:\n{w_shared}")
    print(f"\nw1 [{w1.shape}]:\n{w1}")
    print(f"\nw2 [{w2.shape}]:\n{w2}")

    # 前向传播
    features = torch.matmul(w_shared, x)  # [2, 4] @ [4, 1] = [2, 1]
    out1 = torch.matmul(w1, features)  # [1, 2] @ [2, 1] = [1, 1]
    out2 = torch.matmul(w2, features)  # [1, 2] @ [2, 1] = [1, 1]

    print("\n" + "=" * 60)
    print("前向传播结果:")
    print(f"features [{features.shape}]:\n{features}")
    print(f"\nout1 [{out1.shape}]:\n{out1}")
    print(f"\nout2 [{out2.shape}]:\n{out2}")

    # ========== 任务 1 backward (retain_graph=True) ==========
    # 注意：PyTorch 的 backward 默认累积梯度，所以我们手动清零
    if w_shared.grad is not None:
        w_shared.grad.zero_()
    if w1.grad is not None:
        w1.grad.zero_()

    # 对 out1 做 backward，使用 retain_graph=True
    # 因为 out1 是标量，可以直接 backward
    out1.backward(retain_graph=True)

    print("\n" + "=" * 60)
    print("任务 1 backward 后 (retain_graph=True):")
    print(f"w_shared.grad [{w_shared.grad.shape}]:\n{w_shared.grad}")
    print(f"\nw1.grad [{w1.grad.shape}]:\n{w1.grad}")
    print(f"\nw2.grad: {w2.grad}")  # 应该是 None

    # 保存 w_shared 第一次的梯度
    w_shared_grad_after_task1 = w_shared.grad.clone()

    # ========== 任务 2 backward (retain_graph=False，梯度累积) ==========
    # 注意：不清零梯度，让梯度累积
    out2.backward(retain_graph=False)

    print("\n" + "=" * 60)
    print("任务 2 backward 后 (梯度累积):")
    print(f"w_shared.grad [{w_shared.grad.shape}]:\n{w_shared.grad}")
    print(f"\nw2.grad [{w2.grad.shape}]:\n{w2.grad}")

    print("\n" + "=" * 60)
    print("梯度分析:")
    print(f"w_shared 第一次梯度 (仅 task1):\n{w_shared_grad_after_task1}")
    print(f"\nw_shared 累积后梯度 (task1 + task2):\n{w_shared.grad}")
    print(f"\n验证: task1 贡献 + task2 贡献 = 累积梯度")

    # ========== 生成 Rust 测试用的精确数值 ==========
    print("\n" + "=" * 60)
    print("Rust 测试用数据 (可直接复制):")
    print("\n// 前向传播期望值")
    print(f"// features: {features.detach().numpy().flatten().tolist()}")
    print(f"// out1: {out1.detach().numpy().flatten().tolist()}")
    print(f"// out2: {out2.detach().numpy().flatten().tolist()}")

    print("\n// 任务 1 backward 后的梯度")
    print(f"// w_shared_grad_task1: {w_shared_grad_after_task1.numpy().flatten().tolist()}")
    print(f"// w1_grad: {w1.grad.numpy().flatten().tolist()}")

    # 为了获取 w2 单独的梯度，需要重新计算
    # 重置并单独计算 task2 的梯度
    x2 = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)
    w_shared2 = torch.ones(2, 4, requires_grad=True)
    w2_alone = torch.ones(1, 2, requires_grad=True)
    features2 = torch.matmul(w_shared2, x2)
    out2_alone = torch.matmul(w2_alone, features2)
    out2_alone.backward()

    print(f"// w2_grad (单独): {w2_alone.grad.numpy().flatten().tolist()}")
    print(f"// w_shared_grad_task2 (单独): {w_shared2.grad.numpy().flatten().tolist()}")

    print("\n// 累积后的 w_shared 梯度 (task1 + task2)")
    print(f"// w_shared_grad_accumulated: {w_shared.grad.numpy().flatten().tolist()}")

    # 验证累积是否正确
    expected_accumulated = w_shared_grad_after_task1 + w_shared2.grad
    print(f"\n// 验证: task1_grad + task2_grad = {expected_accumulated.numpy().flatten().tolist()}")
    assert torch.allclose(w_shared.grad, expected_accumulated), "梯度累积验证失败!"
    print("// ✓ 梯度累积验证通过!")


if __name__ == "__main__":
    main()


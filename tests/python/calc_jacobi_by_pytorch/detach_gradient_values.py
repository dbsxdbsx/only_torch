"""
detach 机制梯度数值对照测试

验证 detach 阻断梯度后，下游参数的梯度数值是否正确。
对应 Rust 测试: src/nn/tests/gradient_flow_control.rs::test_detach_gradient_values_match_pytorch

拓扑结构:
    x(input) -> w1 -> h -> w2 -> output
                      ^
                      | (detach 点)

当 h 被 detach 时:
- w1 不应有梯度 (被阻断)
- w2 应有梯度 (在 detach 点之后)
"""

import torch


def test_detach_gradient_values():
    """计算 detach 场景下各参数的精确梯度值"""
    print("=" * 60)
    print("detach 机制梯度数值对照测试")
    print("=" * 60)

    # 输入: [2, 1] 形状
    x = torch.tensor([[1.0], [2.0]], requires_grad=False)

    # w1: [2, 2] -> h: [2, 1]
    w1 = torch.ones(2, 2, requires_grad=True)

    # w2: [1, 2] -> output: [1, 1]
    w2 = torch.ones(1, 2, requires_grad=True)

    print(f"\n输入 x:\n{x}")
    print(f"x.shape: {x.shape}")
    print(f"\nw1 (初始值全1):\n{w1}")
    print(f"w1.shape: {w1.shape}")
    print(f"\nw2 (初始值全1):\n{w2}")
    print(f"w2.shape: {w2.shape}")

    # ========== 场景 1: 无 detach (对照组) ==========
    print("\n" + "=" * 60)
    print("场景 1: 无 detach (对照组)")
    print("=" * 60)

    h_no_detach = torch.matmul(w1, x)
    output_no_detach = torch.matmul(w2, h_no_detach)

    print(f"\nh = w1 @ x:\n{h_no_detach}")
    print(f"h.shape: {h_no_detach.shape}")
    print(f"\noutput = w2 @ h:\n{output_no_detach}")
    print(f"output.shape: {output_no_detach.shape}")

    # 对 output 求和后 backward (模拟 loss)
    output_no_detach.sum().backward()

    assert w1.grad is not None and w2.grad is not None
    print(f"\nw1.grad (无 detach):\n{w1.grad}")
    print(f"w1.grad.shape: {w1.grad.shape}")
    print(f"\nw2.grad (无 detach):\n{w2.grad}")
    print(f"w2.grad.shape: {w2.grad.shape}")

    # 保存无 detach 的梯度
    w1_grad_no_detach = w1.grad.clone()
    w2_grad_no_detach = w2.grad.clone()

    # 清除梯度
    w1.grad.zero_()
    w2.grad.zero_()

    # ========== 场景 2: 有 detach ==========
    print("\n" + "=" * 60)
    print("场景 2: 有 detach (h.detach())")
    print("=" * 60)

    h_with_detach = torch.matmul(w1, x)
    h_detached = h_with_detach.detach()  # 关键: detach!
    output_with_detach = torch.matmul(w2, h_detached)

    print(f"\nh = w1 @ x:\n{h_with_detach}")
    print(f"h_detached (detach 后的值相同):\n{h_detached}")
    print(f"\noutput = w2 @ h_detached:\n{output_with_detach}")

    # backward
    output_with_detach.sum().backward()

    print(f"\nw1.grad (有 detach): {w1.grad}")
    print("  -> 应为 None 或全 0 (梯度被阻断)")
    assert w2.grad is not None
    print(f"\nw2.grad (有 detach):\n{w2.grad}")
    print(f"w2.grad.shape: {w2.grad.shape}")

    # ========== 输出 Rust 测试用的期望值 ==========
    print("\n" + "=" * 60)
    print("Rust 测试用的期望值 (Jacobi 格式: 展平为 [1, n])")
    print("=" * 60)

    # w2 的梯度应该和无 detach 时相同
    # 因为 h 的值没变，只是梯度不再回流到 w1
    print("\n// 场景 1 (无 detach) - 供对照")
    print(f"// w1_grad: {w1_grad_no_detach.flatten().tolist()}")
    print(f"// w2_grad: {w2_grad_no_detach.flatten().tolist()}")

    print("\n// 场景 2 (有 detach) - Rust 测试期望值")
    print("// w1_grad: None (被 detach 阻断)")
    print(f"// w2_grad (Jacobi 格式 [1, 2]): {w2.grad.flatten().tolist()}")

    # 验证 w2 梯度在两种场景下相同
    assert torch.allclose(w2.grad, w2_grad_no_detach), "w2 梯度应该相同!"
    print("\n✓ 验证通过: w2 梯度在 detach 前后相同")

    # ========== 生成 Rust 代码片段 ==========
    print("\n" + "=" * 60)
    print("Rust 测试代码片段")
    print("=" * 60)

    w2_grad_list = w2.grad.flatten().tolist()
    print(f"""
    // PyTorch 计算的期望值
    // w1: 被 detach 阻断，应为 None
    // w2: 梯度正常计算
    let expected_w2_grad = Tensor::new(&{w2_grad_list}, &[1, 2]);

    // 验证
    assert!(graph.get_node_jacobi(w1).unwrap().is_none(),
            "w1 应无梯度 (被 detach 阻断)");
    let actual_w2_grad = graph.get_node_jacobi(w2).unwrap().unwrap();
    assert_eq!(actual_w2_grad, &expected_w2_grad,
            "w2 梯度应与 PyTorch 匹配");
    """)


if __name__ == "__main__":
    test_detach_gradient_values()

"""
PyTorch 参考实现：复杂拓扑下多参数节点的梯度累积测试

用于验证 Rust 实现的正确性，生成预期的梯度值。

拓扑:
  x → w_shared1 → shared_feat1 → w_shared2 → w_shared2_out → w_shared3 → shared_feat2 → w_task1 → out1
                                                                                    └──→ w_task2 → out2

关键测试点: w_shared2 与 w_shared3 是相邻参数节点，验证链式累积是否正确
"""

import torch

def main():
    # ========== 构建拓扑 ==========
    # 输入 (4, 1)
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=False)

    # 共享链参数
    w_shared1 = torch.ones(2, 4, requires_grad=True)  # [2, 4]
    w_shared2 = torch.ones(2, 2, requires_grad=True)  # [2, 2]
    w_shared3 = torch.ones(2, 2, requires_grad=True)  # [2, 2]

    # 分叉参数
    w_task1 = torch.ones(1, 2, requires_grad=True)    # [1, 2]
    w_task2 = torch.ones(1, 2, requires_grad=True)    # [1, 2]

    # ========== 前向传播 ==========
    shared_feat1 = w_shared1 @ x          # [2, 1]
    w_shared2_out = w_shared2 @ shared_feat1  # [2, 1]
    shared_feat2 = w_shared3 @ w_shared2_out  # [2, 1]

    out1 = w_task1 @ shared_feat2  # [1, 1]
    out2 = w_task2 @ shared_feat2  # [1, 1]

    print("========== 前向传播结果 ==========")
    print(f"x:\n{x.T}")
    print(f"shared_feat1:\n{shared_feat1.T}")
    print(f"w_shared2_out:\n{w_shared2_out.T}")
    print(f"shared_feat2:\n{shared_feat2.T}")
    print(f"out1: {out1.item()}")
    print(f"out2: {out2.item()}")

    # ========== 第 1 次 backward (out1, retain_graph=True) ==========
    # 注意：PyTorch 对标量输出自动使用 gradient=1.0
    out1.backward(retain_graph=True)

    print("\n========== 第 1 次 backward (out1) 后的梯度 ==========")
    print(f"w_shared1.grad (shape {w_shared1.grad.shape}):\n{w_shared1.grad}")
    print(f"w_shared2.grad (shape {w_shared2.grad.shape}):\n{w_shared2.grad}")
    print(f"w_shared3.grad (shape {w_shared3.grad.shape}):\n{w_shared3.grad}")
    print(f"w_task1.grad (shape {w_task1.grad.shape}):\n{w_task1.grad}")
    print(f"w_task2.grad: {w_task2.grad}")  # 应该是 None

    # 保存第一次 backward 后的梯度
    w_shared1_after_task1 = w_shared1.grad.clone()
    w_shared2_after_task1 = w_shared2.grad.clone()
    w_shared3_after_task1 = w_shared3.grad.clone()
    w_task1_after_task1 = w_task1.grad.clone()

    # ========== 第 2 次 backward (out2, 梯度累积) ==========
    # 注意：PyTorch 默认会累积梯度，不需要 retain_graph
    out2.backward()

    print("\n========== 第 2 次 backward (out2) 后的累积梯度 ==========")
    print(f"w_shared1.grad (累积):\n{w_shared1.grad}")
    print(f"w_shared2.grad (累积):\n{w_shared2.grad}")
    print(f"w_shared3.grad (累积):\n{w_shared3.grad}")
    print(f"w_task1.grad (不变，因为 out2 不经过 w_task1):\n{w_task1.grad}")
    print(f"w_task2.grad:\n{w_task2.grad}")

    # ========== 验证 ==========
    print("\n========== 验证结果 ==========")

    # 由于拓扑对称，out1 和 out2 对共享参数的贡献应该相同
    # 因此累积后的梯度应该是单次的 2 倍
    print(f"w_shared1 累积 == 2 * 单次: {torch.allclose(w_shared1.grad, w_shared1_after_task1 * 2)}")
    print(f"w_shared2 累积 == 2 * 单次: {torch.allclose(w_shared2.grad, w_shared2_after_task1 * 2)}")
    print(f"w_shared3 累积 == 2 * 单次: {torch.allclose(w_shared3.grad, w_shared3_after_task1 * 2)}")
    print(f"w_task1 不变: {torch.allclose(w_task1.grad, w_task1_after_task1)}")
    print(f"w_task2 == w_task1 单次 (对称): {torch.allclose(w_task2.grad, w_task1_after_task1)}")

    # ========== 输出 Rust 测试需要的精确值 ==========
    print("\n========== Rust 测试需要的精确值 ==========")
    print("// 注意：Rust 中 Jacobi 格式为展平的 [1, n]，而非 PyTorch 原始 shape")
    print()
    print("// 第一次 backward 后各参数的梯度（展平为 Jacobi 格式 [1, n]）")
    print(f"// w_shared1: shape [1, 8], data: {w_shared1_after_task1.flatten().tolist()}")
    print(f"// w_shared2: shape [1, 4], data: {w_shared2_after_task1.flatten().tolist()}")
    print(f"// w_shared3: shape [1, 4], data: {w_shared3_after_task1.flatten().tolist()}")
    print(f"// w_task1: shape [1, 2], data: {w_task1_after_task1.flatten().tolist()}")
    print()
    print("// 第二次 backward 后累积的梯度（展平为 Jacobi 格式 [1, n]）")
    print(f"// w_shared1: shape [1, 8], data: {w_shared1.grad.flatten().tolist()}")
    print(f"// w_shared2: shape [1, 4], data: {w_shared2.grad.flatten().tolist()}")
    print(f"// w_shared3: shape [1, 4], data: {w_shared3.grad.flatten().tolist()}")
    print(f"// w_task1 (不变): shape [1, 2], data: {w_task1.grad.flatten().tolist()}")
    print(f"// w_task2: shape [1, 2], data: {w_task2.grad.flatten().tolist()}")

if __name__ == "__main__":
    main()


"""
PyTorch no_grad 行为验证脚本

验证 only_torch 的 no_grad_scope 与 PyTorch 的 torch.no_grad() 行为一致性。

对应 Rust 测试:
- test_no_grad_scope_backward_still_works
- test_no_grad_scope_nodes_created_inside

运行: python tests/calc_jacobi_by_pytorch/no_grad_scope_behavior.py
"""

import torch


def test_no_grad_backward_still_works():
    """
    测试 1: no_grad 内调用 backward

    结论: PyTorch 的 torch.no_grad() 不阻止 backward() 调用，
    但在 no_grad 内创建的张量没有 requires_grad，所以无法计算梯度。
    """
    print("=" * 60)
    print("测试 1: no_grad 内调用 backward")
    print("=" * 60)

    # 创建参数（在 no_grad 外部）
    w = torch.tensor([[1.0, 2.0]], requires_grad=True)
    x = torch.tensor([[1.0], [2.0]])

    with torch.no_grad():
        print(f"  在 no_grad 内: torch.is_grad_enabled() = {torch.is_grad_enabled()}")

        # forward 正常工作
        y = torch.matmul(w, x)
        print(f"  forward 结果: y = {y.item()}")

        # 尝试 backward - 这会失败，因为 y 没有 grad_fn
        try:
            y.backward()
            print("  backward 成功（意外）")
        except RuntimeError as e:
            print(f"  backward 失败（预期）: {str(e)[:50]}...")

    print(f"  退出后: torch.is_grad_enabled() = {torch.is_grad_enabled()}")

    # 在 no_grad 外部重新计算
    y2 = torch.matmul(w, x)
    y2.backward()
    print(f"  no_grad 外部 backward 成功: w.grad = {w.grad}")
    print()


def test_no_grad_backward_with_external_tensors():
    """
    测试 2: 使用外部创建的张量，在 no_grad 内 forward，然后在外部 backward

    这更接近 only_torch 的静态图模型。
    """
    print("=" * 60)
    print("测试 2: 外部张量 + no_grad 内 forward + 外部 backward")
    print("=" * 60)

    # 在 no_grad 外部创建张量
    w = torch.tensor([[1.0, 2.0]], requires_grad=True)
    x = torch.tensor([[1.0], [2.0]])

    # 在 no_grad 内执行 forward
    with torch.no_grad():
        y_no_grad = torch.matmul(w, x)
        print(f"  no_grad 内 forward: y = {y_no_grad.item()}")
        print(f"  y.requires_grad = {y_no_grad.requires_grad}")  # False
        print(f"  y.grad_fn = {y_no_grad.grad_fn}")  # None

    # 在 no_grad 外部重新 forward 才能 backward
    y = torch.matmul(w, x)
    print(f"  no_grad 外 forward: y = {y.item()}")
    print(f"  y.requires_grad = {y.requires_grad}")  # True
    print(f"  y.grad_fn = {y.grad_fn}")  # MmBackward

    y.backward()
    print(f"  backward 成功: w.grad = {w.grad}")
    print()


def test_tensors_created_inside_no_grad():
    """
    测试 3: 在 no_grad 内创建的张量

    验证: 在 no_grad 内创建的张量默认 requires_grad=False，
    但可以在退出后手动设置 requires_grad=True。
    """
    print("=" * 60)
    print("测试 3: no_grad 内创建的张量")
    print("=" * 60)

    with torch.no_grad():
        # 在 no_grad 内创建张量
        w_inside = torch.ones(1, 2)
        print(f"  no_grad 内创建: w.requires_grad = {w_inside.requires_grad}")  # False

    # 退出后可以启用梯度
    w_inside.requires_grad_(True)
    print(f"  退出后设置: w.requires_grad = {w_inside.requires_grad}")  # True

    # 正常计算
    x = torch.tensor([[1.0], [2.0]])
    y = torch.matmul(w_inside, x)
    y.backward()
    print(f"  backward 成功: w.grad = {w_inside.grad}")
    print()


def test_only_torch_equivalent_pattern():
    """
    测试 4: only_torch 等效模式

    在 only_torch 中，图是静态的，节点创建与梯度模式无关。
    这个测试模拟 only_torch 的行为模式。
    """
    print("=" * 60)
    print("测试 4: only_torch 等效模式（静态图）")
    print("=" * 60)

    # 模拟静态图: 参数始终存在
    w = torch.tensor([[1.0, 2.0]], requires_grad=True)
    x = torch.tensor([[1.0], [2.0]])

    # 场景 A: 训练模式下 forward + backward
    y = torch.matmul(w, x)
    y.backward()
    train_grad = w.grad.clone()
    print(f"  训练模式 backward: w.grad = {train_grad}")
    w.grad.zero_()

    # 场景 B: 评估模式下只 forward（不 backward）
    with torch.no_grad():
        y_eval = torch.matmul(w, x)
        print(f"  评估模式 forward: y = {y_eval.item()}")
        print(f"  评估模式下 w.grad = {w.grad}")  # 仍是之前的零

    # 场景 C: 退出 no_grad 后恢复训练
    y2 = torch.matmul(w, x)
    y2.backward()
    print(f"  恢复训练 backward: w.grad = {w.grad}")

    # 验证梯度值相同
    assert torch.equal(train_grad, w.grad), "梯度应该相同"
    print("  ✓ 梯度值一致")
    print()


def test_same_input_same_output():
    """
    测试 5: 相同输入产生相同输出（核心保证）

    对应 Rust 测试: test_no_grad_scope_same_input_same_loss_no_gradient
    """
    print("=" * 60)
    print("测试 5: 相同输入 → 相同输出（no_grad 核心保证）")
    print("=" * 60)

    w = torch.tensor([[1.0, 2.0]], requires_grad=True)
    x = torch.tensor([[1.0], [2.0]])

    # 训练模式
    y_train = torch.matmul(w, x)
    loss_train = y_train.sum()
    print(f"  训练模式 loss = {loss_train.item()}")

    # 评估模式（相同输入）
    with torch.no_grad():
        y_eval = torch.matmul(w, x)
        loss_eval = y_eval.sum()
        print(f"  评估模式 loss = {loss_eval.item()}")

    # 核心验证: 相同输入 → 相同输出
    assert loss_train.item() == loss_eval.item(), "相同输入应产生相同输出"
    print("  ✓ 核心保证验证通过: 相同输入产生相同 loss")
    print()


def main():
    print("\n" + "=" * 60)
    print("PyTorch no_grad 行为验证")
    print("=" * 60 + "\n")

    test_no_grad_backward_still_works()
    test_no_grad_backward_with_external_tensors()
    test_tensors_created_inside_no_grad()
    test_only_torch_equivalent_pattern()
    test_same_input_same_output()

    print("=" * 60)
    print("总结: only_torch no_grad_scope 与 PyTorch 的关键差异")
    print("=" * 60)
    print("""
    1. PyTorch 动态图: no_grad 影响新创建张量的 requires_grad
       only_torch 静态图: no_grad 影响 forward 时的缓存策略

    2. PyTorch: no_grad 内的 forward 结果没有 grad_fn，无法 backward
       only_torch: backward 可在 no_grad 内调用（图已构建）

    3. 核心一致性: 相同输入产生相同输出，no_grad 只影响梯度计算
    """)

    print("✓ 所有测试通过！\n")


if __name__ == "__main__":
    main()

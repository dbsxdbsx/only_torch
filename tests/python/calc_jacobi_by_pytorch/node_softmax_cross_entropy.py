"""
SoftmaxCrossEntropy 节点的 PyTorch 验证脚本

用于生成 Rust 单元测试的预期值
"""

import torch
import torch.nn.functional as F
import numpy as np

np.set_printoptions(precision=8, suppress=True)


def test_case_1_simple():
    """简单的 3 分类测试"""
    print("=" * 50)
    print("测试用例 1: 简单 3 分类")
    print("=" * 50)

    # 输入 logits（未经 softmax 的原始分数）
    logits = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    # one-hot 标签（真实类别是 2）
    labels = torch.tensor([0.0, 0.0, 1.0])

    # PyTorch 的 cross_entropy 期望 logits 和 class index
    # 我们用 softmax + 手动计算来验证
    softmax = F.softmax(logits, dim=0)
    print(f"softmax = {softmax.detach().numpy()}")

    # 交叉熵损失: -sum(y * log(softmax))
    loss = -torch.sum(labels * torch.log(softmax))
    print(f"loss = {loss.item()}")

    # 反向传播
    loss.backward()
    print(f"grad (dL/d_logits) = {logits.grad.numpy()}")

    # 验证梯度 = softmax - labels
    expected_grad = softmax.detach() - labels
    print(f"expected grad (softmax - labels) = {expected_grad.numpy()}")


def test_case_2_batch():
    """批量输入测试（展平为向量）"""
    print("\n" + "=" * 50)
    print("测试用例 2: 10 分类")
    print("=" * 50)

    # 10 个类别
    logits = torch.tensor([0.5, 1.5, -0.5, 2.0, 0.0, -1.0, 1.0, 0.5, -0.5, 1.5], requires_grad=True)
    # 真实类别是 3
    labels = torch.zeros(10)
    labels[3] = 1.0

    softmax = F.softmax(logits, dim=0)
    print(f"softmax = {softmax.detach().numpy()}")

    loss = -torch.sum(labels * torch.log(softmax))
    print(f"loss = {loss.item()}")

    loss.backward()
    print(f"grad = {logits.grad.numpy()}")


def test_case_3_numerical_stability():
    """数值稳定性测试（大数值）"""
    print("\n" + "=" * 50)
    print("测试用例 3: 数值稳定性（大数值）")
    print("=" * 50)

    # 大数值输入，容易造成 exp 溢出
    logits = torch.tensor([100.0, 200.0, 300.0], requires_grad=True)
    labels = torch.tensor([0.0, 0.0, 1.0])

    softmax = F.softmax(logits, dim=0)
    print(f"softmax = {softmax.detach().numpy()}")

    loss = -torch.sum(labels * torch.log(softmax + 1e-10))  # 加小值防止 log(0)
    print(f"loss = {loss.item()}")

    loss.backward()
    print(f"grad = {logits.grad.numpy()}")


def test_case_4_single_element():
    """单元素测试（边界情况）"""
    print("\n" + "=" * 50)
    print("测试用例 4: 单元素（退化情况）")
    print("=" * 50)

    logits = torch.tensor([2.5], requires_grad=True)
    labels = torch.tensor([1.0])

    softmax = F.softmax(logits, dim=0)
    print(f"softmax = {softmax.detach().numpy()}")

    loss = -torch.sum(labels * torch.log(softmax))
    print(f"loss = {loss.item()}")

    loss.backward()
    print(f"grad = {logits.grad.numpy()}")


def test_case_5_uniform():
    """均匀分布测试"""
    print("\n" + "=" * 50)
    print("测试用例 5: 均匀 logits")
    print("=" * 50)

    # 所有 logits 相同，softmax 应该均匀分布
    logits = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 0.0, 0.0])  # 类别 1

    softmax = F.softmax(logits, dim=0)
    print(f"softmax = {softmax.detach().numpy()}")

    loss = -torch.sum(labels * torch.log(softmax))
    print(f"loss = {loss.item()}")

    loss.backward()
    print(f"grad = {logits.grad.numpy()}")


if __name__ == "__main__":
    test_case_1_simple()
    test_case_2_batch()
    test_case_3_numerical_stability()
    test_case_4_single_element()
    test_case_5_uniform()


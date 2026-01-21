#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
奇偶性检测 PyTorch 参考实现（简化版）

展示 PyTorch RNN 的最简洁用法：
- forward() 一次性处理整个序列
- loss.backward() 自动 BPTT
- 无需手动迭代时间步

此文件作为 Rust 实现 `examples/parity_fixed_len/` 的对照参考。

运行方式:
    python tests/parity_fixed_len_pytorch.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # pyright: ignore[reportAttributeAccessIssue]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class ParityRNN(nn.Module):
    """
    奇偶性检测 RNN 模型

    关键点：forward() 一次性处理整个序列，无需手动循环！
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: [batch, seq_len, 1] → 整个序列一次性输入！
        _, hidden = self.rnn(x)           # hidden: [1, batch, hidden_size]
        last_hidden = hidden.squeeze(0)   # [batch, hidden_size]
        return self.fc(last_hidden)       # [batch, 2]


def generate_parity_data(num_samples: int, seq_len: int, seed: int):
    """生成奇偶性检测数据"""
    np.random.seed(seed)
    sequences = np.random.randint(0, 2, (num_samples, seq_len, 1)).astype(np.float32)
    # 标签：1 的个数是奇数为 1，偶数为 0
    labels = np.array([int(seq.sum()) % 2 for seq in sequences], dtype=np.int64)
    return torch.tensor(sequences), torch.tensor(labels)


def main():
    print("=== 奇偶性检测 PyTorch 参考实现 ===\n")

    # ========== 超参数 ==========
    seq_len = 8
    hidden_size = 16
    batch_size = 32
    train_samples = 1000
    test_samples = 200
    max_epochs = 150
    target_accuracy = 95.0

    print("超参数:")
    print(f"  序列长度: {seq_len}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  批大小: {batch_size}")
    print(f"  优化器: Adam (lr=0.01)")
    print(f"  损失函数: CrossEntropyLoss\n")

    # ========== 数据准备 ==========
    train_x, train_y = generate_parity_data(train_samples, seq_len, seed=42)
    test_x, test_y = generate_parity_data(test_samples, seq_len, seed=1042)

    # 使用 DataLoader（PyTorch 标准做法）
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=False  # 与 Rust 版本保持一致，便于对比
    )

    print(f"数据集: 训练 {train_samples} 样本, 测试 {test_samples} 样本")
    print(f"标签分布: 训练 {train_y.sum().item()}/{train_samples} 奇数, "
          f"测试 {test_y.sum().item()}/{test_samples} 奇数\n")

    # ========== 模型定义 ==========
    model = ParityRNN(hidden_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ========== 训练循环（极简版！）==========
    print("开始训练...\n")
    best_accuracy = 0.0

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)       # 前向：整个序列一次性处理！
            loss = criterion(output, y_batch)
            loss.backward()               # 反向：自动 BPTT！
            optimizer.step()
            epoch_loss += loss.item()

        # 每 10 个 epoch 评估
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / len(train_loader)
            accuracy = evaluate(model, test_x, test_y)
            best_accuracy = max(best_accuracy, accuracy)

            print(f"Epoch {epoch + 1:3}/{max_epochs}: loss={avg_loss:.4f}, test_acc={accuracy:.1f}%")

            if accuracy >= target_accuracy:
                print(f"\n[OK] 达到目标准确率 {target_accuracy}%，提前停止训练")
                break

    # ========== 最终评估 ==========
    final_accuracy = evaluate(model, test_x, test_y)
    print(f"\n========== 最终结果 ==========")
    print(f"测试准确率: {final_accuracy:.1f}%")
    print(f"最佳准确率: {best_accuracy:.1f}%")

    if final_accuracy >= target_accuracy:
        print("[OK] 奇偶性检测任务成功！")
    else:
        print(f"[X] 准确率 {final_accuracy:.1f}% 未达到目标 {target_accuracy}%")


def evaluate(model, test_x, test_y):
    """评估模型准确率"""
    model.eval()
    with torch.no_grad():
        output = model(test_x)            # 一次性评估所有测试样本
        pred = output.argmax(dim=1)
        accuracy = (pred == test_y).float().mean().item() * 100
    return accuracy


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST GAN PyTorch 参考实现

此文件作为 Rust 实现 `examples/mnist_gan/` 的对照参考。
展示 GAN 训练的核心模式：
- Generator 和 Discriminator 双网络
- detach 机制阻止梯度回传
- 交替训练 D 和 G

运行:
    cd tests/python/gan_reference
    python mnist_gan_pytorch.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # pyright: ignore[reportAttributeAccessIssue]

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time


# ========== 网络定义 ==========

class Generator(nn.Module):
    """生成器: z(64) -> 128 -> 784"""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 784),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """判别器: 784 -> 128 -> 1"""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("\n" + "=" * 60)
    print("=== MNIST GAN PyTorch 参考实现 ===")
    print("=" * 60 + "\n")

    # 设置随机种子
    torch.manual_seed(42)

    # ========== 1. 配置 ==========
    batch_size = 256
    train_samples = 5120
    max_epochs = 15
    latent_dim = 64
    lr_d = 0.0005
    lr_g = 0.001
    device = torch.device("cpu")  # 与 Rust 保持一致，只用 CPU

    print("[1/5] 训练配置：")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - 训练样本: {train_samples}")
    print(f"  - 最大 Epochs: {max_epochs}")
    print(f"  - 噪声维度: {latent_dim}")
    print(f"  - 学习率 (D/G): {lr_d}/{lr_g}")

    # ========== 2. 加载数据 ==========
    print("\n[2/5] 加载 MNIST 数据集...")
    load_start = time.time()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 展平为 784
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    # 只使用部分数据
    train_dataset = torch.utils.data.Subset(
        train_dataset,
        range(train_samples)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    print(f"  ✓ 训练集: {len(train_dataset)} 样本，耗时 {time.time() - load_start:.2f}s")

    # ========== 3. 构建网络 ==========
    print("\n[3/5] 构建 GAN 网络...")

    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)

    # 统计参数量
    g_params = sum(p.numel() for p in G.parameters())
    d_params = sum(p.numel() for p in D.parameters())

    print(f"  ✓ Generator: z({latent_dim}) -> 128 -> 784")
    print(f"  ✓ Discriminator: 784 -> 128 -> 1")
    print(f"  ✓ G 参数量: {g_params}")
    print(f"  ✓ D 参数量: {d_params}")

    # ========== 4. 创建优化器和损失函数 ==========
    print("\n[4/5] 创建优化器...")

    # 使用 MSELoss（与 Rust 版本一致）
    criterion = nn.MSELoss()

    # 为 D 和 G 创建独立优化器（beta1=0.5 是 GAN 的常用设置）
    optimizer_D = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.999))

    print("  ✓ Adam 优化器 (beta1=0.5 for GAN stability)")

    # ========== 5. 训练循环 ==========
    print("\n[5/5] 开始训练...\n")

    start_time = time.time()
    d_real_avg = 0.0
    d_fake_avg = 0.0

    for epoch in range(max_epochs):
        epoch_start = time.time()
        d_loss_sum = 0.0
        g_loss_sum = 0.0
        d_real_sum = 0.0
        d_fake_sum = 0.0
        num_batches = 0

        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            batch_size_actual = real_images.size(0)

            # 标签
            real_labels = torch.ones(batch_size_actual, 1, device=device)
            fake_labels = torch.zeros(batch_size_actual, 1, device=device)

            # ========== 训练 Discriminator ==========
            optimizer_D.zero_grad()

            # D 对真实图像的判断
            d_real_out = D(real_images)
            d_loss_real = criterion(d_real_out, real_labels)

            # 生成假图像
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_images = G(z)

            # D 对假图像的判断
            # 关键：detach() 阻止梯度流向 G
            d_fake_out = D(fake_images.detach())
            d_loss_fake = criterion(d_fake_out, fake_labels)

            # D 总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # 记录 D 的判别结果
            d_real_sum += d_real_out.mean().item()
            d_fake_sum += d_fake_out.mean().item()

            # ========== 训练 Generator ==========
            optimizer_G.zero_grad()

            # 重新生成假图像（用新噪声）
            z = torch.randn(batch_size_actual, latent_dim, device=device)
            fake_images = G(z)

            # G 的目标：让 D 认为假图像是真的
            # 注意：这里不 detach，梯度要流回 G
            d_fake_out_for_g = D(fake_images)
            g_loss = criterion(d_fake_out_for_g, real_labels)

            g_loss.backward()
            optimizer_G.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            num_batches += 1

        # Epoch 统计
        avg_d_loss = d_loss_sum / num_batches
        avg_g_loss = g_loss_sum / num_batches
        d_real_avg = d_real_sum / num_batches
        d_fake_avg = d_fake_sum / num_batches

        print(f"Epoch {epoch + 1:2}/{max_epochs}: "
              f"D_loss = {avg_d_loss:.4f}, G_loss = {avg_g_loss:.4f}, "
              f"D(real) = {d_real_avg:.3f}, D(fake) = {d_fake_avg:.3f}, "
              f"耗时 {time.time() - epoch_start:.2f}s")

    total_time = time.time() - start_time
    print(f"\n总耗时: {total_time:.2f}s")

    # ========== 验证 ==========
    print("\n" + "=" * 60)
    print("验证结果：")
    print(f"  - D(real) = {d_real_avg:.3f}")
    print(f"  - D(fake) = {d_fake_avg:.3f}")

    # 验证条件（与 Rust 版本一致）
    test_passed = d_real_avg > 0.3 and 0.2 < d_fake_avg < 0.9

    if test_passed:
        print("\n✅ GAN 训练成功！")
        print("  - D(real) > 0.3 ✓")
        print("  - 0.2 < D(fake) < 0.9 ✓")
    else:
        print("\n❌ GAN 训练未收敛")

    print("=" * 60)

    # ========== 输出网络结构（供参考）==========
    print("\n网络结构：")
    print("\nGenerator:")
    print(G)
    print("\nDiscriminator:")
    print(D)


if __name__ == "__main__":
    main()

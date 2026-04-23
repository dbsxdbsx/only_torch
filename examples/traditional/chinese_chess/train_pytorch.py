#!/usr/bin/env python3
"""
中国象棋棋子 CNN 分类器 — PyTorch 训练脚本

支持从多个数据源（合成 + 真实棋子）加载训练/测试数据，
使用 GPU 训练 CNN，输出每类准确率和混淆矩阵。

用法:
    # 仅合成数据
    python scripts/train_chess_cnn.py

    # 合成 + 真实棋子混合训练
    python scripts/train_chess_cnn.py --real-data data/chinese_chess_real

    # 自定义参数
    python scripts/train_chess_cnn.py --epochs 80 --lr 0.001 --batch-size 128
"""

import argparse
import os
import struct
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T


# ==================== 类别定义 ====================

CLASS_NAMES = [
    "空位", "红帅", "红仕", "红相", "红車", "红馬", "红炮", "红兵",
    "黑将", "黑士", "黑象", "黑車", "黑馬", "黑炮", "黑卒",
]


# ==================== 数据加载 ====================

def load_bin_data(data_dir):
    """加载 images.bin + labels.bin

    Returns: (images_np [N,C,H,W] float32, labels_np [N] uint8)
    """
    img_path = os.path.join(data_dir, "images.bin")
    lbl_path = os.path.join(data_dir, "labels.bin")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"数据文件不存在: {img_path}")

    # 读取图像
    with open(img_path, "rb") as f:
        header = f.read(16)
        n, c, h, w = struct.unpack("<IIII", header)
        img_data = np.frombuffer(f.read(n * c * h * w * 4), dtype=np.float32)
        images = img_data.reshape(n, c, h, w)

    # 读取标签
    with open(lbl_path, "rb") as f:
        header = f.read(4)
        n_labels = struct.unpack("<I", header)[0]
        assert n == n_labels, f"图像数量 {n} != 标签数量 {n_labels}"
        labels = np.frombuffer(f.read(n_labels), dtype=np.uint8)

    return images, labels


def load_and_merge_data(synthetic_dir, real_dir=None, split="train"):
    """加载并合并多个数据源

    Args:
        synthetic_dir: 合成数据目录 (data/chinese_chess/{split}/)
        real_dir: 真实棋子数据目录 (data/chinese_chess_real/{split}/)，可选
        split: "train" or "test"

    Returns: (images_np, labels_np, real_mask_np)
        real_mask_np: bool array, True = 来自真实数据 (用于分开统计)
    """
    all_images = []
    all_labels = []
    all_real_mask = []

    # 合成数据
    syn_path = os.path.join(synthetic_dir, split)
    if os.path.exists(os.path.join(syn_path, "images.bin")):
        images, labels = load_bin_data(syn_path)
        print(f"  合成数据 [{split}]: {len(labels)} 样本")
        all_images.append(images)
        all_labels.append(labels)
        all_real_mask.append(np.zeros(len(labels), dtype=bool))
    else:
        print(f"  合成数据 [{split}]: 未找到 ({syn_path})")

    # 真实棋子数据
    if real_dir:
        real_path = os.path.join(real_dir, split)
        if os.path.exists(os.path.join(real_path, "images.bin")):
            images, labels = load_bin_data(real_path)
            print(f"  真实数据 [{split}]: {len(labels)} 样本")
            all_images.append(images)
            all_labels.append(labels)
            all_real_mask.append(np.ones(len(labels), dtype=bool))
        else:
            print(f"  真实数据 [{split}]: 未找到 ({real_path})")

    if not all_images:
        raise RuntimeError(f"没有找到任何 {split} 数据！")

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    real_mask = np.concatenate(all_real_mask, axis=0)

    # 打乱
    indices = np.random.permutation(len(labels))
    images = images[indices]
    labels = labels[indices]
    real_mask = real_mask[indices]

    return images, labels, real_mask


# ==================== 模型 ====================

class ChessPieceCNN(nn.Module):
    """中国象棋棋子 CNN 分类器

    Input [batch, 3, 28, 28]
      → Conv1 (3→16, 3x3, pad=1) → BN → ReLU → MaxPool(2x2)   [batch, 16, 14, 14]
      → Conv2 (16→32, 3x3, pad=1) → BN → ReLU → MaxPool(2x2)  [batch, 32, 7, 7]
      → Flatten                                                    [batch, 1568]
      → FC1 (1568→128) → ReLU → Dropout(0.3)
      → FC2 (128→15)
    """

    def __init__(self, num_classes=15):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==================== 训练时数据增强 ====================

class TrainTransform:
    """训练时在线数据增强 (作用在 GPU tensor 上)"""

    def __init__(self):
        self.transform = T.Compose([
            T.RandomAffine(degrees=3, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
        ])

    def __call__(self, batch):
        """对一个 batch 逐样本做增强"""
        return torch.stack([self.transform(img) for img in batch])


# ==================== 评估 ====================

def evaluate(model, dataloader, device, real_mask_all=None):
    """评估模型

    Returns: (total_acc, per_class_acc, confusion_matrix, real_acc)
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    num_classes = 15

    # 总体准确率
    total_acc = (all_preds == all_labels).mean() * 100

    # 每类准确率
    per_class_acc = {}
    for cid in range(num_classes):
        mask = all_labels == cid
        if mask.sum() > 0:
            per_class_acc[cid] = (all_preds[mask] == cid).mean() * 100
        else:
            per_class_acc[cid] = 0.0

    # 混淆矩阵
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for pred, true in zip(all_preds, all_labels):
        confusion[true][pred] += 1

    # 真实数据子集准确率
    real_acc = None
    if real_mask_all is not None and real_mask_all.any():
        real_preds = all_preds[real_mask_all]
        real_labels = all_labels[real_mask_all]
        if len(real_labels) > 0:
            real_acc = (real_preds == real_labels).mean() * 100

    return total_acc, per_class_acc, confusion, real_acc


def print_confusion_matrix(confusion, class_names):
    """打印混淆矩阵 (简洁版)"""
    num_classes = len(class_names)
    # 表头
    header = "真\\预 " + " ".join(f"{class_names[i][:2]:>4}" for i in range(num_classes))
    print(header)
    print("-" * len(header))
    for i in range(num_classes):
        row = f"{class_names[i][:2]:>4} "
        for j in range(num_classes):
            val = confusion[i][j]
            if val == 0:
                row += "   ."
            elif i == j:
                row += f" {val:3d}"
            else:
                row += f"  {val:2d}"
        total = confusion[i].sum()
        correct = confusion[i][i]
        acc = correct / total * 100 if total > 0 else 0
        row += f"  | {acc:.0f}%"
        print(row)


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="中国象棋 CNN 训练 (PyTorch)")
    parser.add_argument("--data", type=str, default="data/chinese_chess",
                        help="合成数据根目录")
    parser.add_argument("--real-data", type=str, default=None,
                        help="真实棋子数据根目录 (可选)")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch 大小")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping 耐心")
    parser.add_argument("--save", type=str, default="models/chess_cnn.pth",
                        help="模型保存路径")
    parser.add_argument("--no-augment", action="store_true", help="禁用训练时增强")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 中国象棋 CNN 分类器 (PyTorch) ===")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. 加载数据
    print(f"\n[1/4] 加载数据...")
    train_images, train_labels, train_real_mask = load_and_merge_data(
        args.data, args.real_data, "train"
    )
    test_images, test_labels, test_real_mask = load_and_merge_data(
        args.data, args.real_data, "test"
    )

    print(f"  训练集总计: {len(train_labels)} 样本")
    print(f"  测试集总计: {len(test_labels)} 样本")
    if test_real_mask.any():
        print(f"  测试集中真实棋子: {test_real_mask.sum()} 样本")

    # 转为 PyTorch tensor
    train_x = torch.from_numpy(train_images)
    train_y = torch.from_numpy(train_labels.astype(np.int64))
    test_x = torch.from_numpy(test_images)
    test_y = torch.from_numpy(test_labels.astype(np.int64))

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    # 2. 模型
    print(f"\n[2/4] 构建模型...")
    model = ChessPieceCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  网络: Conv(3→16) → BN → Pool → Conv(16→32) → BN → Pool → FC(1568→128) → FC(128→15)")
    print(f"  参数量: {param_count:,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # 数据增强
    train_augment = TrainTransform() if not args.no_augment else None
    if train_augment:
        print(f"  训练增强: RandomAffine(±3°) + ColorJitter + RandomErasing(20%)")

    # 3. 训练
    print(f"\n[3/4] 开始训练 (epochs={args.epochs}, lr={args.lr}, batch={args.batch_size})...\n")
    best_acc = 0.0
    best_real_acc = 0.0
    no_improve = 0
    train_start = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        num_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # 在线数据增强
            if train_augment:
                images = train_augment(images)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()

        # 评估
        total_acc, per_class_acc, confusion, real_acc = evaluate(
            model, test_loader, device, test_real_mask
        )

        avg_loss = running_loss / num_batches
        elapsed = time.time() - epoch_start
        lr_now = scheduler.get_last_lr()[0]

        real_str = f", 真实={real_acc:.1f}%" if real_acc is not None else ""
        print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, acc={total_acc:.1f}%{real_str}, "
              f"lr={lr_now:.6f}, {elapsed:.1f}s")

        # Best model tracking
        improved = False
        if real_acc is not None and real_acc > best_real_acc:
            best_real_acc = real_acc
            improved = True
        if total_acc > best_acc:
            best_acc = total_acc
            improved = True

        if improved:
            no_improve = 0
            # 保存最佳模型
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "total_acc": total_acc,
                "real_acc": real_acc,
                "per_class_acc": per_class_acc,
            }, args.save)
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= args.patience:
            print(f"\n连续 {args.patience} 轮无提升，提前停止")
            break

        # 高准确率提前停止
        if total_acc >= 99.0 and (real_acc is None or real_acc >= 98.0):
            print(f"\n准确率已达 {total_acc:.1f}%，提前停止")
            break

    train_time = time.time() - train_start
    print(f"\n训练总耗时: {train_time:.1f}s")

    # 4. 最终评估
    print(f"\n[4/4] 最终评估...")

    # 加载最佳模型
    checkpoint = torch.load(args.save, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  加载最佳模型 (epoch {checkpoint['epoch']})")

    total_acc, per_class_acc, confusion, real_acc = evaluate(
        model, test_loader, device, test_real_mask
    )

    print(f"\n总体准确率: {total_acc:.1f}%")
    if real_acc is not None:
        print(f"真实棋子准确率: {real_acc:.1f}% ({int(test_real_mask.sum())} 样本)")

    print(f"\n每类准确率:")
    for cid in range(15):
        acc = per_class_acc.get(cid, 0)
        mark = " OK" if acc >= 95 else " !!" if acc < 80 else ""
        print(f"  [{cid:2d}] {CLASS_NAMES[cid]}: {acc:.1f}%{mark}")

    print(f"\n混淆矩阵:")
    print_confusion_matrix(confusion, CLASS_NAMES)

    # 如果有真实数据，单独统计真实数据的每类准确率
    if test_real_mask.any():
        print(f"\n--- 真实棋子测试集详细 ---")
        real_indices = np.where(test_real_mask)[0]
        real_x = test_x[real_indices]
        real_y = test_y[real_indices]
        real_dataset = TensorDataset(real_x, real_y)
        real_loader = DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False)

        _, real_per_class, real_confusion, _ = evaluate(model, real_loader, device)

        for cid in range(1, 15):
            acc = real_per_class.get(cid, 0)
            mark = " OK" if acc >= 95 else " !!" if acc < 80 else ""
            print(f"  [{cid:2d}] {CLASS_NAMES[cid]}: {acc:.1f}%{mark}")

    print(f"\n模型已保存: {args.save}")
    print(f"最佳总体准确率: {best_acc:.1f}%")
    if best_real_acc > 0:
        print(f"最佳真实棋子准确率: {best_real_acc:.1f}%")

    # 5. 导出 ONNX
    onnx_path = args.save.replace(".pth", ".onnx")
    model.eval()
    model.cpu()
    dummy_input = torch.randn(1, 3, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print(f"ONNX 模型已导出: {onnx_path}")


if __name__ == "__main__":
    main()

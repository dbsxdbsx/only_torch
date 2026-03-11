#!/usr/bin/env python3
"""
真实棋子资源处理器

从本地象棋软件中提取棋子贴图，做增强后输出为训练/测试数据。

支持的软件：
  - 象棋奇兵 2009 (编号制 1-14.bmp)
  - 象棋名手 3.26 (ra/rk/... .png)
  - 兵河五四 3.6  (ra/rk/... .png)
  - 鲨鱼象棋      (ra/rk/... .bmp)

输出格式与 generate_chess_data.py 相同：
  - images.bin: header [u32 N, C, H, W] + float32 data
  - labels.bin: header [u32 N] + u8 data

用法:
    python scripts/prepare_real_pieces.py                         # 默认生成
    python scripts/prepare_real_pieces.py --augments-per-image 50 # 每张原图50个变体
    python scripts/prepare_real_pieces.py --preview               # 仅预览
"""

import argparse
import io
import json
import os
import random
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ==================== 类别定义 ====================

CLASS_NAMES = [
    "空位", "红帅", "红仕", "红相", "红車", "红馬", "红炮", "红兵",
    "黑将", "黑士", "黑象", "黑車", "黑馬", "黑炮", "黑卒",
]

# 标准文件名 → class_id 映射 (适用于象棋名手/兵河五四/鲨鱼象棋)
STANDARD_NAME_MAP = {
    "rk": 1, "ra": 2, "rb": 3, "rr": 4, "rn": 5, "rc": 6, "rp": 7,
    "bk": 8, "ba": 9, "bb": 10, "br": 11, "bn": 12, "bc": 13, "bp": 14,
}

# 象棋奇兵编号 → class_id 映射 (根据 preview_pieces.png 从左到右)
QIBING_NUM_MAP = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14,
}


# ==================== 软件棋子源定义 ====================

PIECE_SOURCES = [
    {
        "name": "象棋奇兵",
        "base": r"D:\SOFTWARE\象棋奇兵2009比赛版\象棋奇兵2009比赛版\bmp",
        "sizes": ["48", "36", "24"],
        "naming": "numbered",  # 1.bmp ~ 14.bmp
        "ext": ".bmp",
    },
    {
        "name": "象棋名手",
        "base": r"D:\SOFTWARE\象棋名手3.26\png",
        "sizes": ["large", "medium", "small"],
        "naming": "standard",  # ra.png, rk.png, ...
        "ext": ".png",
    },
    {
        "name": "兵河五四",
        "base": r"D:\SOFTWARE\兵河五四3.6\Piece",
        "sizes": ["large", "middle", "small"],
        "naming": "standard",
        "ext": ".png",
    },
    {
        "name": "鲨鱼象棋",
        "base": r"D:\SOFTWARE\鲨鱼象棋\img",
        "sizes": ["large", "middle", "small"],
        "naming": "standard",
        "ext": ".bmp",
    },
]


# ==================== 棋盘背景色池 ====================
# (复用 generate_chess_data.py 的棋盘背景)

BOARD_BGS = [
    (200, 210, 190), (220, 190, 140), (200, 180, 130), (210, 200, 170),
    (180, 200, 180), (60, 140, 70), (160, 190, 180), (170, 170, 170),
    (190, 180, 160), (215, 195, 150), (230, 220, 200), (235, 235, 230),
    (165, 145, 120),
]

GRID_COLORS_DARK = [(40, 40, 40), (60, 50, 40), (80, 70, 60), (50, 50, 50), (30, 30, 30)]
GRID_COLORS_LIGHT = [(200, 200, 200), (230, 230, 230), (220, 215, 200)]


def select_grid_color(board_bg):
    """根据棋盘亮度选网格线颜色"""
    lum = 0.299 * board_bg[0] + 0.587 * board_bg[1] + 0.114 * board_bg[2]
    if lum < 140:
        return random.choice(GRID_COLORS_LIGHT)
    else:
        return random.choice(GRID_COLORS_DARK)


def draw_simple_grid(draw, size, board_bg):
    """画简单的十字格线"""
    grid_color = select_grid_color(board_bg)
    cx, cy = size // 2, size // 2
    line_w = max(1, size // 40)
    # 随机决定画哪些线段 (模拟不同格点位置)
    if random.random() < 0.8:
        draw.line([(0, cy), (size - 1, cy)], fill=grid_color, width=line_w)
    if random.random() < 0.8:
        draw.line([(cx, 0), (cx, size - 1)], fill=grid_color, width=line_w)
    # 偶尔画斜线 (九宫格位置)
    if random.random() < 0.15:
        draw.line([(cx, cy), (size - 1, size - 1)], fill=grid_color, width=line_w)


# ==================== 棋子处理 ====================

def load_piece_image(path):
    """加载棋子图片，转为 RGBA"""
    img = Image.open(path).convert("RGBA")
    return img


def extract_circular_piece(img):
    """从棋子贴图中提取圆形区域

    大多数棋子贴图是正方形背景 + 圆形棋子。
    我们用圆形 mask 只保留中心区域。
    """
    w, h = img.size
    size = min(w, h)

    # 裁成正方形
    if w != h:
        left = (w - size) // 2
        top = (h - size) // 2
        img = img.crop((left, top, left + size, top + size))

    # 创建圆形 mask
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    # 稍微缩小半径，避免边缘锯齿
    margin = max(1, size // 30)
    draw.ellipse([margin, margin, size - 1 - margin, size - 1 - margin], fill=255)

    # 应用 mask
    img.putalpha(mask)
    return img


def paste_piece_on_board(piece_img, patch_size, output_size, board_bg):
    """将棋子贴图粘贴到棋盘背景上

    Args:
        piece_img: RGBA 棋子图片
        patch_size: 画布大小 (用于绘制网格)
        output_size: 最终输出尺寸
        board_bg: 棋盘背景色

    Returns:
        RGB Image of output_size x output_size
    """
    # 1. 创建棋盘背景
    canvas = Image.new("RGB", (patch_size, patch_size), board_bg)
    draw = ImageDraw.Draw(canvas)

    # 2. 画网格线
    draw_simple_grid(draw, patch_size, board_bg)

    # 3. 缩放棋子
    piece_diameter = int(patch_size * random.uniform(0.60, 0.92))
    piece_resized = piece_img.resize((piece_diameter, piece_diameter), Image.BILINEAR)

    # 4. 居中粘贴 (带轻微偏移)
    max_offset = max(1, patch_size // 20)
    offset_x = random.randint(-max_offset, max_offset)
    offset_y = random.randint(-max_offset, max_offset)

    paste_x = (patch_size - piece_diameter) // 2 + offset_x
    paste_y = (patch_size - piece_diameter) // 2 + offset_y

    canvas.paste(piece_resized, (paste_x, paste_y), piece_resized)

    # 5. resize 到目标尺寸
    if patch_size != output_size:
        canvas = canvas.resize((output_size, output_size), Image.BILINEAR)

    return canvas


def augment_image(img):
    """对 patch 应用随机增强

    Returns: augmented PIL Image
    """
    # 亮度
    if random.random() < 0.7:
        factor = random.uniform(0.82, 1.18)
        img = ImageEnhance.Brightness(img).enhance(factor)

    # 对比度
    if random.random() < 0.7:
        factor = random.uniform(0.82, 1.18)
        img = ImageEnhance.Contrast(img).enhance(factor)

    # 饱和度
    if random.random() < 0.5:
        factor = random.uniform(0.85, 1.15)
        img = ImageEnhance.Color(img).enhance(factor)

    # 轻微旋转
    if random.random() < 0.5:
        angle = random.uniform(-4, 4)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=None)

    # 轻微模糊 (模拟低分辨率截图)
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # JPEG 压缩模拟
    if random.random() < 0.3:
        quality = random.randint(60, 90)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

    return img


# ==================== 数据集生成 ====================

def scan_all_pieces():
    """扫描所有软件的棋子文件

    Returns: list of (path, class_id, software_name, size_name)
    """
    pieces = []

    for source in PIECE_SOURCES:
        base = Path(source["base"])
        if not base.exists():
            print(f"  [跳过] {source['name']}: 目录不存在 ({base})")
            continue

        for size_name in source["sizes"]:
            size_dir = base / size_name
            if not size_dir.exists():
                continue

            if source["naming"] == "numbered":
                # 象棋奇兵: 1.bmp ~ 14.bmp
                for num in range(1, 15):
                    path = size_dir / f"{num}{source['ext']}"
                    if path.exists():
                        class_id = QIBING_NUM_MAP[num]
                        pieces.append((str(path), class_id, source["name"], size_name))

            elif source["naming"] == "standard":
                # ra.png, rk.png, ... ba.png, bk.png
                for name_prefix, class_id in STANDARD_NAME_MAP.items():
                    path = size_dir / f"{name_prefix}{source['ext']}"
                    if path.exists():
                        pieces.append((str(path), class_id, source["name"], size_name))

    return pieces


def generate_real_dataset(
    pieces,
    augments_per_image=80,
    output_size=28,
    seed=99,
    split="train",
):
    """从真实棋子贴图生成数据集

    Args:
        pieces: scan_all_pieces() 返回的列表
        augments_per_image: 每张原始图生成的变体数量
        output_size: 输出 patch 尺寸
        seed: 随机种子
        split: "train" (做增强) 或 "test" (干净版本)

    Returns:
        (images_array, labels_array)
    """
    random.seed(seed)
    np.random.seed(seed)

    images = []
    labels = []
    total = 0
    source_stats = {}

    for path, class_id, sw_name, size_name in pieces:
        try:
            raw_img = load_piece_image(path)
            piece_img = extract_circular_piece(raw_img)
        except Exception as e:
            print(f"  [警告] 加载失败: {path}: {e}")
            continue

        # 统计
        source_stats[sw_name] = source_stats.get(sw_name, 0) + 1

        if split == "test":
            # 测试集：干净版本，只生成 1 张
            board_bg = random.choice(BOARD_BGS)
            patch_size = random.randint(48, 72)
            patch = paste_piece_on_board(piece_img, patch_size, output_size, board_bg)

            arr = np.array(patch, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            images.append(arr)
            labels.append(class_id)
            total += 1
        else:
            # 训练集：每张原图生成多个增强变体
            num_augs = augments_per_image
            for _ in range(num_augs):
                board_bg = random.choice(BOARD_BGS)
                patch_size = random.randint(40, 80)
                patch = paste_piece_on_board(piece_img, patch_size, output_size, board_bg)
                patch = augment_image(patch)

                arr = np.array(patch, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)  # HWC → CHW
                images.append(arr)
                labels.append(class_id)
                total += 1

                if total % 5000 == 0:
                    print(f"  [{split}] 已生成 {total} 样本...")

    # 打乱
    indices = list(range(len(images)))
    random.shuffle(indices)
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.uint8)

    # 统计信息
    print(f"\n  [{split}] 来源统计:")
    for sw_name, count in sorted(source_stats.items()):
        if split == "train":
            print(f"    {sw_name}: {count} 原图 × {augments_per_image} = {count * augments_per_image} 样本")
        else:
            print(f"    {sw_name}: {count} 样本")

    # 类别统计
    print(f"\n  [{split}] 类别统计:")
    for cid in range(1, 15):
        count = np.sum(labels_arr == cid)
        print(f"    [{cid:2d}] {CLASS_NAMES[cid]}: {count} 样本")

    return images_arr, labels_arr


# ==================== 保存函数 ====================

def save_binary(images, labels, output_dir):
    """保存为二进制格式 (与 generate_chess_data.py 格式一致)"""
    os.makedirs(output_dir, exist_ok=True)

    n, c, h, w = images.shape

    img_path = os.path.join(output_dir, "images.bin")
    with open(img_path, "wb") as f:
        f.write(struct.pack("<IIII", n, c, h, w))
        f.write(images.tobytes())
    print(f"  图像数据: {img_path} ({os.path.getsize(img_path) / 1024 / 1024:.1f} MB)")

    lbl_path = os.path.join(output_dir, "labels.bin")
    with open(lbl_path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(labels.tobytes())
    print(f"  标签数据: {lbl_path} ({os.path.getsize(lbl_path) / 1024:.1f} KB)")

    meta = {
        "num_samples": int(n),
        "channels": int(c),
        "height": int(h),
        "width": int(w),
        "num_classes": 15,
        "source": "real_pieces",
        "format": "float32 [0,1] normalized, CHW layout",
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def save_preview(images, labels, output_dir, split, count=150):
    """保存预览图"""
    os.makedirs(output_dir, exist_ok=True)

    per_class = max(1, count // 14)  # 14 类棋子 (不含空位)
    patch_h = images.shape[2]
    patch_w = images.shape[3]
    margin = 2
    header_h = 20

    class_samples = {}
    for cid in range(1, 15):
        idxs = np.where(labels == cid)[0]
        class_samples[cid] = idxs[:per_class]

    cols = 14
    rows = per_class
    total_w = cols * (patch_w + margin) + margin
    total_h = header_h + rows * (patch_h + margin) + margin

    preview = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(preview)

    try:
        header_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 11)
    except Exception:
        header_font = ImageFont.load_default()

    for idx, cid in enumerate(range(1, 15)):
        x = margin + idx * (patch_w + margin)
        draw.text((x + 2, 2), CLASS_NAMES[cid][:2], fill=(0, 0, 0), font=header_font)

    for idx, cid in enumerate(range(1, 15)):
        for row_idx, global_idx in enumerate(class_samples[cid]):
            x = margin + idx * (patch_w + margin)
            y = header_h + margin + row_idx * (patch_h + margin)

            arr = (images[global_idx].transpose(1, 2, 0) * 255).astype(np.uint8)
            patch_img = Image.fromarray(arr)

            if y + patch_h <= total_h:
                preview.paste(patch_img, (x, y))

    preview_path = os.path.join(output_dir, f"preview_real_{split}.png")
    preview.save(preview_path)
    print(f"  预览图: {preview_path}")
    return preview_path


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="真实棋子资源处理器")
    parser.add_argument("--augments-per-image", type=int, default=80,
                        help="训练集: 每张原图生成的增强变体数 (默认 80)")
    parser.add_argument("--size", type=int, default=28, help="输出 patch 尺寸")
    parser.add_argument("--output", type=str, default="data/chinese_chess_real",
                        help="输出根目录")
    parser.add_argument("--train-seed", type=int, default=99, help="训练集随机种子")
    parser.add_argument("--test-seed", type=int, default=199, help="测试集随机种子")
    parser.add_argument("--preview", action="store_true", help="仅生成预览")
    args = parser.parse_args()

    print("=" * 60)
    print("真实棋子资源处理器")
    print("=" * 60)

    # 1. 扫描所有棋子文件
    print("\n[1/3] 扫描棋子资源...")
    pieces = scan_all_pieces()
    print(f"  共找到 {len(pieces)} 张棋子贴图")

    if not pieces:
        print("\n[错误] 未找到任何棋子资源文件！")
        return

    # 按软件统计
    sw_counts = {}
    for _, _, sw_name, _ in pieces:
        sw_counts[sw_name] = sw_counts.get(sw_name, 0) + 1
    for sw_name, count in sorted(sw_counts.items()):
        print(f"    {sw_name}: {count} 张")

    # 2. 生成训练集
    print(f"\n[2/3] 生成训练集 (每张原图 {args.augments_per_image} 个变体)...")
    train_images, train_labels = generate_real_dataset(
        pieces, args.augments_per_image, args.size, args.train_seed, "train"
    )
    print(f"  训练集: {train_images.shape}")

    train_dir = os.path.join(args.output, "train")
    print(f"\n  保存预览图...")
    save_preview(train_images, train_labels, train_dir, "train")
    if not args.preview:
        print(f"\n  保存二进制数据...")
        save_binary(train_images, train_labels, train_dir)

    # 3. 生成测试集 (干净版本)
    print(f"\n[3/3] 生成测试集 (干净版本, 无增强)...")
    test_images, test_labels = generate_real_dataset(
        pieces, 1, args.size, args.test_seed, "test"
    )
    print(f"  测试集: {test_images.shape}")

    test_dir = os.path.join(args.output, "test")
    print(f"\n  保存预览图...")
    save_preview(test_images, test_labels, test_dir, "test")
    if not args.preview:
        print(f"\n  保存二进制数据...")
        save_binary(test_images, test_labels, test_dir)

    # 摘要
    print(f"\n{'=' * 60}")
    print(f"完成！")
    print(f"  训练集: {train_images.shape[0]} 样本 → {train_dir}")
    print(f"  测试集: {test_images.shape[0]} 样本 → {test_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

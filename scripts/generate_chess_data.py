#!/usr/bin/env python3
"""
中国象棋合成训练数据生成器

为 only_torch CNN 棋子分类器生成多样化训练 patch：
- 15 类：1 空位 + 7 红子 + 7 黑子
- 多字体、多颜色、多棋盘背景、多棋子/格比例
- 输出二进制格式供 Rust 直接读取

用法：
    python scripts/generate_chess_data.py                    # 默认生成
    python scripts/generate_chess_data.py --preview          # 仅预览，不保存二进制
    python scripts/generate_chess_data.py --samples 100000   # 指定样本数量
    python scripts/generate_chess_data.py --output data/chess # 指定输出目录
"""

import argparse
import json
import math
import os
import random
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ==================== 类别定义 ====================

# 15 类: 0=空位, 1-7=红方, 8-14=黑方
CLASSES = {
    0: ("empty", "空位", None, None),
    # 红方 (红色字)
    1: ("red_king", "红帅", "帅", "red"),
    2: ("red_advisor", "红仕", "仕", "red"),
    3: ("red_bishop", "红相", "相", "red"),
    4: ("red_rook", "红車", "車", "red"),
    5: ("red_knight", "红馬", "馬", "red"),
    6: ("red_cannon", "红炮", "炮", "red"),
    7: ("red_pawn", "红兵", "兵", "red"),
    # 黑方 (黑色字)
    8: ("black_king", "黑将", "将", "black"),
    9: ("black_advisor", "黑士", "士", "black"),
    10: ("black_bishop", "黑象", "象", "black"),
    11: ("black_rook", "黑車", "車", "black"),
    12: ("black_knight", "黑馬", "馬", "black"),
    13: ("black_cannon", "黑炮", "炮", "black"),
    14: ("black_pawn", "黑卒", "卒", "black"),
}

# ==================== 样式配置 ====================

# 可用的中文字体 (Windows)
FONT_PATHS = [
    "C:/Windows/Fonts/simkai.ttf",    # 楷体
    "C:/Windows/Fonts/simsun.ttc",    # 宋体
    "C:/Windows/Fonts/simhei.ttf",    # 黑体
    "C:/Windows/Fonts/simfang.ttf",   # 仿宋
    "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
]

# 红方字色变体
RED_TEXT_COLORS = [
    (200, 30, 30),    # 标准红
    (180, 20, 20),    # 深红
    (220, 50, 30),    # 亮红
    (160, 10, 10),    # 暗红
    (190, 40, 40),    # 中红
]

# 黑方字色变体
BLACK_TEXT_COLORS = [
    (30, 30, 30),     # 标准黑
    (50, 50, 50),     # 深灰
    (10, 10, 10),     # 纯黑
    (60, 50, 40),     # 暖黑
]

# 棋子圆形背景色
PIECE_BG_COLORS = [
    (240, 220, 180),  # 米黄
    (245, 230, 190),  # 浅黄
    (235, 210, 170),  # 深米
    (250, 240, 210),  # 象牙白
    (230, 215, 185),  # 暖米
    (255, 245, 220),  # 奶白
    (225, 205, 165),  # 深暖米
]

# 圆形边框色
BORDER_COLORS = [
    (60, 40, 20),     # 深棕
    (40, 30, 15),     # 暗棕
    (80, 50, 25),     # 中棕
    (30, 30, 30),     # 深灰
    (100, 70, 40),    # 浅棕
]

# 棋盘背景色
BOARD_BG_COLORS = [
    (200, 210, 190),  # 浅绿灰（类似象棋奇兵）
    (220, 190, 140),  # 木纹色
    (200, 180, 130),  # 深木色
    (210, 200, 170),  # 米色
    (180, 200, 180),  # 淡绿
    (190, 180, 160),  # 暖灰
    (215, 195, 150),  # 黄木色
    (230, 220, 200),  # 浅米
]

# 格线颜色
GRID_LINE_COLORS = [
    (40, 40, 40),     # 深黑
    (60, 50, 40),     # 暖黑
    (80, 70, 60),     # 灰棕
    (50, 50, 50),     # 灰
    (30, 30, 30),     # 纯黑
]

# ==================== 棋盘网格特征 ====================

# 中国象棋 9 列 10 行，各格点的背景特征
# 特征：交叉类型（十字、T 形、L 形、斜线等）
def get_grid_feature(row: int, col: int) -> str:
    """获取格点 (row, col) 的背景特征类型"""
    # 九宫斜线 (row=0-2/7-9, col=3-5)
    is_palace = (row <= 2 and 3 <= col <= 5) or (row >= 7 and 3 <= col <= 5)

    # 楚河汉界区域 (row=4,5)
    is_river = row in (4, 5)

    # 边界判断
    at_top = row == 0
    at_bottom = row == 9
    at_left = col == 0
    at_right = col == 8

    features = []
    if is_palace:
        features.append("palace")
    if is_river:
        features.append("river")
    if at_top or at_bottom:
        features.append("edge_h")
    if at_left or at_right:
        features.append("edge_v")
    if not features:
        features.append("normal")

    return "_".join(features)


# ==================== 生成函数 ====================

def load_fonts() -> list:
    """加载可用的中文字体"""
    fonts = []
    for path in FONT_PATHS:
        if os.path.exists(path):
            fonts.append(path)
    if not fonts:
        print("警告：未找到任何中文字体，尝试默认字体")
        fonts.append("arial.ttf")
    return fonts


def draw_grid_background(
    draw: ImageDraw.Draw,
    size: int,
    row: int,
    col: int,
    bg_color: tuple,
    line_color: tuple,
    line_width: int = 1,
):
    """在 patch 上绘制对应格点位置的棋盘网格线

    根据 (row, col) 位置绘制正确的格线样式（十字、T 形、L 形、斜线等）
    """
    cx, cy = size // 2, size // 2
    half = size // 2

    # 基本格线：根据位置画对应方向的线段
    at_top = row == 0
    at_bottom = row == 9
    at_left = col == 0
    at_right = col == 8
    is_river_top = row == 4    # 楚河上方
    is_river_bottom = row == 5  # 汉界下方

    # 水平线
    if not at_left:
        draw.line([(0, cy), (cx, cy)], fill=line_color, width=line_width)
    if not at_right:
        draw.line([(cx, cy), (size - 1, cy)], fill=line_color, width=line_width)

    # 垂直线
    if not at_top and not is_river_bottom:
        draw.line([(cx, 0), (cx, cy)], fill=line_color, width=line_width)
    if not at_bottom and not is_river_top:
        draw.line([(cx, cy), (cx, size - 1)], fill=line_color, width=line_width)

    # 九宫斜线
    is_palace_top = (0 <= row <= 2) and (3 <= col <= 5)
    is_palace_bottom = (7 <= row <= 9) and (3 <= col <= 5)

    if is_palace_top or is_palace_bottom:
        # 右下斜线
        if col < 5 and ((is_palace_top and row < 2) or (is_palace_bottom and row < 9)):
            draw.line([(cx, cy), (size - 1, size - 1)], fill=line_color, width=line_width)
        # 左上斜线
        if col > 3 and ((is_palace_top and row > 0) or (is_palace_bottom and row > 7)):
            draw.line([(0, 0), (cx, cy)], fill=line_color, width=line_width)
        # 左下斜线
        if col > 3 and ((is_palace_top and row < 2) or (is_palace_bottom and row < 9)):
            draw.line([(0, size - 1), (cx, cy)], fill=line_color, width=line_width)
        # 右上斜线
        if col < 5 and ((is_palace_top and row > 0) or (is_palace_bottom and row > 7)):
            draw.line([(size - 1, 0), (cx, cy)], fill=line_color, width=line_width)

    # 兵/炮位星位标记（简化版：小十字角标）
    pawn_positions = {
        (3, 0), (3, 2), (3, 4), (3, 6), (3, 8),
        (6, 0), (6, 2), (6, 4), (6, 6), (6, 8),
    }
    cannon_positions = {(2, 1), (2, 7), (7, 1), (7, 7)}

    if (row, col) in pawn_positions or (row, col) in cannon_positions:
        mark_len = max(3, size // 10)
        mark_gap = max(2, size // 16)
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            sx = cx + dx * mark_gap
            sy = cy + dy * mark_gap
            if 0 <= sx + dx * mark_len < size and 0 <= sy < size:
                draw.line(
                    [(sx, sy), (sx + dx * mark_len, sy)],
                    fill=line_color, width=line_width
                )
            if 0 <= sx < size and 0 <= sy + dy * mark_len < size:
                draw.line(
                    [(sx, sy), (sx, sy + dy * mark_len)],
                    fill=line_color, width=line_width
                )


def draw_piece(
    img: Image.Image,
    char: str,
    side: str,
    font_path: str,
    piece_diameter: int,
    text_color: tuple,
    bg_color: tuple,
    border_color: tuple,
    border_width: int = 2,
):
    """在图片中心绘制一个圆形棋子"""
    w, h = img.size
    cx, cy = w // 2, h // 2
    r = piece_diameter // 2

    draw = ImageDraw.Draw(img)

    # 外圆 (边框)
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=bg_color,
        outline=border_color,
        width=border_width,
    )

    # 内圆线 (部分风格有双线边框)
    if random.random() < 0.4:
        inner_gap = max(2, border_width + 1)
        draw.ellipse(
            [cx - r + inner_gap, cy - r + inner_gap,
             cx + r - inner_gap, cy + r - inner_gap],
            outline=border_color,
            width=1,
        )

    # 绘制文字
    font_size = int(piece_diameter * 0.55)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = font.getbbox(char)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = cx - tw // 2 - bbox[0]
    ty = cy - th // 2 - bbox[1]

    draw.text((tx, ty), char, fill=text_color, font=font)


def generate_patch(
    class_id: int,
    patch_size: int,
    output_size: int,
    row: int,
    col: int,
    font_path: str,
    piece_ratio: float,
    board_bg: tuple,
    grid_line_color: tuple,
    piece_bg: tuple,
    border_color: tuple,
    text_color: tuple,
    offset_x: int = 0,
    offset_y: int = 0,
    brightness_factor: float = 1.0,
) -> Image.Image:
    """生成一个训练 patch

    Args:
        class_id: 类别 (0-14)
        patch_size: 原始 patch 大小（模拟裁切尺寸）
        output_size: 最终输出大小（resize 后）
        row, col: 棋盘格点位置 (0-9, 0-8)
        font_path: 字体路径
        piece_ratio: 棋子直径/patch 大小 比例
        board_bg: 棋盘背景色
        grid_line_color: 格线颜色
        piece_bg: 棋子圆形背景色
        border_color: 圆形边框色
        text_color: 文字颜色
        offset_x, offset_y: 微小偏移 (px)
        brightness_factor: 亮度因子
    """
    # 1. 创建背景
    img = Image.new("RGB", (patch_size, patch_size), board_bg)
    draw = ImageDraw.Draw(img)

    # 2. 画格线
    line_width = max(1, patch_size // 40)
    draw_grid_background(draw, patch_size, row, col, board_bg, grid_line_color, line_width)

    # 3. 如果不是空位，画棋子
    if class_id > 0:
        _, _, char, side = CLASSES[class_id]
        piece_diameter = int(patch_size * piece_ratio)
        border_width = max(1, piece_diameter // 20)

        draw_piece(
            img, char, side, font_path,
            piece_diameter, text_color, piece_bg,
            border_color, border_width
        )

    # 4. 应用偏移（通过裁切+重填实现）
    if offset_x != 0 or offset_y != 0:
        shifted = Image.new("RGB", (patch_size, patch_size), board_bg)
        shifted.paste(img, (offset_x, offset_y))
        img = shifted

    # 5. 亮度调整
    if abs(brightness_factor - 1.0) > 0.01:
        arr = np.array(img, dtype=np.float32)
        arr = np.clip(arr * brightness_factor, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    # 6. resize 到目标大小
    if patch_size != output_size:
        img = img.resize((output_size, output_size), Image.BILINEAR)

    return img


def generate_dataset(
    num_samples: int,
    output_size: int = 48,
    seed: int = 42,
) -> tuple:
    """生成完整的训练数据集

    Returns:
        (images_array, labels_array): np.ndarray
        - images: [N, 3, H, W] float32, 归一化到 [0, 1]
        - labels: [N] uint8
    """
    random.seed(seed)
    np.random.seed(seed)

    fonts = load_fonts()
    print(f"  可用字体: {len(fonts)} 种")

    # 每类的基础样本数（空位可以少一些，棋子类别平均分配）
    # class 0 (空位) 占 ~20%，其余 14 类平均分配 ~80%
    empty_count = int(num_samples * 0.2)
    piece_count_per_class = (num_samples - empty_count) // 14
    leftover = num_samples - empty_count - piece_count_per_class * 14

    class_counts = {0: empty_count + leftover}
    for cid in range(1, 15):
        class_counts[cid] = piece_count_per_class

    print(f"  类别分布: 空位 {class_counts[0]}, 每种棋子 ~{piece_count_per_class}")

    images = []
    labels = []
    total = 0

    for class_id, count in class_counts.items():
        for _ in range(count):
            # 随机选取样式参数
            font_path = random.choice(fonts)
            piece_ratio = random.uniform(0.60, 0.92)
            board_bg = random.choice(BOARD_BG_COLORS)
            grid_color = random.choice(GRID_LINE_COLORS)
            piece_bg = random.choice(PIECE_BG_COLORS)
            border_color = random.choice(BORDER_COLORS)

            if class_id == 0:
                text_color = (0, 0, 0)  # 不会用到
            elif class_id <= 7:
                text_color = random.choice(RED_TEXT_COLORS)
            else:
                text_color = random.choice(BLACK_TEXT_COLORS)

            # 随机格点位置（不遵守棋规，覆盖全部背景类型）
            row = random.randint(0, 9)
            col = random.randint(0, 8)

            # 原始 patch 大小（模拟不同棋盘缩放）
            patch_size = random.randint(40, 80)

            # 微小偏移 (±2px in output space, 按比例映射到 patch space)
            max_offset = max(1, patch_size // 20)
            offset_x = random.randint(-max_offset, max_offset)
            offset_y = random.randint(-max_offset, max_offset)

            # 亮度扰动
            brightness = random.uniform(0.88, 1.12)

            img = generate_patch(
                class_id=class_id,
                patch_size=patch_size,
                output_size=output_size,
                row=row, col=col,
                font_path=font_path,
                piece_ratio=piece_ratio,
                board_bg=board_bg,
                grid_line_color=grid_color,
                piece_bg=piece_bg,
                border_color=border_color,
                text_color=text_color,
                offset_x=offset_x,
                offset_y=offset_y,
                brightness_factor=brightness,
            )

            # 转为 numpy 数组 [3, H, W], 归一化到 [0,1]
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            images.append(arr)
            labels.append(class_id)

            total += 1
            if total % 5000 == 0:
                print(f"  已生成 {total}/{num_samples} 样本...")

    # 打乱顺序
    indices = list(range(len(images)))
    random.shuffle(indices)
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.uint8)

    return images_arr, labels_arr


def save_binary(images: np.ndarray, labels: np.ndarray, output_dir: str):
    """保存为 Rust 可读取的二进制格式

    images.bin 格式:
        header: [uint32 N] [uint32 C] [uint32 H] [uint32 W]
        data: N*C*H*W float32 values (little-endian)

    labels.bin 格式:
        header: [uint32 N]
        data: N uint8 values
    """
    os.makedirs(output_dir, exist_ok=True)

    n, c, h, w = images.shape

    # 保存图像
    img_path = os.path.join(output_dir, "images.bin")
    with open(img_path, "wb") as f:
        f.write(struct.pack("<IIII", n, c, h, w))
        f.write(images.tobytes())
    print(f"  图像数据: {img_path} ({os.path.getsize(img_path) / 1024 / 1024:.1f} MB)")

    # 保存标签
    lbl_path = os.path.join(output_dir, "labels.bin")
    with open(lbl_path, "wb") as f:
        f.write(struct.pack("<I", n))
        f.write(labels.tobytes())
    print(f"  标签数据: {lbl_path} ({os.path.getsize(lbl_path) / 1024:.1f} KB)")

    # 保存元数据
    meta = {
        "num_samples": int(n),
        "channels": int(c),
        "height": int(h),
        "width": int(w),
        "num_classes": 15,
        "classes": {str(k): v[1] for k, v in CLASSES.items()},
        "format": "float32 [0,1] normalized, CHW layout",
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元数据: {meta_path}")


def save_preview(images: np.ndarray, labels: np.ndarray, output_dir: str, count: int = 150):
    """保存预览图（每类展示若干样本，按列排列）"""
    os.makedirs(output_dir, exist_ok=True)

    per_class = max(1, count // 15)
    patch_h = images.shape[2]
    patch_w = images.shape[3]
    margin = 2
    header_h = 18

    # 按类收集样本
    class_samples = {}
    for cid in range(15):
        idxs = np.where(labels == cid)[0]
        class_samples[cid] = idxs[:per_class]

    cols = 15
    rows = per_class
    total_w = cols * (patch_w + margin) + margin
    total_h = header_h + rows * (patch_h + margin) + margin

    preview = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(preview)

    # 列标题
    try:
        header_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 11)
    except Exception:
        header_font = ImageFont.load_default()

    for cid in range(15):
        x = margin + cid * (patch_w + margin)
        label_text = CLASSES[cid][1][:2]
        draw.text((x + 2, 2), label_text, fill=(0, 0, 0), font=header_font)

    # 绘制样本
    for cid in range(15):
        for row_idx, global_idx in enumerate(class_samples[cid]):
            x = margin + cid * (patch_w + margin)
            y = header_h + margin + row_idx * (patch_h + margin)

            arr = (images[global_idx].transpose(1, 2, 0) * 255).astype(np.uint8)
            patch_img = Image.fromarray(arr)

            if y + patch_h <= total_h:
                preview.paste(patch_img, (x, y))

    preview_path = os.path.join(output_dir, "preview.png")
    preview.save(preview_path)
    print(f"  预览图: {preview_path}")
    return preview_path


def main():
    parser = argparse.ArgumentParser(description="中国象棋合成训练数据生成器")
    parser.add_argument("--samples", type=int, default=15000, help="总样本数量")
    parser.add_argument("--size", type=int, default=28, help="输出 patch 尺寸")
    parser.add_argument("--output", type=str, default="data/chinese_chess", help="输出目录")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--preview", action="store_true", help="仅生成预览（不保存二进制）")
    parser.add_argument("--preview-count", type=int, default=150, help="预览样本数量")
    args = parser.parse_args()

    print("=" * 60)
    print("中国象棋合成训练数据生成器")
    print("=" * 60)
    print(f"\n配置：")
    print(f"  样本数量: {args.samples}")
    print(f"  输出尺寸: {args.size}x{args.size}")
    print(f"  输出目录: {args.output}")
    print(f"  随机种子: {args.seed}")

    print(f"\n生成数据中...")
    images, labels = generate_dataset(args.samples, args.size, args.seed)
    print(f"  完成！形状: images={images.shape}, labels={labels.shape}")

    # 统计
    print(f"\n类别统计：")
    for cid in range(15):
        count = np.sum(labels == cid)
        name = CLASSES[cid][1]
        print(f"  [{cid:2d}] {name}: {count} 样本")

    # 保存预览
    print(f"\n保存预览图...")
    preview_path = save_preview(images, labels, args.output, args.preview_count)

    if not args.preview:
        # 保存二进制数据
        print(f"\n保存二进制数据...")
        save_binary(images, labels, args.output)

    print(f"\n[OK] 完成！")


if __name__ == "__main__":
    main()

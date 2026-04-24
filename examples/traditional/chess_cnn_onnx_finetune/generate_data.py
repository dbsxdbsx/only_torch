#!/usr/bin/env python3
"""
中国象棋合成训练数据生成器 v3

改进 (v3):
- 三种棋子风格模式: 经典浅底 / 木纹自然色 / 饱和对比色
- 扩充棋盘底色: 纯绿、蓝绿/青、灰色、白色、棕褐等
- 网格线颜色自适应: 深色棋盘自动使用浅色线条
- 字形变体 (繁体/异体字: 帥/俥/傌/砲/將 等)
- 训练集/测试集风格分离 (不同字体、不同配色方案)

用法:
    python scripts/generate_chess_data.py                    # 默认: 生成训练+测试集
    python scripts/generate_chess_data.py --preview          # 仅预览，不保存二进制
    python scripts/generate_chess_data.py --train-samples 12000 --test-samples 3000
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
    0: ("empty", "空位"),
    1: ("red_king", "红帅"),
    2: ("red_advisor", "红仕"),
    3: ("red_bishop", "红相"),
    4: ("red_rook", "红車"),
    5: ("red_knight", "红馬"),
    6: ("red_cannon", "红炮"),
    7: ("red_pawn", "红兵"),
    8: ("black_king", "黑将"),
    9: ("black_advisor", "黑士"),
    10: ("black_bishop", "黑象"),
    11: ("black_rook", "黑車"),
    12: ("black_knight", "黑馬"),
    13: ("black_cannon", "黑炮"),
    14: ("black_pawn", "黑卒"),
}

# 每类可用的字形变体（同一类别，不同写法）
# 训练集和测试集都包含全部变体，确保模型能识别所有写法
CHAR_VARIANTS = {
    # 红方
    1: ["帅", "帥"],                 # 简体/繁体
    2: ["仕"],
    3: ["相"],
    4: ["車", "俥", "车"],           # 传统/红方专用/简体
    5: ["馬", "傌", "马"],           # 传统/红方专用/简体
    6: ["炮", "砲"],                 # 常见/异体
    7: ["兵"],
    # 黑方
    8: ["将", "將"],                 # 简体/繁体
    9: ["士"],
    10: ["象"],
    11: ["車", "车"],                # 传统/简体
    12: ["馬", "马"],                # 传统/简体
    13: ["炮", "砲"],                # 常见/异体
    14: ["卒"],
}


def get_side(class_id: int) -> str | None:
    """红/黑方判断"""
    if class_id == 0:
        return None
    return "red" if class_id <= 7 else "black"


# ==================== 样式池（训练/测试分离） ====================
#
# 核心原则：训练集和测试集使用不同的字体和配色方案，
# 确保测试时模型面对的是"从未见过"的视觉风格。
# 字形变体两边都有，风格差异来自字体和颜色。

FONT_DIR = "C:/Windows/Fonts"

STYLE_POOLS = {
    "train": {
        "fonts": [
            f"{FONT_DIR}/simkai.ttf",       # 楷体
            f"{FONT_DIR}/simsun.ttc",       # 宋体
            f"{FONT_DIR}/simhei.ttf",       # 黑体
            f"{FONT_DIR}/simfang.ttf",      # 仿宋
            f"{FONT_DIR}/msyh.ttc",         # 微软雅黑
        ],
        "board_bgs": [
            (200, 210, 190),    # 浅绿灰
            (220, 190, 140),    # 木纹色
            (200, 180, 130),    # 深木色
            (210, 200, 170),    # 米色
            (180, 200, 180),    # 淡绿
            (60, 140, 70),      # 纯绿 (兵河五四)
            (160, 190, 180),    # 蓝绿/青 (象棋奇兵)
            (170, 170, 170),    # 灰色 (兵河五四)
        ],
        "grid_line_colors_dark": [
            (40, 40, 40),       # 深黑
            (60, 50, 40),       # 暖黑
            (80, 70, 60),       # 灰棕
        ],
        "grid_line_colors_light": [
            (200, 200, 200),    # 浅灰
            (230, 230, 230),    # 白
        ],
    },
    "test": {
        "fonts": [
            f"{FONT_DIR}/simkai.ttf",       # 楷体
            f"{FONT_DIR}/simsun.ttc",       # 宋体
            f"{FONT_DIR}/simhei.ttf",       # 黑体
            f"{FONT_DIR}/simfang.ttf",      # 仿宋
            f"{FONT_DIR}/msyh.ttc",         # 微软雅黑
        ],
        "board_bgs": [
            (190, 180, 160),    # 暖灰
            (215, 195, 150),    # 黄木色
            (230, 220, 200),    # 浅米
            (235, 235, 230),    # 白色/极浅 (兵河五四)
            (165, 145, 120),    # 棕褐/复古 (兵河五四)
        ],
        "grid_line_colors_dark": [
            (50, 50, 50),       # 灰
            (30, 30, 30),       # 纯黑
        ],
        "grid_line_colors_light": [
            (220, 215, 200),    # 米白
        ],
    },
}


# ==================== 棋子风格模式 ====================
#
# 三种棋子视觉风格，覆盖主流象棋软件的设计差异：
#   A — 经典浅底（象棋奇兵风格）：浅米色底 + 红字/黑字
#   B — 木纹自然色（象棋名手风格）：棕色底，红黑两方底色接近
#   C — 饱和对比色（兵河五四风格）：红方饱和红底+白字，黑方深色底+金字
#
# 生成概率：A 40%, B 30%, C 30%

PIECE_MODE_WEIGHTS = [0.4, 0.3, 0.3]  # A, B, C

PIECE_MODES = {
    "train": {
        "A": {
            "piece_bgs": [(240, 220, 180), (245, 230, 190), (235, 210, 170), (250, 240, 210)],
            "red_text": [(200, 30, 30), (180, 20, 20), (220, 50, 30)],
            "black_text": [(30, 30, 30), (50, 50, 50)],
            "border": [(60, 40, 20), (40, 30, 15), (80, 50, 25)],
        },
        "B": {
            "red_piece_bgs": [(160, 110, 70), (170, 120, 80)],
            "black_piece_bgs": [(130, 90, 55), (120, 85, 50)],
            "red_text": [(120, 30, 20), (100, 20, 15)],
            "black_text": [(60, 40, 25), (50, 35, 20)],
            "border": [(90, 60, 30), (80, 55, 25)],
        },
        "C": {
            "red_piece_bgs": [(190, 40, 30), (200, 50, 35)],
            "black_piece_bgs": [(40, 40, 40), (55, 50, 45), (30, 50, 30)],
            "red_text": [(255, 240, 220), (255, 220, 180)],
            "black_text": [(220, 200, 120), (255, 240, 200)],
            "red_border": [(150, 30, 20), (160, 25, 15)],
            "black_border": [(20, 20, 20), (60, 55, 50)],
        },
    },
    "test": {
        "A": {
            "piece_bgs": [(230, 215, 185), (255, 245, 220), (225, 205, 165)],
            "red_text": [(160, 10, 10), (190, 40, 40)],
            "black_text": [(10, 10, 10), (60, 50, 40)],
            "border": [(30, 30, 30), (100, 70, 40)],
        },
        "B": {
            "red_piece_bgs": [(155, 105, 65), (165, 115, 75)],
            "black_piece_bgs": [(125, 85, 50), (115, 80, 45)],
            "red_text": [(110, 25, 15), (95, 15, 10)],
            "black_text": [(55, 35, 20), (45, 30, 15)],
            "border": [(85, 55, 25), (75, 50, 20)],
        },
        "C": {
            "red_piece_bgs": [(180, 35, 25), (170, 45, 30)],
            "black_piece_bgs": [(35, 35, 50), (45, 30, 25)],
            "red_text": [(250, 235, 210), (245, 230, 190)],
            "black_text": [(215, 195, 110), (250, 235, 195)],
            "red_border": [(140, 25, 15), (130, 30, 20)],
            "black_border": [(20, 20, 40), (50, 40, 30)],
        },
    },
}


def select_piece_style(class_id: int, split: str) -> tuple:
    """根据棋子风格模式选择协调的配色方案

    Returns: (piece_bg, text_color, border_color, mode_name)
    """
    mode_name = random.choices(['A', 'B', 'C'], weights=PIECE_MODE_WEIGHTS)[0]
    mode = PIECE_MODES[split][mode_name]
    side = get_side(class_id)

    if mode_name == 'A':
        # 经典浅底：红黑共用底色，靠文字颜色区分
        piece_bg = random.choice(mode["piece_bgs"])
        border_color = random.choice(mode["border"])
        text_color = random.choice(mode["red_text"] if side == "red" else mode["black_text"])
    else:
        # 模式 B/C：红黑两方使用不同底色
        if side == "red":
            piece_bg = random.choice(mode["red_piece_bgs"])
            text_color = random.choice(mode["red_text"])
            border_key = "red_border" if "red_border" in mode else "border"
            border_color = random.choice(mode[border_key])
        else:
            piece_bg = random.choice(mode["black_piece_bgs"])
            text_color = random.choice(mode["black_text"])
            border_key = "black_border" if "black_border" in mode else "border"
            border_color = random.choice(mode[border_key])

    return piece_bg, text_color, border_color, mode_name


def select_grid_color(board_bg: tuple, pool: dict) -> tuple:
    """根据棋盘底色亮度自动选择网格线颜色"""
    lum = 0.299 * board_bg[0] + 0.587 * board_bg[1] + 0.114 * board_bg[2]
    if lum < 140:
        return random.choice(pool["grid_line_colors_light"])
    else:
        return random.choice(pool["grid_line_colors_dark"])


# ==================== 棋盘网格特征 ====================

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

    at_top = row == 0
    at_bottom = row == 9
    at_left = col == 0
    at_right = col == 8
    is_river_top = row == 4
    is_river_bottom = row == 5

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
        if col < 5 and ((is_palace_top and row < 2) or (is_palace_bottom and row < 9)):
            draw.line([(cx, cy), (size - 1, size - 1)], fill=line_color, width=line_width)
        if col > 3 and ((is_palace_top and row > 0) or (is_palace_bottom and row > 7)):
            draw.line([(0, 0), (cx, cy)], fill=line_color, width=line_width)
        if col > 3 and ((is_palace_top and row < 2) or (is_palace_bottom and row < 9)):
            draw.line([(0, size - 1), (cx, cy)], fill=line_color, width=line_width)
        if col < 5 and ((is_palace_top and row > 0) or (is_palace_bottom and row > 7)):
            draw.line([(size - 1, 0), (cx, cy)], fill=line_color, width=line_width)

    # 兵/炮位星位标记
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


# ==================== 绘制函数 ====================

def draw_piece(
    img: Image.Image,
    char: str,
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
    char: str | None,
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
        char: 要绘制的字符（从 CHAR_VARIANTS 中随机选取），空位时为 None
        patch_size: 原始 patch 大小
        output_size: 最终输出大小（resize 后）
        row, col: 棋盘格点位置 (0-9, 0-8)
        font_path: 字体路径
        piece_ratio: 棋子直径/patch 大小 比例
        board_bg / grid_line_color / piece_bg / border_color / text_color: 样式参数
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
    if class_id > 0 and char is not None:
        piece_diameter = int(patch_size * piece_ratio)
        border_width = max(1, piece_diameter // 20)

        draw_piece(
            img, char, font_path,
            piece_diameter, text_color, piece_bg,
            border_color, border_width
        )

    # 4. 应用偏移
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


# ==================== 数据集生成 ====================

def load_fonts(split: str) -> list:
    """加载指定 split 可用的字体"""
    pool = STYLE_POOLS[split]
    fonts = []
    for path in pool["fonts"]:
        if os.path.exists(path):
            fonts.append(path)
    if not fonts:
        print(f"  警告：[{split}] 未找到任何字体，尝试 fallback 到全部字体")
        # fallback: 尝试所有字体
        for sp in STYLE_POOLS.values():
            for path in sp["fonts"]:
                if os.path.exists(path) and path not in fonts:
                    fonts.append(path)
    if not fonts:
        print("  警告：未找到任何中文字体，使用默认字体")
        fonts.append("arial.ttf")
    return fonts


def generate_dataset(
    num_samples: int,
    output_size: int = 48,
    seed: int = 42,
    split: str = "train",
) -> tuple:
    """生成数据集（训练集或测试集）

    Args:
        num_samples: 样本总数
        output_size: 输出 patch 尺寸
        seed: 随机种子
        split: "train" 或 "test"，决定使用哪个样式池

    Returns:
        (images_array, labels_array):
        - images: [N, 3, H, W] float32, [0, 1]
        - labels: [N] uint8
    """
    random.seed(seed)
    np.random.seed(seed)

    fonts = load_fonts(split)
    pool = STYLE_POOLS[split]
    print(f"  [{split}] 可用字体: {len(fonts)} 种")

    # 类别样本分配: 空位 ~20%，其余 14 类平均分配 ~80%
    empty_count = int(num_samples * 0.2)
    piece_count_per_class = (num_samples - empty_count) // 14
    leftover = num_samples - empty_count - piece_count_per_class * 14

    class_counts = {0: empty_count + leftover}
    for cid in range(1, 15):
        class_counts[cid] = piece_count_per_class

    print(f"  [{split}] 类别分布: 空位 {class_counts[0]}, 每种棋子 ~{piece_count_per_class}")

    images = []
    labels = []
    total = 0

    # 统计字形变体和风格模式使用情况
    variant_stats = {}
    mode_stats = {}

    for class_id, count in class_counts.items():
        for _ in range(count):
            # 随机选取样式参数
            font_path = random.choice(fonts)
            piece_ratio = random.uniform(0.60, 0.92)
            board_bg = random.choice(pool["board_bgs"])
            grid_color = select_grid_color(board_bg, pool)

            # 选择字形变体 + 棋子配色
            if class_id == 0:
                char = None
                text_color = (0, 0, 0)
                piece_bg = (0, 0, 0)      # 空位不画棋子
                border_color = (0, 0, 0)
            else:
                variants = CHAR_VARIANTS[class_id]
                char = random.choice(variants)
                # 统计
                key = f"{CLASSES[class_id][1]}:{char}"
                variant_stats[key] = variant_stats.get(key, 0) + 1

                piece_bg, text_color, border_color, mode = select_piece_style(class_id, split)
                mode_stats[mode] = mode_stats.get(mode, 0) + 1

            # 随机格点位置
            row = random.randint(0, 9)
            col = random.randint(0, 8)

            # 原始 patch 大小
            patch_size = random.randint(40, 80)

            # 微小偏移
            max_offset = max(1, patch_size // 20)
            offset_x = random.randint(-max_offset, max_offset)
            offset_y = random.randint(-max_offset, max_offset)

            # 亮度扰动
            brightness = random.uniform(0.88, 1.12)

            img = generate_patch(
                class_id=class_id,
                char=char,
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

            # 转为 numpy 数组 [3, H, W], [0,1]
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
            images.append(arr)
            labels.append(class_id)

            total += 1
            if total % 5000 == 0:
                print(f"  [{split}] 已生成 {total}/{num_samples} 样本...")

    # 打乱顺序
    indices = list(range(len(images)))
    random.shuffle(indices)
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]

    images_arr = np.array(images, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.uint8)

    # 打印字形变体统计
    print(f"\n  [{split}] 字形变体统计:")
    for key in sorted(variant_stats.keys()):
        print(f"    {key}: {variant_stats[key]} 样本")

    # 打印棋子风格模式统计
    if mode_stats:
        total_pieces = sum(mode_stats.values())
        mode_labels = {'A': '经典浅底', 'B': '木纹自然色', 'C': '饱和对比色'}
        print(f"\n  [{split}] 棋子风格模式统计:")
        for m in ['A', 'B', 'C']:
            cnt = mode_stats.get(m, 0)
            pct = cnt / total_pieces * 100 if total_pieces > 0 else 0
            print(f"    模式 {m} ({mode_labels[m]}): {cnt} 样本 ({pct:.1f}%)")

    return images_arr, labels_arr


# ==================== 保存函数 ====================

def save_binary(images: np.ndarray, labels: np.ndarray, output_dir: str):
    """保存为 Rust 可读取的二进制格式

    images.bin: header [u32 N, C, H, W] + float32 data
    labels.bin: header [u32 N] + u8 data
    """
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
        "classes": {str(k): v[1] for k, v in CLASSES.items()},
        "char_variants": {str(k): v for k, v in CHAR_VARIANTS.items()},
        "format": "float32 [0,1] normalized, CHW layout",
    }
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元数据: {meta_path}")


def save_preview(
    images: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    split: str,
    count: int = 150,
):
    """保存预览图（每类展示若干样本，按列排列）"""
    os.makedirs(output_dir, exist_ok=True)

    per_class = max(1, count // 15)
    patch_h = images.shape[2]
    patch_w = images.shape[3]
    margin = 2
    header_h = 20

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

    try:
        header_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 11)
    except Exception:
        header_font = ImageFont.load_default()

    for cid in range(15):
        x = margin + cid * (patch_w + margin)
        label_text = CLASSES[cid][1][:2]
        draw.text((x + 2, 2), label_text, fill=(0, 0, 0), font=header_font)

    for cid in range(15):
        for row_idx, global_idx in enumerate(class_samples[cid]):
            x = margin + cid * (patch_w + margin)
            y = header_h + margin + row_idx * (patch_h + margin)

            arr = (images[global_idx].transpose(1, 2, 0) * 255).astype(np.uint8)
            patch_img = Image.fromarray(arr)

            if y + patch_h <= total_h:
                preview.paste(patch_img, (x, y))

    preview_path = os.path.join(output_dir, f"preview_{split}.png")
    preview.save(preview_path)
    print(f"  预览图: {preview_path}")
    return preview_path


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="中国象棋合成训练数据生成器 v3")
    parser.add_argument("--train-samples", type=int, default=12000, help="训练集样本数")
    parser.add_argument("--test-samples", type=int, default=3000, help="测试集样本数")
    parser.add_argument("--size", type=int, default=28, help="输出 patch 尺寸")
    parser.add_argument("--output", type=str, default="data/chess_cnn_onnx_finetune", help="输出根目录")
    parser.add_argument("--train-seed", type=int, default=42, help="训练集随机种子")
    parser.add_argument("--test-seed", type=int, default=12345, help="测试集随机种子")
    parser.add_argument("--preview", action="store_true", help="仅生成预览（不保存二进制）")
    parser.add_argument("--preview-count", type=int, default=150, help="预览中每个 split 的样本数")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"],
                        help="生成哪个 split")
    args = parser.parse_args()

    print("=" * 60)
    print("中国象棋合成训练数据生成器 v3")
    print("=" * 60)
    print(f"\n配置：")
    print(f"  输出尺寸: {args.size}x{args.size}")
    print(f"  输出目录: {args.output}")

    # 打印字形变体信息
    print(f"\n字形变体：")
    for cid in range(1, 15):
        name = CLASSES[cid][1]
        variants = CHAR_VARIANTS[cid]
        print(f"  {name}: {' / '.join(variants)}")

    splits_to_generate = []
    if args.split in ("train", "both"):
        splits_to_generate.append(("train", args.train_samples, args.train_seed))
    if args.split in ("test", "both"):
        splits_to_generate.append(("test", args.test_samples, args.test_seed))

    for split, num_samples, seed in splits_to_generate:
        print(f"\n{'─' * 50}")
        print(f"生成 [{split}] 数据集 ({num_samples} 样本, seed={seed})")
        print(f"{'─' * 50}")

        # 打印该 split 使用的字体
        pool = STYLE_POOLS[split]
        available_fonts = [p for p in pool["fonts"] if os.path.exists(p)]
        print(f"  字体: {[os.path.basename(f) for f in available_fonts]}")

        images, labels = generate_dataset(num_samples, args.size, seed, split)
        print(f"  完成！形状: images={images.shape}, labels={labels.shape}")

        # 类别统计
        print(f"\n  类别统计：")
        for cid in range(15):
            count = np.sum(labels == cid)
            name = CLASSES[cid][1]
            print(f"    [{cid:2d}] {name}: {count} 样本")

        # 保存
        split_dir = os.path.join(args.output, split)

        print(f"\n  保存预览图...")
        save_preview(images, labels, split_dir, split, args.preview_count)

        if not args.preview:
            print(f"\n  保存二进制数据...")
            save_binary(images, labels, split_dir)

    # 风格分离说明
    print(f"\n{'=' * 60}")
    print("风格分离说明：")
    print("  字体: 楷体, 宋体, 黑体, 仿宋, 微软雅黑 (共享)")
    print("  泛化测试: 基于棋子配色方案 + 棋盘底色差异")
    print("  棋子风格: 3 种模式 (经典浅底 / 木纹自然色 / 饱和对比色)")
    print("  棋盘底色: 含暖木/绿/蓝绿/灰/白/复古 等色系")
    print("  网格线: 根据棋盘亮度自动选择深色/浅色线条")
    print("  训练集和测试集使用不同的配色方案")
    print("  字形变体 (繁体/异体字) 在两个集合中均包含")
    print(f"{'=' * 60}")
    print(f"\n[OK] 完成！")


if __name__ == "__main__":
    main()

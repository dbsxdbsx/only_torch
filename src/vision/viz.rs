//! 展示画布工具：调色板、像素放大、alpha 混合、5x3 字体标签。
//!
//! 与 [`crate::vision::draw`] 的分工：
//!
//! - `draw`：在**原图**上画几何（bbox / 矩形 / 圆），结果会落到训练 / 推理图本身
//! - `viz`：在**展示画布**上做"为了给人看"的可视化（放大像素、叠加 mask 半透明、
//!   写预测分数标签等），结果通常单独保存为 PNG，不影响原图
//!
//! 所有函数都基于 `image::RgbImage`，不进入 Tensor 域。

use image::{Rgb, RgbImage};

// ============================================================================
// Palette
// ============================================================================

/// 类别 / slot 颜色调色板。
///
/// 提供"按索引取色"的统一接口，索引超过 `len()` 时按 `len()` 取模回卷，
/// 不会 panic。配合 segmentation / instance / detection 任务的可视化使用。
///
/// # 用法
///
/// ```ignore
/// use only_torch::vision::viz::Palette;
///
/// let palette = Palette::default_categorical();
/// let color_for_class_3 = palette.color(3);   // 类别索引取色
/// let color_for_slot_5 = palette.color(5);    // 自动按 len() 取模
///
/// // 自定义调色板
/// let custom = Palette::new(vec![[255, 0, 0], [0, 255, 0], [0, 0, 255]]);
/// ```
#[derive(Debug, Clone)]
pub struct Palette {
    colors: Vec<[u8; 3]>,
}

impl Palette {
    /// 自定义调色板。`colors` 为空时 panic。
    pub fn new(colors: Vec<[u8; 3]>) -> Self {
        assert!(!colors.is_empty(), "Palette: 颜色数必须 >= 1");
        Self { colors }
    }

    /// 默认分类调色板（Tab10 风格 8 色）。
    ///
    /// 索引 0 是深灰，常用作 background；后续 7 色是高对比度的红 / 蓝 / 绿 /
    /// 黄 / 紫 / 橙 / 青。索引超界会按 8 取模回卷。
    pub fn default_categorical() -> Self {
        Self::new(vec![
            [32, 32, 32],
            [240, 76, 76],
            [76, 160, 255],
            [80, 220, 120],
            [255, 192, 64],
            [180, 96, 255],
            [255, 128, 0],
            [80, 220, 220],
        ])
    }

    /// 按索引取色，超界时按 `len()` 取模。
    pub fn color(&self, idx: usize) -> [u8; 3] {
        self.colors[idx % self.colors.len()]
    }

    /// 颜色数。
    pub fn len(&self) -> usize {
        self.colors.len()
    }

    /// 是否为空（构造保证非空，因此始终为 `false`）。
    pub fn is_empty(&self) -> bool {
        self.colors.is_empty()
    }
}

impl Default for Palette {
    fn default() -> Self {
        Self::default_categorical()
    }
}

// ============================================================================
// 像素放大与 alpha 混合
// ============================================================================

/// 在 `canvas` 上把 `(block_x, block_y)` 这个"逻辑像素"放大成 `scale x scale`
/// 的色块。
///
/// 用于把 16x16 / 64x64 的小图放大成可观察的 PNG。`scale == 0` 时不绘制。
/// 落在 canvas 外的部分会被自动裁剪。
pub fn pixel_block_scale(
    canvas: &mut RgbImage,
    block_x: u32,
    block_y: u32,
    color: [u8; 3],
    scale: u32,
) {
    if scale == 0 {
        return;
    }
    let x0 = block_x * scale;
    let y0 = block_y * scale;
    let (cw, ch) = (canvas.width(), canvas.height());
    for dy in 0..scale {
        for dx in 0..scale {
            let x = x0 + dx;
            let y = y0 + dy;
            if x < cw && y < ch {
                canvas.put_pixel(x, y, Rgb(color));
            }
        }
    }
}

/// 基础色与 overlay 色按 `alpha ∈ [0, 1]` 做线性混合。
///
/// `alpha <= 0.0` 返回 `base`、`alpha >= 1.0` 返回 `overlay`，中间按通道
/// 线性插值。常用于在原图上叠加半透明 mask 着色。
pub fn blend_alpha(base: [u8; 3], overlay: [u8; 3], alpha: f32) -> [u8; 3] {
    let a = alpha.clamp(0.0, 1.0);
    [
        blend_channel(base[0], overlay[0], a),
        blend_channel(base[1], overlay[1], a),
        blend_channel(base[2], overlay[2], a),
    ]
}

fn blend_channel(base: u8, overlay: u8, alpha: f32) -> u8 {
    ((base as f32 * (1.0 - alpha)) + (overlay as f32 * alpha)).round() as u8
}

// ============================================================================
// 5x3 像素字体（用于在可视化图上写 IoU / score 等标签）
// ============================================================================

/// 5x3 像素字体（每字 5 行 × 3 列像素 + 1 列字符间距）。
///
/// 内置数字 `0-9`、大写 `A-Z`、常用标点 `% : . #`、空格；小写自动映射到大写。
/// 字符表外的字符渲染为空白。
pub struct TinyFont;

impl TinyFont {
    /// 单个字符宽度（含 1 列字符间距）。
    pub const CHAR_WIDTH: u32 = 4;
    /// 字符高度。
    pub const CHAR_HEIGHT: u32 = 5;

    /// 估算字符串渲染后的像素宽度（末尾不带间距）。
    pub fn text_width(text: &str) -> u32 {
        let n = text.chars().count() as u32;
        if n == 0 { 0 } else { n * Self::CHAR_WIDTH - 1 }
    }

    /// 在 `canvas` 上从左上角 `(x, y)` 开始绘制 `text`。
    ///
    /// 落在 canvas 外的像素会被自动跳过。
    pub fn draw(canvas: &mut RgbImage, x: u32, y: u32, text: &str, color: [u8; 3]) {
        let mut cx = x;
        for ch in text.chars() {
            draw_char(canvas, cx, y, ch, color);
            cx += Self::CHAR_WIDTH;
        }
    }

    /// 在 `canvas` 上画"label + 背景框"，用于 IoU / score 等标注。
    ///
    /// 在 `(x, y)` 处先填充 `bg_color` 作为底框（自动按文字宽 + 4 像素 padding
    /// 计算尺寸），然后在底框内画 `text`。落在 canvas 外的部分自动裁剪。
    pub fn draw_with_box(
        canvas: &mut RgbImage,
        x: u32,
        y: u32,
        text: &str,
        text_color: [u8; 3],
        bg_color: [u8; 3],
    ) {
        let bg_w = Self::text_width(text) + 4;
        let bg_h = Self::CHAR_HEIGHT + 4;
        let (cw, ch) = (canvas.width(), canvas.height());
        for dy in 0..bg_h {
            for dx in 0..bg_w {
                let px = x + dx;
                let py = y + dy;
                if px < cw && py < ch {
                    canvas.put_pixel(px, py, Rgb(bg_color));
                }
            }
        }
        Self::draw(canvas, x + 2, y + 2, text, text_color);
    }
}

fn draw_char(canvas: &mut RgbImage, x: u32, y: u32, ch: char, color: [u8; 3]) {
    let pattern = char_pattern(ch);
    let (cw, ch_size) = (canvas.width(), canvas.height());
    for (dy, row) in pattern.iter().enumerate() {
        for (dx, bit) in row.as_bytes().iter().enumerate() {
            if *bit == b'1' {
                let px = x + dx as u32;
                let py = y + dy as u32;
                if px < cw && py < ch_size {
                    canvas.put_pixel(px, py, Rgb(color));
                }
            }
        }
    }
}

fn char_pattern(ch: char) -> [&'static str; 5] {
    match ch {
        '0' => ["111", "101", "101", "101", "111"],
        '1' => ["010", "110", "010", "010", "111"],
        '2' => ["111", "001", "111", "100", "111"],
        '3' => ["111", "001", "111", "001", "111"],
        '4' => ["101", "101", "111", "001", "001"],
        '5' => ["111", "100", "111", "001", "111"],
        '6' => ["111", "100", "111", "101", "111"],
        '7' => ["111", "001", "010", "010", "010"],
        '8' => ["111", "101", "111", "101", "111"],
        '9' => ["111", "101", "111", "001", "111"],
        'A' => ["010", "101", "111", "101", "101"],
        'B' => ["110", "101", "110", "101", "110"],
        'C' => ["111", "100", "100", "100", "111"],
        'D' => ["110", "101", "101", "101", "110"],
        'E' => ["111", "100", "111", "100", "111"],
        'F' => ["111", "100", "111", "100", "100"],
        'G' => ["111", "100", "101", "101", "111"],
        'H' => ["101", "101", "111", "101", "101"],
        'I' => ["111", "010", "010", "010", "111"],
        'J' => ["001", "001", "001", "101", "111"],
        'K' => ["101", "101", "110", "101", "101"],
        'L' => ["100", "100", "100", "100", "111"],
        'M' => ["101", "111", "111", "101", "101"],
        'N' => ["101", "111", "111", "111", "101"],
        'O' => ["111", "101", "101", "101", "111"],
        'P' => ["111", "101", "111", "100", "100"],
        'Q' => ["111", "101", "101", "111", "001"],
        'R' => ["110", "101", "110", "101", "101"],
        'S' => ["111", "100", "111", "001", "111"],
        'T' => ["111", "010", "010", "010", "010"],
        'U' => ["101", "101", "101", "101", "111"],
        'V' => ["101", "101", "101", "111", "010"],
        'W' => ["101", "101", "111", "111", "101"],
        'X' => ["101", "101", "010", "101", "101"],
        'Y' => ["101", "101", "010", "010", "010"],
        'Z' => ["111", "001", "010", "100", "111"],
        '%' => ["101", "001", "010", "100", "101"],
        '.' => ["000", "000", "000", "000", "010"],
        ':' => ["000", "010", "000", "010", "000"],
        '#' => ["101", "111", "101", "111", "101"],
        ' ' => ["000", "000", "000", "000", "000"],
        c if c.is_ascii_lowercase() => char_pattern(c.to_ascii_uppercase()),
        _ => ["000", "000", "000", "000", "000"],
    }
}

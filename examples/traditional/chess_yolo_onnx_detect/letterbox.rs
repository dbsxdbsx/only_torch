//! Letterbox 预处理：等比缩放 + 灰色填充到 target × target
//!
//! YOLOv5 标准预处理：保持原图宽高比，短边居中填充灰色 (114, 114, 114)。
//! 输出 (img, scale, pad) 三件套，下游用 scale/pad 把 letterbox 空间的 bbox
//! 反向映射回原图坐标。

use image::{DynamicImage, GenericImageView, ImageBuffer, Rgb};

/// Letterbox 预处理结果
pub struct LetterboxResult {
    /// 处理后的 RGB 图像（恰好 target × target）
    pub image: DynamicImage,
    /// 等比缩放因子（new_dim = orig_dim × scale）
    pub scale: f32,
    /// 上左填充像素数 (pad_x, pad_y)，用于把 letterbox 坐标反映射回原图
    pub pad: (u32, u32),
}

impl LetterboxResult {
    /// 把 letterbox 坐标系下的 (cx, cy) 反映射到原始图像坐标
    pub fn to_origin(&self, cx_letterbox: f32, cy_letterbox: f32) -> (f32, f32) {
        let cx_origin = (cx_letterbox - self.pad.0 as f32) / self.scale;
        let cy_origin = (cy_letterbox - self.pad.1 as f32) / self.scale;
        (cx_origin, cy_origin)
    }
}

/// 对原图执行 letterbox 预处理
///
/// # 参数
/// - `img`：原始图像（RGB 或可转 RGB）
/// - `target`：目标方形边长（YOLOv5 默认 640）
///
/// # 返回
/// `LetterboxResult { image, scale, pad }`
///
/// # 算法
/// 1. 计算 `scale = min(target/w, target/h)`，保持宽高比
/// 2. 缩放到 `(new_w, new_h)` 其中 `new_dim = floor(orig_dim × scale)`
/// 3. 在 `target × target` 灰色画布上居中放置缩放后的图，
///    填充 `(target - new_dim) / 2` 像素灰边
pub fn letterbox(img: &DynamicImage, target: u32) -> LetterboxResult {
    let (orig_w, orig_h) = img.dimensions();
    let scale = (target as f32 / orig_w as f32).min(target as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).floor() as u32;
    let new_h = (orig_h as f32 * scale).floor() as u32;

    let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
    let resized_rgb = resized.to_rgb8();

    let pad_x = (target.saturating_sub(new_w)) / 2;
    let pad_y = (target.saturating_sub(new_h)) / 2;

    let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
        ImageBuffer::from_pixel(target, target, Rgb([114, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized_rgb, pad_x as i64, pad_y as i64);

    LetterboxResult {
        image: DynamicImage::ImageRgb8(canvas),
        scale,
        pad: (pad_x, pad_y),
    }
}

/// 把 letterboxed RGB 图像转为 NCHW 排布的 f32 张量数据，归一化到 [0, 1]
///
/// 输出 `Vec<f32>` 长度 = 1 × 3 × target × target
pub fn image_to_nchw_normalized(img: &DynamicImage, target: u32) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let mut out = vec![0f32; (3 * target * target) as usize];
    let plane = (target * target) as usize;
    for (x, y, p) in rgb.enumerate_pixels() {
        let idx = (y * target + x) as usize;
        out[idx] = p.0[0] as f32 / 255.0;
        out[plane + idx] = p.0[1] as f32 / 255.0;
        out[2 * plane + idx] = p.0[2] as f32 / 255.0;
    }
    out
}

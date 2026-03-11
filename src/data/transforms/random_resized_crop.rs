//! 随机裁切并缩放
//!
//! 先在随机面积比 + 宽高比下裁取一个区域，再用双线性插值 resize 到目标尺寸。
//! 这是 ImageNet 标准训练流水线的核心增强组件。
//!
//! 对应 PyTorch `torchvision.transforms.RandomResizedCrop`。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 随机裁切并缩放（RandomResizedCrop）
///
/// 以随机面积比和宽高比裁取原图的一个区域，再用双线性插值缩放到 `(target_h, target_w)`。
///
/// # 参数
/// - `target_h`, `target_w`: 输出尺寸
/// - `scale`: 裁切区域面积占原图面积的比例范围，默认 `(0.08, 1.0)`
/// - `ratio`: 裁切区域的宽高比范围，默认 `(3/4, 4/3)`
///
/// # 算法
/// 1. 在 `scale` 和 `ratio` 范围内随机采样，计算裁切区域的 (h, w)
/// 2. 在原图中随机选取裁切位置
/// 3. 用双线性插值将裁切区域 resize 到 `(target_h, target_w)`
/// 4. 若多次采样均无法找到合法区域，fallback 为中心裁切
///
/// # 示例
///
/// ```ignore
/// // 输出 224x224，默认参数
/// let crop = RandomResizedCrop::new(224, 224);
/// let output = crop.apply(&image_tensor);  // [C, 224, 224]
///
/// // 自定义面积比和宽高比
/// let crop = RandomResizedCrop::new(32, 32)
///     .scale(0.5, 1.0)
///     .ratio(0.75, 1.33);
/// ```
pub struct RandomResizedCrop {
    target_h: usize,
    target_w: usize,
    scale: (f64, f64),
    ratio: (f64, f64),
}

impl RandomResizedCrop {
    /// 创建 RandomResizedCrop
    ///
    /// # 参数
    /// - `target_h`: 输出高度
    /// - `target_w`: 输出宽度
    pub fn new(target_h: usize, target_w: usize) -> Self {
        assert!(target_h > 0 && target_w > 0);
        Self {
            target_h,
            target_w,
            scale: (0.08, 1.0),
            ratio: (3.0 / 4.0, 4.0 / 3.0),
        }
    }

    /// 设置面积比范围
    pub fn scale(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max && max <= 1.0);
        self.scale = (min, max);
        self
    }

    /// 设置宽高比范围
    pub fn ratio(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max);
        self.ratio = (min, max);
        self
    }
}

impl Transform for RandomResizedCrop {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        let ndim = shape.len();
        assert!(
            ndim == 2 || ndim == 3,
            "RandomResizedCrop: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
        );

        let (c, h, w) = if ndim == 2 {
            (1, shape[0], shape[1])
        } else {
            (shape[0], shape[1], shape[2])
        };

        let (top, left, crop_h, crop_w) = get_crop_params(h, w, self.scale, self.ratio);

        // 裁切 + 双线性插值 resize
        let flat: Vec<f32> = tensor.flatten_view().to_vec();
        let mut out = vec![0.0f32; c * self.target_h * self.target_w];

        for ch in 0..c {
            let ch_offset = ch * h * w;
            let out_ch_offset = ch * self.target_h * self.target_w;

            for out_y in 0..self.target_h {
                for out_x in 0..self.target_w {
                    // 将输出坐标映射回裁切区域内的坐标（浮点）
                    let src_y = top as f64 + (out_y as f64 + 0.5) * crop_h as f64
                        / self.target_h as f64
                        - 0.5;
                    let src_x = left as f64 + (out_x as f64 + 0.5) * crop_w as f64
                        / self.target_w as f64
                        - 0.5;

                    out[out_ch_offset + out_y * self.target_w + out_x] =
                        bilinear_sample(&flat, ch_offset, h, w, src_y, src_x);
                }
            }
        }

        if ndim == 2 {
            Tensor::new(&out, &[self.target_h, self.target_w])
        } else {
            Tensor::new(&out, &[c, self.target_h, self.target_w])
        }
    }
}

/// 计算裁切参数 (top, left, crop_h, crop_w)
///
/// 与 PyTorch `RandomResizedCrop.get_params()` 逻辑一致：
/// 尝试最多 10 次随机采样，失败则 fallback 为中心裁切。
fn get_crop_params(
    h: usize,
    w: usize,
    scale: (f64, f64),
    ratio: (f64, f64),
) -> (usize, usize, usize, usize) {
    let mut rng = rand::thread_rng();
    let area = (h * w) as f64;

    for _ in 0..10 {
        let target_area = rng.gen_range(scale.0..=scale.1) * area;
        let log_ratio_min = ratio.0.ln();
        let log_ratio_max = ratio.1.ln();
        let aspect_ratio = rng.gen_range(log_ratio_min..=log_ratio_max).exp();

        let crop_w = (target_area * aspect_ratio).sqrt().round() as usize;
        let crop_h = (target_area / aspect_ratio).sqrt().round() as usize;

        if crop_w > 0 && crop_h > 0 && crop_w <= w && crop_h <= h {
            let top = rng.gen_range(0..=h - crop_h);
            let left = rng.gen_range(0..=w - crop_w);
            return (top, left, crop_h, crop_w);
        }
    }

    // Fallback: 中心裁切，保持宽高比
    let in_ratio = w as f64 / h as f64;
    let mid_ratio = ((ratio.0 * ratio.1) as f64).sqrt();

    let (crop_h, crop_w) = if in_ratio < mid_ratio {
        // 原图太窄，以宽度为准
        (((w as f64) / mid_ratio).round() as usize, w)
    } else {
        // 原图太宽，以高度为准
        (h, ((h as f64) * mid_ratio).round() as usize)
    };

    let crop_h = crop_h.min(h).max(1);
    let crop_w = crop_w.min(w).max(1);
    let top = (h - crop_h) / 2;
    let left = (w - crop_w) / 2;

    (top, left, crop_h, crop_w)
}

/// 双线性插值采样
///
/// 边界外像素用 clamp 处理（重复边缘像素）。
fn bilinear_sample(
    flat: &[f32],
    ch_offset: usize,
    h: usize,
    w: usize,
    y: f64,
    x: f64,
) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f64);
    let y = y.clamp(0.0, (h - 1) as f64);

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);

    let dx = (x - x0 as f64) as f32;
    let dy = (y - y0 as f64) as f32;

    let v00 = flat[ch_offset + y0 * w + x0];
    let v01 = flat[ch_offset + y0 * w + x1];
    let v10 = flat[ch_offset + y1 * w + x0];
    let v11 = flat[ch_offset + y1 * w + x1];

    v00 * (1.0 - dx) * (1.0 - dy) + v01 * dx * (1.0 - dy) + v10 * (1.0 - dx) * dy + v11 * dx * dy
}

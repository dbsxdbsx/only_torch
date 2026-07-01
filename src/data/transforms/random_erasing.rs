//! 随机区域擦除
//!
//! 以概率 p 在图像上随机选择一个矩形区域，用指定值填充。
//!
//! 同时为 `ClassificationSample` / `DetectionSample` / `SegmentationSample`
//! 实现 [`SampleTransform`]，遵循 torchvision v2 的 image-only 语义——**只擦
//! 图像，不同步擦 label / mask / bbox**。理由：RandomErasing 的训练价值就是
//! "让模型学会在部分遮挡下仍能识别"，如果同步把 bbox / mask 也擦掉，就反向
//! 抵消了增强效果。

use super::Transform;
use super::sample_transform::SampleTransform;
use crate::data::DetectionSample;
use crate::data::sample::{ClassificationSample, SegmentationSample};
use crate::tensor::Tensor;
use rand::Rng;
use rand::rngs::ThreadRng;

/// 随机区域擦除
///
/// 支持 2D `[H, W]` 灰度图像与 3D `[C, H, W]` 彩色图像；3D 情况下所有通道
/// 在同一矩形区域被擦除。以概率 `p` 掷骰决定是否擦除，命中后在 `scale` /
/// `ratio` 约束下采样矩形窗口。
///
/// # 参数说明
/// - `scale`: 擦除区域面积与原图面积之比的范围
/// - `ratio`: 擦除区域宽高比的范围
/// - `value`: 擦除值
///
/// # 示例
///
/// ```ignore
/// let erasing = RandomErasing::new(0.5);
/// let output = erasing.apply(&image_tensor);
/// ```
pub struct RandomErasing {
    p: f64,
    scale: (f64, f64),
    ratio: (f64, f64),
    value: f32,
}

impl RandomErasing {
    /// 创建随机擦除变换
    ///
    /// # 参数
    /// - `p`: 擦除概率，范围 [0, 1]
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "RandomErasing: 概率 p 必须在 [0, 1] 范围内，得到 {p}"
        );
        Self {
            p,
            scale: (0.02, 0.33),
            ratio: (0.3, 3.3),
            value: 0.0,
        }
    }

    /// 设置擦除区域面积比范围
    pub fn scale(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max && max <= 1.0);
        self.scale = (min, max);
        self
    }

    /// 设置擦除区域宽高比范围
    pub fn ratio(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max);
        self.ratio = (min, max);
        self
    }

    /// 设置擦除值
    pub fn value(mut self, value: f32) -> Self {
        self.value = value;
        self
    }

    /// 按概率掷一次骰子，并采样擦除窗口 `(top, left, erase_h, erase_w)`。
    ///
    /// - 返回 `None` 表示 **不擦**（概率未命中，或 10 次采样都无法找到合法窗口）。
    /// - 调用方只需对返回值 `if let Some((t, l, eh, ew)) = ...` 一层判断。
    ///
    /// paired 路径必须保证 image 侧和可能的其他路径共用**同一次**采样，所以
    /// 该方法被抽出。
    fn sample_erase_window(&self, h: usize, w: usize) -> Option<(usize, usize, usize, usize)> {
        let mut rng: ThreadRng = rand::thread_rng();
        if rng.gen_range(0.0_f64..1.0) >= self.p {
            return None;
        }

        let area = (h * w) as f64;
        let max_attempts = 10;
        for _ in 0..max_attempts {
            let target_area = rng.gen_range(self.scale.0..=self.scale.1) * area;
            let log_ratio_min = self.ratio.0.ln();
            let log_ratio_max = self.ratio.1.ln();
            let aspect_ratio = rng.gen_range(log_ratio_min..=log_ratio_max).exp();

            let erase_w = (target_area * aspect_ratio).sqrt() as usize;
            let erase_h = (target_area / aspect_ratio).sqrt() as usize;

            if erase_w == 0 || erase_h == 0 || erase_w >= w || erase_h >= h {
                continue;
            }

            let top = rng.gen_range(0..h - erase_h);
            let left = rng.gen_range(0..w - erase_w);
            return Some((top, left, erase_h, erase_w));
        }
        None
    }
}

impl Transform for RandomErasing {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let (h, w) = image_h_w(tensor);
        match self.sample_erase_window(h, w) {
            Some((top, left, erase_h, erase_w)) => {
                erase_region(tensor, top, left, erase_h, erase_w, self.value)
            }
            None => tensor.clone(),
        }
    }
}

// ============================================================================
// SampleTransform 实现：按 A 方案——只擦 image，label / mask / bbox 不动
// ============================================================================

impl SampleTransform<ClassificationSample> for RandomErasing {
    fn apply_to(&self, mut sample: ClassificationSample) -> ClassificationSample {
        let (h, w) = image_h_w(&sample.image);
        if let Some((top, left, erase_h, erase_w)) = self.sample_erase_window(h, w) {
            sample.image = erase_region(&sample.image, top, left, erase_h, erase_w, self.value);
        }
        sample
    }
}

impl SampleTransform<DetectionSample> for RandomErasing {
    fn apply_to(&self, sample: DetectionSample) -> DetectionSample {
        let DetectionSample { image, labels } = sample;
        let (h, w) = image_h_w(&image);
        let new_image = match self.sample_erase_window(h, w) {
            Some((top, left, erase_h, erase_w)) => {
                erase_region(&image, top, left, erase_h, erase_w, self.value)
            }
            None => image,
        };
        // A 方案：labels 原样保留——模型需要在部分遮挡下仍识别出原物体。
        DetectionSample::new(new_image, labels)
    }
}

impl SampleTransform<SegmentationSample> for RandomErasing {
    fn apply_to(&self, mut sample: SegmentationSample) -> SegmentationSample {
        let (h, w) = image_h_w(&sample.image);
        if let Some((top, left, erase_h, erase_w)) = self.sample_erase_window(h, w) {
            sample.image = erase_region(&sample.image, top, left, erase_h, erase_w, self.value);
        }
        // A 方案：mask 原样保留——模型需要在部分遮挡下仍给出原分割结果。
        sample
    }
}

// ============================================================================
// 底层 helper
// ============================================================================

/// 在给定 `[top, top+erase_h) × [left, left+erase_w)` 矩形范围内把所有
/// 通道填成 `value`。支持 2D `[H, W]` 和 3D `[C, H, W]` 两种 layout。
///
/// 调用方保证窗口合法（由 `sample_erase_window` 输出的窗口天然合法）。
pub(crate) fn erase_region(
    tensor: &Tensor,
    top: usize,
    left: usize,
    erase_h: usize,
    erase_w: usize,
    value: f32,
) -> Tensor {
    let shape = tensor.shape();
    // to_vec 按逻辑行主序展开、对任意布局都成立：输入可能是非连续视图，
    // flatten_view（into_shape）在非连续上会 panic。
    let mut data: Vec<f32> = tensor.to_vec();
    match shape.len() {
        2 => {
            let (h, w) = (shape[0], shape[1]);
            debug_assert!(top + erase_h <= h && left + erase_w <= w);
            for row in top..top + erase_h {
                for col in left..left + erase_w {
                    data[row * w + col] = value;
                }
            }
        }
        3 => {
            let (c, h, w) = (shape[0], shape[1], shape[2]);
            debug_assert!(top + erase_h <= h && left + erase_w <= w);
            for ch in 0..c {
                let ch_offset = ch * h * w;
                for row in top..top + erase_h {
                    for col in left..left + erase_w {
                        data[ch_offset + row * w + col] = value;
                    }
                }
            }
        }
        n => panic!("RandomErasing: 期望图像形状 [H, W] 或 [C, H, W]，得到 {n}D"),
    }
    Tensor::new(&data, shape)
}

/// 推断图像 Tensor 的 `(height, width)`。支持 `[H, W]` 与 `[C, H, W]`。
fn image_h_w(tensor: &Tensor) -> (usize, usize) {
    let shape = tensor.shape();
    match shape.len() {
        2 => (shape[0], shape[1]),
        3 => (shape[1], shape[2]),
        _ => panic!(
            "RandomErasing: 期望图像形状 [H, W] 或 [C, H, W]，得到 {:?}",
            shape
        ),
    }
}

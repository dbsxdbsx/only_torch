//! 随机仿射变换
//!
//! 对图像应用随机的旋转、平移、缩放、剪切组合变换，使用双线性插值。
//! 对应 PyTorch `torchvision.transforms.RandomAffine`。
//!
//! 同时为 `ClassificationSample` / `DetectionSample` / `SegmentationSample`
//! 实现 [`SampleTransform`]——所有 paired 路径下 image / mask / bbox 必须
//! 共用**同一组**随机采样参数，因此 `apply` 与三档 paired 实现都走统一的
//! `sample_params` → `affine_kernel` 组合。

use super::Transform;
use super::affine_kernel::{AffineParams, affine_bbox, affine_bilinear, affine_nearest};
use super::sample_transform::SampleTransform;
use crate::data::DetectionSample;
use crate::data::sample::{ClassificationSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::{DetectionLabelFilter, GroundTruthBox, clip_filter_labels};
use rand::Rng;

/// 随机仿射变换
///
/// 对输入张量 [C, H, W] 或 [H, W] 应用随机仿射变换（旋转+平移+缩放+剪切）。
/// 超出边界的像素用 `fill_value` 填充。
///
/// # 参数
/// - `degrees`: 旋转角度范围 [-degrees, +degrees]（度）
/// - `translate`: 平移范围 (max_dx, max_dy)，以图像尺寸的比例表示
/// - `scale`: 缩放范围 (min, max)
/// - `shear`: 剪切角度范围 [-shear, +shear]（度）
///
/// # 示例
///
/// ```ignore
/// // 仅旋转 ±10°
/// let t = RandomAffine::new(10.0);
///
/// // 旋转 ±5° + 平移 5% + 缩放 0.9~1.1
/// let t = RandomAffine::new(5.0)
///     .translate(0.05, 0.05)
///     .scale(0.9, 1.1);
///
/// // 完整参数
/// let t = RandomAffine::new(15.0)
///     .translate(0.1, 0.1)
///     .scale(0.8, 1.2)
///     .shear(10.0);
/// ```
pub struct RandomAffine {
    degrees: f64,
    translate: Option<(f64, f64)>,
    scale_range: Option<(f64, f64)>,
    shear: Option<f64>,
    fill_value: f32,
    label_filter: DetectionLabelFilter,
}

impl RandomAffine {
    /// 创建随机仿射变换
    ///
    /// # 参数
    /// - `degrees`: 最大旋转角度（绝对值），单位为度
    pub fn new(degrees: f64) -> Self {
        assert!(degrees >= 0.0, "RandomAffine: degrees 必须 >= 0");
        Self {
            degrees,
            translate: None,
            scale_range: None,
            shear: None,
            fill_value: 0.0,
            label_filter: DetectionLabelFilter::default(),
        }
    }

    /// 设置最大平移量（以图像尺寸的比例表示）
    ///
    /// # 参数
    /// - `max_dx`: 水平最大平移比例，范围 [0, 1]
    /// - `max_dy`: 垂直最大平移比例，范围 [0, 1]
    pub fn translate(mut self, max_dx: f64, max_dy: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&max_dx) && (0.0..=1.0).contains(&max_dy),
            "RandomAffine: translate 范围必须在 [0, 1]"
        );
        self.translate = Some((max_dx, max_dy));
        self
    }

    /// 设置缩放范围
    pub fn scale(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max, "RandomAffine: scale 范围无效");
        self.scale_range = Some((min, max));
        self
    }

    /// 设置最大剪切角度（度）
    pub fn shear(mut self, degrees: f64) -> Self {
        assert!(degrees >= 0.0, "RandomAffine: shear 必须 >= 0");
        self.shear = Some(degrees);
        self
    }

    /// 设置超出边界的填充值
    pub fn fill_value(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }

    /// 设置 detection label 过滤规则；仅在 `SampleTransform<DetectionSample>`
    /// 路径下生效。仿射后的 bbox 会先 clip 到图像边界，再按此 filter 丢弃
    /// 面积过小的框（典型：大幅缩小或移到图外）。
    pub fn with_label_filter(mut self, filter: DetectionLabelFilter) -> Self {
        self.label_filter = filter;
        self
    }

    /// 采样一次随机仿射参数。
    ///
    /// `w / h` 仅用于把 `translate` 比例转换为像素位移——因此 paired 路径下
    /// image / mask / bbox 只要尺寸一致（都是同一张图），采样结果就一致。
    fn sample_params(&self, h: usize, w: usize) -> AffineParams {
        let mut rng = rand::thread_rng();

        let angle_rad = if self.degrees > 0.0 {
            rng.gen_range(-self.degrees..=self.degrees).to_radians()
        } else {
            0.0
        };

        let (tx, ty) = if let Some((max_dx, max_dy)) = self.translate {
            let tx = rng.gen_range(-max_dx..=max_dx) * w as f64;
            let ty = rng.gen_range(-max_dy..=max_dy) * h as f64;
            (tx, ty)
        } else {
            (0.0, 0.0)
        };

        let scale = if let Some((min_s, max_s)) = self.scale_range {
            rng.gen_range(min_s..=max_s)
        } else {
            1.0
        };

        let shear_rad = if let Some(max_shear) = self.shear {
            rng.gen_range(-max_shear..=max_shear).to_radians()
        } else {
            0.0
        };

        AffineParams {
            angle_rad,
            tx,
            ty,
            scale,
            shear_rad,
        }
    }
}

impl Transform for RandomAffine {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let (h, w) = image_h_w(tensor);
        let params = self.sample_params(h, w);
        if params.is_identity() {
            return tensor.clone();
        }
        affine_bilinear(tensor, params, self.fill_value)
    }
}

// ============================================================================
// SampleTransform 实现：保持 image / label 几何同步
// ============================================================================

impl SampleTransform<ClassificationSample> for RandomAffine {
    fn apply_to(&self, mut sample: ClassificationSample) -> ClassificationSample {
        let (h, w) = image_h_w(&sample.image);
        let params = self.sample_params(h, w);
        if params.is_identity() {
            return sample;
        }
        sample.image = affine_bilinear(&sample.image, params, self.fill_value);
        sample
    }
}

impl SampleTransform<DetectionSample> for RandomAffine {
    fn apply_to(&self, sample: DetectionSample) -> DetectionSample {
        let DetectionSample { image, labels } = sample;
        let (h, w) = image_h_w(&image);
        let params = self.sample_params(h, w);
        if params.is_identity() {
            return DetectionSample::new(image, labels);
        }
        let new_image = affine_bilinear(&image, params, self.fill_value);
        let transformed: Vec<GroundTruthBox> = labels
            .into_iter()
            .map(|gt| {
                GroundTruthBox::new(
                    affine_bbox(gt.bbox, params, w as f32, h as f32),
                    gt.class_id,
                )
            })
            .collect();
        let filtered = clip_filter_labels(&transformed, w as f32, h as f32, self.label_filter);
        DetectionSample::new(new_image, filtered)
    }
}

impl SampleTransform<SegmentationSample> for RandomAffine {
    fn apply_to(&self, sample: SegmentationSample) -> SegmentationSample {
        let (h, w) = image_h_w(&sample.image);
        let params = self.sample_params(h, w);
        if params.is_identity() {
            return sample;
        }
        let SegmentationSample { image, mask } = sample;
        let new_image = affine_bilinear(&image, params, self.fill_value);
        let new_mask = affine_nearest(&mask, params, self.fill_value);
        SegmentationSample::new(new_image, new_mask)
    }
}

/// 推断图像 Tensor 的 `(height, width)`。支持 `[H, W]` 与 `[C, H, W]`。
fn image_h_w(tensor: &Tensor) -> (usize, usize) {
    let shape = tensor.shape();
    match shape.len() {
        2 => (shape[0], shape[1]),
        3 => (shape[1], shape[2]),
        _ => panic!(
            "RandomAffine: 期望图像形状 [H, W] 或 [C, H, W]，得到 {:?}",
            shape
        ),
    }
}

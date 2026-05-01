//! 随机裁切
//!
//! 可选填充后随机裁切到目标尺寸；对 detection / segmentation sample 同步裁标签。

use super::Transform;
use super::crop_helpers::{
    crop_and_filter_bboxes, narrow_image, pad_image, shift_bboxes_by_padding, tensor_h_w,
};
use super::sample_transform::SampleTransform;
use crate::data::DetectionSample;
use crate::data::sample::{ClassificationSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::DetectionLabelFilter;
use rand::Rng;

/// 随机裁切
///
/// 对输入张量 `[C, H, W]` 或 `[H, W]`，可选填充后随机裁切到
/// `(target_h, target_w)`。同时为 `ClassificationSample` / `DetectionSample` /
/// `SegmentationSample` 实现 [`SampleTransform`]，让 image 与 label 同步
/// 裁剪。
///
/// # 示例
///
/// ```ignore
/// use only_torch::data::transforms::{RandomCrop, SampleTransform};
/// use only_torch::vision::detection::DetectionLabelFilter;
///
/// let crop = RandomCrop::new(32, 32)
///     .padding(4)
///     .with_label_filter(DetectionLabelFilter::new(2.0));
/// let new_sample = crop.apply_to(detection_sample);
/// ```
pub struct RandomCrop {
    target_h: usize,
    target_w: usize,
    padding: usize,
    fill_value: f32,
    label_filter: DetectionLabelFilter,
}

impl RandomCrop {
    /// 创建随机裁切变换
    ///
    /// # 参数
    /// - `target_h`: 目标高度
    /// - `target_w`: 目标宽度
    pub fn new(target_h: usize, target_w: usize) -> Self {
        Self {
            target_h,
            target_w,
            padding: 0,
            fill_value: 0.0,
            label_filter: DetectionLabelFilter::default(),
        }
    }

    /// 设置填充量（四周等量填充）
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// 设置填充值
    pub fn fill_value(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }

    /// 设置 detection label 过滤规则；仅在 `SampleTransform<DetectionSample>`
    /// 路径下生效。
    pub fn with_label_filter(mut self, filter: DetectionLabelFilter) -> Self {
        self.label_filter = filter;
        self
    }

    /// 验证 padded 尺寸足够大，并随机选择 crop 起点。
    fn random_origin(&self, padded_h: usize, padded_w: usize) -> (usize, usize) {
        assert!(
            padded_h >= self.target_h && padded_w >= self.target_w,
            "RandomCrop: 填充后尺寸 ({padded_h}x{padded_w}) 必须 >= 目标尺寸 ({}x{})",
            self.target_h,
            self.target_w
        );
        let mut rng = rand::thread_rng();
        let top = if padded_h == self.target_h {
            0
        } else {
            rng.gen_range(0..=padded_h - self.target_h)
        };
        let left = if padded_w == self.target_w {
            0
        } else {
            rng.gen_range(0..=padded_w - self.target_w)
        };
        (top, left)
    }
}

impl Transform for RandomCrop {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let ndim = tensor.shape().len();
        assert!(
            ndim == 2 || ndim == 3,
            "RandomCrop: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
        );
        let padded = pad_image(tensor, self.padding, self.fill_value);
        let (h, w) = tensor_h_w(&padded);
        let (top, left) = self.random_origin(h, w);
        narrow_image(&padded, top, left, self.target_h, self.target_w)
    }
}

impl SampleTransform<ClassificationSample> for RandomCrop {
    fn apply_to(&self, sample: ClassificationSample) -> ClassificationSample {
        let padded_image = pad_image(&sample.image, self.padding, self.fill_value);
        let (h, w) = tensor_h_w(&padded_image);
        let (top, left) = self.random_origin(h, w);
        ClassificationSample::new(
            narrow_image(&padded_image, top, left, self.target_h, self.target_w),
            sample.label,
        )
    }
}

impl SampleTransform<DetectionSample> for RandomCrop {
    fn apply_to(&self, sample: DetectionSample) -> DetectionSample {
        let DetectionSample { image, labels } = sample;
        let padded_image = pad_image(&image, self.padding, self.fill_value);
        let padded_labels = shift_bboxes_by_padding(labels, self.padding);
        let (h, w) = tensor_h_w(&padded_image);
        let (top, left) = self.random_origin(h, w);
        let cropped_image = narrow_image(&padded_image, top, left, self.target_h, self.target_w);
        let cropped_labels = crop_and_filter_bboxes(
            &padded_labels,
            top,
            left,
            self.target_h,
            self.target_w,
            self.label_filter,
        );
        DetectionSample::new(cropped_image, cropped_labels)
    }
}

impl SampleTransform<SegmentationSample> for RandomCrop {
    fn apply_to(&self, sample: SegmentationSample) -> SegmentationSample {
        // 注意：mask 与 image 共用 fill_value，符合"crop window 之外按训练
        // 边界默认值处理"的简化语义；如果项目需要 ignore_index 等更复杂的
        // mask 填充策略，可在调用方先做好 padding。
        let padded_image = pad_image(&sample.image, self.padding, self.fill_value);
        let padded_mask = pad_image(&sample.mask, self.padding, self.fill_value);
        let (h, w) = tensor_h_w(&padded_image);
        let (top, left) = self.random_origin(h, w);
        SegmentationSample::new(
            narrow_image(&padded_image, top, left, self.target_h, self.target_w),
            narrow_image(&padded_mask, top, left, self.target_h, self.target_w),
        )
    }
}

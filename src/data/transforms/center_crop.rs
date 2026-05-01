//! 中心裁切
//!
//! 从图像中心裁切到目标尺寸；对 detection / segmentation sample 同步裁标签。

use super::Transform;
use super::crop_helpers::{crop_and_filter_bboxes, narrow_image, tensor_h_w};
use super::sample_transform::SampleTransform;
use crate::data::DetectionSample;
use crate::data::sample::{ClassificationSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::DetectionLabelFilter;

/// 中心裁切
///
/// 对输入张量 `[C, H, W]` 或 `[H, W]` 从中心裁切到 `(target_h, target_w)`。
/// 同时为 `ClassificationSample` / `DetectionSample` / `SegmentationSample`
/// 实现 [`SampleTransform`]，让 image 与 label 同步裁剪。
///
/// # 示例
///
/// ```ignore
/// use only_torch::data::transforms::{CenterCrop, SampleTransform, Transform};
///
/// // image-only 流水线
/// let crop = CenterCrop::new(224, 224);
/// let output = crop.apply(&image_tensor);
///
/// // detection 流水线，label 自动跟随
/// let new_sample = crop.apply_to(detection_sample);
/// ```
pub struct CenterCrop {
    target_h: usize,
    target_w: usize,
    label_filter: DetectionLabelFilter,
}

impl CenterCrop {
    /// 创建中心裁切变换
    ///
    /// # 参数
    /// - `target_h`: 目标高度
    /// - `target_w`: 目标宽度
    pub fn new(target_h: usize, target_w: usize) -> Self {
        Self {
            target_h,
            target_w,
            label_filter: DetectionLabelFilter::default(),
        }
    }

    /// 设置 detection label 过滤规则；仅在 `SampleTransform<DetectionSample>`
    /// 路径下生效。
    pub fn with_label_filter(mut self, filter: DetectionLabelFilter) -> Self {
        self.label_filter = filter;
        self
    }

    /// 计算中心裁切的 `(top, left)` 起点；输入尺寸不足时 panic。
    fn center_origin(&self, h: usize, w: usize) -> (usize, usize) {
        assert!(
            h >= self.target_h && w >= self.target_w,
            "CenterCrop: 输入尺寸 ({h}x{w}) 必须 >= 目标尺寸 ({}x{})",
            self.target_h,
            self.target_w
        );
        ((h - self.target_h) / 2, (w - self.target_w) / 2)
    }
}

impl Transform for CenterCrop {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let ndim = tensor.shape().len();
        assert!(
            ndim == 2 || ndim == 3,
            "CenterCrop: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
        );
        let (h, w) = tensor_h_w(tensor);
        let (top, left) = self.center_origin(h, w);
        narrow_image(tensor, top, left, self.target_h, self.target_w)
    }
}

impl SampleTransform<ClassificationSample> for CenterCrop {
    fn apply_to(&self, sample: ClassificationSample) -> ClassificationSample {
        let (h, w) = tensor_h_w(&sample.image);
        let (top, left) = self.center_origin(h, w);
        ClassificationSample::new(
            narrow_image(&sample.image, top, left, self.target_h, self.target_w),
            sample.label,
        )
    }
}

impl SampleTransform<DetectionSample> for CenterCrop {
    fn apply_to(&self, sample: DetectionSample) -> DetectionSample {
        let DetectionSample { image, labels } = sample;
        let (h, w) = tensor_h_w(&image);
        let (top, left) = self.center_origin(h, w);
        let cropped_image = narrow_image(&image, top, left, self.target_h, self.target_w);
        let cropped_labels = crop_and_filter_bboxes(
            &labels,
            top,
            left,
            self.target_h,
            self.target_w,
            self.label_filter,
        );
        DetectionSample::new(cropped_image, cropped_labels)
    }
}

impl SampleTransform<SegmentationSample> for CenterCrop {
    fn apply_to(&self, sample: SegmentationSample) -> SegmentationSample {
        let (h, w) = tensor_h_w(&sample.image);
        let (top, left) = self.center_origin(h, w);
        SegmentationSample::new(
            narrow_image(&sample.image, top, left, self.target_h, self.target_w),
            narrow_image(&sample.mask, top, left, self.target_h, self.target_w),
        )
    }
}

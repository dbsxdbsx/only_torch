//! Detection 数据结构与 batch 组装。

use crate::data::DataError;
use crate::tensor::Tensor;
use crate::vision::detection::{GroundTruthBox, clip_filter_ground_truths};
use crate::vision::preprocess::LetterboxResult;

/// 单张图像及其变长检测标签。
#[derive(Clone, Debug)]
pub struct DetectionSample {
    pub image: Tensor,
    pub labels: Vec<GroundTruthBox>,
}

impl DetectionSample {
    pub fn new(image: Tensor, labels: Vec<GroundTruthBox>) -> Self {
        Self { image, labels }
    }
}

/// Detection batch：图像可堆叠为 Tensor，标签保持每图变长列表。
#[derive(Clone, Debug)]
pub struct DetectionBatch {
    pub images: Tensor,
    pub labels: Vec<Vec<GroundTruthBox>>,
}

impl DetectionBatch {
    pub fn from_samples(samples: &[DetectionSample]) -> Result<Self, DataError> {
        let first = samples.first().ok_or_else(|| {
            DataError::FormatError("DetectionBatch 至少需要 1 个样本".to_string())
        })?;
        let sample_shape = first.image.shape().to_vec();
        let sample_size: usize = sample_shape.iter().product();
        let mut data = Vec::with_capacity(samples.len() * sample_size);
        let mut labels = Vec::with_capacity(samples.len());

        for sample in samples {
            if sample.image.shape() != sample_shape {
                return Err(DataError::ShapeMismatch {
                    expected: sample_shape.clone(),
                    got: sample.image.shape().to_vec(),
                });
            }
            data.extend_from_slice(&sample.image.flatten_view().to_vec());
            labels.push(sample.labels.clone());
        }

        let mut batch_shape = vec![samples.len()];
        batch_shape.extend_from_slice(&sample_shape);
        Ok(Self {
            images: Tensor::new(&data, &batch_shape),
            labels,
        })
    }
}

/// 检测标签过滤规则。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionLabelFilter {
    /// 裁剪后保留的最小面积，使用当前坐标空间单位。
    pub min_area: f32,
}

impl DetectionLabelFilter {
    pub const fn new(min_area: f32) -> Self {
        Self { min_area }
    }
}

impl Default for DetectionLabelFilter {
    fn default() -> Self {
        Self { min_area: 0.0 }
    }
}

/// 把原图坐标标签同步映射到 letterbox 输出坐标，并裁剪 / 过滤无效框。
pub fn letterbox_labels(
    labels: &[GroundTruthBox],
    letterbox: &LetterboxResult,
    filter: DetectionLabelFilter,
) -> Vec<GroundTruthBox> {
    let mapped = labels
        .iter()
        .map(|label| GroundTruthBox::new(letterbox.bbox_to_letterbox(label.bbox), label.class_id))
        .collect::<Vec<_>>();
    clip_filter_labels(
        &mapped,
        letterbox.output_size.0 as f32,
        letterbox.output_size.1 as f32,
        filter,
    )
}

/// 把 letterbox 输出坐标标签同步反映射到原图坐标，并裁剪 / 过滤无效框。
pub fn restore_letterbox_labels(
    labels: &[GroundTruthBox],
    letterbox: &LetterboxResult,
    filter: DetectionLabelFilter,
) -> Vec<GroundTruthBox> {
    let mapped = labels
        .iter()
        .map(|label| GroundTruthBox::new(letterbox.bbox_to_origin(label.bbox), label.class_id))
        .collect::<Vec<_>>();
    clip_filter_labels(
        &mapped,
        letterbox.original_size.0 as f32,
        letterbox.original_size.1 as f32,
        filter,
    )
}

/// 水平翻转检测标签。
pub fn horizontal_flip_labels(labels: &[GroundTruthBox], image_width: f32) -> Vec<GroundTruthBox> {
    labels
        .iter()
        .map(|label| GroundTruthBox::new(label.bbox.horizontal_flip(image_width), label.class_id))
        .collect()
}

/// 裁剪标签到图像边界并过滤面积不足的框。
pub fn clip_filter_labels(
    labels: &[GroundTruthBox],
    image_width: f32,
    image_height: f32,
    filter: DetectionLabelFilter,
) -> Vec<GroundTruthBox> {
    clip_filter_ground_truths(labels, image_width, image_height, filter.min_area)
}

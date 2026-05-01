//! Detection 标签的几何变换。
//!
//! 这里只做"图像几何 + label"的同步映射，不绑定具体训练器。配合
//! `vision::preprocess::letterbox` 等图像几何工具使用。

use super::{GroundTruthBox, clip_filter_ground_truths};
use crate::vision::preprocess::LetterboxResult;

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

//! 训练样本数据类型。
//!
//! 这里定义"image + 配套 label"的标准 sample 结构，供 `SampleTransform` 在
//! detection / segmentation / classification 任务中保持几何一致。
//!
//! 注意：`DetectionSample` 已存在于 `data::detection`（与变长 GT 列表相关），
//! 本文件只补充 `ClassificationSample` 与 `SegmentationSample`。

use crate::tensor::Tensor;

/// 分类任务样本：图像 + 单一类别 ID。
#[derive(Clone, Debug)]
pub struct ClassificationSample {
    pub image: Tensor,
    pub label: usize,
}

impl ClassificationSample {
    pub fn new(image: Tensor, label: usize) -> Self {
        Self { image, label }
    }
}

/// 语义 / 二值分割样本：图像 + 像素级 mask。
///
/// `image` 形状通常 `[C, H, W]` 或 `[H, W]`；`mask` 必须能匹配 image 的空间
/// 维度（`[H, W]` 或 `[Classes, H, W]`）。验证由具体 transform 实现按需执行。
#[derive(Clone, Debug)]
pub struct SegmentationSample {
    pub image: Tensor,
    pub mask: Tensor,
}

impl SegmentationSample {
    pub fn new(image: Tensor, mask: Tensor) -> Self {
        Self { image, mask }
    }
}

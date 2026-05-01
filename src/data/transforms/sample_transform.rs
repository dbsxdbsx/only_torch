//! 以 sample 为单位的 transform 契约。
//!
//! 与 image-only [`Transform`](super::Transform) 正交：
//!
//! | trait | 输入 | 输出 | 适用 |
//! |---|---|---|---|
//! | `Transform` | `&Tensor` | `Tensor` | image-only 流水线（分类、自监督） |
//! | `SampleTransform<S>` | `S`（owned） | `S` | image + label 同步几何变换（检测、分割） |
//!
//! 同一个 transform 类型可以同时为多种 `Sample` 实现 [`SampleTransform`]，
//! 按调用上下文自动 dispatch——例如 `RandomHorizontalFlip` 既能翻图像，也能
//! 翻图像 + bbox（detection）或图像 + mask（segmentation）。
//!
//! # 示例
//!
//! ```ignore
//! use only_torch::data::{DetectionSample, SampleTransform};
//! use only_torch::data::transforms::RandomHorizontalFlip;
//!
//! let flip = RandomHorizontalFlip::new(0.5);
//! let new_sample = flip.apply_to(detection_sample);
//! ```

/// 以 sample 为单位的 transform 契约。
///
/// 接收 owned sample 并返回（可能修改过的）新 sample。move 语义符合
/// "消费一个 sample，产出一个新 sample" 的训练数据流水线模式。
pub trait SampleTransform<S> {
    fn apply_to(&self, sample: S) -> S;
}

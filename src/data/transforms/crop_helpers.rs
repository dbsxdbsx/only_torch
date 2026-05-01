//! Crop / pad 在 image-only 与 paired sample transform 之间的共用辅助函数。
//!
//! 这里只放纯函数（无随机性），由 `CenterCrop` / `RandomCrop` 的具体实现按需
//! 调用。模块对外保持 `pub(crate)`，不暴露到框架外部。

use crate::tensor::Tensor;
use crate::vision::detection::{DetectionLabelFilter, GroundTruthBox, clip_filter_labels};

/// 推断图像 Tensor 的 `(height, width)`。支持 `[H, W]` 与 `[C, H, W]`。
pub(crate) fn tensor_h_w(tensor: &Tensor) -> (usize, usize) {
    let shape = tensor.shape();
    match shape.len() {
        2 => (shape[0], shape[1]),
        3 => (shape[1], shape[2]),
        _ => panic!("crop/pad: 期望图像形状 [H, W] 或 [C, H, W]，得到 {shape:?}"),
    }
}

/// 对 `[H, W]` 或 `[C, H, W]` 的图像 Tensor 做 `narrow` 裁剪。
pub(crate) fn narrow_image(
    tensor: &Tensor,
    top: usize,
    left: usize,
    target_h: usize,
    target_w: usize,
) -> Tensor {
    match tensor.shape().len() {
        2 => tensor.narrow(0, top, target_h).narrow(1, left, target_w),
        3 => tensor.narrow(1, top, target_h).narrow(2, left, target_w),
        n => panic!("crop/pad: 期望图像形状 [H, W] 或 [C, H, W]，得到 {n}D"),
    }
}

/// 对 `[H, W]` 或 `[C, H, W]` 的图像 Tensor 做四周等量 padding。
///
/// `padding == 0` 时直接 clone 返回，与原 image-only `RandomCrop` 行为一致。
pub(crate) fn pad_image(tensor: &Tensor, padding: usize, fill_value: f32) -> Tensor {
    if padding == 0 {
        return tensor.clone();
    }
    let p = padding;
    match tensor.shape().len() {
        2 => tensor.pad(&[(p, p), (p, p)], fill_value),
        3 => tensor.pad(&[(0, 0), (p, p), (p, p)], fill_value),
        n => panic!("crop/pad: 期望图像形状 [H, W] 或 [C, H, W]，得到 {n}D"),
    }
}

/// 把 detection labels 平移 `padding` 像素（pad 后、crop 之前的中间步骤）。
pub(crate) fn shift_bboxes_by_padding(
    labels: Vec<GroundTruthBox>,
    padding: usize,
) -> Vec<GroundTruthBox> {
    if padding == 0 {
        return labels;
    }
    let dp = padding as f32;
    labels
        .into_iter()
        .map(|gt| GroundTruthBox::new(gt.bbox.scale_translate(1.0, 1.0, dp, dp), gt.class_id))
        .collect()
}

/// 对 detection labels 做 crop 同步：先平移到 crop 坐标系，再裁剪到 crop
/// window，最后按 `filter` 过滤面积不足的 bbox。
pub(crate) fn crop_and_filter_bboxes(
    labels: &[GroundTruthBox],
    top: usize,
    left: usize,
    target_h: usize,
    target_w: usize,
    filter: DetectionLabelFilter,
) -> Vec<GroundTruthBox> {
    let dx = -(left as f32);
    let dy = -(top as f32);
    let translated: Vec<GroundTruthBox> = labels
        .iter()
        .map(|gt| GroundTruthBox::new(gt.bbox.scale_translate(1.0, 1.0, dx, dy), gt.class_id))
        .collect();
    clip_filter_labels(&translated, target_w as f32, target_h as f32, filter)
}

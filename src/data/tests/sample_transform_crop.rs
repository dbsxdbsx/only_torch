//! `SampleTransform` 在 CenterCrop / RandomCrop 上的 paired 实现测试。

use approx::assert_abs_diff_eq;

use crate::data::transforms::{CenterCrop, RandomCrop, SampleTransform};
use crate::data::{ClassificationSample, DetectionSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, DetectionLabelFilter, GroundTruthBox};

/// 1×4×4 单通道图像，方便人眼追踪 crop 起点。
fn make_image_4x4() -> Tensor {
    let mut data = Vec::with_capacity(16);
    for v in 0..16 {
        data.push(v as f32);
    }
    Tensor::new(&data, &[1, 4, 4])
}

#[test]
fn test_paired_center_crop_classification_keeps_label() {
    let sample = ClassificationSample::new(make_image_4x4(), 7);
    let crop = CenterCrop::new(2, 2);
    let result = crop.apply_to(sample);

    assert_eq!(result.image.shape(), &[1, 2, 2]);
    assert_eq!(result.label, 7);
    // 中心 2×2 应为原图 (1..3, 1..3) 的元素：5, 6, 9, 10
    assert_abs_diff_eq!(result.image[[0, 0, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.image[[0, 1, 1]], 10.0, epsilon = 1e-6);
}

#[test]
fn test_paired_center_crop_detection_synchronizes_bbox() {
    let labels = vec![GroundTruthBox::new(BBox::from_xyxy(1.0, 1.0, 3.0, 3.0), 0)];
    let sample = DetectionSample::new(make_image_4x4(), labels);
    let crop = CenterCrop::new(2, 2);
    let result = crop.apply_to(sample);

    assert_eq!(result.image.shape(), &[1, 2, 2]);
    assert_eq!(result.labels.len(), 1);
    // CenterCrop top=1, left=1：bbox 平移 (-1, -1) 后裁到 [0, 2]×[0, 2]
    let bbox = result.labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(bbox[0], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[2], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[3], 2.0, epsilon = 1e-6);
}

#[test]
fn test_paired_center_crop_detection_filters_outside_bbox() {
    let labels = vec![
        // 完全位于 crop window (top-left=(1,1), size 2×2) 之外
        GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 0.5, 0.5), 0),
        // 部分位于 crop window 内
        GroundTruthBox::new(BBox::from_xyxy(1.0, 1.0, 3.0, 3.0), 1),
    ];
    let sample = DetectionSample::new(make_image_4x4(), labels);
    let crop = CenterCrop::new(2, 2);
    let result = crop.apply_to(sample);

    // 第一个 bbox 完全在 crop 外 → 被过滤；保留 class_id=1 的那个
    assert_eq!(result.labels.len(), 1);
    assert_eq!(result.labels[0].class_id, 1);
}

#[test]
fn test_paired_center_crop_segmentation_synchronizes_mask() {
    let mask_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let mask = Tensor::new(&mask_data, &[4, 4]);
    let sample = SegmentationSample::new(make_image_4x4(), mask);
    let crop = CenterCrop::new(2, 2);
    let result = crop.apply_to(sample);

    assert_eq!(result.image.shape(), &[1, 2, 2]);
    assert_eq!(result.mask.shape(), &[2, 2]);
    // mask 中心 2×2 应为原 mask (1..3, 1..3)：5, 6, 9, 10
    assert_abs_diff_eq!(result.mask[[0, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.mask[[1, 1]], 10.0, epsilon = 1e-6);
}

#[test]
fn test_paired_random_crop_padded_to_target_is_deterministic() {
    // 4×4 图像 + padding=2 → padded 8×8；target 也 8×8 时 random_origin 必为 (0, 0)
    let labels = vec![GroundTruthBox::new(BBox::from_xyxy(1.0, 1.0, 3.0, 3.0), 0)];
    let sample = DetectionSample::new(make_image_4x4(), labels);
    let crop = RandomCrop::new(8, 8).padding(2);
    let result = crop.apply_to(sample);

    assert_eq!(result.image.shape(), &[1, 8, 8]);
    assert_eq!(result.labels.len(), 1);
    // padded 后 bbox 平移 +2，再 crop top=0/left=0 → 平移 -0 → 最终 (3, 3, 5, 5)
    let bbox = result.labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(bbox[0], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[1], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[2], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[3], 5.0, epsilon = 1e-6);
}

#[test]
fn test_paired_center_crop_label_filter_drops_tiny_bbox() {
    let labels = vec![
        // crop 后会变成 (0, 0, 0.4, 0.4)，面积 0.16 — 小于 min_area=1.0
        GroundTruthBox::new(BBox::from_xyxy(1.0, 1.0, 1.4, 1.4), 0),
        // crop 后会变成 (0, 0, 2, 2)，面积 4 — 满足
        GroundTruthBox::new(BBox::from_xyxy(1.0, 1.0, 3.0, 3.0), 1),
    ];
    let sample = DetectionSample::new(make_image_4x4(), labels);
    let crop = CenterCrop::new(2, 2).with_label_filter(DetectionLabelFilter::new(1.0));
    let result = crop.apply_to(sample);

    assert_eq!(result.labels.len(), 1);
    assert_eq!(result.labels[0].class_id, 1);
}

#[test]
fn test_paired_random_crop_segmentation_padded_to_target_is_deterministic() {
    let mask_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let mask = Tensor::new(&mask_data, &[4, 4]);
    let sample = SegmentationSample::new(make_image_4x4(), mask);
    let crop = RandomCrop::new(8, 8).padding(2).fill_value(99.0);
    let result = crop.apply_to(sample);

    assert_eq!(result.image.shape(), &[1, 8, 8]);
    assert_eq!(result.mask.shape(), &[8, 8]);
    // 左上 padding 区域用 fill_value 填充
    assert_abs_diff_eq!(result.image[[0, 0, 0]], 99.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.mask[[0, 0]], 99.0, epsilon = 1e-6);
}

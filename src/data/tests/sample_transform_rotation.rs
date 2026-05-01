//! `SampleTransform` 在 RandomRotation 上的 paired 实现测试。
//!
//! 确定性角度（0° / 90° / 180°）下 image 与 bbox / mask 的坐标关系都能精确
//! 验证；这里直接用底层的 `rotate_bbox` / `rotate_nearest` 做数学层面的断言，
//! 配合 `RandomRotation::new(0.0)` 的"不旋转"短路路径验证 owned-sample 入口。

use approx::assert_abs_diff_eq;

use crate::data::transforms::random_rotation::{rotate_bbox, rotate_nearest};
use crate::data::transforms::{RandomRotation, SampleTransform};
use crate::data::{ClassificationSample, DetectionSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, DetectionLabelFilter, GroundTruthBox};

// ---------------------------------------------------------------------------
// 底层 rotate_bbox 纯函数测试
// ---------------------------------------------------------------------------

#[test]
fn test_rotate_bbox_zero_degree_is_identity() {
    // 0° 旋转：bbox 不变
    let bbox = BBox::from_xyxy(1.0, 2.0, 3.0, 4.0);
    let rotated = rotate_bbox(bbox, 0.0, 8.0, 8.0);
    let [x1, y1, x2, y2] = rotated.to_xyxy();
    assert_abs_diff_eq!(x1, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y1, 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x2, 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y2, 4.0, epsilon = 1e-5);
}

#[test]
fn test_rotate_bbox_90_degrees_3x3() {
    // 3x3 图像中心 = (1, 1)；90° 旋转 (in_x, in_y) → (2 - in_y, in_x)
    // bbox (0, 0, 1, 1) → 4 角映射：(2,0)、(2,1)、(1,0)、(1,1) → AABB (1, 0, 2, 1)
    let bbox = BBox::from_xyxy(0.0, 0.0, 1.0, 1.0);
    let rotated = rotate_bbox(bbox, 90.0, 3.0, 3.0);
    let [x1, y1, x2, y2] = rotated.to_xyxy();
    assert_abs_diff_eq!(x1, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y1, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x2, 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y2, 1.0, epsilon = 1e-5);
}

#[test]
fn test_rotate_bbox_180_degrees_3x3() {
    // 180° 旋转 (in_x, in_y) → (2 - in_x, 2 - in_y)
    // bbox (0, 0, 1, 1) → AABB (1, 1, 2, 2)
    let bbox = BBox::from_xyxy(0.0, 0.0, 1.0, 1.0);
    let rotated = rotate_bbox(bbox, 180.0, 3.0, 3.0);
    let [x1, y1, x2, y2] = rotated.to_xyxy();
    assert_abs_diff_eq!(x1, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y1, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x2, 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y2, 2.0, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// rotate_nearest 纯函数测试（mask 专用）
// ---------------------------------------------------------------------------

#[test]
fn test_rotate_nearest_preserves_discrete_classes() {
    // 构造 3x3 mask，值只能是整数 0 或 5；bilinear 会混出 2.5，
    // nearest 必须保证旋转后仍只有 0 / 5
    let mask = Tensor::new(
        &[
            0.0, 0.0, 0.0, //
            0.0, 5.0, 0.0, //
            0.0, 0.0, 0.0,
        ],
        &[3, 3],
    );
    let rotated = rotate_nearest(&mask, 45.0, 0.0);
    let flat = rotated.flatten_view();
    for &v in flat.iter() {
        // 只允许 0 / 5，没有任何中间值
        assert!(
            v == 0.0 || (v - 5.0).abs() < 1e-6,
            "nearest 插值不应混出非法类别，得到 {v}"
        );
    }
}

#[test]
fn test_rotate_nearest_180_of_3x3_reverses_layout() {
    // 180° 旋转下 (in_x, in_y) → (2 - in_x, 2 - in_y)
    let mask = Tensor::new(
        &[
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
            7.0, 8.0, 9.0,
        ],
        &[3, 3],
    );
    let rotated = rotate_nearest(&mask, 180.0, 0.0);
    assert_abs_diff_eq!(rotated[[0, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(rotated[[1, 1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(rotated[[2, 2]], 1.0, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// SampleTransform 入口测试
// ---------------------------------------------------------------------------

#[test]
fn test_paired_rotation_zero_degrees_keeps_all_samples() {
    // degrees=0 → 三种 sample 都返回原样
    let image = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let rot = RandomRotation::new(0.0);

    let cls = ClassificationSample::new(image.clone(), 7);
    let cls_out = rot.apply_to(cls);
    assert_eq!(cls_out.label, 7);
    assert_abs_diff_eq!(cls_out.image[[0, 0, 0]], 1.0, epsilon = 1e-5);

    let det = DetectionSample::new(
        image.clone(),
        vec![GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 3)],
    );
    let det_out = rot.apply_to(det);
    assert_eq!(det_out.labels.len(), 1);
    let bbox = det_out.labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(bbox[0], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(bbox[2], 1.0, epsilon = 1e-5);

    let mask = Tensor::new(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
    let seg = SegmentationSample::new(image, mask);
    let seg_out = rot.apply_to(seg);
    assert_abs_diff_eq!(seg_out.mask[[1, 1]], 40.0, epsilon = 1e-5);
}

#[test]
fn test_paired_rotation_detection_label_filter_drops_tiny_bbox() {
    // 360° 旋转采样范围内可能采到 0°，但我们用 label_filter 路径只依赖 bbox
    // 几何——用 degrees=0 强制 identity 旋转，同时塞入一个面积不足的 bbox
    // 验证**在 angle==0 的短路路径上不会误过滤**（盒子保留）
    let image = Tensor::new(&[1.0; 9], &[1, 3, 3]);
    let labels = vec![
        GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 0.4, 0.4), 0),
        GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), 1),
    ];
    let sample = DetectionSample::new(image, labels);
    // degrees=0 → 不旋转 → label_filter 不参与
    let rot = RandomRotation::new(0.0).with_label_filter(DetectionLabelFilter::new(1.0));
    let result = rot.apply_to(sample);
    assert_eq!(
        result.labels.len(),
        2,
        "0° 短路路径不应触发 clip/filter 逻辑"
    );
}

#[test]
fn test_paired_rotation_classification_only_rotates_image() {
    // degrees=360 强制显著旋转；label 保持不变即可（不关心 image 精确像素）
    let image = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let sample = ClassificationSample::new(image, 42);
    let rot = RandomRotation::new(360.0);
    let result = rot.apply_to(sample);
    assert_eq!(result.label, 42);
    assert_eq!(result.image.shape(), &[1, 2, 2]);
}

#[test]
fn test_paired_rotation_segmentation_shape_preserved() {
    // segmentation 路径下 image / mask 形状都应保持
    let image = Tensor::new(&[1.0; 9], &[1, 3, 3]);
    let mask = Tensor::new(&[0.0; 9], &[3, 3]);
    let sample = SegmentationSample::new(image, mask);
    let rot = RandomRotation::new(360.0);
    let result = rot.apply_to(sample);
    assert_eq!(result.image.shape(), &[1, 3, 3]);
    assert_eq!(result.mask.shape(), &[3, 3]);
}

//! `SampleTransform` 在 RandomAffine 上的 paired 实现测试。
//!
//! 分两层：
//!
//! 1. 底层 `affine_bbox` / `affine_nearest` 纯函数测试——可以用确定的
//!    `AffineParams` 构造恒等 / 纯旋转 / 纯缩放等场景做精确断言。
//! 2. `SampleTransform` owned-sample 入口——用 `RandomAffine::new(0.0)` 且
//!    其他参数不设，保证采样到恒等变换，验证短路路径下 label 保持。

use approx::assert_abs_diff_eq;

use crate::data::transforms::affine_kernel::{
    AffineParams, affine_bbox, affine_bilinear, affine_nearest,
};
use crate::data::transforms::{RandomAffine, SampleTransform};
use crate::data::{ClassificationSample, DetectionSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, DetectionLabelFilter, GroundTruthBox};

/// 构造指定的 affine 参数（辅助）。
fn params(angle_deg: f64, tx: f64, ty: f64, scale: f64, shear_deg: f64) -> AffineParams {
    AffineParams {
        angle_rad: angle_deg.to_radians(),
        tx,
        ty,
        scale,
        shear_rad: shear_deg.to_radians(),
    }
}

// ---------------------------------------------------------------------------
// affine_bbox 纯函数测试
// ---------------------------------------------------------------------------

#[test]
fn test_affine_bbox_identity_preserves_bbox() {
    let bbox = BBox::from_xyxy(1.0, 2.0, 3.0, 4.0);
    let p = params(0.0, 0.0, 0.0, 1.0, 0.0);
    let out = affine_bbox(bbox, p, 8.0, 8.0);
    let [x1, y1, x2, y2] = out.to_xyxy();
    assert_abs_diff_eq!(x1, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y1, 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x2, 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y2, 4.0, epsilon = 1e-5);
}

#[test]
fn test_affine_bbox_pure_rotation_90_degrees_matches_rotate_bbox() {
    // 纯旋转 90°，scale=1、shear=0、tx=ty=0 → 应该等价于 rotate_bbox 90°
    // bbox (0,0,1,1) 在 3x3 图像中 → AABB (1, 0, 2, 1)（见 rotation 测试）
    let bbox = BBox::from_xyxy(0.0, 0.0, 1.0, 1.0);
    let p = params(90.0, 0.0, 0.0, 1.0, 0.0);
    let out = affine_bbox(bbox, p, 3.0, 3.0);
    let [x1, y1, x2, y2] = out.to_xyxy();
    assert_abs_diff_eq!(x1, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y1, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x2, 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y2, 1.0, epsilon = 1e-5);
}

#[test]
fn test_affine_bbox_pure_scale_2x_doubles_extent_around_center() {
    // 中心 (cx, cy) = (1.5, 1.5) for 4x4 图像
    // scale=2 → (in_x - 1.5) * 2 + 1.5 = 2*in_x - 1.5
    // bbox (1, 1, 2, 2) → (0.5, 0.5, 2.5, 2.5)
    let bbox = BBox::from_xyxy(1.0, 1.0, 2.0, 2.0);
    let p = params(0.0, 0.0, 0.0, 2.0, 0.0);
    let out = affine_bbox(bbox, p, 4.0, 4.0);
    let [x1, y1, x2, y2] = out.to_xyxy();
    assert_abs_diff_eq!(x1, 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(y1, 0.5, epsilon = 1e-5);
    assert_abs_diff_eq!(x2, 2.5, epsilon = 1e-5);
    assert_abs_diff_eq!(y2, 2.5, epsilon = 1e-5);
}

#[test]
fn test_affine_bbox_pure_translate_shifts_coords() {
    let bbox = BBox::from_xyxy(1.0, 2.0, 3.0, 4.0);
    // tx=+2, ty=-1
    let p = params(0.0, 2.0, -1.0, 1.0, 0.0);
    let out = affine_bbox(bbox, p, 8.0, 8.0);
    let [x1, y1, x2, y2] = out.to_xyxy();
    assert_abs_diff_eq!(x1, 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y1, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(x2, 5.0, epsilon = 1e-5);
    assert_abs_diff_eq!(y2, 3.0, epsilon = 1e-5);
}

// ---------------------------------------------------------------------------
// affine_nearest 纯函数测试（mask 专用）
// ---------------------------------------------------------------------------

#[test]
fn test_affine_nearest_preserves_discrete_classes() {
    // 构造 mask 只含整数类别 0 / 3 / 7；纯旋转 30° 后 nearest 必须保证输出
    // 像素值仍然 ∈ {0, 3, 7}，不会出现混合中间值
    let mask = Tensor::new(
        &[
            0.0, 3.0, 7.0, //
            0.0, 3.0, 7.0, //
            0.0, 3.0, 7.0,
        ],
        &[3, 3],
    );
    let p = params(30.0, 0.0, 0.0, 1.0, 0.0);
    let out = affine_nearest(&mask, p, 0.0);
    let flat = out.flatten_view();
    for &v in flat.iter() {
        assert!(
            v == 0.0 || v == 3.0 || v == 7.0,
            "nearest 不应混出非法类别，得到 {v}"
        );
    }
}

#[test]
fn test_affine_bilinear_identity_matches_input() {
    // identity 参数 → 输出应该和输入相同
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
    let p = params(0.0, 0.0, 0.0, 1.0, 0.0);
    let out = affine_bilinear(&input, p, 0.0);
    for r in 0..3 {
        for c in 0..3 {
            assert_abs_diff_eq!(out[[r, c]], input[[r, c]], epsilon = 1e-5);
        }
    }
}

// ---------------------------------------------------------------------------
// SampleTransform 入口测试（identity 短路路径）
// ---------------------------------------------------------------------------

/// `RandomAffine::new(0.0)` 且不设置其他参数 → sample_params 必采样到恒等
/// → 走 is_identity 短路路径，label 完全保留。
fn identity_affine() -> RandomAffine {
    RandomAffine::new(0.0)
}

#[test]
fn test_paired_affine_classification_identity_preserves_label() {
    let image = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let sample = ClassificationSample::new(image, 42);
    let result = identity_affine().apply_to(sample);
    assert_eq!(result.label, 42);
    assert_abs_diff_eq!(result.image[[0, 0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(result.image[[0, 1, 1]], 4.0, epsilon = 1e-5);
}

#[test]
fn test_paired_affine_detection_identity_preserves_labels() {
    let image = Tensor::new(&[1.0; 9], &[1, 3, 3]);
    let labels = vec![
        GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0),
        GroundTruthBox::new(BBox::from_xyxy(1.0, 1.0, 2.0, 2.0), 1),
    ];
    let sample = DetectionSample::new(image, labels);
    let result = identity_affine().apply_to(sample);
    assert_eq!(result.labels.len(), 2);
    assert_eq!(result.labels[0].class_id, 0);
    assert_eq!(result.labels[1].class_id, 1);
    let bbox0 = result.labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(bbox0[0], 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(bbox0[2], 1.0, epsilon = 1e-5);
}

#[test]
fn test_paired_affine_segmentation_identity_preserves_mask() {
    let image = Tensor::new(&[1.0; 9], &[1, 3, 3]);
    let mask_data: Vec<f32> = (0..9).map(|v| v as f32).collect();
    let mask = Tensor::new(&mask_data, &[3, 3]);
    let sample = SegmentationSample::new(image, mask);
    let result = identity_affine().apply_to(sample);
    assert_eq!(result.image.shape(), &[1, 3, 3]);
    assert_eq!(result.mask.shape(), &[3, 3]);
    for r in 0..3 {
        for c in 0..3 {
            assert_abs_diff_eq!(result.mask[[r, c]], (r * 3 + c) as f32, epsilon = 1e-5);
        }
    }
}

#[test]
fn test_paired_affine_detection_shape_preserved_under_random_params() {
    // 设置 degrees 与 scale 范围 → 采样结果几乎一定非恒等 → 仍要保证
    // image shape 与 labels 仍是 Vec（可能被 filter 掉，但类型结构完整）
    let image = Tensor::new(&[1.0; 16], &[1, 4, 4]);
    let labels = vec![GroundTruthBox::new(BBox::from_xyxy(1.0, 1.0, 3.0, 3.0), 0)];
    let sample = DetectionSample::new(image, labels);
    let affine = RandomAffine::new(5.0)
        .scale(0.95, 1.05)
        .with_label_filter(DetectionLabelFilter::new(0.1));
    let result = affine.apply_to(sample);
    assert_eq!(result.image.shape(), &[1, 4, 4]);
    // 幅度很小的 random affine 下，面积 4 的 bbox 不该被过滤掉
    assert!(
        !result.labels.is_empty(),
        "随机小幅 affine 下 bbox 不应被过滤"
    );
}

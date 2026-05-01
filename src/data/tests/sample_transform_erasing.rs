//! `SampleTransform` 在 RandomErasing 上的 paired 实现测试。
//!
//! 核心语义（A 方案，与 torchvision v2 对齐）：**只擦 image，label / bbox /
//! mask 原样保留**。下面的断言围绕这一点展开。

use approx::assert_abs_diff_eq;

use crate::data::transforms::random_erasing::erase_region;
use crate::data::transforms::{RandomErasing, SampleTransform};
use crate::data::{ClassificationSample, DetectionSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, GroundTruthBox};

// ---------------------------------------------------------------------------
// erase_region 纯函数测试
// ---------------------------------------------------------------------------

#[test]
fn test_erase_region_3d_fills_rectangle_on_all_channels() {
    let input = Tensor::new(&[1.0; 27], &[3, 3, 3]);
    let out = erase_region(&input, 1, 1, 2, 2, -1.0);
    assert_eq!(out.shape(), &[3, 3, 3]);
    // 被擦的位置：每个通道的 (1..3) × (1..3)
    for ch in 0..3 {
        for row in 1..3 {
            for col in 1..3 {
                assert_abs_diff_eq!(out[[ch, row, col]], -1.0, epsilon = 1e-6);
            }
        }
    }
    // 未被擦的位置：保持 1.0
    assert_abs_diff_eq!(out[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(out[[2, 0, 2]], 1.0, epsilon = 1e-6);
}

#[test]
fn test_erase_region_2d_fills_rectangle() {
    let input = Tensor::new(&[1.0; 16], &[4, 4]);
    let out = erase_region(&input, 0, 0, 2, 2, 9.0);
    assert_eq!(out.shape(), &[4, 4]);
    for r in 0..2 {
        for c in 0..2 {
            assert_abs_diff_eq!(out[[r, c]], 9.0, epsilon = 1e-6);
        }
    }
    assert_abs_diff_eq!(out[[3, 3]], 1.0, epsilon = 1e-6);
}

// ---------------------------------------------------------------------------
// SampleTransform 入口测试
// ---------------------------------------------------------------------------

#[test]
fn test_paired_erasing_p_zero_keeps_all_samples() {
    // p=0 → 永不擦除，三档 sample 完全原样
    let image = Tensor::new(&[1.0; 27], &[3, 3, 3]);
    let erasing = RandomErasing::new(0.0).value(-99.0);

    let cls = ClassificationSample::new(image.clone(), 7);
    let out = erasing.apply_to(cls);
    assert_eq!(out.label, 7);
    let flat = out.image.flatten_view();
    assert!(flat.iter().all(|&v| (v - 1.0).abs() < 1e-6));

    let det = DetectionSample::new(
        image.clone(),
        vec![GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), 3)],
    );
    let out = erasing.apply_to(det);
    assert_eq!(out.labels.len(), 1);
    assert_eq!(out.labels[0].class_id, 3);

    let mask = Tensor::new(&[5.0; 9], &[3, 3]);
    let seg = SegmentationSample::new(image, mask);
    let out = erasing.apply_to(seg);
    let flat = out.mask.flatten_view();
    assert!(flat.iter().all(|&v| (v - 5.0).abs() < 1e-6));
}

#[test]
fn test_paired_erasing_detection_labels_are_always_preserved() {
    // A 方案：即使 p=1.0 擦了图像，bbox / class_id 依然保留
    let image = Tensor::new(&[1.0; 48], &[3, 4, 4]);
    let labels = vec![
        GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 2.0, 2.0), 0),
        GroundTruthBox::new(BBox::from_xyxy(2.0, 2.0, 4.0, 4.0), 1),
    ];
    let sample = DetectionSample::new(image, labels.clone());
    let erasing = RandomErasing::new(1.0).scale(0.1, 0.3);
    let result = erasing.apply_to(sample);

    assert_eq!(result.labels.len(), 2, "A 方案下 labels 必须全部保留");
    assert_eq!(result.labels[0].class_id, 0);
    assert_eq!(result.labels[1].class_id, 1);
    // bbox 坐标也完全保留
    let bbox0 = result.labels[0].bbox.to_xyxy();
    let orig0 = labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(bbox0[0], orig0[0], epsilon = 1e-6);
    assert_abs_diff_eq!(bbox0[2], orig0[2], epsilon = 1e-6);
}

#[test]
fn test_paired_erasing_segmentation_mask_is_always_preserved() {
    // A 方案：mask 永远不动，哪怕 image 被擦
    let image = Tensor::new(&[1.0; 48], &[3, 4, 4]);
    let mask_data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let mask = Tensor::new(&mask_data, &[4, 4]);
    let sample = SegmentationSample::new(image, mask.clone());
    let erasing = RandomErasing::new(1.0).value(-1.0).scale(0.1, 0.3);

    // 多次采样其中有可能命中擦除；mask 必须始终保持
    for _ in 0..20 {
        let result = erasing.apply_to(sample.clone());
        let out_mask = result.mask.flatten_view();
        for (i, &v) in out_mask.iter().enumerate() {
            assert_abs_diff_eq!(v, i as f32, epsilon = 1e-6);
        }
    }
}

#[test]
fn test_paired_erasing_classification_actually_erases_image_when_p_one() {
    // p=1.0 + 可行 scale → 大概率真的擦到像素；label 保持不变
    let image = Tensor::new(&[1.0; 48], &[3, 4, 4]);
    let sample = ClassificationSample::new(image, 42);
    let erasing = RandomErasing::new(1.0).value(-1.0).scale(0.1, 0.3);

    let mut ever_erased = false;
    for _ in 0..50 {
        let result = erasing.apply_to(sample.clone());
        assert_eq!(result.label, 42);
        let flat = result.image.flatten_view();
        if flat.iter().any(|&v| v == -1.0) {
            ever_erased = true;
            break;
        }
    }
    assert!(ever_erased, "p=1.0 且 scale 合理时，应至少擦到一次");
}

#[test]
fn test_paired_erasing_2d_classification_image_supported() {
    // paired 路径额外支持 2D 灰度图；p=0 路径下无副作用
    let image = Tensor::new(&[1.0; 16], &[4, 4]);
    let sample = ClassificationSample::new(image, 1);
    let erasing = RandomErasing::new(0.0);
    let result = erasing.apply_to(sample);
    assert_eq!(result.image.shape(), &[4, 4]);
    assert_eq!(result.label, 1);
}

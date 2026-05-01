//! `SampleTransform` 在 RandomHorizontalFlip 上的 paired 实现测试。

use approx::assert_abs_diff_eq;

use crate::data::transforms::{RandomHorizontalFlip, SampleTransform};
use crate::data::{ClassificationSample, DetectionSample, SegmentationSample};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, GroundTruthBox};

/// 1×3×4 单通道图像，方便人眼验证翻转。
fn make_image_3x4() -> Tensor {
    Tensor::new(
        &[
            1.0, 2.0, 3.0, 4.0, // row 0
            5.0, 6.0, 7.0, 8.0, // row 1
            9.0, 10.0, 11.0, 12.0, // row 2
        ],
        &[1, 3, 4],
    )
}

#[test]
fn test_paired_hflip_classification_p_zero_keeps_sample() {
    let sample = ClassificationSample::new(make_image_3x4(), 5);
    let flip = RandomHorizontalFlip::new(0.0);
    let result = flip.apply_to(sample.clone());
    assert_eq!(result.label, 5);
    assert_eq!(result.image.to_vec(), sample.image.to_vec());
}

#[test]
fn test_paired_hflip_classification_p_one_flips_image_keeps_label() {
    let sample = ClassificationSample::new(make_image_3x4(), 5);
    let flip = RandomHorizontalFlip::new(1.0);
    let result = flip.apply_to(sample);
    assert_eq!(result.label, 5);
    // 第一行 [1, 2, 3, 4] → [4, 3, 2, 1]
    assert_abs_diff_eq!(result.image[[0, 0, 0]], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.image[[0, 0, 3]], 1.0, epsilon = 1e-6);
}

#[test]
fn test_paired_hflip_detection_synchronizes_bbox() {
    let image = make_image_3x4(); // shape [1, 3, 4]，width = 4
    let labels = vec![GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 2.0, 3.0), 0)];
    let sample = DetectionSample::new(image, labels);
    let flip = RandomHorizontalFlip::new(1.0);
    let result = flip.apply_to(sample);

    // image_width = 4，原 bbox (0, 0, 2, 3) → 翻转后 (4-2, 0, 4-0, 3) = (2, 0, 4, 3)
    let bbox = result.labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(bbox[0], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[2], 4.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[3], 3.0, epsilon = 1e-6);
    assert_eq!(result.labels[0].class_id, 0);
    // image 也被翻转
    assert_abs_diff_eq!(result.image[[0, 0, 0]], 4.0, epsilon = 1e-6);
}

#[test]
fn test_paired_hflip_detection_p_zero_keeps_sample() {
    let image = make_image_3x4();
    let labels = vec![GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 2.0, 3.0), 0)];
    let sample = DetectionSample::new(image, labels);
    let flip = RandomHorizontalFlip::new(0.0);
    let result = flip.apply_to(sample);

    let bbox = result.labels[0].bbox.to_xyxy();
    assert_abs_diff_eq!(bbox[0], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(bbox[2], 2.0, epsilon = 1e-6);
}

#[test]
fn test_paired_hflip_segmentation_flips_image_and_mask() {
    let image = make_image_3x4();
    let mask = Tensor::new(
        &[
            0.0, 0.0, 0.0, 1.0, // row 0
            0.0, 0.0, 0.0, 0.0, // row 1
            0.0, 0.0, 0.0, 0.0, // row 2
        ],
        &[3, 4],
    );
    let sample = SegmentationSample::new(image, mask);
    let flip = RandomHorizontalFlip::new(1.0);
    let result = flip.apply_to(sample);

    // mask 第一行 [0, 0, 0, 1] → [1, 0, 0, 0]
    assert_abs_diff_eq!(result.mask[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.mask[[0, 3]], 0.0, epsilon = 1e-6);
    // image 也被翻转
    assert_abs_diff_eq!(result.image[[0, 0, 0]], 4.0, epsilon = 1e-6);
}

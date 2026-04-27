//! 分割指标测试

use crate::metrics::{binary_iou, pixel_accuracy};
use crate::tensor::Tensor;

#[test]
fn test_pixel_accuracy_basic() {
    let predictions = Tensor::new(&[0.9, 0.8, 0.2, 0.1], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 0.0, 0.0], &[1, 1, 2, 2]);

    let result = pixel_accuracy(&predictions, &actuals, 0.5);

    assert!((result.value() - 0.75).abs() < 1e-6);
    assert_eq!(result.n_samples(), 4);
}

#[test]
fn test_pixel_accuracy_perfect() {
    let predictions = Tensor::new(&[0.9, 0.1, 0.8, 0.2], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 2, 2]);

    let result = pixel_accuracy(&predictions, &actuals, 0.5);

    assert!((result.value() - 1.0).abs() < 1e-6);
    assert_eq!(result.n_samples(), 4);
}

#[test]
fn test_binary_iou_basic() {
    let predictions = Tensor::new(&[0.9, 0.8, 0.2, 0.1], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 2, 2]);

    let result = binary_iou(&predictions, &actuals, 0.5);

    // pred positives: {0,1}; actual positives: {0,2}; intersection=1, union=3
    assert!((result.value() - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(result.n_samples(), 4);
}

#[test]
fn test_binary_iou_empty_masks_are_perfect_match() {
    let predictions = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 2, 2]);

    let result = binary_iou(&predictions, &actuals, 0.5);

    assert!((result.value() - 1.0).abs() < 1e-6);
    assert_eq!(result.n_samples(), 4);
}

#[test]
fn test_segmentation_metrics_empty_tensor() {
    let predictions = Tensor::new(&[] as &[f32], &[0, 1, 2, 2]);
    let actuals = Tensor::new(&[] as &[f32], &[0, 1, 2, 2]);

    let acc = pixel_accuracy(&predictions, &actuals, 0.5);
    let iou = binary_iou(&predictions, &actuals, 0.5);

    assert_eq!(acc.value(), 0.0);
    assert_eq!(acc.n_samples(), 0);
    assert_eq!(iou.value(), 0.0);
    assert_eq!(iou.n_samples(), 0);
}

#[test]
#[should_panic(expected = "pixel_accuracy: 形状不匹配")]
fn test_pixel_accuracy_shape_mismatch_panics() {
    let predictions = Tensor::new(&[0.9, 0.1], &[1, 1, 1, 2]);
    let actuals = Tensor::new(&[1.0, 0.0], &[1, 2]);

    let _ = pixel_accuracy(&predictions, &actuals, 0.5);
}

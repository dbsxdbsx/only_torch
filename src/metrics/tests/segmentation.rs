//! 分割指标测试

use crate::metrics::{
    binary_iou, dice_score, mean_iou, per_class_iou, pixel_accuracy, semantic_pixel_accuracy,
};
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
fn test_dice_score_basic() {
    let predictions = Tensor::new(&[0.9, 0.8, 0.2, 0.1], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[1, 1, 2, 2]);

    let result = dice_score(&predictions, &actuals, 0.5);

    // pred positives: {0,1}; actual positives: {0,2}; intersection=1
    assert!((result.value() - 0.5).abs() < 1e-6);
    assert_eq!(result.n_samples(), 4);
}

#[test]
fn test_dice_score_empty_masks_are_perfect_match() {
    let predictions = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 2, 2]);

    let result = dice_score(&predictions, &actuals, 0.5);

    assert!((result.value() - 1.0).abs() < 1e-6);
    assert_eq!(result.n_samples(), 4);
}

#[test]
fn test_semantic_pixel_accuracy_basic() {
    // 两类、三个像素正确，一个像素错误。
    let predictions = Tensor::new(
        &[
            0.9, 0.2, 0.8, 0.4, // class 0
            0.1, 0.8, 0.2, 0.6, // class 1
        ],
        &[1, 2, 2, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 1.0, // class 0
            0.0, 1.0, 0.0, 0.0, // class 1
        ],
        &[1, 2, 2, 2],
    );

    let result = semantic_pixel_accuracy(&predictions, &actuals);

    assert!((result.value() - 0.75).abs() < 1e-6);
    assert_eq!(result.n_samples(), 4);
}

#[test]
fn test_per_class_iou_and_mean_iou_basic() {
    let predictions = Tensor::new(
        &[
            0.9, 0.2, 0.8, 0.4, // class 0 => pixels {0,2}
            0.1, 0.8, 0.2, 0.6, // class 1 => pixels {1,3}
        ],
        &[1, 2, 2, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 1.0, // class 0 => pixels {0,2,3}
            0.0, 1.0, 0.0, 0.0, // class 1 => pixel {1}
        ],
        &[1, 2, 2, 2],
    );

    let per_class = per_class_iou(&predictions, &actuals);
    let miou = mean_iou(&predictions, &actuals);

    assert_eq!(per_class.len(), 2);
    assert!((per_class[0].value() - 2.0 / 3.0).abs() < 1e-6);
    assert!((per_class[1].value() - 0.5).abs() < 1e-6);
    assert!((miou.value() - (2.0 / 3.0 + 0.5) / 2.0).abs() < 1e-6);
    assert_eq!(miou.n_samples(), 4);
}

#[test]
fn test_per_class_iou_absent_class_is_perfect_match() {
    let predictions = Tensor::new(
        &[
            0.9, 0.8, // class 0
            0.1, 0.2, // class 1
            0.0, 0.0, // class 2 absent
        ],
        &[1, 3, 1, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 1.0, // class 0
            0.0, 0.0, // class 1
            0.0, 0.0, // class 2 absent
        ],
        &[1, 3, 1, 2],
    );

    let per_class = per_class_iou(&predictions, &actuals);

    assert!((per_class[2].value() - 1.0).abs() < 1e-6);
}

#[test]
fn test_segmentation_metrics_empty_tensor() {
    let predictions = Tensor::new(&[] as &[f32], &[0, 1, 2, 2]);
    let actuals = Tensor::new(&[] as &[f32], &[0, 1, 2, 2]);

    let acc = pixel_accuracy(&predictions, &actuals, 0.5);
    let iou = binary_iou(&predictions, &actuals, 0.5);
    let dice = dice_score(&predictions, &actuals, 0.5);

    assert_eq!(acc.value(), 0.0);
    assert_eq!(acc.n_samples(), 0);
    assert_eq!(iou.value(), 0.0);
    assert_eq!(iou.n_samples(), 0);
    assert_eq!(dice.value(), 0.0);
    assert_eq!(dice.n_samples(), 0);
}

#[test]
#[should_panic(expected = "pixel_accuracy: 形状不匹配")]
fn test_pixel_accuracy_shape_mismatch_panics() {
    let predictions = Tensor::new(&[0.9, 0.1], &[1, 1, 1, 2]);
    let actuals = Tensor::new(&[1.0, 0.0], &[1, 2]);

    let _ = pixel_accuracy(&predictions, &actuals, 0.5);
}

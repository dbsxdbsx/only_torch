//! 分割指标测试

use crate::metrics::{
    binary_iou, dice_score, empty_slot_accuracy, mean_instance_iou, mean_iou, mean_valid_slot_iou,
    per_class_iou, pixel_accuracy, semantic_pixel_accuracy,
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
fn test_binary_iou_batch() {
    let predictions = Tensor::new(
        &[
            0.9, 0.8, 0.2, 0.1, // sample 0: positives {0,1}
            0.1, 0.9, 0.9, 0.1, // sample 1: positives {1,2}
        ],
        &[2, 1, 2, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 0.0, // sample 0: positives {0,2}
            0.0, 1.0, 0.0, 1.0, // sample 1: positives {1,3}
        ],
        &[2, 1, 2, 2],
    );

    let result = binary_iou(&predictions, &actuals, 0.5);

    // 整个 batch 上的 micro IoU：intersection=2，union=6。
    assert!((result.value() - 1.0 / 3.0).abs() < 1e-6);
    assert_eq!(result.n_samples(), 8);
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

#[test]
fn test_mean_instance_iou_two_slots_perfect_match() {
    // [1, 2, 2, 2]:slot 0 完全匹配,slot 1 完全匹配 → mean = 1.0
    let preds = Tensor::new(
        &[
            0.9, 0.9, 0.1, 0.1, // slot 0
            0.1, 0.1, 0.9, 0.9, // slot 1
        ],
        &[1, 2, 2, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 1.0, 0.0, 0.0, // slot 0
            0.0, 0.0, 1.0, 1.0, // slot 1
        ],
        &[1, 2, 2, 2],
    );

    let metric = mean_instance_iou(&preds, &actuals, 0.5);
    assert!((metric.value() - 1.0).abs() < 1e-6);
    assert_eq!(metric.n_samples(), 2);
}

#[test]
fn test_mean_instance_iou_partial_overlap() {
    // [1, 2, 2, 2]:slot 0 完美,slot 1 IoU = 1/3 → mean = 2/3
    let preds = Tensor::new(
        &[
            0.9, 0.9, 0.1, 0.1, // slot 0:positives {0,1}
            0.9, 0.1, 0.1, 0.9, // slot 1:positives {0,3}
        ],
        &[1, 2, 2, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 1.0, 0.0, 0.0, // slot 0:positives {0,1} → IoU = 1.0
            0.0, 0.0, 1.0,
            1.0, // slot 1:positives {2,3},inter={3},union={0,2,3} → IoU = 1/3
        ],
        &[1, 2, 2, 2],
    );

    let metric = mean_instance_iou(&preds, &actuals, 0.5);
    let expected = (1.0 + 1.0 / 3.0) / 2.0;
    assert!((metric.value() - expected).abs() < 1e-6);
    assert_eq!(metric.n_samples(), 2);
}

#[test]
fn test_mean_instance_iou_empty_slot_treats_match_as_perfect() {
    // 两个 slot 都全空 → IoU = 1.0(union=0 时按"完美匹配空 mask"处理)
    let preds = Tensor::new(&[0.0; 8], &[1, 2, 2, 2]);
    let actuals = Tensor::new(&[0.0; 8], &[1, 2, 2, 2]);

    let metric = mean_instance_iou(&preds, &actuals, 0.5);
    assert!((metric.value() - 1.0).abs() < 1e-6);
    assert_eq!(metric.n_samples(), 2);
}

#[test]
fn test_mean_valid_slot_iou_skips_empty_target_slots() {
    // [1, 2, 2, 2]:slot 0 target 非空,slot 1 target 全空
    // → 只评估 slot 0,即使 slot 1 的预测有错也不计入
    let preds = Tensor::new(
        &[
            0.9, 0.9, 0.1, 0.1, // slot 0:与 target 完全一致
            0.9, 0.0, 0.0, 0.0, // slot 1:错误地预测了 1 个像素
        ],
        &[1, 2, 2, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 1.0, 0.0, 0.0, // slot 0:有 target
            0.0, 0.0, 0.0, 0.0, // slot 1:全空
        ],
        &[1, 2, 2, 2],
    );

    let metric = mean_valid_slot_iou(&preds, &actuals, 0.5);
    assert!((metric.value() - 1.0).abs() < 1e-6);
    // valid slot 数 = 1
    assert_eq!(metric.n_samples(), 1);
}

#[test]
fn test_mean_valid_slot_iou_zero_when_no_valid() {
    // 全空 target → 没有 valid slot,返回 (0.0, 0)
    let preds = Tensor::new(&[0.0; 4], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[0.0; 4], &[1, 1, 2, 2]);

    let metric = mean_valid_slot_iou(&preds, &actuals, 0.5);
    assert_eq!(metric.value(), 0.0);
    assert_eq!(metric.n_samples(), 0);
}

#[test]
fn test_empty_slot_accuracy_basic() {
    // [1, 2, 2, 2]:slot 0 target 非空(不计入),slot 1 target 全空
    // slot 1 prediction 也全空 → 1/1 = 100%
    let preds = Tensor::new(
        &[
            0.9, 0.9, 0.1, 0.1, // slot 0:有预测
            0.0, 0.0, 0.0, 0.0, // slot 1:全空预测
        ],
        &[1, 2, 2, 2],
    );
    let actuals = Tensor::new(
        &[
            1.0, 1.0, 0.0, 0.0, // slot 0:非空
            0.0, 0.0, 0.0, 0.0, // slot 1:空
        ],
        &[1, 2, 2, 2],
    );

    let metric = empty_slot_accuracy(&preds, &actuals, 0.5);
    assert!((metric.value() - 1.0).abs() < 1e-6);
    assert_eq!(metric.n_samples(), 1);
}

#[test]
fn test_empty_slot_accuracy_wrong_prediction() {
    // 1 个空 slot,但 prediction 有正像素 → 0/1 = 0%
    let preds = Tensor::new(&[0.9, 0.0, 0.0, 0.0], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 1, 2, 2]);

    let metric = empty_slot_accuracy(&preds, &actuals, 0.5);
    assert_eq!(metric.value(), 0.0);
    assert_eq!(metric.n_samples(), 1);
}

#[test]
fn test_empty_slot_accuracy_no_empty_returns_full() {
    // 没有空 slot → 返回 (1.0, 0),约定调用方按 n_samples=0 判断
    let preds = Tensor::new(&[0.9, 0.9, 0.0, 0.0], &[1, 1, 2, 2]);
    let actuals = Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[1, 1, 2, 2]);

    let metric = empty_slot_accuracy(&preds, &actuals, 0.5);
    assert_eq!(metric.value(), 1.0);
    assert_eq!(metric.n_samples(), 0);
}

#[test]
#[should_panic(expected = "mean_instance_iou: 期望 shape=[N, S, H, W]")]
fn test_mean_instance_iou_panics_on_wrong_dim() {
    let preds = Tensor::new(&[0.9, 0.1], &[1, 2]);
    let actuals = Tensor::new(&[1.0, 0.0], &[1, 2]);

    let _ = mean_instance_iou(&preds, &actuals, 0.5);
}

//! 目标检测指标测试

use approx::assert_abs_diff_eq;

use crate::metrics::{
    DetectionMetricOptions, VOC_IOU_THRESHOLDS, mean_average_precision,
    mean_average_precision_with_options, mean_box_iou_cxcywh, precision_recall_at_iou,
    precision_recall_at_iou_with_options,
};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, Detection, GroundTruthBox};

#[test]
fn test_mean_box_iou_cxcywh_perfect() {
    let predictions = Tensor::new(&[0.5, 0.5, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2], &[2, 4]);
    let actuals = Tensor::new(&[0.5, 0.5, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2], &[2, 4]);

    let result = mean_box_iou_cxcywh(&predictions, &actuals);

    assert_abs_diff_eq!(result.value(), 1.0, epsilon = 1e-6);
    assert_eq!(result.n_samples(), 2);
}

#[test]
fn test_mean_box_iou_cxcywh_partial_overlap() {
    let predictions = Tensor::new(&[0.5, 0.5, 0.4, 0.4], &[1, 4]);
    let actuals = Tensor::new(&[0.6, 0.5, 0.4, 0.4], &[1, 4]);

    let result = mean_box_iou_cxcywh(&predictions, &actuals);

    // 两个 0.4x0.4 框在 x 方向错开 0.1：intersection=0.3*0.4，union=0.2
    assert_abs_diff_eq!(result.value(), 0.6, epsilon = 1e-6);
    assert_eq!(result.n_samples(), 1);
}

#[test]
fn test_mean_box_iou_cxcywh_no_overlap() {
    let predictions = Tensor::new(&[0.2, 0.2, 0.2, 0.2], &[1, 4]);
    let actuals = Tensor::new(&[0.8, 0.8, 0.2, 0.2], &[1, 4]);

    let result = mean_box_iou_cxcywh(&predictions, &actuals);

    assert_eq!(result.value(), 0.0);
    assert_eq!(result.n_samples(), 1);
}

#[test]
fn test_mean_box_iou_cxcywh_empty_tensor() {
    let predictions = Tensor::new(&[] as &[f32], &[0, 4]);
    let actuals = Tensor::new(&[] as &[f32], &[0, 4]);

    let result = mean_box_iou_cxcywh(&predictions, &actuals);

    assert_eq!(result.value(), 0.0);
    assert_eq!(result.n_samples(), 0);
}

#[test]
fn test_mean_box_iou_cxcywh_clips_to_unit_square() {
    let predictions = Tensor::new(&[0.0, 0.0, 0.4, 0.4], &[1, 4]);
    let actuals = Tensor::new(&[0.0, 0.0, 0.4, 0.4], &[1, 4]);

    let result = mean_box_iou_cxcywh(&predictions, &actuals);

    assert_abs_diff_eq!(result.value(), 1.0, epsilon = 1e-6);
}

#[test]
#[should_panic(expected = "mean_box_iou_cxcywh: 形状不匹配")]
fn test_mean_box_iou_cxcywh_shape_mismatch_panics() {
    let predictions = Tensor::new(&[0.5, 0.5, 0.2, 0.2], &[1, 4]);
    let actuals = Tensor::new(&[0.5, 0.5, 0.2, 0.2], &[4]);

    let _ = mean_box_iou_cxcywh(&predictions, &actuals);
}

#[test]
#[should_panic(expected = "mean_box_iou_cxcywh: bbox Tensor 必须是 [N, 4]")]
fn test_mean_box_iou_cxcywh_invalid_bbox_width_panics() {
    let predictions = Tensor::new(&[0.5, 0.5, 0.2], &[1, 3]);
    let actuals = Tensor::new(&[0.5, 0.5, 0.2], &[1, 3]);

    let _ = mean_box_iou_cxcywh(&predictions, &actuals);
}

#[test]
fn test_mean_average_precision_perfect_predictions() {
    let predictions = vec![
        vec![Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.9, 0)],
        vec![Detection::new(BBox::from_xyxy(2.0, 2.0, 3.0, 3.0), 0.8, 1)],
    ];
    let ground_truths = vec![
        vec![GroundTruthBox::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0)],
        vec![GroundTruthBox::new(BBox::from_xyxy(2.0, 2.0, 3.0, 3.0), 1)],
    ];

    let result = mean_average_precision(&predictions, &ground_truths, 2, VOC_IOU_THRESHOLDS);

    assert_abs_diff_eq!(result.value(), 1.0, epsilon = 1e-6);
    assert_eq!(result.n_samples(), 2);
    assert_eq!(result.per_threshold_ap().len(), 1);
    assert_abs_diff_eq!(result.per_threshold_ap()[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.map_50().unwrap(), 1.0, epsilon = 1e-6);
    assert_eq!(result.per_class_ap().len(), 2);
    assert_abs_diff_eq!(result.per_class_ap()[0].unwrap(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.per_class_ap()[1].unwrap(), 1.0, epsilon = 1e-6);
}

#[test]
fn test_average_precision_penalizes_high_score_false_positive() {
    let predictions = vec![vec![
        Detection::new(BBox::from_xyxy(2.0, 2.0, 3.0, 3.0), 0.9, 0),
        Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.8, 0),
    ]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let result = mean_average_precision(&predictions, &ground_truths, 1, VOC_IOU_THRESHOLDS);

    assert_abs_diff_eq!(result.value(), 0.5, epsilon = 1e-6);
}

#[test]
fn test_mean_average_precision_options_filter_score_and_limit_max_detections() {
    let predictions = vec![vec![
        Detection::new(BBox::from_xyxy(2.0, 2.0, 3.0, 3.0), 0.95, 0),
        Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.90, 0),
        Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.10, 0),
    ]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let limited = mean_average_precision_with_options(
        &predictions,
        &ground_truths,
        1,
        &DetectionMetricOptions::voc().with_max_detections(1),
    );
    let filtered = mean_average_precision_with_options(
        &predictions,
        &ground_truths,
        1,
        &DetectionMetricOptions::voc().with_score_threshold(0.5),
    );

    assert_abs_diff_eq!(limited.value(), 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(filtered.value(), 0.5, epsilon = 1e-6);
}

#[test]
fn test_mean_average_precision_empty_ground_truth_preserves_thresholds() {
    let predictions = vec![vec![Detection::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0.9,
        0,
    )]];
    let ground_truths = vec![Vec::new()];

    let result = mean_average_precision(&predictions, &ground_truths, 1, VOC_IOU_THRESHOLDS);

    assert_eq!(result.n_samples(), 0);
    assert_eq!(result.iou_thresholds(), VOC_IOU_THRESHOLDS);
    assert_eq!(result.per_threshold_ap().len(), 1);
    assert_abs_diff_eq!(result.map_50().unwrap(), 0.0, epsilon = 1e-6);
    assert_eq!(result.per_class_ap(), &[None]);
}

#[test]
fn test_coco_style_result_exposes_thresholds_and_map_50_95() {
    let predictions = vec![vec![Detection::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0.9,
        0,
    )]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let result = mean_average_precision_with_options(
        &predictions,
        &ground_truths,
        1,
        &DetectionMetricOptions::coco_style(),
    );

    assert_eq!(result.iou_thresholds().len(), 10);
    assert_abs_diff_eq!(result.map_50().unwrap(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.map_50_95(), 1.0, epsilon = 1e-6);
}

#[test]
fn test_mean_average_precision_multiple_thresholds() {
    let predictions = vec![vec![Detection::new(
        BBox::from_xyxy(0.25, 0.0, 1.25, 1.0),
        0.9,
        0,
    )]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let result = mean_average_precision(&predictions, &ground_truths, 1, &[0.5, 0.75]);

    assert_abs_diff_eq!(result.per_threshold_ap()[0], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.per_threshold_ap()[1], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.value(), 0.5, epsilon = 1e-6);
}

#[test]
fn test_precision_recall_counts_empty_class_predictions_as_false_positive() {
    let predictions = vec![vec![
        Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.9, 0),
        Detection::new(BBox::from_xyxy(2.0, 2.0, 3.0, 3.0), 0.8, 1),
    ]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let result = precision_recall_at_iou(&predictions, &ground_truths, 2, 0.5);

    assert_eq!(result.true_positives(), 1);
    assert_eq!(result.false_positives(), 1);
    assert_eq!(result.false_negatives(), 0);
    assert_abs_diff_eq!(result.precision(), 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(result.recall(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.f1_score(), 2.0 / 3.0, epsilon = 1e-6);
}

#[test]
fn test_precision_recall_options_drop_low_score_false_positive() {
    let predictions = vec![vec![
        Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.9, 0),
        Detection::new(BBox::from_xyxy(2.0, 2.0, 3.0, 3.0), 0.1, 0),
    ]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let result = precision_recall_at_iou_with_options(
        &predictions,
        &ground_truths,
        1,
        0.5,
        &DetectionMetricOptions::voc().with_score_threshold(0.5),
    );

    assert_eq!(result.true_positives(), 1);
    assert_eq!(result.false_positives(), 0);
    assert_abs_diff_eq!(result.precision(), 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(result.recall(), 1.0, epsilon = 1e-6);
}

#[test]
fn test_precision_recall_duplicate_predictions_match_one_gt_once() {
    let predictions = vec![vec![
        Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.9, 0),
        Detection::new(BBox::from_xyxy(0.0, 0.0, 1.0, 1.0), 0.8, 0),
    ]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let result = precision_recall_at_iou(&predictions, &ground_truths, 1, 0.5);

    assert_eq!(result.true_positives(), 1);
    assert_eq!(result.false_positives(), 1);
    assert_eq!(result.false_negatives(), 0);
    assert_abs_diff_eq!(result.precision(), 0.5, epsilon = 1e-6);
    assert_abs_diff_eq!(result.recall(), 1.0, epsilon = 1e-6);
}

#[test]
fn test_mean_average_precision_ignores_classes_without_ground_truth() {
    let predictions = vec![vec![Detection::new(
        BBox::from_xyxy(2.0, 2.0, 3.0, 3.0),
        0.9,
        1,
    )]];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0,
    )]];

    let result = mean_average_precision(&predictions, &ground_truths, 2, VOC_IOU_THRESHOLDS);

    assert_abs_diff_eq!(result.value(), 0.0, epsilon = 1e-6);
    assert_eq!(result.n_samples(), 1);
}

#[test]
#[should_panic(expected = "predictions 和 ground_truths 图片数量不一致")]
fn test_mean_average_precision_rejects_mismatched_image_count() {
    let predictions = vec![Vec::new(), Vec::new()];
    let ground_truths = vec![Vec::new()];

    let _ = mean_average_precision(&predictions, &ground_truths, 1, VOC_IOU_THRESHOLDS);
}

#[test]
#[should_panic(expected = "prediction class_id 1 越界")]
fn test_mean_average_precision_rejects_prediction_class_out_of_range() {
    let predictions = vec![vec![Detection::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        0.9,
        1,
    )]];
    let ground_truths = vec![Vec::new()];

    let _ = mean_average_precision(&predictions, &ground_truths, 1, VOC_IOU_THRESHOLDS);
}

#[test]
#[should_panic(expected = "ground truth class_id 1 越界")]
fn test_mean_average_precision_rejects_ground_truth_class_out_of_range() {
    let predictions = vec![Vec::new()];
    let ground_truths = vec![vec![GroundTruthBox::new(
        BBox::from_xyxy(0.0, 0.0, 1.0, 1.0),
        1,
    )]];

    let _ = mean_average_precision(&predictions, &ground_truths, 1, VOC_IOU_THRESHOLDS);
}

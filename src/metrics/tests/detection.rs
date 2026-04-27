//! 目标检测指标测试

use crate::metrics::mean_box_iou_cxcywh;
use crate::tensor::Tensor;

#[test]
fn test_mean_box_iou_cxcywh_perfect() {
    let predictions = Tensor::new(&[0.5, 0.5, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2], &[2, 4]);
    let actuals = Tensor::new(&[0.5, 0.5, 0.4, 0.2, 0.3, 0.4, 0.2, 0.2], &[2, 4]);

    let result = mean_box_iou_cxcywh(&predictions, &actuals);

    assert!((result.value() - 1.0).abs() < 1e-6);
    assert_eq!(result.n_samples(), 2);
}

#[test]
fn test_mean_box_iou_cxcywh_partial_overlap() {
    let predictions = Tensor::new(&[0.5, 0.5, 0.4, 0.4], &[1, 4]);
    let actuals = Tensor::new(&[0.6, 0.5, 0.4, 0.4], &[1, 4]);

    let result = mean_box_iou_cxcywh(&predictions, &actuals);

    // 两个 0.4x0.4 框在 x 方向错开 0.1：intersection=0.3*0.4，union=0.2
    assert!((result.value() - 0.6).abs() < 1e-6);
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

    assert!((result.value() - 1.0).abs() < 1e-6);
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

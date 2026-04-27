//! # 分割评估指标
//!
//! 用于语义分割等像素级二分类任务的轻量指标。

use super::ClassificationMetric;
use crate::tensor::Tensor;

/// 计算二值分割的像素准确率（Pixel Accuracy）。
///
/// `predictions` 与 `actuals` 必须形状一致。两者都会按 `threshold`
/// 二值化，常见用法是传入 sigmoid 概率图与 0/1 mask，阈值为 0.5。
pub fn pixel_accuracy(
    predictions: &Tensor,
    actuals: &Tensor,
    threshold: f32,
) -> ClassificationMetric {
    assert_same_shape("pixel_accuracy", predictions, actuals);

    let total = predictions.size();
    if total == 0 {
        return ClassificationMetric::new(0.0, 0);
    }

    let correct = predictions
        .to_vec()
        .into_iter()
        .zip(actuals.to_vec())
        .filter(|(pred, actual)| (*pred >= threshold) == (*actual >= threshold))
        .count();

    ClassificationMetric::new(correct as f32 / total as f32, total)
}

/// 计算二值分割的 IoU（Intersection over Union）。
///
/// 当预测和真实 mask 都没有正像素时，按“空 mask 完全匹配”返回 1.0。
pub fn binary_iou(predictions: &Tensor, actuals: &Tensor, threshold: f32) -> ClassificationMetric {
    assert_same_shape("binary_iou", predictions, actuals);

    let total = predictions.size();
    if total == 0 {
        return ClassificationMetric::new(0.0, 0);
    }

    let mut intersection = 0usize;
    let mut union = 0usize;

    for (pred, actual) in predictions.to_vec().into_iter().zip(actuals.to_vec()) {
        let pred_positive = pred >= threshold;
        let actual_positive = actual >= threshold;

        if pred_positive && actual_positive {
            intersection += 1;
        }
        if pred_positive || actual_positive {
            union += 1;
        }
    }

    let value = if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    };

    ClassificationMetric::new(value, total)
}

fn assert_same_shape(metric_name: &str, predictions: &Tensor, actuals: &Tensor) {
    assert!(
        predictions.shape() == actuals.shape(),
        "{metric_name}: 形状不匹配，predictions={:?}, actuals={:?}",
        predictions.shape(),
        actuals.shape()
    );
}

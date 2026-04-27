//! # 目标检测评估指标
//!
//! 用于轻量目标检测示例的 bbox 指标。当前只覆盖固定形状 `[N, 4]`
//! 的单框回归，坐标格式为归一化 `cx, cy, w, h`。

use super::ClassificationMetric;
use crate::tensor::Tensor;

/// 计算归一化 `cx, cy, w, h` bbox 的平均 IoU。
///
/// `predictions` 与 `actuals` 必须同为 `[N, 4]`，每一行表示：
/// `[center_x, center_y, width, height]`。坐标会裁剪到 `[0, 1]`，宽高小于
/// 0 时按 0 处理。
pub fn mean_box_iou_cxcywh(predictions: &Tensor, actuals: &Tensor) -> ClassificationMetric {
    assert_bbox_shape("mean_box_iou_cxcywh", predictions, actuals);

    let n = predictions.shape()[0];
    if n == 0 {
        return ClassificationMetric::new(0.0, 0);
    }

    let mut iou_sum = 0.0f32;
    for i in 0..n {
        let pred = [
            predictions[[i, 0]],
            predictions[[i, 1]],
            predictions[[i, 2]],
            predictions[[i, 3]],
        ];
        let actual = [
            actuals[[i, 0]],
            actuals[[i, 1]],
            actuals[[i, 2]],
            actuals[[i, 3]],
        ];
        iou_sum += box_iou_cxcywh(pred, actual);
    }

    ClassificationMetric::new(iou_sum / n as f32, n)
}

fn box_iou_cxcywh(a: [f32; 4], b: [f32; 4]) -> f32 {
    let [a_x1, a_y1, a_x2, a_y2] = cxcywh_to_xyxy(a);
    let [b_x1, b_y1, b_x2, b_y2] = cxcywh_to_xyxy(b);

    let inter_w = (a_x2.min(b_x2) - a_x1.max(b_x1)).max(0.0);
    let inter_h = (a_y2.min(b_y2) - a_y1.max(b_y1)).max(0.0);
    let intersection = inter_w * inter_h;

    let a_area = (a_x2 - a_x1).max(0.0) * (a_y2 - a_y1).max(0.0);
    let b_area = (b_x2 - b_x1).max(0.0) * (b_y2 - b_y1).max(0.0);
    let union = a_area + b_area - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn cxcywh_to_xyxy([cx, cy, w, h]: [f32; 4]) -> [f32; 4] {
    let half_w = w.max(0.0) * 0.5;
    let half_h = h.max(0.0) * 0.5;
    [
        (cx - half_w).clamp(0.0, 1.0),
        (cy - half_h).clamp(0.0, 1.0),
        (cx + half_w).clamp(0.0, 1.0),
        (cy + half_h).clamp(0.0, 1.0),
    ]
}

fn assert_bbox_shape(metric_name: &str, predictions: &Tensor, actuals: &Tensor) {
    assert!(
        predictions.shape() == actuals.shape(),
        "{metric_name}: 形状不匹配，predictions={:?}, actuals={:?}",
        predictions.shape(),
        actuals.shape()
    );
    assert!(
        predictions.shape().len() == 2 && predictions.shape()[1] == 4,
        "{metric_name}: bbox Tensor 必须是 [N, 4]，实际 shape={:?}",
        predictions.shape()
    );
}

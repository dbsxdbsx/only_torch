//! # 目标检测评估指标
//!
//! 支持单框回归指标与多目标检测 mAP / precision / recall。

use super::{ClassificationMetric, Metric};
use crate::tensor::Tensor;
use crate::vision::detection::{BBox, BoxFormat, Detection, GroundTruthBox};
use std::cmp::Ordering;

/// VOC 风格 `mAP@0.5` 阈值。
pub const VOC_IOU_THRESHOLDS: &[f32] = &[0.5];

/// COCO 风格 `mAP@0.5:0.95` 阈值。
pub const COCO_IOU_THRESHOLDS: &[f32] =
    &[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

/// 多阈值 mAP 指标结果。
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionMapMetric {
    mean_ap: f32,
    num_ground_truths: usize,
    per_threshold_ap: Vec<f32>,
}

impl DetectionMapMetric {
    pub(crate) fn new(mean_ap: f32, num_ground_truths: usize, per_threshold_ap: Vec<f32>) -> Self {
        Self {
            mean_ap,
            num_ground_truths,
            per_threshold_ap,
        }
    }

    /// 获取 mAP 主值。
    pub fn value(&self) -> f32 {
        self.mean_ap
    }

    /// 获取 GT 框数量。
    pub const fn n_samples(&self) -> usize {
        self.num_ground_truths
    }

    /// 获取百分比形式。
    pub fn percent(&self) -> f32 {
        self.mean_ap * 100.0
    }

    /// 获取加权值。
    pub fn weighted(&self) -> f32 {
        self.mean_ap * self.num_ground_truths as f32
    }

    /// 每个 IoU 阈值上的 AP。
    pub fn per_threshold_ap(&self) -> &[f32] {
        &self.per_threshold_ap
    }
}

impl Metric for DetectionMapMetric {
    fn value(&self) -> f32 {
        self.value()
    }

    fn n_samples(&self) -> usize {
        self.n_samples()
    }

    fn percent(&self) -> f32 {
        self.percent()
    }

    fn weighted(&self) -> f32 {
        self.weighted()
    }
}

/// 检测 precision / recall 指标结果。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionPrMetric {
    precision: f32,
    recall: f32,
    true_positives: usize,
    false_positives: usize,
    false_negatives: usize,
}

impl DetectionPrMetric {
    pub(crate) const fn new(
        precision: f32,
        recall: f32,
        true_positives: usize,
        false_positives: usize,
        false_negatives: usize,
    ) -> Self {
        Self {
            precision,
            recall,
            true_positives,
            false_positives,
            false_negatives,
        }
    }

    pub const fn precision(&self) -> f32 {
        self.precision
    }

    pub const fn recall(&self) -> f32 {
        self.recall
    }

    pub const fn true_positives(&self) -> usize {
        self.true_positives
    }

    pub const fn false_positives(&self) -> usize {
        self.false_positives
    }

    pub const fn false_negatives(&self) -> usize {
        self.false_negatives
    }
}

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

/// 计算多目标检测 mean Average Precision。
///
/// `predictions` / `ground_truths` 外层按图片排列，内层是该图上的预测 / 真值框。
/// AP 使用 all-points interpolation，并只对存在 GT 的类别参与类别平均。
pub fn mean_average_precision(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruthBox>],
    num_classes: usize,
    iou_thresholds: &[f32],
) -> DetectionMapMetric {
    assert_detection_inputs(predictions, ground_truths, num_classes, iou_thresholds);

    let num_ground_truths = ground_truths.iter().map(Vec::len).sum();
    if num_ground_truths == 0 || iou_thresholds.is_empty() {
        return DetectionMapMetric::new(0.0, num_ground_truths, Vec::new());
    }

    let mut per_threshold_ap = Vec::with_capacity(iou_thresholds.len());
    for &threshold in iou_thresholds {
        let mut class_aps = Vec::new();
        for class_id in 0..num_classes {
            if let Some(ap) =
                average_precision_for_class(predictions, ground_truths, class_id, threshold)
            {
                class_aps.push(ap);
            }
        }
        per_threshold_ap.push(mean_or_zero(&class_aps));
    }

    let mean_ap = mean_or_zero(&per_threshold_ap);
    DetectionMapMetric::new(mean_ap, num_ground_truths, per_threshold_ap)
}

/// 在单个 IoU 阈值上计算 micro precision / recall。
///
/// 与 mAP 不同，空类别上的预测会计入 false positive，便于暴露误检问题。
pub fn precision_recall_at_iou(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruthBox>],
    num_classes: usize,
    iou_threshold: f32,
) -> DetectionPrMetric {
    assert_detection_inputs(predictions, ground_truths, num_classes, &[iou_threshold]);

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut total_gt = 0usize;

    for class_id in 0..num_classes {
        let outcome = match_class_predictions(predictions, ground_truths, class_id, iou_threshold);
        tp += outcome.true_positives;
        fp += outcome.false_positives;
        total_gt += outcome.num_ground_truths;
    }

    let fn_count = total_gt.saturating_sub(tp);
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f32 / (tp + fp) as f32
    };
    let recall = if total_gt == 0 {
        0.0
    } else {
        tp as f32 / total_gt as f32
    };
    DetectionPrMetric::new(precision, recall, tp, fp, fn_count)
}

fn box_iou_cxcywh(a: [f32; 4], b: [f32; 4]) -> f32 {
    let a = BBox::from_array(a, BoxFormat::CxCyWh).clip(0.0, 1.0);
    let b = BBox::from_array(b, BoxFormat::CxCyWh).clip(0.0, 1.0);
    a.iou(b)
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

fn assert_detection_inputs(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruthBox>],
    num_classes: usize,
    iou_thresholds: &[f32],
) {
    assert_eq!(
        predictions.len(),
        ground_truths.len(),
        "detection metric: predictions 和 ground_truths 图片数量不一致"
    );
    assert!(num_classes > 0, "detection metric: num_classes 必须大于 0");
    for &threshold in iou_thresholds {
        assert!(
            (0.0..=1.0).contains(&threshold),
            "detection metric: IoU threshold 必须在 [0, 1]，得到 {threshold}"
        );
    }
}

fn average_precision_for_class(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruthBox>],
    class_id: usize,
    iou_threshold: f32,
) -> Option<f32> {
    let outcome = match_class_predictions(predictions, ground_truths, class_id, iou_threshold);
    if outcome.num_ground_truths == 0 {
        return None;
    }
    if outcome.true_positives == 0 && outcome.false_positives == 0 {
        return Some(0.0);
    }

    let mut precisions = Vec::with_capacity(outcome.matches.len());
    let mut recalls = Vec::with_capacity(outcome.matches.len());
    let mut tp = 0usize;
    let mut fp = 0usize;

    for matched in outcome.matches {
        if matched {
            tp += 1;
        } else {
            fp += 1;
        }
        precisions.push(tp as f32 / (tp + fp) as f32);
        recalls.push(tp as f32 / outcome.num_ground_truths as f32);
    }

    Some(interpolated_ap(&recalls, &precisions))
}

fn interpolated_ap(recalls: &[f32], precisions: &[f32]) -> f32 {
    debug_assert_eq!(recalls.len(), precisions.len());
    if recalls.is_empty() {
        return 0.0;
    }

    let mut mrec = Vec::with_capacity(recalls.len() + 2);
    let mut mpre = Vec::with_capacity(precisions.len() + 2);
    mrec.push(0.0);
    mrec.extend_from_slice(recalls);
    mrec.push(1.0);
    mpre.push(0.0);
    mpre.extend_from_slice(precisions);
    mpre.push(0.0);

    for i in (0..mpre.len() - 1).rev() {
        mpre[i] = mpre[i].max(mpre[i + 1]);
    }

    let mut ap = 0.0;
    for i in 1..mrec.len() {
        let delta = mrec[i] - mrec[i - 1];
        if delta > 0.0 {
            ap += delta * mpre[i];
        }
    }
    ap
}

#[derive(Debug)]
struct ClassMatchOutcome {
    matches: Vec<bool>,
    true_positives: usize,
    false_positives: usize,
    num_ground_truths: usize,
}

fn match_class_predictions(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruthBox>],
    class_id: usize,
    iou_threshold: f32,
) -> ClassMatchOutcome {
    let mut class_predictions = Vec::new();
    let mut class_ground_truths: Vec<Vec<&GroundTruthBox>> = vec![Vec::new(); ground_truths.len()];

    for (image_idx, image_predictions) in predictions.iter().enumerate() {
        for prediction in image_predictions {
            if prediction.class_id == class_id {
                class_predictions.push((image_idx, prediction));
            }
        }
    }
    for (image_idx, image_ground_truths) in ground_truths.iter().enumerate() {
        for gt in image_ground_truths {
            if gt.class_id == class_id {
                class_ground_truths[image_idx].push(gt);
            }
        }
    }

    class_predictions
        .sort_by(|(_, a), (_, b)| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

    let num_ground_truths = class_ground_truths.iter().map(Vec::len).sum();
    let mut matched: Vec<Vec<bool>> = class_ground_truths
        .iter()
        .map(|image_gts| vec![false; image_gts.len()])
        .collect();
    let mut matches = Vec::with_capacity(class_predictions.len());
    let mut true_positives = 0usize;
    let mut false_positives = 0usize;

    for (image_idx, prediction) in class_predictions {
        let mut best_gt_idx = None;
        let mut best_iou = iou_threshold;
        for (gt_idx, gt) in class_ground_truths[image_idx].iter().enumerate() {
            if matched[image_idx][gt_idx] {
                continue;
            }
            let iou = prediction.bbox.iou(gt.bbox);
            if iou >= best_iou {
                best_iou = iou;
                best_gt_idx = Some(gt_idx);
            }
        }

        if let Some(gt_idx) = best_gt_idx {
            matched[image_idx][gt_idx] = true;
            matches.push(true);
            true_positives += 1;
        } else {
            matches.push(false);
            false_positives += 1;
        }
    }

    ClassMatchOutcome {
        matches,
        true_positives,
        false_positives,
        num_ground_truths,
    }
}

fn mean_or_zero(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

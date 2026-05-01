//! # 目标检测评估指标
//!
//! 支持单框回归指标与多目标检测 mAP / precision / recall。

use super::{ClassificationMetric, Metric};
use crate::vision::detection::{BBox, Detection, GroundTruthBox};
use std::cmp::Ordering;

/// VOC 风格 `mAP@0.5` 阈值。
pub const VOC_IOU_THRESHOLDS: &[f32] = &[0.5];

/// COCO 风格 `mAP@0.5:0.95` 阈值。
pub const COCO_IOU_THRESHOLDS: &[f32] =
    &[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95];

/// 多目标检测评估选项。
///
/// 当前实现使用 VOC 风格 all-points interpolation，并支持 COCO-style 的多 IoU
/// 阈值集合；它不完整复刻 pycocotools 的 crowd / area range / 101-point 细节。
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionMetricOptions {
    pub iou_thresholds: Vec<f32>,
    pub score_threshold: f32,
    pub max_detections: Option<usize>,
}

impl DetectionMetricOptions {
    pub fn new(iou_thresholds: &[f32]) -> Self {
        Self {
            iou_thresholds: iou_thresholds.to_vec(),
            score_threshold: f32::NEG_INFINITY,
            max_detections: None,
        }
    }

    pub fn voc() -> Self {
        Self::new(VOC_IOU_THRESHOLDS)
    }

    pub fn coco_style() -> Self {
        Self::new(COCO_IOU_THRESHOLDS)
    }

    pub fn with_score_threshold(mut self, score_threshold: f32) -> Self {
        self.score_threshold = score_threshold;
        self
    }

    pub fn with_max_detections(mut self, max_detections: usize) -> Self {
        self.max_detections = Some(max_detections);
        self
    }
}

/// 多阈值 mAP 指标结果。
#[derive(Debug, Clone, PartialEq)]
pub struct DetectionMapMetric {
    mean_ap: f32,
    num_ground_truths: usize,
    iou_thresholds: Vec<f32>,
    per_threshold_ap: Vec<f32>,
    per_class_ap: Vec<Option<f32>>,
}

impl DetectionMapMetric {
    pub(crate) fn new(
        mean_ap: f32,
        num_ground_truths: usize,
        iou_thresholds: Vec<f32>,
        per_threshold_ap: Vec<f32>,
        per_class_ap: Vec<Option<f32>>,
    ) -> Self {
        Self {
            mean_ap,
            num_ground_truths,
            iou_thresholds,
            per_threshold_ap,
            per_class_ap,
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

    /// 每个阈值对应的 IoU threshold。
    pub fn iou_thresholds(&self) -> &[f32] {
        &self.iou_thresholds
    }

    /// 每个类别的 AP；没有 GT 的类别为 `None`。
    pub fn per_class_ap(&self) -> &[Option<f32>] {
        &self.per_class_ap
    }

    /// 指定 IoU 阈值上的 mAP。
    pub fn map_at(&self, iou_threshold: f32) -> Option<f32> {
        self.iou_thresholds
            .iter()
            .position(|&threshold| (threshold - iou_threshold).abs() < 1e-6)
            .map(|idx| self.per_threshold_ap[idx])
    }

    /// VOC 常用 `mAP@0.5`。
    pub fn map_50(&self) -> Option<f32> {
        self.map_at(0.5)
    }

    /// 当前阈值集合上的平均值，COCO-style 调用时即 `mAP@0.5:0.95`。
    pub fn map_50_95(&self) -> f32 {
        self.mean_ap
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

    pub fn f1_score(&self) -> f32 {
        let denom = self.precision + self.recall;
        if denom <= 0.0 {
            0.0
        } else {
            2.0 * self.precision * self.recall / denom
        }
    }
}

/// 计算配对 `BBox` 的平均 IoU。
///
/// `predictions[i]` 与 `actuals[i]` 必须形成 1-1 配对，常用于单框回归 / 单目标
/// 检测的 toy benchmark；多目标检测请使用 [`mean_average_precision`]。
///
/// 调用方负责坐标格式转换与裁剪——例如归一化 `cxcywh` 输入可在转换 `BBox` 时
/// 调用 `BBox::from_array(arr, BoxFormat::CxCyWh).clip(0.0, 1.0)`。
pub fn mean_box_iou(predictions: &[BBox], actuals: &[BBox]) -> ClassificationMetric {
    assert_eq!(
        predictions.len(),
        actuals.len(),
        "mean_box_iou: predictions 与 actuals 长度必须一致，分别为 {} / {}",
        predictions.len(),
        actuals.len(),
    );

    let n = predictions.len();
    if n == 0 {
        return ClassificationMetric::new(0.0, 0);
    }

    let iou_sum: f32 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(pred, actual)| pred.iou(*actual))
        .sum();

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
    mean_average_precision_with_options(
        predictions,
        ground_truths,
        num_classes,
        &DetectionMetricOptions::new(iou_thresholds),
    )
}

/// 使用显式评估选项计算多目标检测 mean Average Precision。
pub fn mean_average_precision_with_options(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruthBox>],
    num_classes: usize,
    options: &DetectionMetricOptions,
) -> DetectionMapMetric {
    assert_detection_inputs(
        predictions,
        ground_truths,
        num_classes,
        &options.iou_thresholds,
    );

    let num_ground_truths = ground_truths.iter().map(Vec::len).sum();
    if options.iou_thresholds.is_empty() {
        return DetectionMapMetric::new(
            0.0,
            num_ground_truths,
            Vec::new(),
            Vec::new(),
            vec![None; num_classes],
        );
    }
    if num_ground_truths == 0 {
        return DetectionMapMetric::new(
            0.0,
            0,
            options.iou_thresholds.clone(),
            vec![0.0; options.iou_thresholds.len()],
            vec![None; num_classes],
        );
    }

    let predictions = prepare_predictions(predictions, options);
    let mut per_threshold_ap = Vec::with_capacity(options.iou_thresholds.len());
    let mut class_ap_values: Vec<Vec<f32>> = vec![Vec::new(); num_classes];
    for &threshold in &options.iou_thresholds {
        let mut class_aps = Vec::new();
        for (class_id, class_values) in class_ap_values.iter_mut().enumerate() {
            if let Some(ap) =
                average_precision_for_class(&predictions, ground_truths, class_id, threshold)
            {
                class_values.push(ap);
                class_aps.push(ap);
            }
        }
        per_threshold_ap.push(mean_or_zero(&class_aps));
    }

    let mean_ap = mean_or_zero(&per_threshold_ap);
    let per_class_ap = class_ap_values
        .iter()
        .map(|values| (!values.is_empty()).then(|| mean_or_zero(values)))
        .collect();
    DetectionMapMetric::new(
        mean_ap,
        num_ground_truths,
        options.iou_thresholds.clone(),
        per_threshold_ap,
        per_class_ap,
    )
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
    precision_recall_at_iou_with_options(
        predictions,
        ground_truths,
        num_classes,
        iou_threshold,
        &DetectionMetricOptions::new(&[iou_threshold]),
    )
}

/// 在单个 IoU 阈值上按显式评估选项计算 micro precision / recall。
pub fn precision_recall_at_iou_with_options(
    predictions: &[Vec<Detection>],
    ground_truths: &[Vec<GroundTruthBox>],
    num_classes: usize,
    iou_threshold: f32,
    options: &DetectionMetricOptions,
) -> DetectionPrMetric {
    assert_detection_inputs(predictions, ground_truths, num_classes, &[iou_threshold]);
    let predictions = prepare_predictions(predictions, options);

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut total_gt = 0usize;

    for class_id in 0..num_classes {
        let outcome = match_class_predictions(&predictions, ground_truths, class_id, iou_threshold);
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

fn prepare_predictions(
    predictions: &[Vec<Detection>],
    options: &DetectionMetricOptions,
) -> Vec<Vec<Detection>> {
    predictions
        .iter()
        .map(|image_predictions| {
            let mut filtered = image_predictions
                .iter()
                .filter(|prediction| {
                    prediction.score.is_finite() && prediction.score >= options.score_threshold
                })
                .cloned()
                .collect::<Vec<_>>();
            filtered.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.class_id.cmp(&b.class_id))
            });
            if let Some(max_detections) = options.max_detections {
                filtered.truncate(max_detections);
            }
            filtered
        })
        .collect()
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
    for (image_idx, image_predictions) in predictions.iter().enumerate() {
        for prediction in image_predictions {
            assert!(
                prediction.class_id < num_classes,
                "detection metric: prediction class_id {} 越界，num_classes={}，image_idx={}",
                prediction.class_id,
                num_classes,
                image_idx
            );
        }
    }
    for (image_idx, image_ground_truths) in ground_truths.iter().enumerate() {
        for gt in image_ground_truths {
            assert!(
                gt.class_id < num_classes,
                "detection metric: ground truth class_id {} 越界，num_classes={}，image_idx={}",
                gt.class_id,
                num_classes,
                image_idx
            );
        }
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

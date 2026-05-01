//! # 分割评估指标
//!
//! 用于语义分割等像素级任务的轻量指标。

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

/// 计算二值分割的 Dice score。
///
/// 当预测和真实 mask 都没有正像素时，按“空 mask 完全匹配”返回 1.0。
pub fn dice_score(predictions: &Tensor, actuals: &Tensor, threshold: f32) -> ClassificationMetric {
    assert_same_shape("dice_score", predictions, actuals);

    let total = predictions.size();
    if total == 0 {
        return ClassificationMetric::new(0.0, 0);
    }

    let mut intersection = 0usize;
    let mut pred_positive_count = 0usize;
    let mut actual_positive_count = 0usize;

    for (pred, actual) in predictions.to_vec().into_iter().zip(actuals.to_vec()) {
        let pred_positive = pred >= threshold;
        let actual_positive = actual >= threshold;

        if pred_positive {
            pred_positive_count += 1;
        }
        if actual_positive {
            actual_positive_count += 1;
        }
        if pred_positive && actual_positive {
            intersection += 1;
        }
    }

    let denom = pred_positive_count + actual_positive_count;
    let value = if denom == 0 {
        1.0
    } else {
        2.0 * intersection as f32 / denom as f32
    };

    ClassificationMetric::new(value, total)
}

/// 计算多类别语义分割的像素准确率。
///
/// `predictions` 与 `actuals` 均使用 `[N, C, H, W]` 形状。预测侧按通道
/// argmax 解码；标签侧可传 one-hot 或 soft label，同样按通道 argmax 解码。
pub fn semantic_pixel_accuracy(predictions: &Tensor, actuals: &Tensor) -> ClassificationMetric {
    assert_semantic_shapes("semantic_pixel_accuracy", predictions, actuals);

    let [n, _classes, h, w] = semantic_shape(predictions);
    let total = n * h * w;
    if total == 0 {
        return ClassificationMetric::new(0.0, 0);
    }

    let mut correct = 0usize;
    for sample in 0..n {
        for y in 0..h {
            for x in 0..w {
                if argmax_channel(predictions, sample, y, x)
                    == argmax_channel(actuals, sample, y, x)
                {
                    correct += 1;
                }
            }
        }
    }

    ClassificationMetric::new(correct as f32 / total as f32, total)
}

/// 计算多类别语义分割的每类 IoU。
///
/// 返回向量长度等于类别数。若某个类别在预测和标签中都不存在，则该类 IoU 记为 1.0。
pub fn per_class_iou(predictions: &Tensor, actuals: &Tensor) -> Vec<ClassificationMetric> {
    assert_semantic_shapes("per_class_iou", predictions, actuals);

    let [n, classes, h, w] = semantic_shape(predictions);
    if n * h * w == 0 {
        return (0..classes)
            .map(|_| ClassificationMetric::new(0.0, 0))
            .collect();
    }

    let mut intersections = vec![0usize; classes];
    let mut unions = vec![0usize; classes];

    for sample in 0..n {
        for y in 0..h {
            for x in 0..w {
                let pred_class = argmax_channel(predictions, sample, y, x);
                let actual_class = argmax_channel(actuals, sample, y, x);

                for class_idx in 0..classes {
                    let pred_match = pred_class == class_idx;
                    let actual_match = actual_class == class_idx;
                    if pred_match && actual_match {
                        intersections[class_idx] += 1;
                    }
                    if pred_match || actual_match {
                        unions[class_idx] += 1;
                    }
                }
            }
        }
    }

    let total = n * h * w;
    (0..classes)
        .map(|class_idx| {
            let value = if unions[class_idx] == 0 {
                1.0
            } else {
                intersections[class_idx] as f32 / unions[class_idx] as f32
            };
            ClassificationMetric::new(value, total)
        })
        .collect()
}

/// 计算多类别语义分割的 Mean IoU。
pub fn mean_iou(predictions: &Tensor, actuals: &Tensor) -> ClassificationMetric {
    let class_metrics = per_class_iou(predictions, actuals);
    if class_metrics.is_empty() {
        return ClassificationMetric::new(0.0, 0);
    }

    let value = class_metrics
        .iter()
        .map(|metric| metric.value())
        .sum::<f32>()
        / class_metrics.len() as f32;
    let n_samples = class_metrics[0].n_samples();
    ClassificationMetric::new(value, n_samples)
}

/// 实例分割的 slot-wise IoU 平均（所有 slot 都参与平均）。
///
/// 输入形状必须是 `[N, S, H, W]`，`S` 是固定 slot 数（每个 slot 一个 instance mask）。
/// 对每个 `(sample, slot)` 计算二值 IoU 后求均值；`union == 0` 的 slot
/// （pred 和 target 都为空）按"完美匹配空 slot"返回 1.0，参与平均。
///
/// 适合 multi_instance_segmentation 这种"slot 数量固定且每个 slot 都有意义"
/// 的场景。如果允许部分 slot 为空、且只想评估有 GT 的 slot，参考
/// [`mean_valid_slot_iou`] 与 [`empty_slot_accuracy`] 配套使用。
pub fn mean_instance_iou(
    predictions: &Tensor,
    actuals: &Tensor,
    threshold: f32,
) -> ClassificationMetric {
    assert_instance_shape("mean_instance_iou", predictions, actuals);
    let shape = predictions.shape();
    let (n, slots) = (shape[0], shape[1]);
    let total_slots = n * slots;
    if total_slots == 0 {
        return ClassificationMetric::new(0.0, 0);
    }

    let mut total_iou = 0.0f32;
    for sample in 0..n {
        for slot in 0..slots {
            total_iou += slot_iou(predictions, actuals, sample, slot, threshold);
        }
    }
    ClassificationMetric::new(total_iou / total_slots as f32, total_slots)
}

/// 实例分割的 slot-wise IoU，但**只在 target 非空**的 slot 上做平均。
///
/// 输入形状必须是 `[N, S, H, W]`。"target 是否非空"的判断阈值固定为 `0.5`，
/// 与一般 0/1 mask 的语义保持一致；预测侧的二值化阈值由 `threshold` 控制。
///
/// 适合"允许 slot 为空"的固定 slot 实例分割任务，单独评估有真值的 slot
/// 上分割质量；空 slot 性能用 [`empty_slot_accuracy`] 评估，两者互补。
pub fn mean_valid_slot_iou(
    predictions: &Tensor,
    actuals: &Tensor,
    threshold: f32,
) -> ClassificationMetric {
    assert_instance_shape("mean_valid_slot_iou", predictions, actuals);
    let shape = predictions.shape();
    let (n, slots) = (shape[0], shape[1]);

    let mut total_iou = 0.0f32;
    let mut valid_slots = 0usize;
    for sample in 0..n {
        for slot in 0..slots {
            if !slot_has_positive(actuals, sample, slot, 0.5) {
                continue;
            }
            total_iou += slot_iou(predictions, actuals, sample, slot, threshold);
            valid_slots += 1;
        }
    }
    if valid_slots == 0 {
        return ClassificationMetric::new(0.0, 0);
    }
    ClassificationMetric::new(total_iou / valid_slots as f32, valid_slots)
}

/// 空 slot 准确率：target 全 0 的 slot 中，prediction 也（按 `threshold`）
/// 全 0 的占比。
///
/// 输入形状必须是 `[N, S, H, W]`。`target 是否非空`的判断阈值固定为 `0.5`。
/// 没有任何空 slot 时返回 `value = 1.0, n_samples = 0`，调用方按需判断。
pub fn empty_slot_accuracy(
    predictions: &Tensor,
    actuals: &Tensor,
    threshold: f32,
) -> ClassificationMetric {
    assert_instance_shape("empty_slot_accuracy", predictions, actuals);
    let shape = predictions.shape();
    let (n, slots) = (shape[0], shape[1]);

    let mut empty_slots = 0usize;
    let mut correct_empty = 0usize;
    for sample in 0..n {
        for slot in 0..slots {
            if slot_has_positive(actuals, sample, slot, 0.5) {
                continue;
            }
            empty_slots += 1;
            if !slot_has_positive(predictions, sample, slot, threshold) {
                correct_empty += 1;
            }
        }
    }
    if empty_slots == 0 {
        return ClassificationMetric::new(1.0, 0);
    }
    ClassificationMetric::new(correct_empty as f32 / empty_slots as f32, empty_slots)
}

fn assert_same_shape(metric_name: &str, predictions: &Tensor, actuals: &Tensor) {
    assert!(
        predictions.shape() == actuals.shape(),
        "{metric_name}: 形状不匹配，predictions={:?}, actuals={:?}",
        predictions.shape(),
        actuals.shape()
    );
}

fn assert_semantic_shapes(metric_name: &str, predictions: &Tensor, actuals: &Tensor) {
    assert_same_shape(metric_name, predictions, actuals);
    assert!(
        predictions.shape().len() == 4,
        "{metric_name}: 期望 shape=[N, C, H, W]，实际 {:?}",
        predictions.shape()
    );
    assert!(
        predictions.shape()[1] > 0,
        "{metric_name}: 类别通道数必须 > 0，实际 {:?}",
        predictions.shape()
    );
}

fn semantic_shape(tensor: &Tensor) -> [usize; 4] {
    let shape = tensor.shape();
    [shape[0], shape[1], shape[2], shape[3]]
}

fn argmax_channel(tensor: &Tensor, sample: usize, y: usize, x: usize) -> usize {
    let classes = tensor.shape()[1];
    let mut best_class = 0usize;
    let mut best_value = tensor[[sample, 0, y, x]];
    for class_idx in 1..classes {
        let value = tensor[[sample, class_idx, y, x]];
        if value > best_value {
            best_value = value;
            best_class = class_idx;
        }
    }
    best_class
}

fn assert_instance_shape(metric_name: &str, predictions: &Tensor, actuals: &Tensor) {
    assert_same_shape(metric_name, predictions, actuals);
    assert!(
        predictions.shape().len() == 4,
        "{metric_name}: 期望 shape=[N, S, H, W]，实际 {:?}",
        predictions.shape()
    );
}

/// 计算单个 `(sample, slot)` 上的二值 IoU。
///
/// `union == 0` 时按"两边都是空 mask 完全匹配"返回 1.0，与 [`binary_iou`]
/// 在全空场景下的行为一致。
fn slot_iou(
    predictions: &Tensor,
    actuals: &Tensor,
    sample: usize,
    slot: usize,
    threshold: f32,
) -> f32 {
    let shape = predictions.shape();
    let (h, w) = (shape[2], shape[3]);
    let mut intersection = 0usize;
    let mut union = 0usize;
    for y in 0..h {
        for x in 0..w {
            let pred = predictions[[sample, slot, y, x]] >= threshold;
            let actual = actuals[[sample, slot, y, x]] >= threshold;
            if pred && actual {
                intersection += 1;
            }
            if pred || actual {
                union += 1;
            }
        }
    }
    if union == 0 {
        1.0
    } else {
        intersection as f32 / union as f32
    }
}

fn slot_has_positive(tensor: &Tensor, sample: usize, slot: usize, threshold: f32) -> bool {
    let shape = tensor.shape();
    let (h, w) = (shape[2], shape[3]);
    for y in 0..h {
        for x in 0..w {
            if tensor[[sample, slot, y, x]] >= threshold {
                return true;
            }
        }
    }
    false
}

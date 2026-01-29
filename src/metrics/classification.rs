//! # 分类评估指标
//!
//! 用于评估分类模型性能的指标函数。
//!
//! ## 统一接口
//!
//! 所有分类指标函数返回 [`ClassificationMetric`](super::ClassificationMetric)，
//! 实现了 [`Metric`](super::Metric) trait，提供统一的访问方式：
//!
//! - `.value()` - 获取指标值（0.0 ~ 1.0）
//! - `.n_samples()` - 获取样本数
//! - `.percent()` - 百分比形式
//! - `.weighted()` - 加权值（批处理累加用）
//!
//! ## 多态输入
//!
//! 所有函数都支持多种输入类型：
//! - `&[usize]` / `Vec<usize>` - 类别索引数组
//! - `&[i32]` / `Vec<i32>` - 整数数组（自动转换）
//! - `&Tensor` - 自动处理：
//!   - `[batch]` 形状 → 直接作为类别索引
//!   - `[batch, num_classes]` 形状 → 自动 argmax
//!
//! ## 多分类支持
//!
//! 本模块的指标函数（如 `precision`、`recall`、`f1_score`）均采用 **Macro-Average** 策略，
//! 天然支持多分类场景，无需额外参数。

use super::ClassificationMetric;
use super::traits::IntoClassLabels;
use std::collections::HashSet;

/// 计算准确率（Accuracy）
///
/// 准确率 = 预测正确的样本数 / 总样本数
///
/// ## 参数
///
/// - `predictions`: 模型预测（支持多种类型，见模块文档）
/// - `actuals`: 真实标签（支持多种类型，见模块文档）
///
/// ## 返回值
///
/// 返回 [`ClassificationMetric`](super::ClassificationMetric)，通过统一接口访问：
/// - `.value()` - 准确率（0.0 ~ 1.0）
/// - `.n_samples()` - 样本数
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::accuracy;
/// use only_torch::tensor::Tensor;
///
/// // 方式 1：直接传 slice
/// let result = accuracy(&[0, 1, 1, 0, 1], &[0, 1, 0, 0, 1]);
/// assert!((result.value() - 0.8).abs() < 1e-6);  // 4/5 = 0.8
/// assert_eq!(result.n_samples(), 5);
///
/// // 方式 2：传 Tensor（自动 argmax）
/// let logits = Tensor::new(&[0.1f32, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2]);
/// let labels = Tensor::new(&[0.0f32, 1.0, 1.0, 0.0, 0.0, 1.0], &[3, 2]);
/// let result = accuracy(&logits, &labels);
/// assert!((result.value() - 1.0).abs() < 1e-6);  // 完美预测
/// ```
///
/// ## 边界情况
///
/// - 空输入 → 返回 value=0.0, n_samples=0
/// - 长度不一致 → 取较短者计算
pub fn accuracy(
    predictions: &(impl IntoClassLabels + ?Sized),
    actuals: &(impl IntoClassLabels + ?Sized),
) -> ClassificationMetric {
    let pred_labels = predictions.to_class_labels();
    let true_labels = actuals.to_class_labels();

    if pred_labels.is_empty() || true_labels.is_empty() {
        return ClassificationMetric::new(0.0, 0);
    }

    let n = pred_labels.len().min(true_labels.len());

    let correct = pred_labels[..n]
        .iter()
        .zip(true_labels[..n].iter())
        .filter(|(pred, actual)| pred == actual)
        .count();

    ClassificationMetric::new(correct as f32 / n as f32, n)
}

/// 计算精确率（Precision）- Macro-Average
///
/// Precision 衡量"模型预测为正的样本中，有多少是真正的正样本"。
///
/// ## 计算方式（Macro-Average）
///
/// 1. 对每个类别 c，计算 `Precision_c = TP_c / (TP_c + FP_c)`
/// 2. 取所有类别的简单平均：`Macro-Precision = Σ Precision_c / 类别数`
///
/// ## 参数
///
/// - `predictions`: 模型预测（支持多种类型，见模块文档）
/// - `actuals`: 真实标签（支持多种类型，见模块文档）
///
/// ## 返回值
///
/// 返回 [`ClassificationMetric`](super::ClassificationMetric)，通过统一接口访问：
/// - `.value()` - Macro-Precision（0.0 ~ 1.0）
/// - `.n_samples()` - 样本数
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::{precision, Metric};
///
/// // 二分类
/// let result = precision(&[1, 1, 0, 1, 0], &[1, 0, 0, 1, 1]);
/// // 类0: TP=1, FP=1 → Precision_0 = 0.5
/// // 类1: TP=2, FP=1 → Precision_1 = 0.667
/// // Macro = (0.5 + 0.667) / 2 ≈ 0.583
/// assert!((result.value() - 0.583).abs() < 0.01);
/// ```
///
/// ## 边界情况
///
/// - 空输入 → 返回 value=0.0, n_samples=0
/// - 某个类从未被预测过（TP+FP=0）→ 该类不参与平均
pub fn precision(
    predictions: &(impl IntoClassLabels + ?Sized),
    actuals: &(impl IntoClassLabels + ?Sized),
) -> ClassificationMetric {
    let pred_labels = predictions.to_class_labels();
    let true_labels = actuals.to_class_labels();

    if pred_labels.is_empty() || true_labels.is_empty() {
        return ClassificationMetric::new(0.0, 0);
    }

    let n = pred_labels.len().min(true_labels.len());
    let pred_labels = &pred_labels[..n];
    let true_labels = &true_labels[..n];

    // 收集所有出现的类别
    let all_classes: HashSet<usize> = pred_labels
        .iter()
        .chain(true_labels.iter())
        .cloned()
        .collect();

    let mut precision_sum = 0.0;
    let mut valid_classes = 0;

    for class in &all_classes {
        // 计算该类的 TP 和 FP
        let mut tp = 0;
        let mut fp = 0;

        for (pred, actual) in pred_labels.iter().zip(true_labels.iter()) {
            if pred == class {
                if actual == class {
                    tp += 1; // True Positive
                } else {
                    fp += 1; // False Positive
                }
            }
        }

        // 只有当该类被预测过时才计入平均
        if tp + fp > 0 {
            precision_sum += tp as f32 / (tp + fp) as f32;
            valid_classes += 1;
        }
    }

    let value = if valid_classes == 0 {
        0.0
    } else {
        precision_sum / valid_classes as f32
    };

    ClassificationMetric::new(value, n)
}

/// 计算召回率（Recall）- Macro-Average
///
/// Recall 衡量"真实为正的样本中，有多少被模型正确预测"。
///
/// ## 计算方式（Macro-Average）
///
/// 1. 对每个类别 c，计算 `Recall_c = TP_c / (TP_c + FN_c)`
/// 2. 取所有类别的简单平均：`Macro-Recall = Σ Recall_c / 类别数`
///
/// ## 参数
///
/// - `predictions`: 模型预测（支持多种类型，见模块文档）
/// - `actuals`: 真实标签（支持多种类型，见模块文档）
///
/// ## 返回值
///
/// 返回 [`ClassificationMetric`](super::ClassificationMetric)，通过统一接口访问：
/// - `.value()` - Macro-Recall（0.0 ~ 1.0）
/// - `.n_samples()` - 样本数
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::{recall, Metric};
///
/// // 二分类
/// let result = recall(&[1, 1, 0, 1, 0], &[1, 0, 0, 1, 1]);
/// // 类0: TP=1, FN=1 → Recall_0 = 0.5
/// // 类1: TP=2, FN=1 → Recall_1 = 2/3 ≈ 0.667
/// // Macro = (0.5 + 0.667) / 2 ≈ 0.583
/// assert!((result.value() - 0.583).abs() < 0.01);
/// ```
///
/// ## 边界情况
///
/// - 空输入 → 返回 value=0.0, n_samples=0
/// - 某个类从未出现在真实标签中（TP+FN=0）→ 该类不参与平均
pub fn recall(
    predictions: &(impl IntoClassLabels + ?Sized),
    actuals: &(impl IntoClassLabels + ?Sized),
) -> ClassificationMetric {
    let pred_labels = predictions.to_class_labels();
    let true_labels = actuals.to_class_labels();

    if pred_labels.is_empty() || true_labels.is_empty() {
        return ClassificationMetric::new(0.0, 0);
    }

    let n = pred_labels.len().min(true_labels.len());
    let pred_labels = &pred_labels[..n];
    let true_labels = &true_labels[..n];

    // 收集所有出现的类别
    let all_classes: HashSet<usize> = pred_labels
        .iter()
        .chain(true_labels.iter())
        .cloned()
        .collect();

    let mut recall_sum = 0.0;
    let mut valid_classes = 0;

    for class in &all_classes {
        // 计算该类的 TP 和 FN
        let mut tp = 0;
        let mut fn_count = 0;

        for (pred, actual) in pred_labels.iter().zip(true_labels.iter()) {
            if actual == class {
                if pred == class {
                    tp += 1; // True Positive
                } else {
                    fn_count += 1; // False Negative
                }
            }
        }

        // 只有当该类在真实标签中出现过时才计入平均
        if tp + fn_count > 0 {
            recall_sum += tp as f32 / (tp + fn_count) as f32;
            valid_classes += 1;
        }
    }

    let value = if valid_classes == 0 {
        0.0
    } else {
        recall_sum / valid_classes as f32
    };

    ClassificationMetric::new(value, n)
}

/// 计算 F1 分数（F1 Score）- Macro-Average
///
/// F1 是 Precision 和 Recall 的调和平均，兼顾两者的平衡。
///
/// ## 计算方式（Macro-Average）
///
/// 1. 对每个类别 c，计算 `F1_c = 2 × Precision_c × Recall_c / (Precision_c + Recall_c)`
/// 2. 取所有类别的简单平均：`Macro-F1 = Σ F1_c / 类别数`
///
/// ## 参数
///
/// - `predictions`: 模型预测（支持多种类型，见模块文档）
/// - `actuals`: 真实标签（支持多种类型，见模块文档）
///
/// ## 返回值
///
/// 返回 [`ClassificationMetric`](super::ClassificationMetric)，通过统一接口访问：
/// - `.value()` - Macro-F1（0.0 ~ 1.0）
/// - `.n_samples()` - 样本数
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::{f1_score, Metric};
///
/// // 二分类
/// let result = f1_score(&[1, 1, 0, 1, 0], &[1, 0, 0, 1, 1]);
/// // 类0: Precision=0.5, Recall=0.5 → F1_0 = 0.5
/// // 类1: Precision=0.667, Recall=0.667 → F1_1 ≈ 0.667
/// // Macro = (0.5 + 0.667) / 2 ≈ 0.583
/// assert!((result.value() - 0.583).abs() < 0.01);
/// ```
///
/// ## 边界情况
///
/// - 空输入 → 返回 value=0.0, n_samples=0
/// - 某个类的 Precision + Recall = 0 → 该类的 F1 视为 0
pub fn f1_score(
    predictions: &(impl IntoClassLabels + ?Sized),
    actuals: &(impl IntoClassLabels + ?Sized),
) -> ClassificationMetric {
    let pred_labels = predictions.to_class_labels();
    let true_labels = actuals.to_class_labels();

    if pred_labels.is_empty() || true_labels.is_empty() {
        return ClassificationMetric::new(0.0, 0);
    }

    let n = pred_labels.len().min(true_labels.len());
    let pred_labels = &pred_labels[..n];
    let true_labels = &true_labels[..n];

    // 收集所有出现的类别
    let all_classes: HashSet<usize> = pred_labels
        .iter()
        .chain(true_labels.iter())
        .cloned()
        .collect();

    let mut f1_sum = 0.0;
    let mut valid_classes = 0;

    for class in &all_classes {
        // 计算该类的 TP, FP, FN
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;

        for (pred, actual) in pred_labels.iter().zip(true_labels.iter()) {
            if pred == class && actual == class {
                tp += 1; // True Positive
            } else if pred == class && actual != class {
                fp += 1; // False Positive
            } else if pred != class && actual == class {
                fn_count += 1; // False Negative
            }
        }

        // 计算该类的 Precision 和 Recall
        let precision = if tp + fp > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            0.0
        };

        let recall = if tp + fn_count > 0 {
            tp as f32 / (tp + fn_count) as f32
        } else {
            0.0
        };

        // 计算该类的 F1
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        // 只有当该类在预测或真实标签中出现过时才计入平均
        if tp + fp + fn_count > 0 {
            f1_sum += f1;
            valid_classes += 1;
        }
    }

    let value = if valid_classes == 0 {
        0.0
    } else {
        f1_sum / valid_classes as f32
    };

    ClassificationMetric::new(value, n)
}

/// 计算混淆矩阵（Confusion Matrix）
///
/// 混淆矩阵是分类任务中最常用的评估工具，直观展示模型的预测情况。
///
/// ## 返回值
///
/// 返回 `num_classes × num_classes` 的二维矩阵，其中：
/// - `matrix[true_class][pred_class]` = 该组合出现的次数
/// - 对角线元素为正确预测数
/// - 非对角线元素为错误预测数
///
/// ## 参数
///
/// - `predictions`: 模型预测（支持多种类型，见模块文档）
/// - `actuals`: 真实标签（支持多种类型，见模块文档）
///
/// ## 示例
///
/// ```rust
/// use only_torch::metrics::confusion_matrix;
///
/// // 二分类：predictions=[0,1,1,0,1], actuals=[0,1,0,0,1]
/// let cm = confusion_matrix(&[0, 1, 1, 0, 1], &[0, 1, 0, 0, 1]);
/// // 返回 matrix[actual][pred]:
/// // [[2, 1],   // actual=0: 2个正确(pred=0), 1个误判为1
/// //  [0, 2]]   // actual=1: 0个误判为0, 2个正确(pred=1)
/// assert_eq!(cm[0][0], 2);  // TN (actual=0, pred=0)
/// assert_eq!(cm[0][1], 1);  // FP (actual=0, pred=1)
/// assert_eq!(cm[1][0], 0);  // FN (actual=1, pred=0)
/// assert_eq!(cm[1][1], 2);  // TP (actual=1, pred=1)
/// ```
///
/// ## 多分类示例
///
/// ```rust
/// use only_torch::metrics::confusion_matrix;
/// use only_torch::tensor::Tensor;
///
/// // 三分类，直接传 Tensor（自动 argmax）
/// let logits = Tensor::new(
///     &[0.9f32, 0.1, 0.0,   // 预测类0
///       0.1, 0.8, 0.1,       // 预测类1
///       0.0, 0.2, 0.8],      // 预测类2
///     &[3, 3],
/// );
/// let labels = [0, 1, 2];  // 真实标签
///
/// let cm = confusion_matrix(&logits, &labels);
/// // 完美预测，对角线全是1
/// assert_eq!(cm, vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
/// ```
///
/// ## 边界情况
///
/// - 空输入 → 返回空矩阵 `vec![]`
/// - 类别从 0 开始，矩阵大小为 `max_label + 1`
pub fn confusion_matrix(
    predictions: &(impl IntoClassLabels + ?Sized),
    actuals: &(impl IntoClassLabels + ?Sized),
) -> Vec<Vec<usize>> {
    let pred_labels = predictions.to_class_labels();
    let true_labels = actuals.to_class_labels();

    if pred_labels.is_empty() || true_labels.is_empty() {
        return vec![];
    }

    let n = pred_labels.len().min(true_labels.len());
    let pred_labels = &pred_labels[..n];
    let true_labels = &true_labels[..n];

    // 自动检测类别数量（max label + 1）
    let num_classes = pred_labels
        .iter()
        .chain(true_labels.iter())
        .max()
        .map(|&x| x + 1)
        .unwrap_or(0);

    // 构建混淆矩阵
    let mut matrix = vec![vec![0usize; num_classes]; num_classes];
    for (&pred, &actual) in pred_labels.iter().zip(true_labels.iter()) {
        matrix[actual][pred] += 1;
    }

    matrix
}

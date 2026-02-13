//! Multilabel Loose/Strict Accuracy 相关测试

use crate::metrics::{multilabel_loose_accuracy, multilabel_strict_accuracy};
use crate::tensor::Tensor;

// ==================== Multilabel Loose Accuracy 测试 ====================

/// 基本功能测试
#[test]
fn test_multilabel_loose_accuracy_basic() {
    // 3 个样本，4 个标签
    let predictions = Tensor::new(
        &[
            0.8, 0.3, 0.6, 0.1, // 样本1: [1,0,1,0]
            0.2, 0.9, 0.4, 0.7, // 样本2: [0,1,0,1]
            0.6, 0.6, 0.6, 0.6, // 样本3: [1,1,1,1]
        ],
        &[3, 4],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 0.0, // 样本1: 全对
            0.0, 1.0, 0.0, 1.0, // 样本2: 全对
            1.0, 1.0, 0.0, 1.0, // 样本3: 第3个标签错误
        ],
        &[3, 4],
    );

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    // 正确: 11/12 (样本3的第3个标签预测为1，实际为0)
    assert!(
        (result.value() - 11.0 / 12.0).abs() < 1e-6,
        "Multi-label accuracy = {}, 期望 ≈ 0.917",
        result.value()
    );
    assert_eq!(result.n_samples(), 12); // 3 × 4 = 12 个标签
}

/// 完美预测
#[test]
fn test_multilabel_loose_accuracy_perfect() {
    let predictions = Tensor::new(&[0.9, 0.1, 0.8, 0.2, 0.7, 0.3], &[2, 3]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]);

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "完美预测应返回 1.0，实际 = {}",
        result.value()
    );
    assert_eq!(result.n_samples(), 6);
}

/// 完全错误
#[test]
fn test_multilabel_loose_accuracy_all_wrong() {
    let predictions = Tensor::new(&[0.9, 0.9, 0.1, 0.1], &[2, 2]);
    let actuals = Tensor::new(&[0.0, 0.0, 1.0, 1.0], &[2, 2]);

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    assert!(
        result.value().abs() < 1e-6,
        "全错应返回 0.0，实际 = {}",
        result.value()
    );
}

/// 50% 准确率
#[test]
fn test_multilabel_loose_accuracy_half() {
    let predictions = Tensor::new(&[0.6, 0.4, 0.6, 0.4], &[2, 2]);
    let actuals = Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[2, 2]);

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    // 样本1: pred=[1,0], actual=[1,1] → 1对1错
    // 样本2: pred=[1,0], actual=[0,0] → 1错1对
    // 总计: 2/4 = 0.5
    assert!(
        (result.value() - 0.5).abs() < 1e-6,
        "Half correct = 0.5，实际 = {}",
        result.value()
    );
}

/// 不同阈值测试
#[test]
fn test_multilabel_loose_accuracy_threshold() {
    let predictions = Tensor::new(&[0.3, 0.7], &[1, 2]);
    let actuals = Tensor::new(&[0.0, 1.0], &[1, 2]);

    // 阈值 0.5: pred=[0,1], actual=[0,1] → 2/2 = 1.0
    let result_05 = multilabel_loose_accuracy(&predictions, &actuals, 0.5);
    assert!((result_05.value() - 1.0).abs() < 1e-6, "阈值0.5应全对");

    // 阈值 0.25: pred=[1,1], actual=[0,1] → 1/2 = 0.5
    let result_025 = multilabel_loose_accuracy(&predictions, &actuals, 0.25);
    assert!((result_025.value() - 0.5).abs() < 1e-6, "阈值0.25应半对");

    // 阈值 0.8: pred=[0,0], actual=[0,1] → 1/2 = 0.5
    let result_08 = multilabel_loose_accuracy(&predictions, &actuals, 0.8);
    assert!((result_08.value() - 0.5).abs() < 1e-6, "阈值0.8应半对");
}

/// 单样本单标签
#[test]
fn test_multilabel_loose_accuracy_single() {
    let pred_correct = Tensor::new(&[0.6], &[1, 1]);
    let pred_wrong = Tensor::new(&[0.4], &[1, 1]);
    let actual = Tensor::new(&[1.0], &[1, 1]);

    assert!((multilabel_loose_accuracy(&pred_correct, &actual, 0.5).value() - 1.0).abs() < 1e-6);
    assert!(
        multilabel_loose_accuracy(&pred_wrong, &actual, 0.5)
            .value()
            .abs()
            < 1e-6
    );
}

/// 边界情况：空 Tensor
#[test]
fn test_multilabel_loose_accuracy_empty() {
    let empty = Tensor::new(&[] as &[f32], &[0, 4]);

    let result = multilabel_loose_accuracy(&empty, &empty, 0.5);
    assert_eq!(result.value(), 0.0);
    assert_eq!(result.n_samples(), 0);
    assert_eq!(result.num_labels(), 0);
    assert_eq!(result.num_samples(), 0);
    assert!(result.per_label().is_empty());
}

/// 大批量测试
#[test]
fn test_multilabel_loose_accuracy_large_batch() {
    // 100 个样本，10 个标签
    let mut pred_data = vec![0.0f32; 1000];
    let mut actual_data = vec![0.0f32; 1000];

    // 设置前 800 个标签全对，后 200 个全错
    for i in 0..800 {
        pred_data[i] = 0.9;
        actual_data[i] = 1.0;
    }
    for i in 800..1000 {
        pred_data[i] = 0.9;
        actual_data[i] = 0.0;
    }

    let predictions = Tensor::new(&pred_data, &[100, 10]);
    let actuals = Tensor::new(&actual_data, &[100, 10]);

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    // 800/1000 = 0.8
    assert!(
        (result.value() - 0.8).abs() < 1e-6,
        "大批量测试：accuracy = {}, 期望 0.8",
        result.value()
    );
    assert_eq!(result.n_samples(), 1000);
}

/// 测试 Metric trait 方法
#[test]
fn test_multilabel_loose_accuracy_metric_trait() {
    let predictions = Tensor::new(&[0.9, 0.1, 0.9, 0.1], &[2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    // percent() 测试
    assert!((result.percent() - 100.0).abs() < 1e-4);

    // weighted() 测试
    assert!((result.weighted() - 4.0).abs() < 1e-4); // 1.0 × 4
}

/// 测试 per_label() 方法
#[test]
fn test_multilabel_loose_accuracy_per_label() {
    // 3 个样本，4 个标签
    // 设计明确的测试数据：
    //         标签0  标签1  标签2  标签3
    // 样本1:  pred=1 pred=1 pred=1 pred=1
    //        actual=1 actual=0 actual=1 actual=0  → 标签0对, 标签1错, 标签2对, 标签3错
    // 样本2:  pred=1 pred=0 pred=0 pred=1
    //        actual=1 actual=0 actual=1 actual=0  → 标签0对, 标签1对, 标签2错, 标签3错
    // 样本3:  pred=1 pred=1 pred=0 pred=1
    //        actual=1 actual=1 actual=0 actual=0  → 标签0对, 标签1对, 标签2对, 标签3错
    //
    // 标签0: 3/3 = 100%
    // 标签1: 2/3 ≈ 66.7%
    // 标签2: 2/3 ≈ 66.7%
    // 标签3: 0/3 = 0%
    let predictions = Tensor::new(
        &[
            0.9, 0.9, 0.9, 0.9, // 样本1: pred=[1,1,1,1]
            0.9, 0.1, 0.1, 0.9, // 样本2: pred=[1,0,0,1]
            0.9, 0.9, 0.1, 0.9, // 样本3: pred=[1,1,0,1]
        ],
        &[3, 4],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 0.0, // 样本1: actual=[1,0,1,0]
            1.0, 0.0, 1.0, 0.0, // 样本2: actual=[1,0,1,0]
            1.0, 1.0, 0.0, 0.0, // 样本3: actual=[1,1,0,0]
        ],
        &[3, 4],
    );

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    let per_label = result.per_label();
    assert_eq!(per_label.len(), 4);
    assert_eq!(result.num_labels(), 4);
    assert_eq!(result.num_samples(), 3);

    // 标签0: 3/3 = 100%
    assert!(
        (per_label[0].value() - 1.0).abs() < 1e-6,
        "标签0准确率 = {}, 期望 1.0",
        per_label[0].value()
    );
    assert_eq!(per_label[0].n_samples(), 3);

    // 标签1: 2/3 ≈ 66.7% (样本1错, 样本2对, 样本3对)
    assert!(
        (per_label[1].value() - 2.0 / 3.0).abs() < 1e-6,
        "标签1准确率 = {}, 期望 ≈ 0.667",
        per_label[1].value()
    );

    // 标签2: 2/3 ≈ 66.7% (样本1对, 样本2错, 样本3对)
    assert!(
        (per_label[2].value() - 2.0 / 3.0).abs() < 1e-6,
        "标签2准确率 = {}, 期望 ≈ 0.667",
        per_label[2].value()
    );

    // 标签3: 0/3 = 0% (全错)
    assert!(
        per_label[3].value().abs() < 1e-6,
        "标签3准确率 = {}, 期望 0.0",
        per_label[3].value()
    );

    // 总体准确率: (3+2+2+0) / 12 = 7/12 ≈ 58.3%
    assert!(
        (result.value() - 7.0 / 12.0).abs() < 1e-6,
        "总体准确率 = {}, 期望 ≈ 0.583",
        result.value()
    );
}

/// 与示例一致的用例（验证概念理解）
#[test]
fn test_multilabel_loose_accuracy_example_scenario() {
    // 模拟示例中的场景：
    // 样本1: 真实 [1,0,1,0], 预测 [1,0,1,0] → 4/4 对
    // 样本2: 真实 [1,1,0,0], 预测 [1,0,0,0] → 3/4 对
    // 样本3: 真实 [0,0,1,1], 预测 [0,1,1,0] → 2/4 对
    let predictions = Tensor::new(
        &[
            0.9, 0.1, 0.9, 0.1, // 样本1: pred=[1,0,1,0]
            0.9, 0.1, 0.1, 0.1, // 样本2: pred=[1,0,0,0]
            0.1, 0.9, 0.9, 0.1, // 样本3: pred=[0,1,1,0]
        ],
        &[3, 4],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 0.0, // 样本1
            1.0, 1.0, 0.0, 0.0, // 样本2
            0.0, 0.0, 1.0, 1.0, // 样本3
        ],
        &[3, 4],
    );

    let result = multilabel_loose_accuracy(&predictions, &actuals, 0.5);

    // 总计: (4 + 3 + 2) / 12 = 9/12 = 0.75
    assert!(
        (result.value() - 0.75).abs() < 1e-6,
        "示例场景准确率 = {}, 期望 0.75",
        result.value()
    );
    assert_eq!(result.n_samples(), 12);
    assert_eq!(result.num_labels(), 4);
    assert_eq!(result.num_samples(), 3);

    // 验证 per_label 数据可用
    assert_eq!(result.per_label().len(), 4);
}

// ==================== Subset Accuracy 测试 ====================

/// 基本功能测试
#[test]
fn test_multilabel_strict_accuracy_basic() {
    // 3 个样本，4 个标签
    // 样本1: pred=[1,0,1,0], actual=[1,0,1,0] → 完全匹配 ✓
    // 样本2: pred=[1,0,0,0], actual=[1,1,0,0] → 标签1错误 ✗
    // 样本3: pred=[0,1,1,0], actual=[0,0,1,1] → 标签1,3错误 ✗
    let predictions = Tensor::new(
        &[
            0.9, 0.1, 0.9, 0.1, // 样本1
            0.9, 0.1, 0.1, 0.1, // 样本2
            0.1, 0.9, 0.9, 0.1, // 样本3
        ],
        &[3, 4],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 0.0, // 样本1
            1.0, 1.0, 0.0, 0.0, // 样本2
            0.0, 0.0, 1.0, 1.0, // 样本3
        ],
        &[3, 4],
    );

    let result = multilabel_strict_accuracy(&predictions, &actuals, 0.5);

    // 只有样本1完全匹配
    assert!(
        (result.value() - 1.0 / 3.0).abs() < 1e-6,
        "Subset accuracy = {}, 期望 ≈ 0.333",
        result.value()
    );
    assert_eq!(result.n_samples(), 3); // 样本数，非标签数
}

/// 完美预测
#[test]
fn test_multilabel_strict_accuracy_perfect() {
    let predictions = Tensor::new(&[0.9, 0.1, 0.8, 0.2, 0.7, 0.3], &[2, 3]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 3]);

    let result = multilabel_strict_accuracy(&predictions, &actuals, 0.5);

    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "完美预测应返回 1.0，实际 = {}",
        result.value()
    );
    assert_eq!(result.n_samples(), 2);
}

/// 完全错误（无完全匹配）
#[test]
fn test_multilabel_strict_accuracy_all_wrong() {
    // 每个样本至少有一个标签错误
    let predictions = Tensor::new(&[0.9, 0.9, 0.1, 0.1], &[2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);

    let result = multilabel_strict_accuracy(&predictions, &actuals, 0.5);

    // 样本1: pred=[1,1], actual=[1,0] → 标签1错误
    // 样本2: pred=[0,0], actual=[0,1] → 标签1错误
    assert!(
        result.value().abs() < 1e-6,
        "无完全匹配应返回 0.0，实际 = {}",
        result.value()
    );
}

/// 50% 准确率
#[test]
fn test_multilabel_strict_accuracy_half() {
    let predictions = Tensor::new(&[0.9, 0.1, 0.9, 0.1], &[2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 1.0], &[2, 2]);

    let result = multilabel_strict_accuracy(&predictions, &actuals, 0.5);

    // 样本1: pred=[1,0], actual=[1,0] → 完全匹配 ✓
    // 样本2: pred=[1,0], actual=[1,1] → 标签1错误 ✗
    assert!(
        (result.value() - 0.5).abs() < 1e-6,
        "Half correct = 0.5，实际 = {}",
        result.value()
    );
}

/// 不同阈值测试
#[test]
fn test_multilabel_strict_accuracy_threshold() {
    let predictions = Tensor::new(&[0.3, 0.7, 0.6, 0.4], &[2, 2]);
    let actuals = Tensor::new(&[0.0, 1.0, 1.0, 0.0], &[2, 2]);

    // 阈值 0.5:
    // 样本1: pred=[0,1], actual=[0,1] → ✓
    // 样本2: pred=[1,0], actual=[1,0] → ✓
    let result_05 = multilabel_strict_accuracy(&predictions, &actuals, 0.5);
    assert!((result_05.value() - 1.0).abs() < 1e-6, "阈值0.5应全对");

    // 阈值 0.25:
    // 样本1: pred=[1,1], actual=[0,1] → ✗
    // 样本2: pred=[1,1], actual=[1,0] → ✗
    let result_025 = multilabel_strict_accuracy(&predictions, &actuals, 0.25);
    assert!(result_025.value().abs() < 1e-6, "阈值0.25应全错");
}

/// 单样本单标签
#[test]
fn test_multilabel_strict_accuracy_single() {
    let pred_correct = Tensor::new(&[0.6], &[1, 1]);
    let pred_wrong = Tensor::new(&[0.4], &[1, 1]);
    let actual = Tensor::new(&[1.0], &[1, 1]);

    assert!((multilabel_strict_accuracy(&pred_correct, &actual, 0.5).value() - 1.0).abs() < 1e-6);
    assert!(
        multilabel_strict_accuracy(&pred_wrong, &actual, 0.5)
            .value()
            .abs()
            < 1e-6
    );
}

/// 边界情况：空 Tensor
#[test]
fn test_multilabel_strict_accuracy_empty() {
    let empty = Tensor::new(&[] as &[f32], &[0, 4]);

    let result = multilabel_strict_accuracy(&empty, &empty, 0.5);
    assert_eq!(result.value(), 0.0);
    assert_eq!(result.n_samples(), 0);
}

/// 大批量测试
#[test]
fn test_multilabel_strict_accuracy_large_batch() {
    // 100 个样本，10 个标签
    // 前 30 个样本完全匹配，后 70 个样本有错误
    let pred_data = vec![0.9f32; 1000]; // 全预测为正
    let mut actual_data = vec![1.0f32; 1000]; // 前 300 个全为正（前 30 个样本）

    // 后 70 个样本的第一个标签设为 0（制造错误）
    for i in 30..100 {
        actual_data[i * 10] = 0.0;
    }

    let predictions = Tensor::new(&pred_data, &[100, 10]);
    let actuals = Tensor::new(&actual_data, &[100, 10]);

    let result = multilabel_strict_accuracy(&predictions, &actuals, 0.5);

    // 30/100 = 0.3
    assert!(
        (result.value() - 0.3).abs() < 1e-6,
        "大批量测试：accuracy = {}, 期望 0.3",
        result.value()
    );
    assert_eq!(result.n_samples(), 100);
}

/// 测试 Metric trait 方法
#[test]
fn test_multilabel_strict_accuracy_metric_trait() {
    let predictions = Tensor::new(&[0.9, 0.1, 0.9, 0.1], &[2, 2]);
    let actuals = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);

    let result = multilabel_strict_accuracy(&predictions, &actuals, 0.5);

    // percent() 测试
    assert!((result.percent() - 100.0).abs() < 1e-4);

    // weighted() 测试
    assert!((result.weighted() - 2.0).abs() < 1e-4); // 1.0 × 2
}

/// 与 multilabel_loose_accuracy 对比测试
#[test]
fn test_subset_vs_multilabel_loose_accuracy() {
    // 3 个样本，4 个标签
    // 样本1: 全对 (4/4)
    // 样本2: 3对1错 (3/4)
    // 样本3: 2对2错 (2/4)
    let predictions = Tensor::new(
        &[
            0.9, 0.1, 0.9, 0.1, // 样本1: pred=[1,0,1,0]
            0.9, 0.1, 0.1, 0.1, // 样本2: pred=[1,0,0,0]
            0.1, 0.9, 0.9, 0.1, // 样本3: pred=[0,1,1,0]
        ],
        &[3, 4],
    );
    let actuals = Tensor::new(
        &[
            1.0, 0.0, 1.0, 0.0, // 样本1
            1.0, 1.0, 0.0, 0.0, // 样本2
            0.0, 0.0, 1.0, 1.0, // 样本3
        ],
        &[3, 4],
    );

    let multi_label = multilabel_loose_accuracy(&predictions, &actuals, 0.5);
    let subset = multilabel_strict_accuracy(&predictions, &actuals, 0.5);

    // multilabel_loose_accuracy: (4+3+2) / 12 = 9/12 = 0.75
    assert!(
        (multi_label.value() - 0.75).abs() < 1e-6,
        "multilabel_loose_accuracy = {}",
        multi_label.value()
    );

    // multilabel_strict_accuracy: 1/3 ≈ 0.333
    assert!(
        (subset.value() - 1.0 / 3.0).abs() < 1e-6,
        "multilabel_strict_accuracy = {}",
        subset.value()
    );

    // subset 应该 <= multi_label（更严格）
    assert!(
        subset.value() <= multi_label.value(),
        "subset ({}) 应该 <= multi_label ({})",
        subset.value(),
        multi_label.value()
    );
}

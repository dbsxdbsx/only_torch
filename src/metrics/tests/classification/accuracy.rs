//! Accuracy 相关测试

use crate::metrics::accuracy;
use crate::tensor::Tensor;

/// 基本功能测试
#[test]
fn test_accuracy_basic() {
    let predictions = vec![0, 1, 1, 0, 1];
    let actuals = vec![0, 1, 0, 0, 1];

    let result = accuracy(&predictions, &actuals);

    // 4/5 = 0.8
    assert!(
        (result.value() - 0.8).abs() < 1e-6,
        "Accuracy = {}, 期望 0.8",
        result.value()
    );
    assert_eq!(result.n_samples(), 5);
}

/// 完美预测：Accuracy = 1.0
#[test]
fn test_accuracy_perfect() {
    let labels = vec![0, 1, 2, 3, 4];
    let result = accuracy(&labels, &labels);

    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "完美预测应返回 Accuracy = 1.0，实际 = {}",
        result.value()
    );
}

/// 完全错误：Accuracy = 0.0
#[test]
fn test_accuracy_all_wrong() {
    let predictions = vec![1, 0, 1, 0];
    let actuals = vec![0, 1, 0, 1];

    let result = accuracy(&predictions, &actuals);

    assert!(
        result.value().abs() < 1e-6,
        "全错应返回 Accuracy = 0.0，实际 = {}",
        result.value()
    );
}

/// 多分类支持
#[test]
fn test_accuracy_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0, 2];
    let actuals = vec![0, 1, 2, 2, 0, 1];

    let result = accuracy(&predictions, &actuals);

    // 4/6 ≈ 0.6667
    assert!(
        (result.value() - 4.0 / 6.0).abs() < 1e-6,
        "Accuracy = {}, 期望 ≈ 0.6667",
        result.value()
    );
}

/// 边界情况：空输入
#[test]
fn test_accuracy_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(accuracy(&empty, &values).value(), 0.0);
    assert_eq!(accuracy(&values, &empty).value(), 0.0);
    assert_eq!(accuracy(&empty, &empty).value(), 0.0);
    // 空输入时 n_samples 应为 0
    assert_eq!(accuracy(&empty, &values).n_samples(), 0);
}

/// 边界情况：单个样本
#[test]
fn test_accuracy_single_sample() {
    assert_eq!(accuracy(&[1], &[1]).value(), 1.0);
    assert_eq!(accuracy(&[1], &[0]).value(), 0.0);
    assert_eq!(accuracy(&[1], &[1]).n_samples(), 1);
}

/// 长度不一致时取较短者
#[test]
fn test_accuracy_length_mismatch() {
    let predictions = vec![0, 1, 1];
    let actuals = vec![0, 1]; // 较短

    let result = accuracy(&predictions, &actuals);

    // 只用前 2 个元素，完美预测
    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "长度不一致时应取较短者，实际 Accuracy = {}",
        result.value()
    );
    assert_eq!(result.n_samples(), 2);
}

/// 支持不同整数类型
#[test]
fn test_accuracy_different_types() {
    // u8 类型
    let pred_u8: Vec<u8> = vec![0, 1, 1, 0];
    let actual_u8: Vec<u8> = vec![0, 1, 0, 0];
    assert!((accuracy(&pred_u8, &actual_u8).value() - 0.75).abs() < 1e-6);

    // i64 类型
    let pred_i64: Vec<i64> = vec![0, 1, 1, 0];
    let actual_i64: Vec<i64> = vec![0, 1, 0, 0];
    assert!((accuracy(&pred_i64, &actual_i64).value() - 0.75).abs() < 1e-6);

    // usize 类型（常用于 argmax 结果）
    let pred_usize: Vec<usize> = vec![0, 1, 1, 0];
    let actual_usize: Vec<usize> = vec![0, 1, 0, 0];
    assert!((accuracy(&pred_usize, &actual_usize).value() - 0.75).abs() < 1e-6);
}

/// 测试 Metric trait 方法
#[test]
fn test_accuracy_metric_trait() {
    let result = accuracy(&[0, 1, 1, 0, 1], &[0, 1, 0, 0, 1]);

    // percent() 测试
    assert!((result.percent() - 80.0).abs() < 1e-4);

    // weighted() 测试：value × n_samples
    assert!((result.weighted() - 0.8 * 5.0).abs() < 1e-4);
}

/// 测试 Tensor 作为 logits 输入（自动 argmax）
#[test]
fn test_accuracy_tensor_logits() {
    // logits: [3, 2] 形状，3 个样本，2 个类别
    // 预测类别分别为 1, 0, 1
    let logits = Tensor::new(&[0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2]);
    // 真实标签（one-hot）: 1, 0, 1
    let labels = Tensor::new(&[0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[3, 2]);

    let result = accuracy(&logits, &labels);
    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "Tensor logits 完美预测应返回 1.0，实际 = {}",
        result.value()
    );
    assert_eq!(result.n_samples(), 3);
}

/// 测试 Tensor 与 slice 混合输入
#[test]
fn test_accuracy_tensor_mixed_input() {
    // logits: 预测类别 0, 1, 0
    let logits = Tensor::new(&[0.9, 0.1, 0.2, 0.8, 0.7, 0.3], &[3, 2]);
    // 真实标签：slice
    let labels = [0_usize, 1, 0];

    let result = accuracy(&logits, &labels);
    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "Tensor 与 slice 混合完美预测应返回 1.0，实际 = {}",
        result.value()
    );
}

/// 测试 1D Tensor 作为类别索引输入
#[test]
fn test_accuracy_tensor_1d_labels() {
    // 1D Tensor：直接作为类别索引
    let preds = Tensor::new(&[0.0, 1.0, 1.0, 0.0, 1.0], &[5]);
    let labels = Tensor::new(&[0.0, 1.0, 0.0, 0.0, 1.0], &[5]);

    let result = accuracy(&preds, &labels);
    // 4/5 = 0.8
    assert!(
        (result.value() - 0.8).abs() < 1e-6,
        "1D Tensor 应正确计算 Accuracy，实际 = {}",
        result.value()
    );
}

//! 回归指标测试

use crate::metrics::r2_score;

/// 基本功能测试（与 sklearn 对照）
#[test]
fn test_r2_score_basic() {
    // 测试数据来自 sklearn.metrics.r2_score 文档示例
    let predictions = vec![2.5, 0.0, 2.0, 8.0];
    let actuals = vec![3.0, -0.5, 2.0, 7.0];

    let result = r2_score(&predictions, &actuals);

    // sklearn 结果：0.9486081370449679
    assert!(
        (result.value() - 0.9486).abs() < 0.001,
        "R² = {}, 期望 ≈ 0.9486",
        result.value()
    );
    assert_eq!(result.n_samples(), 4);
}

/// 完美预测：R² = 1.0
#[test]
fn test_r2_score_perfect() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = r2_score(&values, &values);

    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "完美预测应返回 R² = 1.0，实际 = {}",
        result.value()
    );
}

/// 均值预测：R² ≈ 0.0
#[test]
fn test_r2_score_mean_prediction() {
    let actuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mean = 3.0; // actuals 的均值
    let predictions = vec![mean; 5];

    let result = r2_score(&predictions, &actuals);

    assert!(
        result.value().abs() < 1e-6,
        "均值预测应返回 R² ≈ 0.0，实际 = {}",
        result.value()
    );
}

/// 负 R²：模型比均值预测更差
#[test]
fn test_r2_score_negative() {
    let actuals = vec![1.0, 2.0, 3.0];
    // 预测值与真实值完全相反的趋势
    let predictions = vec![3.0, 2.0, 1.0];

    let result = r2_score(&predictions, &actuals);

    assert!(
        result.value() < 0.0,
        "反向预测应返回负 R²，实际 = {}",
        result.value()
    );
}

/// 边界情况：空输入
#[test]
fn test_r2_score_empty() {
    let empty: &[f32] = &[];
    let values = [1.0, 2.0, 3.0];

    assert_eq!(r2_score(empty, &values).value(), 0.0);
    assert_eq!(r2_score(&values, empty).value(), 0.0);
    assert_eq!(r2_score(empty, empty).value(), 0.0);
    // 空输入时 n_samples 应为 0
    assert_eq!(r2_score(empty, &values).n_samples(), 0);
}

/// 边界情况：所有真实值相同
#[test]
fn test_r2_score_constant_actual() {
    let actuals = vec![5.0, 5.0, 5.0, 5.0];

    // 完美预测
    let perfect_pred = vec![5.0, 5.0, 5.0, 5.0];
    assert_eq!(r2_score(&perfect_pred, &actuals).value(), 1.0);

    // 有误差的预测
    let bad_pred = vec![4.0, 5.0, 6.0, 5.0];
    assert_eq!(r2_score(&bad_pred, &actuals).value(), 0.0);
}

/// 单个样本
#[test]
fn test_r2_score_single_sample() {
    let pred = vec![1.0];
    let actual = vec![1.0];
    // 单样本且完美预测
    assert_eq!(r2_score(&pred, &actual).value(), 1.0);
    assert_eq!(r2_score(&pred, &actual).n_samples(), 1);

    let pred2 = vec![2.0];
    // 单样本有误差，但 SS_tot = 0（只有一个点）
    assert_eq!(r2_score(&pred2, &actual).value(), 0.0);
}

/// 长度不一致时取较短者
#[test]
fn test_r2_score_length_mismatch() {
    let predictions = vec![1.0, 2.0, 3.0];
    let actuals = vec![1.0, 2.0]; // 较短

    let result = r2_score(&predictions, &actuals);

    // 只用前 2 个元素，完美预测
    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "长度不一致时应取较短者，实际 R² = {}",
        result.value()
    );
    assert_eq!(result.n_samples(), 2);
}

/// 测试 Metric trait 方法
#[test]
fn test_r2_score_metric_trait() {
    let result = r2_score(&[2.5, 0.0, 2.0, 8.0], &[3.0, -0.5, 2.0, 7.0]);

    // percent() 测试
    assert!((result.percent() - 94.86).abs() < 0.1);

    // weighted() 测试：value × n_samples
    assert!((result.weighted() - result.value() * 4.0).abs() < 1e-4);
}

// ==================== Tensor 输入测试 ====================

use crate::tensor::Tensor;

/// 测试 Tensor 作为输入
#[test]
fn test_r2_score_tensor() {
    // sklearn 对照数据
    let preds = Tensor::new(&[2.5, 0.0, 2.0, 8.0], &[4]);
    let actuals = Tensor::new(&[3.0, -0.5, 2.0, 7.0], &[4]);

    let result = r2_score(&preds, &actuals);
    assert!(
        (result.value() - 0.9486).abs() < 0.001,
        "Tensor 输入应正确计算 R²，实际 = {}",
        result.value()
    );
}

/// 测试 [batch, 1] 形状的 Tensor
#[test]
fn test_r2_score_tensor_2d() {
    let preds = Tensor::new(&[2.5, 0.0, 2.0, 8.0], &[4, 1]);
    let actuals = Tensor::new(&[3.0, -0.5, 2.0, 7.0], &[4, 1]);

    let result = r2_score(&preds, &actuals);
    assert!(
        (result.value() - 0.9486).abs() < 0.001,
        "[batch, 1] Tensor 应正确计算 R²，实际 = {}",
        result.value()
    );
}

/// 测试 Tensor 与 slice 混合输入
#[test]
fn test_r2_score_tensor_mixed() {
    let preds = Tensor::new(&[2.5, 0.0, 2.0, 8.0], &[4]);

    let result = r2_score(&preds, &[3.0, -0.5, 2.0, 7.0]);
    assert!(
        (result.value() - 0.9486).abs() < 0.001,
        "Tensor 与 slice 混合应正确计算 R²，实际 = {}",
        result.value()
    );
}

//! Precision、Recall、F1 Score 相关测试

use crate::metrics::{f1_score, precision, recall};

// ==================== Precision 测试 ====================

/// Precision 基本功能测试（二分类）
#[test]
fn test_precision_binary() {
    let predictions = vec![1, 1, 0, 1, 0];
    let actuals = vec![1, 0, 0, 1, 1];

    let result = precision(&predictions, &actuals);

    // 类0: TP=1, FP=1 → Precision_0 = 0.5
    // 类1: TP=2, FP=1 → Precision_1 = 2/3 ≈ 0.667
    // Macro = (0.5 + 0.667) / 2 ≈ 0.583
    let expected = (0.5 + 2.0 / 3.0) / 2.0;
    assert!(
        (result.value() - expected).abs() < 1e-4,
        "Precision = {}, 期望 ≈ {expected}",
        result.value()
    );
    assert_eq!(result.n_samples(), 5);
}

/// Precision 多分类测试
#[test]
fn test_precision_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0];
    let actuals = vec![0, 1, 2, 2, 0];

    let result = precision(&predictions, &actuals);

    // 类0: TP=2, FP=0 → Precision_0 = 1.0
    // 类1: TP=1, FP=1 → Precision_1 = 0.5
    // 类2: TP=1, FP=0 → Precision_2 = 1.0
    // Macro = (1.0 + 0.5 + 1.0) / 3 ≈ 0.833
    let expected = (1.0 + 0.5 + 1.0) / 3.0;
    assert!(
        (result.value() - expected).abs() < 1e-4,
        "Precision = {}, 期望 ≈ {expected}",
        result.value()
    );
}

/// 完美预测：Precision = 1.0
#[test]
fn test_precision_perfect() {
    let labels = vec![0, 1, 2, 0, 1];
    let result = precision(&labels, &labels);

    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "完美预测应返回 Precision = 1.0，实际 = {}",
        result.value()
    );
}

/// 完全错误（二分类互换）
#[test]
fn test_precision_all_wrong_binary() {
    let predictions = vec![1, 1, 1, 1];
    let actuals = vec![0, 0, 0, 0];

    let result = precision(&predictions, &actuals);

    // 类0: 从未被预测 → 不参与平均
    // 类1: TP=0, FP=4 → Precision_1 = 0.0
    // Macro = 0.0 / 1 = 0.0
    assert!(
        result.value().abs() < 1e-6,
        "全错应返回 Precision = 0.0，实际 = {}",
        result.value()
    );
}

/// 边界情况：空输入
#[test]
fn test_precision_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(precision(&empty, &values).value(), 0.0);
    assert_eq!(precision(&values, &empty).value(), 0.0);
    assert_eq!(precision(&empty, &empty).value(), 0.0);
}

/// 边界情况：单个样本
#[test]
fn test_precision_single_sample() {
    // 正确预测
    assert!((precision(&[1], &[1]).value() - 1.0).abs() < 1e-6);

    // 错误预测
    assert!(precision(&[1], &[0]).value().abs() < 1e-6);
}

/// 某个类从未被预测过（不参与平均）
#[test]
fn test_precision_class_never_predicted() {
    // 类2 存在于 actuals 但从未被预测
    let predictions = vec![0, 0, 1, 1];
    let actuals = vec![0, 2, 1, 2];

    let result = precision(&predictions, &actuals);

    // 类0: TP=1, FP=1 → Precision_0 = 0.5
    // 类1: TP=1, FP=1 → Precision_1 = 0.5
    // 类2: 从未被预测 → 不参与平均
    // Macro = (0.5 + 0.5) / 2 = 0.5
    assert!(
        (result.value() - 0.5).abs() < 1e-6,
        "Precision = {}, 期望 0.5",
        result.value()
    );
}

/// 类别不平衡场景：验证 Macro-Average 不被大类主导
///
/// 这是 Macro vs Micro 的核心区别测试
#[test]
fn test_precision_imbalanced_classes() {
    // 极端不平衡：类0 有 100 个样本，类1 只有 2 个样本
    // 模型对大类(0)预测很好，对小类(1)预测很差
    let mut predictions = vec![0; 100]; // 100 个预测为 0
    predictions.extend(vec![0, 0]); // 2 个也预测为 0（本应是 1）

    let mut actuals = vec![0; 100]; // 100 个真实为 0
    actuals.extend(vec![1, 1]); // 2 个真实为 1

    let result = precision(&predictions, &actuals);

    // 类0: TP=100, FP=2 → Precision_0 = 100/102 ≈ 0.980
    // 类1: 从未被预测 → 不参与平均
    // Macro = 0.980 / 1 ≈ 0.980
    //
    // 如果是 Micro：全局 TP=100, FP=2
    // Micro-Precision = 100/102 ≈ 0.980（与 Macro 相同，因为只有一个类被预测）

    // 关键点：小类(1)完全被忽略，但由于从未被预测，不参与平均
    // 这是合理的——如果模型从不预测某个类，该类的 Precision 无定义
    let precision_0 = 100.0 / 102.0;
    assert!(
        (result.value() - precision_0).abs() < 1e-4,
        "Precision = {}, 期望 ≈ {precision_0}",
        result.value()
    );
}

/// 类别不平衡场景2：小类被预测但全错
///
/// 验证小类的差表现会拉低 Macro 平均
#[test]
fn test_precision_imbalanced_small_class_all_wrong() {
    // 大类(0): 10 个样本，全部正确预测
    // 小类(1): 2 个样本，全部被错误预测为 0
    // 但模型也错误地把一些 0 预测为 1
    let mut predictions = vec![0; 10]; // 10 个正确预测为 0
    predictions.extend(vec![0, 0]); // 2 个错误预测为 0（本应是 1）
    predictions.extend(vec![1, 1]); // 2 个错误预测为 1（本应是 0）

    let mut actuals = vec![0; 10]; // 10 个真实为 0
    actuals.extend(vec![1, 1]); // 2 个真实为 1
    actuals.extend(vec![0, 0]); // 2 个真实为 0

    let result = precision(&predictions, &actuals);

    // 类0: TP=10, FP=2 → Precision_0 = 10/12 ≈ 0.833
    // 类1: TP=0, FP=2 → Precision_1 = 0/2 = 0.0
    // Macro = (0.833 + 0.0) / 2 ≈ 0.417
    //
    // 对比：如果是 Micro
    // 全局 TP=10, 全局 FP=4
    // Micro-Precision = 10/14 ≈ 0.714
    //
    // Macro (0.417) < Micro (0.714)
    // 说明 Macro 更能暴露小类的问题！
    let precision_0 = 10.0 / 12.0;
    let precision_1 = 0.0;
    let expected_macro = (precision_0 + precision_1) / 2.0;

    assert!(
        (result.value() - expected_macro).abs() < 1e-4,
        "Macro-Precision = {}, 期望 ≈ {expected_macro}",
        result.value()
    );

    // 验证 Macro 确实比 Micro 低（暴露小类问题）
    let micro_precision = 10.0 / 14.0;
    assert!(
        result.value() < micro_precision,
        "Macro ({}) 应该 < Micro ({micro_precision})，说明小类问题被暴露",
        result.value()
    );
}

// ==================== Recall 测试 ====================

/// Recall 基本功能测试（二分类）
#[test]
fn test_recall_binary() {
    let predictions = vec![1, 1, 0, 1, 0];
    let actuals = vec![1, 0, 0, 1, 1];

    let result = recall(&predictions, &actuals);

    // 类0: TP=1, FN=1 → Recall_0 = 0.5
    // 类1: TP=2, FN=1 → Recall_1 = 2/3 ≈ 0.667
    // Macro = (0.5 + 0.667) / 2 ≈ 0.583
    let expected = (0.5 + 2.0 / 3.0) / 2.0;
    assert!(
        (result.value() - expected).abs() < 1e-4,
        "Recall = {}, 期望 ≈ {expected}",
        result.value()
    );
}

/// Recall 多分类测试
#[test]
fn test_recall_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0];
    let actuals = vec![0, 1, 2, 2, 0];

    let result = recall(&predictions, &actuals);

    // 类0: TP=2, FN=0 → Recall_0 = 1.0
    // 类1: TP=1, FN=0 → Recall_1 = 1.0
    // 类2: TP=1, FN=1 → Recall_2 = 0.5
    // Macro = (1.0 + 1.0 + 0.5) / 3 ≈ 0.833
    let expected = (1.0 + 1.0 + 0.5) / 3.0;
    assert!(
        (result.value() - expected).abs() < 1e-4,
        "Recall = {}, 期望 ≈ {expected}",
        result.value()
    );
}

/// 完美预测：Recall = 1.0
#[test]
fn test_recall_perfect() {
    let labels = vec![0, 1, 2, 0, 1];
    let result = recall(&labels, &labels);

    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "完美预测应返回 Recall = 1.0，实际 = {}",
        result.value()
    );
}

/// 完全错误（二分类互换）：Recall = 0.0
#[test]
fn test_recall_all_wrong_binary() {
    let predictions = vec![1, 1, 1, 1];
    let actuals = vec![0, 0, 0, 0];

    let result = recall(&predictions, &actuals);

    // 类0: TP=0, FN=4 → Recall_0 = 0.0
    // 类1: 从未在真实标签中出现 → 不参与平均
    // Macro = 0.0 / 1 = 0.0
    assert!(
        result.value().abs() < 1e-6,
        "全错应返回 Recall = 0.0，实际 = {}",
        result.value()
    );
}

/// 边界情况：空输入
#[test]
fn test_recall_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(recall(&empty, &values).value(), 0.0);
    assert_eq!(recall(&values, &empty).value(), 0.0);
    assert_eq!(recall(&empty, &empty).value(), 0.0);
}

/// 边界情况：单个样本
#[test]
fn test_recall_single_sample() {
    // 正确预测
    assert!((recall(&[1], &[1]).value() - 1.0).abs() < 1e-6);

    // 错误预测
    assert!(recall(&[1], &[0]).value().abs() < 1e-6);
}

/// 某个类从未在真实标签中出现（不参与平均）
#[test]
fn test_recall_class_never_in_actuals() {
    // 类2 被预测过但从未在真实标签中出现
    let predictions = vec![0, 2, 1, 2];
    let actuals = vec![0, 0, 1, 1];

    let result = recall(&predictions, &actuals);

    // 类0: TP=1, FN=1 → Recall_0 = 0.5
    // 类1: TP=1, FN=1 → Recall_1 = 0.5
    // 类2: 从未在真实标签中出现 → 不参与平均
    // Macro = (0.5 + 0.5) / 2 = 0.5
    assert!(
        (result.value() - 0.5).abs() < 1e-6,
        "Recall = {}, 期望 0.5",
        result.value()
    );
}

/// 类别不平衡场景：验证 Macro-Average 暴露小类问题
#[test]
fn test_recall_imbalanced_small_class_all_missed() {
    // 大类(0): 10 个样本，全部正确预测
    // 小类(1): 2 个样本，全部漏掉（预测为 0）
    let mut predictions = vec![0; 10]; // 10 个正确预测为 0
    predictions.extend(vec![0, 0]); // 2 个错误预测为 0（本应是 1）

    let mut actuals = vec![0; 10]; // 10 个真实为 0
    actuals.extend(vec![1, 1]); // 2 个真实为 1

    let result = recall(&predictions, &actuals);

    // 类0: TP=10, FN=0 → Recall_0 = 1.0
    // 类1: TP=0, FN=2 → Recall_1 = 0.0
    // Macro = (1.0 + 0.0) / 2 = 0.5
    //
    // 对比：如果是 Micro
    // 全局 TP=10, 全局 FN=2
    // Micro-Recall = 10/12 ≈ 0.833
    //
    // Macro (0.5) < Micro (0.833)
    // 说明 Macro 更能暴露小类被漏掉的问题！
    let expected_macro = (1.0 + 0.0) / 2.0;

    assert!(
        (result.value() - expected_macro).abs() < 1e-4,
        "Macro-Recall = {}, 期望 ≈ {expected_macro}",
        result.value()
    );

    // 验证 Macro 确实比 Micro 低（暴露小类问题）
    let micro_recall = 10.0 / 12.0;
    assert!(
        result.value() < micro_recall,
        "Macro ({}) 应该 < Micro ({micro_recall})，说明小类漏检问题被暴露",
        result.value()
    );
}

// ==================== F1 Score 测试 ====================

/// F1 基本功能测试（二分类）
#[test]
fn test_f1_score_binary() {
    let predictions = vec![1, 1, 0, 1, 0];
    let actuals = vec![1, 0, 0, 1, 1];

    let result = f1_score(&predictions, &actuals);

    // 类0: TP=1, FP=1, FN=1 → P=0.5, R=0.5 → F1_0 = 0.5
    // 类1: TP=2, FP=1, FN=1 → P=2/3, R=2/3 → F1_1 = 2/3
    // Macro = (0.5 + 2/3) / 2 ≈ 0.583
    let f1_0 = 0.5;
    let f1_1 = 2.0 / 3.0;
    let expected = (f1_0 + f1_1) / 2.0;
    assert!(
        (result.value() - expected).abs() < 1e-4,
        "F1 = {}, 期望 ≈ {expected}",
        result.value()
    );
}

/// F1 多分类测试
#[test]
fn test_f1_score_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0];
    let actuals = vec![0, 1, 2, 2, 0];

    let result = f1_score(&predictions, &actuals);

    // 类0: P=1.0, R=1.0 → F1_0 = 1.0
    // 类1: P=0.5, R=1.0 → F1_1 = 2*0.5*1.0/(0.5+1.0) = 2/3 ≈ 0.667
    // 类2: P=1.0, R=0.5 → F1_2 = 2*1.0*0.5/(1.0+0.5) = 2/3 ≈ 0.667
    // Macro = (1.0 + 0.667 + 0.667) / 3 ≈ 0.778
    let f1_0 = 1.0;
    let f1_1 = 2.0 * 0.5 * 1.0 / (0.5 + 1.0);
    let f1_2 = 2.0 * 1.0 * 0.5 / (1.0 + 0.5);
    let expected = (f1_0 + f1_1 + f1_2) / 3.0;
    assert!(
        (result.value() - expected).abs() < 1e-4,
        "F1 = {}, 期望 ≈ {expected}",
        result.value()
    );
}

/// 完美预测：F1 = 1.0
#[test]
fn test_f1_score_perfect() {
    let labels = vec![0, 1, 2, 0, 1];
    let result = f1_score(&labels, &labels);

    assert!(
        (result.value() - 1.0).abs() < 1e-6,
        "完美预测应返回 F1 = 1.0，实际 = {}",
        result.value()
    );
}

/// 完全错误（二分类互换）：F1 = 0.0
#[test]
fn test_f1_score_all_wrong_binary() {
    let predictions = vec![1, 1, 1, 1];
    let actuals = vec![0, 0, 0, 0];

    let result = f1_score(&predictions, &actuals);

    // 类0: TP=0, FP=0, FN=4 → P=0, R=0 → F1_0 = 0
    // 类1: TP=0, FP=4, FN=0 → P=0, R=undefined(0) → F1_1 = 0
    // Macro = 0.0
    assert!(
        result.value().abs() < 1e-6,
        "全错应返回 F1 = 0.0，实际 = {}",
        result.value()
    );
}

/// 边界情况：空输入
#[test]
fn test_f1_score_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(f1_score(&empty, &values).value(), 0.0);
    assert_eq!(f1_score(&values, &empty).value(), 0.0);
    assert_eq!(f1_score(&empty, &empty).value(), 0.0);
}

/// 边界情况：单个样本
#[test]
fn test_f1_score_single_sample() {
    // 正确预测
    assert!((f1_score(&[1], &[1]).value() - 1.0).abs() < 1e-6);

    // 错误预测
    assert!(f1_score(&[1], &[0]).value().abs() < 1e-6);
}

/// 类别不平衡场景：验证 Macro-F1 暴露小类问题
#[test]
fn test_f1_score_imbalanced() {
    // 大类(0): 10 个样本，全部正确
    // 小类(1): 2 个样本，全部漏掉
    let predictions = vec![0; 12];
    let mut actuals = vec![0; 10];
    actuals.extend(vec![1, 1]);

    let result = f1_score(&predictions, &actuals);

    // 类0: TP=10, FP=2, FN=0 → P=10/12, R=1.0 → F1_0 = 2*(10/12)*1/(10/12+1) ≈ 0.909
    // 类1: TP=0, FP=0, FN=2 → P=0, R=0 → F1_1 = 0
    // Macro = (0.909 + 0) / 2 ≈ 0.455
    let p_0 = 10.0 / 12.0;
    let r_0 = 1.0;
    let f1_0 = 2.0 * p_0 * r_0 / (p_0 + r_0);
    let f1_1 = 0.0;
    let expected = (f1_0 + f1_1) / 2.0;

    assert!(
        (result.value() - expected).abs() < 1e-4,
        "Macro-F1 = {}, 期望 ≈ {expected}",
        result.value()
    );

    // 验证 Macro-F1 确实较低（暴露小类问题）
    // 如果是 Micro-F1 = Accuracy = 10/12 ≈ 0.833
    let micro_f1 = 10.0 / 12.0;
    assert!(
        result.value() < micro_f1,
        "Macro-F1 ({}) 应该 < Micro-F1 ({micro_f1})，说明小类问题被暴露",
        result.value()
    );
}

/// 验证 F1 是 Precision 和 Recall 的调和平均
#[test]
fn test_f1_is_harmonic_mean() {
    let predictions = vec![0, 1, 1, 0, 1, 0];
    let actuals = vec![0, 0, 1, 1, 1, 0];

    let p = precision(&predictions, &actuals).value();
    let r = recall(&predictions, &actuals).value();
    let f1 = f1_score(&predictions, &actuals).value();

    // 注意：Macro-F1 ≠ 2*Macro-P*Macro-R/(Macro-P+Macro-R)
    // 因为 Macro-F1 是先算每个类的 F1 再平均，而不是先算 Macro-P/R 再算 F1
    // 所以这里我们只验证 F1 在 min(P,R) 和 max(P,R) 之间
    let min_pr = p.min(r);
    let max_pr = p.max(r);

    assert!(
        f1 >= min_pr - 1e-4 && f1 <= max_pr + 1e-4,
        "F1 ({f1}) 应该在 P ({p}) 和 R ({r}) 之间（或接近）"
    );
}

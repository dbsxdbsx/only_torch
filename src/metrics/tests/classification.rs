//! 分类指标测试

use crate::metrics::{accuracy, f1_score, precision, recall};

/// 基本功能测试
#[test]
fn test_accuracy_basic() {
    let predictions = vec![0, 1, 1, 0, 1];
    let actuals = vec![0, 1, 0, 0, 1];

    let acc = accuracy(&predictions, &actuals);

    // 4/5 = 0.8
    assert!((acc - 0.8).abs() < 1e-6, "Accuracy = {acc}, 期望 0.8");
}

/// 完美预测：Accuracy = 1.0
#[test]
fn test_accuracy_perfect() {
    let labels = vec![0, 1, 2, 3, 4];
    let acc = accuracy(&labels, &labels);

    assert!(
        (acc - 1.0).abs() < 1e-6,
        "完美预测应返回 Accuracy = 1.0，实际 = {acc}"
    );
}

/// 完全错误：Accuracy = 0.0
#[test]
fn test_accuracy_all_wrong() {
    let predictions = vec![1, 0, 1, 0];
    let actuals = vec![0, 1, 0, 1];

    let acc = accuracy(&predictions, &actuals);

    assert!(acc.abs() < 1e-6, "全错应返回 Accuracy = 0.0，实际 = {acc}");
}

/// 多分类支持
#[test]
fn test_accuracy_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0, 2];
    let actuals = vec![0, 1, 2, 2, 0, 1];

    let acc = accuracy(&predictions, &actuals);

    // 4/6 ≈ 0.6667
    assert!(
        (acc - 4.0 / 6.0).abs() < 1e-6,
        "Accuracy = {acc}, 期望 ≈ 0.6667"
    );
}

/// 边界情况：空输入
#[test]
fn test_accuracy_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(accuracy(&empty, &values), 0.0);
    assert_eq!(accuracy(&values, &empty), 0.0);
    assert_eq!(accuracy(&empty, &empty), 0.0);
}

/// 边界情况：单个样本
#[test]
fn test_accuracy_single_sample() {
    assert_eq!(accuracy(&[1], &[1]), 1.0);
    assert_eq!(accuracy(&[1], &[0]), 0.0);
}

/// 长度不一致时取较短者
#[test]
fn test_accuracy_length_mismatch() {
    let predictions = vec![0, 1, 1];
    let actuals = vec![0, 1]; // 较短

    let acc = accuracy(&predictions, &actuals);

    // 只用前 2 个元素，完美预测
    assert!(
        (acc - 1.0).abs() < 1e-6,
        "长度不一致时应取较短者，实际 Accuracy = {acc}"
    );
}

/// 支持不同整数类型
#[test]
fn test_accuracy_different_types() {
    // u8 类型
    let pred_u8: Vec<u8> = vec![0, 1, 1, 0];
    let actual_u8: Vec<u8> = vec![0, 1, 0, 0];
    assert!((accuracy(&pred_u8, &actual_u8) - 0.75).abs() < 1e-6);

    // i64 类型
    let pred_i64: Vec<i64> = vec![0, 1, 1, 0];
    let actual_i64: Vec<i64> = vec![0, 1, 0, 0];
    assert!((accuracy(&pred_i64, &actual_i64) - 0.75).abs() < 1e-6);

    // usize 类型（常用于 argmax 结果）
    let pred_usize: Vec<usize> = vec![0, 1, 1, 0];
    let actual_usize: Vec<usize> = vec![0, 1, 0, 0];
    assert!((accuracy(&pred_usize, &actual_usize) - 0.75).abs() < 1e-6);
}

// ==================== Precision 测试 ====================

/// Precision 基本功能测试（二分类）
#[test]
fn test_precision_binary() {
    let predictions = vec![1, 1, 0, 1, 0];
    let actuals = vec![1, 0, 0, 1, 1];

    let prec = precision(&predictions, &actuals);

    // 类0: TP=1, FP=1 → Precision_0 = 0.5
    // 类1: TP=2, FP=1 → Precision_1 = 2/3 ≈ 0.667
    // Macro = (0.5 + 0.667) / 2 ≈ 0.583
    let expected = (0.5 + 2.0 / 3.0) / 2.0;
    assert!(
        (prec - expected).abs() < 1e-4,
        "Precision = {prec}, 期望 ≈ {expected}"
    );
}

/// Precision 多分类测试
#[test]
fn test_precision_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0];
    let actuals = vec![0, 1, 2, 2, 0];

    let prec = precision(&predictions, &actuals);

    // 类0: TP=2, FP=0 → Precision_0 = 1.0
    // 类1: TP=1, FP=1 → Precision_1 = 0.5
    // 类2: TP=1, FP=0 → Precision_2 = 1.0
    // Macro = (1.0 + 0.5 + 1.0) / 3 ≈ 0.833
    let expected = (1.0 + 0.5 + 1.0) / 3.0;
    assert!(
        (prec - expected).abs() < 1e-4,
        "Precision = {prec}, 期望 ≈ {expected}"
    );
}

/// 完美预测：Precision = 1.0
#[test]
fn test_precision_perfect() {
    let labels = vec![0, 1, 2, 0, 1];
    let prec = precision(&labels, &labels);

    assert!(
        (prec - 1.0).abs() < 1e-6,
        "完美预测应返回 Precision = 1.0，实际 = {prec}"
    );
}

/// 完全错误（二分类互换）
#[test]
fn test_precision_all_wrong_binary() {
    let predictions = vec![1, 1, 1, 1];
    let actuals = vec![0, 0, 0, 0];

    let prec = precision(&predictions, &actuals);

    // 类0: 从未被预测 → 不参与平均
    // 类1: TP=0, FP=4 → Precision_1 = 0.0
    // Macro = 0.0 / 1 = 0.0
    assert!(
        prec.abs() < 1e-6,
        "全错应返回 Precision = 0.0，实际 = {prec}"
    );
}

/// 边界情况：空输入
#[test]
fn test_precision_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(precision(&empty, &values), 0.0);
    assert_eq!(precision(&values, &empty), 0.0);
    assert_eq!(precision(&empty, &empty), 0.0);
}

/// 边界情况：单个样本
#[test]
fn test_precision_single_sample() {
    // 正确预测
    assert!((precision(&[1], &[1]) - 1.0).abs() < 1e-6);

    // 错误预测
    assert!(precision(&[1], &[0]).abs() < 1e-6);
}

/// 某个类从未被预测过（不参与平均）
#[test]
fn test_precision_class_never_predicted() {
    // 类2 存在于 actuals 但从未被预测
    let predictions = vec![0, 0, 1, 1];
    let actuals = vec![0, 2, 1, 2];

    let prec = precision(&predictions, &actuals);

    // 类0: TP=1, FP=1 → Precision_0 = 0.5
    // 类1: TP=1, FP=1 → Precision_1 = 0.5
    // 类2: 从未被预测 → 不参与平均
    // Macro = (0.5 + 0.5) / 2 = 0.5
    assert!((prec - 0.5).abs() < 1e-6, "Precision = {prec}, 期望 0.5");
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

    let prec = precision(&predictions, &actuals);

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
        (prec - precision_0).abs() < 1e-4,
        "Precision = {prec}, 期望 ≈ {precision_0}"
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

    let prec = precision(&predictions, &actuals);

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
        (prec - expected_macro).abs() < 1e-4,
        "Macro-Precision = {prec}, 期望 ≈ {expected_macro}"
    );

    // 验证 Macro 确实比 Micro 低（暴露小类问题）
    let micro_precision = 10.0 / 14.0;
    assert!(
        prec < micro_precision,
        "Macro ({prec}) 应该 < Micro ({micro_precision})，说明小类问题被暴露"
    );
}

// ==================== Recall 测试 ====================

/// Recall 基本功能测试（二分类）
#[test]
fn test_recall_binary() {
    let predictions = vec![1, 1, 0, 1, 0];
    let actuals = vec![1, 0, 0, 1, 1];

    let rec = recall(&predictions, &actuals);

    // 类0: TP=1, FN=1 → Recall_0 = 0.5
    // 类1: TP=2, FN=1 → Recall_1 = 2/3 ≈ 0.667
    // Macro = (0.5 + 0.667) / 2 ≈ 0.583
    let expected = (0.5 + 2.0 / 3.0) / 2.0;
    assert!(
        (rec - expected).abs() < 1e-4,
        "Recall = {rec}, 期望 ≈ {expected}"
    );
}

/// Recall 多分类测试
#[test]
fn test_recall_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0];
    let actuals = vec![0, 1, 2, 2, 0];

    let rec = recall(&predictions, &actuals);

    // 类0: TP=2, FN=0 → Recall_0 = 1.0
    // 类1: TP=1, FN=0 → Recall_1 = 1.0
    // 类2: TP=1, FN=1 → Recall_2 = 0.5
    // Macro = (1.0 + 1.0 + 0.5) / 3 ≈ 0.833
    let expected = (1.0 + 1.0 + 0.5) / 3.0;
    assert!(
        (rec - expected).abs() < 1e-4,
        "Recall = {rec}, 期望 ≈ {expected}"
    );
}

/// 完美预测：Recall = 1.0
#[test]
fn test_recall_perfect() {
    let labels = vec![0, 1, 2, 0, 1];
    let rec = recall(&labels, &labels);

    assert!(
        (rec - 1.0).abs() < 1e-6,
        "完美预测应返回 Recall = 1.0，实际 = {rec}"
    );
}

/// 完全错误（二分类互换）：Recall = 0.0
#[test]
fn test_recall_all_wrong_binary() {
    let predictions = vec![1, 1, 1, 1];
    let actuals = vec![0, 0, 0, 0];

    let rec = recall(&predictions, &actuals);

    // 类0: TP=0, FN=4 → Recall_0 = 0.0
    // 类1: 从未在真实标签中出现 → 不参与平均
    // Macro = 0.0 / 1 = 0.0
    assert!(rec.abs() < 1e-6, "全错应返回 Recall = 0.0，实际 = {rec}");
}

/// 边界情况：空输入
#[test]
fn test_recall_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(recall(&empty, &values), 0.0);
    assert_eq!(recall(&values, &empty), 0.0);
    assert_eq!(recall(&empty, &empty), 0.0);
}

/// 边界情况：单个样本
#[test]
fn test_recall_single_sample() {
    // 正确预测
    assert!((recall(&[1], &[1]) - 1.0).abs() < 1e-6);

    // 错误预测
    assert!(recall(&[1], &[0]).abs() < 1e-6);
}

/// 某个类从未在真实标签中出现（不参与平均）
#[test]
fn test_recall_class_never_in_actuals() {
    // 类2 被预测过但从未在真实标签中出现
    let predictions = vec![0, 2, 1, 2];
    let actuals = vec![0, 0, 1, 1];

    let rec = recall(&predictions, &actuals);

    // 类0: TP=1, FN=1 → Recall_0 = 0.5
    // 类1: TP=1, FN=1 → Recall_1 = 0.5
    // 类2: 从未在真实标签中出现 → 不参与平均
    // Macro = (0.5 + 0.5) / 2 = 0.5
    assert!((rec - 0.5).abs() < 1e-6, "Recall = {rec}, 期望 0.5");
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

    let rec = recall(&predictions, &actuals);

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
        (rec - expected_macro).abs() < 1e-4,
        "Macro-Recall = {rec}, 期望 ≈ {expected_macro}"
    );

    // 验证 Macro 确实比 Micro 低（暴露小类问题）
    let micro_recall = 10.0 / 12.0;
    assert!(
        rec < micro_recall,
        "Macro ({rec}) 应该 < Micro ({micro_recall})，说明小类漏检问题被暴露"
    );
}

// ==================== F1 Score 测试 ====================

/// F1 基本功能测试（二分类）
#[test]
fn test_f1_score_binary() {
    let predictions = vec![1, 1, 0, 1, 0];
    let actuals = vec![1, 0, 0, 1, 1];

    let f1 = f1_score(&predictions, &actuals);

    // 类0: TP=1, FP=1, FN=1 → P=0.5, R=0.5 → F1_0 = 0.5
    // 类1: TP=2, FP=1, FN=1 → P=2/3, R=2/3 → F1_1 = 2/3
    // Macro = (0.5 + 2/3) / 2 ≈ 0.583
    let f1_0 = 0.5;
    let f1_1 = 2.0 / 3.0;
    let expected = (f1_0 + f1_1) / 2.0;
    assert!((f1 - expected).abs() < 1e-4, "F1 = {f1}, 期望 ≈ {expected}");
}

/// F1 多分类测试
#[test]
fn test_f1_score_multiclass() {
    // 三分类：0=猫, 1=狗, 2=鸟
    let predictions = vec![0, 1, 2, 1, 0];
    let actuals = vec![0, 1, 2, 2, 0];

    let f1 = f1_score(&predictions, &actuals);

    // 类0: P=1.0, R=1.0 → F1_0 = 1.0
    // 类1: P=0.5, R=1.0 → F1_1 = 2*0.5*1.0/(0.5+1.0) = 2/3 ≈ 0.667
    // 类2: P=1.0, R=0.5 → F1_2 = 2*1.0*0.5/(1.0+0.5) = 2/3 ≈ 0.667
    // Macro = (1.0 + 0.667 + 0.667) / 3 ≈ 0.778
    let f1_0 = 1.0;
    let f1_1 = 2.0 * 0.5 * 1.0 / (0.5 + 1.0);
    let f1_2 = 2.0 * 1.0 * 0.5 / (1.0 + 0.5);
    let expected = (f1_0 + f1_1 + f1_2) / 3.0;
    assert!((f1 - expected).abs() < 1e-4, "F1 = {f1}, 期望 ≈ {expected}");
}

/// 完美预测：F1 = 1.0
#[test]
fn test_f1_score_perfect() {
    let labels = vec![0, 1, 2, 0, 1];
    let f1 = f1_score(&labels, &labels);

    assert!(
        (f1 - 1.0).abs() < 1e-6,
        "完美预测应返回 F1 = 1.0，实际 = {f1}"
    );
}

/// 完全错误（二分类互换）：F1 = 0.0
#[test]
fn test_f1_score_all_wrong_binary() {
    let predictions = vec![1, 1, 1, 1];
    let actuals = vec![0, 0, 0, 0];

    let f1 = f1_score(&predictions, &actuals);

    // 类0: TP=0, FP=0, FN=4 → P=0, R=0 → F1_0 = 0
    // 类1: TP=0, FP=4, FN=0 → P=0, R=undefined(0) → F1_1 = 0
    // Macro = 0.0
    assert!(f1.abs() < 1e-6, "全错应返回 F1 = 0.0，实际 = {f1}");
}

/// 边界情况：空输入
#[test]
fn test_f1_score_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert_eq!(f1_score(&empty, &values), 0.0);
    assert_eq!(f1_score(&values, &empty), 0.0);
    assert_eq!(f1_score(&empty, &empty), 0.0);
}

/// 边界情况：单个样本
#[test]
fn test_f1_score_single_sample() {
    // 正确预测
    assert!((f1_score(&[1], &[1]) - 1.0).abs() < 1e-6);

    // 错误预测
    assert!(f1_score(&[1], &[0]).abs() < 1e-6);
}

/// 类别不平衡场景：验证 Macro-F1 暴露小类问题
#[test]
fn test_f1_score_imbalanced() {
    // 大类(0): 10 个样本，全部正确
    // 小类(1): 2 个样本，全部漏掉
    let predictions = vec![0; 12];
    let mut actuals = vec![0; 10];
    actuals.extend(vec![1, 1]);

    let f1 = f1_score(&predictions, &actuals);

    // 类0: TP=10, FP=2, FN=0 → P=10/12, R=1.0 → F1_0 = 2*(10/12)*1/(10/12+1) ≈ 0.909
    // 类1: TP=0, FP=0, FN=2 → P=0, R=0 → F1_1 = 0
    // Macro = (0.909 + 0) / 2 ≈ 0.455
    let p_0 = 10.0 / 12.0;
    let r_0 = 1.0;
    let f1_0 = 2.0 * p_0 * r_0 / (p_0 + r_0);
    let f1_1 = 0.0;
    let expected = (f1_0 + f1_1) / 2.0;

    assert!(
        (f1 - expected).abs() < 1e-4,
        "Macro-F1 = {f1}, 期望 ≈ {expected}"
    );

    // 验证 Macro-F1 确实较低（暴露小类问题）
    // 如果是 Micro-F1 = Accuracy = 10/12 ≈ 0.833
    let micro_f1 = 10.0 / 12.0;
    assert!(
        f1 < micro_f1,
        "Macro-F1 ({f1}) 应该 < Micro-F1 ({micro_f1})，说明小类问题被暴露"
    );
}

/// 验证 F1 是 Precision 和 Recall 的调和平均
#[test]
fn test_f1_is_harmonic_mean() {
    let predictions = vec![0, 1, 1, 0, 1, 0];
    let actuals = vec![0, 0, 1, 1, 1, 0];

    let p = precision(&predictions, &actuals);
    let r = recall(&predictions, &actuals);
    let f1 = f1_score(&predictions, &actuals);

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

// ==================== Tensor 输入测试 ====================

use crate::tensor::Tensor;

/// 测试 Tensor 作为 logits 输入（自动 argmax）
#[test]
fn test_accuracy_tensor_logits() {
    // logits: [3, 2] 形状，3 个样本，2 个类别
    // 预测类别分别为 1, 0, 1
    let logits = Tensor::new(&[0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2]);
    // 真实标签（one-hot）: 1, 0, 1
    let labels = Tensor::new(&[0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[3, 2]);

    let acc = accuracy(&logits, &labels);
    assert!(
        (acc - 1.0).abs() < 1e-6,
        "Tensor logits 完美预测应返回 1.0，实际 = {acc}"
    );
}

/// 测试 Tensor 与 slice 混合输入
#[test]
fn test_accuracy_tensor_mixed_input() {
    // logits: 预测类别 0, 1, 0
    let logits = Tensor::new(&[0.9, 0.1, 0.2, 0.8, 0.7, 0.3], &[3, 2]);
    // 真实标签：slice
    let labels = [0_usize, 1, 0];

    let acc = accuracy(&logits, &labels);
    assert!(
        (acc - 1.0).abs() < 1e-6,
        "Tensor 与 slice 混合完美预测应返回 1.0，实际 = {acc}"
    );
}

/// 测试 1D Tensor 作为类别索引输入
#[test]
fn test_accuracy_tensor_1d_labels() {
    // 1D Tensor：直接作为类别索引
    let preds = Tensor::new(&[0.0, 1.0, 1.0, 0.0, 1.0], &[5]);
    let labels = Tensor::new(&[0.0, 1.0, 0.0, 0.0, 1.0], &[5]);

    let acc = accuracy(&preds, &labels);
    // 4/5 = 0.8
    assert!(
        (acc - 0.8).abs() < 1e-6,
        "1D Tensor 应正确计算 Accuracy，实际 = {acc}"
    );
}

// ==================== Confusion Matrix 测试 ====================

use crate::metrics::confusion_matrix;

/// 基本功能测试（二分类）
#[test]
fn test_confusion_matrix_binary() {
    let predictions = vec![0, 1, 1, 0, 1];
    let actuals = vec![0, 1, 0, 0, 1];

    let cm = confusion_matrix(&predictions, &actuals);

    // 期望:
    // [[2, 1],   // 真实类0：2个预测对(TN)，1个预测错(FP)
    //  [0, 2]]   // 真实类1：0个预测错(FN)，2个预测对(TP)
    assert_eq!(cm.len(), 2, "应该有2个类别");
    assert_eq!(cm[0][0], 2, "TN = 2");
    assert_eq!(cm[0][1], 1, "FP = 1");
    assert_eq!(cm[1][0], 0, "FN = 0");
    assert_eq!(cm[1][1], 2, "TP = 2");
}

/// 多分类测试
#[test]
fn test_confusion_matrix_multiclass() {
    // 三分类：0, 1, 2
    let predictions = vec![0, 1, 2, 1, 0, 2];
    let actuals = vec![0, 1, 2, 2, 0, 1];

    let cm = confusion_matrix(&predictions, &actuals);

    // 期望:
    // [[2, 0, 0],   // 真实类0：2个预测对
    //  [0, 1, 1],   // 真实类1：1个预测对，1个预测成类2
    //  [0, 1, 1]]   // 真实类2：1个预测成类1，1个预测对
    assert_eq!(cm.len(), 3, "应该有3个类别");
    assert_eq!(cm[0], vec![2, 0, 0]);
    assert_eq!(cm[1], vec![0, 1, 1]);
    assert_eq!(cm[2], vec![0, 1, 1]);
}

/// 完美预测：对角线全是正数
#[test]
fn test_confusion_matrix_perfect() {
    let labels = vec![0, 1, 2, 0, 1, 2];
    let cm = confusion_matrix(&labels, &labels);

    // 完美预测，所有非对角线元素为0
    assert_eq!(cm.len(), 3);
    for i in 0..3 {
        for j in 0..3 {
            if i == j {
                assert!(cm[i][j] > 0, "对角线元素应 > 0");
            } else {
                assert_eq!(cm[i][j], 0, "非对角线元素应 = 0");
            }
        }
    }
}

/// 完全错误（二分类互换）
#[test]
fn test_confusion_matrix_all_wrong() {
    let predictions = vec![1, 1, 1, 1];
    let actuals = vec![0, 0, 0, 0];

    let cm = confusion_matrix(&predictions, &actuals);

    // 期望:
    // [[0, 4],   // 真实类0全部预测成类1
    //  [0, 0]]   // 真实类1没有样本
    assert_eq!(cm[0][0], 0, "TN = 0");
    assert_eq!(cm[0][1], 4, "FP = 4");
}

/// 边界情况：空输入
#[test]
fn test_confusion_matrix_empty() {
    let empty: Vec<i32> = vec![];
    let values = vec![0, 1, 2];

    assert!(confusion_matrix(&empty, &values).is_empty());
    assert!(confusion_matrix(&values, &empty).is_empty());
    assert!(confusion_matrix(&empty, &empty).is_empty());
}

/// 边界情况：单个样本
#[test]
fn test_confusion_matrix_single_sample() {
    let cm = confusion_matrix(&[1], &[1]);
    // 只有一个类别1，所以矩阵是 2x2（类别0和1）
    assert_eq!(cm.len(), 2);
    assert_eq!(cm[1][1], 1);
}

/// 类别不连续（只有类0和类2，没有类1）
#[test]
fn test_confusion_matrix_non_contiguous_classes() {
    let predictions = vec![0, 2, 0, 2];
    let actuals = vec![0, 2, 2, 0];

    let cm = confusion_matrix(&predictions, &actuals);

    // 矩阵大小应该是 3x3（包含类0, 1, 2）
    assert_eq!(cm.len(), 3);
    // 类1的行和列应该全是0
    assert_eq!(cm[1], vec![0, 0, 0]);
    for row in &cm {
        assert_eq!(row[1], 0);
    }
}

/// 测试 Tensor 输入（自动 argmax）
#[test]
fn test_confusion_matrix_tensor() {
    // logits: [3, 2] 形状，预测类别 1, 0, 1
    let logits = Tensor::new(&[0.1, 0.9, 0.8, 0.2, 0.3, 0.7], &[3, 2]);
    // 真实标签（one-hot）: 1, 0, 1
    let labels = Tensor::new(&[0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[3, 2]);

    let cm = confusion_matrix(&logits, &labels);

    // 完美预测
    assert_eq!(cm.len(), 2);
    assert_eq!(cm[0][0], 1, "类0预测正确1次");
    assert_eq!(cm[1][1], 2, "类1预测正确2次");
    assert_eq!(cm[0][1], 0, "无 FP");
    assert_eq!(cm[1][0], 0, "无 FN");
}

/// 测试 Tensor 与 slice 混合输入
#[test]
fn test_confusion_matrix_tensor_mixed() {
    // logits: 预测类别 0, 1, 0
    let logits = Tensor::new(&[0.9, 0.1, 0.2, 0.8, 0.7, 0.3], &[3, 2]);
    let labels = [0_usize, 1, 0];

    let cm = confusion_matrix(&logits, &labels);

    // 完美预测
    assert_eq!(cm[0][0], 2);
    assert_eq!(cm[1][1], 1);
}

/// 验证混淆矩阵的行和等于真实类别数量
#[test]
fn test_confusion_matrix_row_sums() {
    let predictions = vec![0, 1, 2, 1, 0, 2, 1, 2, 0];
    let actuals = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    let cm = confusion_matrix(&predictions, &actuals);

    // 每行的和应该等于该真实类别的样本数（每类3个）
    for row in &cm {
        let row_sum: usize = row.iter().sum();
        assert_eq!(row_sum, 3, "每个真实类别应有3个样本");
    }
}

/// 验证混淆矩阵的列和等于预测类别数量
#[test]
fn test_confusion_matrix_col_sums() {
    let predictions = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
    let actuals = vec![0, 1, 2, 0, 1, 2, 0, 1, 2];

    let cm = confusion_matrix(&predictions, &actuals);

    // 每列的和应该等于该预测类别的次数（每类3次）
    for j in 0..3 {
        let col_sum: usize = (0..3).map(|i| cm[i][j]).sum();
        assert_eq!(col_sum, 3, "每个预测类别应出现3次");
    }
}

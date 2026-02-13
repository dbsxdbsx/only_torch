//! Confusion Matrix 相关测试

use crate::metrics::confusion_matrix;
use crate::tensor::Tensor;

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

/*
 * @Author       : 老董
 * @Date         : 2026-01-29
 * @Description  : BCE（Binary Cross Entropy）节点单元测试
 *
 * BCE 采用 BCEWithLogitsLoss 形式（内置 Sigmoid），数值稳定。
 * 适用于二分类和多标签分类任务。
 */

use crate::nn::{GraphError, GraphInner, Reduction};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ========== 基本功能测试 ==========

#[cfg(any())]
#[test]
fn test_bce_loss_creation() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    // 验证节点存在且预期形状正确
    assert_eq!(
        graph.get_node_value_expected_shape(loss_id).unwrap(),
        &[1, 1]
    );
}

#[cfg(any())]
#[test]
fn test_bce_loss_shape_mismatch() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 4], Some("target")).unwrap(); // 形状不匹配

    let result = graph.new_bce_loss_node(logits_id, target_id, None);
    assert!(result.is_err());
}

// ========== Mean Reduction 测试 ==========

/// PyTorch 验证:
/// ```python
/// import torch
/// import torch.nn as nn
///
/// logits = torch.tensor([[0.5, -0.5, 1.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 0.0, 1.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 0.4204719067
/// ```
#[cfg(any())]
#[test]
fn test_bce_loss_forward_mean_basic() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    // 设置输入值
    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    // 前向传播
    graph.forward(loss_id).unwrap();

    // 验证损失值（PyTorch: 0.4204719067）
    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    assert_abs_diff_eq!(loss[[0, 0]], 0.420_471_9, epsilon = 1e-5);
}

/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[1.0, 2.0], [-1.0, -2.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 0.2200948894
/// ```
#[cfg(any())]
#[test]
fn test_bce_loss_forward_2d_matrix() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[2, 2], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[1.0, 2.0, -1.0, -2.0], &[2, 2])))
        .unwrap();
    #[rustfmt::skip]
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[2, 2])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 0.2200948894
    assert_abs_diff_eq!(loss[[0, 0]], 0.220_094_9, epsilon = 1e-5);
}

// ========== Sum Reduction 测试 ==========

/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[0.5, -0.5, 1.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 0.0, 1.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='sum')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 1.2614157200
/// ```
#[cfg(any())]
#[test]
fn test_bce_loss_forward_sum() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node_with_reduction(logits_id, target_id, Reduction::Sum, Some("loss"))
        .unwrap();

    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 1.2614157200
    assert_abs_diff_eq!(loss[[0, 0]], 1.261_415_7, epsilon = 1e-5);
}

// ========== 端到端反向传播测试（通过 graph.backward）==========

/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[0.5, -0.5, 1.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 0.0, 1.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// loss.backward()
/// print(f"grad = {logits.grad}")
/// # grad = tensor([[-0.1258,  0.1258, -0.0896]])
/// # 梯度公式: (sigmoid(logits) - target) / N
/// ```
#[cfg(any())]
#[test]
fn test_bce_loss_backward_e2e_mean() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    // 获取梯度
    let grad = graph.get_node(logits_id).unwrap().grad().unwrap();

    // 预期梯度: (sigmoid(logits) - target) / N
    // sigmoid([0.5, -0.5, 1.0]) ≈ [0.6225, 0.3775, 0.7311]
    // (sigmoid - target) = [-0.3775, 0.3775, -0.2689]
    // / 3 = [-0.1258, 0.1258, -0.0896]
    let expected_grad = Tensor::new(&[-0.125_846_9, 0.125_846_9, -0.089_647_1], &[1, 3]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[0.5, -0.5, 1.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 0.0, 1.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='sum')(logits, target)
/// loss.backward()
/// print(f"grad = {logits.grad}")
/// # grad = tensor([[-0.3775,  0.3775, -0.2689]])
/// ```
#[cfg(any())]
#[test]
fn test_bce_loss_backward_e2e_sum() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node_with_reduction(logits_id, target_id, Reduction::Sum, Some("loss"))
        .unwrap();

    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(logits_id).unwrap().grad().unwrap();
    // sigmoid - target（不除以 N）
    let expected_grad = Tensor::new(&[-0.377_540_7, 0.377_540_7, -0.268_941_4], &[1, 3]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

/// 测试 2D 矩阵的梯度
/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([[1.0, 2.0], [-1.0, -2.0]], requires_grad=True)
/// target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// loss.backward()
/// print(f"grad = {logits.grad}")
/// # grad = tensor([[-0.0672, -0.0298],
/// #                [ 0.0672,  0.0298]])
/// ```
#[cfg(any())]
#[test]
fn test_bce_loss_backward_e2e_2d() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[2, 2], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[2, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[1.0, 2.0, -1.0, -2.0], &[2, 2])))
        .unwrap();
    #[rustfmt::skip]
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 1.0, 0.0, 0.0], &[2, 2])))
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(logits_id).unwrap().grad().unwrap();
    // PyTorch 验证的梯度值
    #[rustfmt::skip]
    let expected_grad = Tensor::new(&[
        -0.067_235_4, -0.029_800_7,
         0.067_235_4,  0.029_800_7,
    ], &[2, 2]);

    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

// ========== 批量输入测试 ==========

/// PyTorch 验证:
/// ```python
/// logits = torch.tensor([
///     [0.5, -0.5, 1.0, -1.0],
///     [1.5, -1.5, 2.0, -2.0],
///     [0.0, 0.0, 0.5, -0.5]
/// ], requires_grad=True)
/// target = torch.tensor([
///     [1.0, 0.0, 1.0, 0.0],
///     [1.0, 0.0, 1.0, 0.0],
///     [0.0, 1.0, 0.0, 1.0]
/// ])
/// loss = nn.BCEWithLogitsLoss(reduction='mean')(logits, target)
/// print(f"loss = {loss.item()}")  # loss = 0.4638172686
/// ```
#[cfg(any())]
#[test]
fn test_bce_loss_batch_forward() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[3, 4], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[3, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    let logits_data = Tensor::new(&[
        0.5, -0.5, 1.0, -1.0,
        1.5, -1.5, 2.0, -2.0,
        0.0,  0.0, 0.5, -0.5,
    ], &[3, 4]);

    #[rustfmt::skip]
    let target_data = Tensor::new(&[
        1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
    ], &[3, 4]);

    graph.set_node_value(logits_id, Some(&logits_data)).unwrap();
    graph.set_node_value(target_id, Some(&target_data)).unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 0.4638172686
    assert_abs_diff_eq!(loss[[0, 0]], 0.463_817_3, epsilon = 1e-5);
}

/// 批量输入的反向传播测试
#[cfg(any())]
#[test]
fn test_bce_loss_batch_backward() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[3, 4], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[3, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    let logits_data = Tensor::new(&[
        0.5, -0.5, 1.0, -1.0,
        1.5, -1.5, 2.0, -2.0,
        0.0,  0.0, 0.5, -0.5,
    ], &[3, 4]);

    #[rustfmt::skip]
    let target_data = Tensor::new(&[
        1.0, 0.0, 1.0, 0.0,
        1.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 1.0,
    ], &[3, 4]);

    graph.set_node_value(logits_id, Some(&logits_data)).unwrap();
    graph.set_node_value(target_id, Some(&target_data)).unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(logits_id).unwrap().grad().unwrap();

    // PyTorch 验证的梯度值（sigmoid - target）/ 12
    // 确保梯度形状正确
    assert_eq!(grad.shape(), &[3, 4]);
    // 梯度值应该在合理范围内
    for &val in grad.flatten_view().iter() {
        assert!(val.abs() < 1.0, "梯度值应该在合理范围内");
    }
}

// ========== 数值稳定性测试 ==========

/// 测试大正数 logits 的数值稳定性
/// 大正数 logits 时 sigmoid ≈ 1，BCE 应该趋近于 0（当 target=1）
#[cfg(any())]
#[test]
fn test_bce_loss_large_positive_logits() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    // 大正数 logits
    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[10.0, 20.0, 30.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 1.0, 1.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 当 target=1 且 logits 很大时，BCE ≈ 0
    assert!(
        loss[[0, 0]] < 1e-4,
        "大正数 logits 配合 target=1 时损失应接近 0"
    );
    assert!(loss[[0, 0]] >= 0.0, "损失不应为负");
}

/// 测试大负数 logits 的数值稳定性
/// 大负数 logits 时 sigmoid ≈ 0，BCE 应该趋近于 0（当 target=0）
#[cfg(any())]
#[test]
fn test_bce_loss_large_negative_logits() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    // 大负数 logits
    graph
        .set_node_value(
            logits_id,
            Some(&Tensor::new(&[-10.0, -20.0, -30.0], &[1, 3])),
        )
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 当 target=0 且 logits 很小时，BCE ≈ 0
    assert!(
        loss[[0, 0]] < 1e-4,
        "大负数 logits 配合 target=0 时损失应接近 0"
    );
    assert!(loss[[0, 0]] >= 0.0, "损失不应为负");
}

/// 测试错误预测时的高损失
#[cfg(any())]
#[test]
fn test_bce_loss_wrong_prediction_high_loss() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_basic_input_node(&[1, 2], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    // 大正数预测 0，大负数预测 1（完全错误）
    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[10.0, -10.0], &[1, 2])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[0.0, 1.0], &[1, 2])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 错误预测时损失应该很大
    assert!(loss[[0, 0]] > 5.0, "错误预测时损失应该很大");
}

// ========== 梯度累积测试 ==========

#[cfg(any())]
#[test]
fn test_bce_loss_gradient_accumulation() {
    let mut graph = GraphInner::new();

    let logits_id = graph.new_parameter_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[0.5, -0.5, 1.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 0.0, 1.0], &[1, 3])))
        .unwrap();

    // 第一次前向+反向
    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad_first = graph.get_node(logits_id).unwrap().grad().unwrap().clone();

    // 第二次反向传播（梯度累积）
    graph.forward(loss_id).unwrap();
    graph.backward(loss_id).unwrap();
    let grad_second = graph.get_node(logits_id).unwrap().grad().unwrap();
    assert_abs_diff_eq!(grad_second, &(&grad_first * 2.0), epsilon = 1e-6);

    // zero_grad 后重新计算
    graph.zero_grad().unwrap();
    graph.forward(loss_id).unwrap();
    graph.backward(loss_id).unwrap();
    let grad_after_clear = graph.get_node(logits_id).unwrap().grad().unwrap();
    assert_abs_diff_eq!(grad_after_clear, &grad_first, epsilon = 1e-6);
}

// ========== 端到端训练测试 ==========

/// 简单的二分类训练测试
/// 目标: 学习一个线性分类器 y = sigmoid(w * x)
#[cfg(any())]
#[test]
fn test_bce_loss_simple_binary_classification_training() {
    let mut graph = GraphInner::new_with_seed(42);

    // 创建网络: logit = x * w, loss = BCE(logit, target)
    let x_id = graph.new_basic_input_node(&[1, 1], Some("x")).unwrap();
    let w_id = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let logit_id = graph.new_mat_mul_node(x_id, w_id, Some("logit")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 1], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logit_id, target_id, Some("loss"))
        .unwrap();

    // 初始化权重为 0
    graph
        .set_node_value(w_id, Some(&Tensor::new(&[0.0], &[1, 1])))
        .unwrap();

    let lr = 0.5;

    // 训练数据: x > 0 -> y = 1, x < 0 -> y = 0
    let training_data = [
        (1.0_f32, 1.0_f32),
        (2.0, 1.0),
        (3.0, 1.0),
        (-1.0, 0.0),
        (-2.0, 0.0),
        (-3.0, 0.0),
    ];

    // 训练 100 个 epoch
    for _ in 0..100 {
        for &(x_val, y_val) in &training_data {
            graph
                .set_node_value(x_id, Some(&Tensor::new(&[x_val], &[1, 1])))
                .unwrap();
            graph
                .set_node_value(target_id, Some(&Tensor::new(&[y_val], &[1, 1])))
                .unwrap();

            graph.zero_grad().unwrap();
            graph.forward(loss_id).unwrap();
            graph.backward(loss_id).unwrap();

            // 手动 SGD 更新
            let w_val = graph.get_node_value(w_id).unwrap().unwrap();
            let w_grad = graph.get_node_grad(w_id).unwrap().unwrap();
            let new_w = w_val - lr * &w_grad;
            graph.set_node_value(w_id, Some(&new_w)).unwrap();
        }
    }

    // 验证学习到的权重为正数（正样本 x>0 应该预测为 1）
    let learned_w = graph.get_node_value(w_id).unwrap().unwrap();
    assert!(
        learned_w[[0, 0]] > 0.5,
        "学习到的权重应为正数（实际: {}）",
        learned_w[[0, 0]]
    );
}

// ==================== 多标签分类测试 ====================

/// 测试多标签分类场景
/// 这是 BCE 相对于 Softmax CE 的核心优势
#[cfg(any())]
#[test]
fn test_bce_loss_multi_label_classification() {
    let mut graph = GraphInner::new();

    // 多标签场景：3 个独立的二分类标签
    let logits_id = graph.new_basic_input_node(&[1, 3], Some("logits")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_bce_loss_node(logits_id, target_id, Some("loss"))
        .unwrap();

    // 预测: [高概率, 高概率, 低概率]
    // 标签: [1, 1, 0] - 同时属于前两个类别
    graph
        .set_node_value(logits_id, Some(&Tensor::new(&[2.0, 2.0, -2.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[1.0, 1.0, 0.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 预测正确，损失应该很小
    assert!(loss[[0, 0]] < 0.2, "正确预测时损失应该很小");
}

// ==================== 动态形状测试 ====================

/// 测试 BCE 节点的动态形状（输出固定为标量 [1, 1]）
#[cfg(any())]
#[test]
fn test_bce_loss_dynamic_shape_output_fixed() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let logits = graph.new_basic_input_node(&[4, 8], Some("logits"))?;
    let target = graph.new_basic_input_node(&[4, 8], Some("target"))?;
    let loss = graph.new_bce_loss_node(logits, target, Some("loss"))?;

    // BCE 输出形状始终是 [1, 1]（标量）
    let node = graph.get_node(loss)?;
    let dyn_shape = node.dynamic_expected_shape();

    // 输出形状固定
    assert!(!dyn_shape.is_dynamic(0), "BCE 输出维度 0 应固定");
    assert!(!dyn_shape.is_dynamic(1), "BCE 输出维度 1 应固定");
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));

    Ok(())
}

/// 测试 BCE 接受动态 batch 输入
#[cfg(any())]
#[test]
fn test_bce_loss_dynamic_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let logits = graph.new_basic_input_node(&[2, 4], Some("logits"))?;
    let target = graph.new_basic_input_node(&[2, 4], Some("target"))?;
    let loss = graph.new_bce_loss_node(logits, target, Some("loss"))?;

    // 第一次 forward：batch=2
    graph.set_node_value(logits, Some(&Tensor::ones(&[2, 4])))?;
    graph.set_node_value(target, Some(&Tensor::ones(&[2, 4])))?;
    graph.forward(loss)?;
    let loss_val1 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(loss_val1.shape(), &[1, 1], "BCE 输出应为标量");

    // 第二次 forward：batch=6（不同 batch 大小）
    graph.set_node_value(logits, Some(&Tensor::ones(&[6, 4])))?;
    graph.set_node_value(target, Some(&Tensor::ones(&[6, 4])))?;
    graph.forward(loss)?;
    let loss_val2 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(loss_val2.shape(), &[1, 1], "BCE 输出应始终为标量");

    Ok(())
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use crate::nn::Graph;
use std::rc::Rc;

#[test]
fn test_create_bce_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("logits"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("target"))
        .unwrap();

    let bce = inner
        .borrow_mut()
        .create_bce_mean_node(logits.clone(), target.clone(), Some("bce"))
        .unwrap();

    // BCE 输出形状固定为 [1, 1]
    assert_eq!(bce.shape(), vec![1, 1]);
    assert_eq!(bce.name(), Some("bce"));
    assert!(!bce.is_leaf());
    assert_eq!(bce.parents().len(), 2);
}

#[test]
fn test_create_bce_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None) // 形状不匹配
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_bce_mean_node(logits, target, None);

    assert!(result.is_err());
}

#[test]
fn test_create_bce_node_output_always_scalar() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let logits = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[5, 10], None)
        .unwrap();
    let bce = inner
        .borrow_mut()
        .create_bce_mean_node(logits, target, None)
        .unwrap();
    assert_eq!(bce.shape(), vec![1, 1]);
}

#[test]
fn test_create_bce_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_bce;
    let weak_logits;
    let weak_target;
    {
        let logits = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_logits = Rc::downgrade(&logits);

        let target = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_target = Rc::downgrade(&target);

        let bce = inner
            .borrow_mut()
            .create_bce_mean_node(logits, target, None)
            .unwrap();
        weak_bce = Rc::downgrade(&bce);

        assert!(weak_bce.upgrade().is_some());
        assert!(weak_logits.upgrade().is_some());
        assert!(weak_target.upgrade().is_some());
    }
    assert!(weak_bce.upgrade().is_none());
    assert!(weak_logits.upgrade().is_none());
    assert!(weak_target.upgrade().is_none());
}

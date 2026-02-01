/*
 * @Author       : 老董
 * @Date         : 2026-01-29
 * @Description  : Huber Loss（Smooth L1 Loss）节点单元测试
 *
 * Huber Loss 结合 MSE（小误差）和 MAE（大误差）的优点，
 * 是强化学习（DQN 等）的标准损失函数。
 */

use crate::nn::{GraphError, GraphInner, Reduction};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ========== 基本功能测试 ==========

#[test]
fn test_huber_loss_creation() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    // 验证节点存在且预期形状正确
    assert_eq!(
        graph.get_node_value_expected_shape(loss_id).unwrap(),
        &[1, 1]
    );
}

#[test]
fn test_huber_loss_shape_mismatch() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 4], Some("target")).unwrap(); // 形状不匹配

    let result = graph.new_huber_loss_node(input_id, target_id, None);
    assert!(result.is_err());
}

#[test]
fn test_huber_loss_invalid_delta() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();

    // δ=0 应该失败
    let result = graph.new_huber_loss_node_with_delta(input_id, target_id, 0.0, None);
    assert!(result.is_err());

    // δ<0 应该失败
    let result = graph.new_huber_loss_node_with_delta(input_id, target_id, -1.0, None);
    assert!(result.is_err());
}

// ========== 小误差测试（MSE 行为）==========

/// 当 |error| ≤ δ 时，Huber Loss 行为像 MSE
///
/// PyTorch 验证:
/// ```python
/// import torch
/// import torch.nn.functional as F
///
/// input = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
/// target = torch.tensor([[0.15, 0.25, 0.35]])  # diff = [-0.05, -0.05, -0.05]
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// # loss = 0.00125 (与 MSE/2 相同：0.5 * mean(0.05^2) = 0.5 * 0.0025 = 0.00125)
/// ```
#[test]
fn test_huber_loss_small_error_mse_behavior() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    // 小误差 (|diff| = 0.05 < δ=1.0)
    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.1, 0.2, 0.3], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[0.15, 0.25, 0.35], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // Huber(小误差) = 0.5 * diff² = 0.5 * 0.0025 = 0.00125 (每个元素)
    // mean = 0.00125
    assert_abs_diff_eq!(loss[[0, 0]], 0.00125, epsilon = 1e-6);
}

// ========== 大误差测试（MAE 行为）==========

/// 当 |error| > δ 时，Huber Loss 行为像 MAE
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 3.0, 4.0]])  # diff = [-2, -3, -4], |diff| > 1
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// # 每个元素: δ * |a| - 0.5 * δ² = 1 * |diff| - 0.5
/// # = (2-0.5) + (3-0.5) + (4-0.5) = 1.5 + 2.5 + 3.5 = 7.5
/// # mean = 7.5 / 3 = 2.5
/// ```
#[test]
fn test_huber_loss_large_error_mae_behavior() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    // 大误差 (|diff| = 2, 3, 4 > δ=1.0)
    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // Huber(大误差) = δ * |a| - 0.5 * δ² = |a| - 0.5 (当 δ=1)
    // = (2-0.5) + (3-0.5) + (4-0.5) = 7.5
    // mean = 7.5 / 3 = 2.5
    assert_abs_diff_eq!(loss[[0, 0]], 2.5, epsilon = 1e-6);
}

// ========== 混合误差测试 ==========

/// 同时包含小误差和大误差的情况
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// # diff = [-0.5, -1.0, -1.5, -2.0]
/// # |diff| <= 1: 0.5, 1.0 -> 0.5 * 0.25, 0.5 * 1.0 = 0.125, 0.5
/// # |diff| > 1: 1.5, 2.0 -> 1.5 - 0.5, 2.0 - 0.5 = 1.0, 1.5
/// # sum = 0.125 + 0.5 + 1.0 + 1.5 = 3.125
/// # mean = 3.125 / 4 = 0.78125
/// ```
#[test]
fn test_huber_loss_mixed_errors() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 4], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 4])))
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[0.5, 1.0, 1.5, 2.0], &[1, 4])),
        )
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // PyTorch: 0.78125
    assert_abs_diff_eq!(loss[[0, 0]], 0.78125, epsilon = 1e-6);
}

// ========== Sum Reduction 测试 ==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 3.0, 4.0]])
/// loss = F.smooth_l1_loss(input, target, reduction='sum', beta=1.0)
/// # sum = 7.5
/// ```
#[test]
fn test_huber_loss_sum_reduction() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node_with_params(input_id, target_id, Reduction::Sum, 1.0, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    assert_abs_diff_eq!(loss[[0, 0]], 7.5, epsilon = 1e-6);
}

// ========== 自定义 δ 测试 ==========

/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[0.3, 1.0]])  # diff = [-0.3, -1.0]
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=0.5)
/// # |0.3| <= 0.5: 0.5 * 0.09 = 0.045
/// # |1.0| > 0.5: 0.5 * 1.0 - 0.5 * 0.25 = 0.5 - 0.125 = 0.375
/// # mean = (0.045 + 0.375) / 2 = 0.21
/// ```
#[test]
fn test_huber_loss_custom_delta() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 2], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 2], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node_with_delta(input_id, target_id, 0.5, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0], &[1, 2])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[0.3, 1.0], &[1, 2])))
        .unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 手动计算:
    // |0.3| <= 0.5: 0.5 * 0.3² = 0.045
    // |1.0| > 0.5: 0.5 * 1.0 - 0.5 * 0.25 = 0.375
    // mean = (0.045 + 0.375) / 2 = 0.21
    assert_abs_diff_eq!(loss[[0, 0]], 0.21, epsilon = 1e-6);
}

// ========== 反向传播测试（小误差）==========

/// 小误差时梯度与 MSE 相同
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
/// target = torch.tensor([[0.15, 0.25, 0.35]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// loss.backward()
/// # grad = diff / N = [-0.05, -0.05, -0.05] / 3 ≈ [-0.0167, -0.0167, -0.0167]
/// ```
#[test]
fn test_huber_loss_backward_small_error() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.1, 0.2, 0.3], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[0.15, 0.25, 0.35], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();

    // 小误差时梯度 = diff / N = (input - target) / N
    // diff = [-0.05, -0.05, -0.05], N = 3
    // grad = [-0.0167, -0.0167, -0.0167]
    let expected_grad = Tensor::new(&[-0.016_666_67, -0.016_666_67, -0.016_666_67], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

// ========== 反向传播测试（大误差）==========

/// 大误差时梯度被"裁剪"到 ±δ/N
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 3.0, 4.0]])  # 全部 > δ
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// loss.backward()
/// # grad = δ * sign(diff) / N = 1 * [-1, -1, -1] / 3 ≈ [-0.333, -0.333, -0.333]
/// ```
#[test]
fn test_huber_loss_backward_large_error() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();

    // 大误差时梯度 = δ * sign(diff) / N = 1 * sign(input - target) / 3
    // diff = [0-2, 0-3, 0-4] = [-2, -3, -4], sign = [-1, -1, -1]
    // grad = [-1, -1, -1] / 3 ≈ [-0.333, -0.333, -0.333]
    let expected_grad = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

// ========== 反向传播测试（混合误差）==========

/// 混合误差时梯度根据 |diff| 分段计算
#[test]
fn test_huber_loss_backward_mixed_error() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 4], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 4])))
        .unwrap();
    graph
        .set_node_value(
            target_id,
            Some(&Tensor::new(&[0.5, 1.0, 1.5, 2.0], &[1, 4])),
        )
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();

    // diff = [-0.5, -1.0, -1.5, -2.0]
    // |diff| <= 1: grad = diff = [-0.5, -1.0]
    // |diff| > 1: grad = δ * sign(diff) = 1 * [-1, -1]
    // 未归一化: [-0.5, -1.0, -1.0, -1.0]
    // Mean reduction: / 4 = [-0.125, -0.25, -0.25, -0.25]
    let expected_grad = Tensor::new(&[-0.125, -0.25, -0.25, -0.25], &[1, 4]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-5);
}

// ========== Sum Reduction 反向传播 ==========

#[test]
fn test_huber_loss_backward_sum() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node_with_params(input_id, target_id, Reduction::Sum, 1.0, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();

    // Sum reduction: 不除以 N
    // grad = δ * sign(diff) = [-1, -1, -1]
    let expected_grad = Tensor::new(&[-1.0, -1.0, -1.0], &[1, 3]);
    assert_abs_diff_eq!(grad, &expected_grad, epsilon = 1e-6);
}

// ========== 批量输入测试 ==========

#[test]
fn test_huber_loss_batch_forward() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[3, 4], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[3, 4], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    let input_data = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0,
        0.5, 1.5, 2.5, 3.5,
        2.0, 3.0, 4.0, 5.0,
    ], &[3, 4]);

    #[rustfmt::skip]
    let target_data = Tensor::new(&[
        1.2, 2.1, 2.9, 4.1,
        0.6, 1.4, 2.6, 3.4,
        1.9, 3.1, 4.0, 5.2,
    ], &[3, 4]);

    graph.set_node_value(input_id, Some(&input_data)).unwrap();
    graph.set_node_value(target_id, Some(&target_data)).unwrap();

    graph.forward(loss_id).unwrap();

    let loss = graph.get_node_value(loss_id).unwrap().unwrap();
    // 所有 |diff| < 1，所以使用 MSE 公式: 0.5 * diff²
    // 与 MSE 测试类似，但乘以 0.5
    // MSE = 0.014166653, Huber = 0.5 * MSE_unnormalized / N
    assert!(loss[[0, 0]] > 0.0);
    assert!(loss[[0, 0]] < 0.1); // 小误差，损失应该较小
}

#[test]
fn test_huber_loss_batch_backward() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[2, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[2, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    #[rustfmt::skip]
    let input_data = Tensor::new(&[
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ], &[2, 3]);

    #[rustfmt::skip]
    let target_data = Tensor::new(&[
        0.5, 1.5, 2.5,  // 小、大、大
        0.3, 1.0, 2.0,  // 小、边界、大
    ], &[2, 3]);

    graph.set_node_value(input_id, Some(&input_data)).unwrap();
    graph.set_node_value(target_id, Some(&target_data)).unwrap();

    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad = graph.get_node(input_id).unwrap().grad().unwrap();
    assert_eq!(grad.shape(), &[2, 3]);

    // 验证梯度形状和方向正确
    // 所有 diff 为负（input < target），所以所有梯度应为负
    for &g in grad.flatten_view().iter() {
        assert!(g <= 0.0, "梯度应为负（input < target）");
    }
}

// ========== 梯度累积测试 ==========

#[test]
fn test_huber_loss_gradient_accumulation() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_parameter_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(input_id, target_id, Some("loss"))
        .unwrap();

    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();

    // 第一次前向+反向
    graph.forward(loss_id).unwrap();
    graph.zero_grad().unwrap();
    graph.backward(loss_id).unwrap();

    let grad_first = graph.get_node(input_id).unwrap().grad().unwrap().clone();

    // 第二次反向传播（梯度累积）
    graph.forward(loss_id).unwrap();
    graph.backward(loss_id).unwrap();
    let grad_second = graph.get_node(input_id).unwrap().grad().unwrap();
    assert_eq!(grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad().unwrap();
    graph.forward(loss_id).unwrap();
    graph.backward(loss_id).unwrap();
    let grad_after_clear = graph.get_node(input_id).unwrap().grad().unwrap();
    assert_eq!(grad_after_clear, &grad_first);
}

// ========== 端到端训练测试 ==========

/// 简单的回归训练测试
/// 验证 Huber Loss 能正确训练模型
#[test]
fn test_huber_loss_simple_regression_training() {
    let mut graph = GraphInner::new_with_seed(42);

    // 创建网络: y_pred = x * w
    let x_id = graph.new_basic_input_node(&[1, 1], Some("x")).unwrap();
    let w_id = graph.new_parameter_node(&[1, 1], Some("w")).unwrap();
    let y_pred_id = graph.new_mat_mul_node(x_id, w_id, Some("y_pred")).unwrap();
    let y_true_id = graph.new_basic_input_node(&[1, 1], Some("y_true")).unwrap();
    let loss_id = graph
        .new_huber_loss_node(y_pred_id, y_true_id, Some("loss"))
        .unwrap();

    // 初始化权重为 0.5（目标是 2.0）
    graph
        .set_node_value(w_id, Some(&Tensor::new(&[0.5], &[1, 1])))
        .unwrap();

    let lr = 0.1;

    // 训练数据: x=1 -> y=2, x=2 -> y=4, x=3 -> y=6
    let training_data = [(1.0_f32, 2.0_f32), (2.0, 4.0), (3.0, 6.0)];

    // 训练 50 个 epoch
    for _ in 0..50 {
        for &(x_val, y_val) in &training_data {
            graph
                .set_node_value(x_id, Some(&Tensor::new(&[x_val], &[1, 1])))
                .unwrap();
            graph
                .set_node_value(y_true_id, Some(&Tensor::new(&[y_val], &[1, 1])))
                .unwrap();

            graph.zero_grad().unwrap();
            graph.forward(loss_id).unwrap();
            graph.backward(loss_id).unwrap();

            // 手动 SGD 更新：w = w - lr * grad
            let w_val = graph.get_node_value(w_id).unwrap().unwrap();
            let w_grad = graph.get_node_grad(w_id).unwrap().unwrap();
            let new_w = w_val - lr * &w_grad;
            graph.set_node_value(w_id, Some(&new_w)).unwrap();
        }
    }

    // 验证学习到的权重接近 2.0
    let learned_w = graph.get_node_value(w_id).unwrap().unwrap();
    assert_abs_diff_eq!(learned_w[[0, 0]], 2.0, epsilon = 0.1);
}

// ========== 动态形状测试 ==========

/// 测试 Huber Loss 节点的动态形状（输出固定为标量 [1, 1]）
#[test]
fn test_huber_loss_dynamic_shape_output_fixed() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[4, 8], Some("input"))?;
    let target = graph.new_basic_input_node(&[4, 8], Some("target"))?;
    let loss = graph.new_huber_loss_node(input, target, Some("loss"))?;

    // Huber 输出形状始终是 [1, 1]（标量）
    let node = graph.get_node(loss)?;
    let dyn_shape = node.dynamic_expected_shape();

    // 输出形状固定
    assert!(!dyn_shape.is_dynamic(0), "Huber 输出维度 0 应固定");
    assert!(!dyn_shape.is_dynamic(1), "Huber 输出维度 1 应固定");
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));

    Ok(())
}

/// 测试 Huber Loss 接受动态 batch 输入
#[test]
fn test_huber_loss_dynamic_batch_forward() -> Result<(), GraphError> {
    let mut graph = GraphInner::new();

    let input = graph.new_basic_input_node(&[2, 4], Some("input"))?;
    let target = graph.new_basic_input_node(&[2, 4], Some("target"))?;
    let loss = graph.new_huber_loss_node(input, target, Some("loss"))?;

    // 第一次 forward：batch=2
    graph.set_node_value(input, Some(&Tensor::ones(&[2, 4])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[2, 4])))?;
    graph.forward(loss)?;
    let loss_val1 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(loss_val1.shape(), &[1, 1], "Huber 输出应为标量");
    assert!(loss_val1[[0, 0]] > 0.0);

    // 第二次 forward：batch=6（不同 batch 大小）
    graph.set_node_value(input, Some(&Tensor::ones(&[6, 4])))?;
    graph.set_node_value(target, Some(&Tensor::zeros(&[6, 4])))?;
    graph.forward(loss)?;
    let loss_val2 = graph.get_node_value(loss)?.unwrap();
    assert_eq!(loss_val2.shape(), &[1, 1], "Huber 输出应始终为标量");

    Ok(())
}

// ========== 与 MSE/MAE 的对比测试 ==========

/// 验证小误差时 Huber ≈ 0.5 * MSE
#[test]
fn test_huber_vs_mse_small_error() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();

    let mse_id = graph
        .new_mse_loss_node(input_id, target_id, Some("mse"))
        .unwrap();
    let huber_id = graph
        .new_huber_loss_node(input_id, target_id, Some("huber"))
        .unwrap();

    // 小误差
    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.1, 0.2, 0.3], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[0.15, 0.25, 0.35], &[1, 3])))
        .unwrap();

    graph.forward(mse_id).unwrap();
    graph.forward(huber_id).unwrap();

    let mse_loss = graph.get_node_value(mse_id).unwrap().unwrap()[[0, 0]];
    let huber_loss = graph.get_node_value(huber_id).unwrap().unwrap()[[0, 0]];

    // Huber(小误差) = 0.5 * MSE
    assert_abs_diff_eq!(huber_loss, 0.5 * mse_loss, epsilon = 1e-6);
}

/// 验证大误差时 Huber ≈ MAE - 0.5*δ（当 δ=1）
#[test]
fn test_huber_vs_mae_large_error() {
    let mut graph = GraphInner::new();

    let input_id = graph.new_basic_input_node(&[1, 3], Some("input")).unwrap();
    let target_id = graph.new_basic_input_node(&[1, 3], Some("target")).unwrap();

    let mae_id = graph
        .new_mae_loss_node(input_id, target_id, Some("mae"))
        .unwrap();
    let huber_id = graph
        .new_huber_loss_node(input_id, target_id, Some("huber"))
        .unwrap();

    // 大误差
    graph
        .set_node_value(input_id, Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    graph
        .set_node_value(target_id, Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();

    graph.forward(mae_id).unwrap();
    graph.forward(huber_id).unwrap();

    let mae_loss = graph.get_node_value(mae_id).unwrap().unwrap()[[0, 0]];
    let huber_loss = graph.get_node_value(huber_id).unwrap().unwrap()[[0, 0]];

    // Huber(大误差) = MAE - 0.5*δ = MAE - 0.5（当 δ=1）
    assert_abs_diff_eq!(huber_loss, mae_loss - 0.5, epsilon = 1e-6);
}

// ==================== 方案 C：新节点创建 API 测试 ====================

use crate::nn::Graph;
use std::rc::Rc;

#[test]
fn test_create_huber_node() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], Some("target"))
        .unwrap();

    let huber = inner
        .borrow_mut()
        .create_huber_default_node(input.clone(), target.clone(), Some("huber"))
        .unwrap();

    // Huber 输出形状固定为 [1, 1]
    assert_eq!(huber.shape(), vec![1, 1]);
    assert_eq!(huber.name(), Some("huber"));
    assert!(!huber.is_leaf());
    assert_eq!(huber.parents().len(), 2);
}

#[test]
fn test_create_huber_node_custom_delta() {
    use crate::nn::nodes::raw_node::Reduction;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    // 自定义 delta = 0.5
    let huber = inner
        .borrow_mut()
        .create_huber_node(input, target, Reduction::Mean, 0.5, None)
        .unwrap();

    assert_eq!(huber.shape(), vec![1, 1]);
}

#[test]
fn test_create_huber_node_invalid_delta() {
    use crate::nn::nodes::raw_node::Reduction;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();

    // delta <= 0 应该失败
    let result = inner
        .borrow_mut()
        .create_huber_node(input, target, Reduction::Mean, -1.0, None);

    assert!(result.is_err());
}

#[test]
fn test_create_huber_node_shape_mismatch() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 3], None)
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], None) // 形状不匹配
        .unwrap();

    let result = inner
        .borrow_mut()
        .create_huber_default_node(input, target, None);

    assert!(result.is_err());
}

#[test]
fn test_create_huber_node_drop_releases() {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let weak_huber;
    let weak_input;
    let weak_target;
    {
        let input = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_input = Rc::downgrade(&input);

        let target = inner
            .borrow_mut()
            .create_basic_input_node(&[2, 3], None)
            .unwrap();
        weak_target = Rc::downgrade(&target);

        let huber = inner
            .borrow_mut()
            .create_huber_default_node(input, target, None)
            .unwrap();
        weak_huber = Rc::downgrade(&huber);

        assert!(weak_huber.upgrade().is_some());
        assert!(weak_input.upgrade().is_some());
        assert!(weak_target.upgrade().is_some());
    }
    assert!(weak_huber.upgrade().is_none());
    assert!(weak_input.upgrade().is_none());
    assert!(weak_target.upgrade().is_none());
}

/*
 * @Author       : 老董
 * @Description  : Huber Loss（Smooth L1 Loss）损失节点单元测试
 *
 * 测试策略：
 * 1. 前向传播测试（高层 API）→ 小误差 MSE 行为; 大误差 MAE 行为; 混合; cannot_set_value
 * 2. VJP 单元测试（底层）→ 小误差 VJP; 大误差 VJP; 混合 VJP; Sum VJP; 自定义 delta VJP
 * 3. 端到端反向传播测试（高层）
 * 4. 梯度累积测试
 * 5. 动态形状测试
 * 6. 新节点创建 API 测试（KEEP AS-IS）
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试（高层 Graph + Var API）====================

/// 小误差时 Huber Loss 行为像 MSE
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
/// target = torch.tensor([[0.15, 0.25, 0.35]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// # loss = 0.00125
/// ```
#[test]
fn test_huber_forward_small_error_mse_behavior() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.1, 0.2, 0.3], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.15, 0.25, 0.35], &[1, 3]))
        .unwrap();
    let loss = input.huber_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    // Huber(小误差) = 0.5 * diff² = 0.5 * 0.0025 = 0.00125 (每个元素)
    // mean = 0.00125
    assert_abs_diff_eq!(loss_val, 0.00125, epsilon = 1e-6);
}

/// 大误差时 Huber Loss 行为像 MAE
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 3.0, 4.0]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// # = (1.5 + 2.5 + 3.5) / 3 = 2.5
/// ```
#[test]
fn test_huber_forward_large_error_mae_behavior() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3]))
        .unwrap();
    let loss = input.huber_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    // Huber(大误差) = δ * |a| - 0.5 * δ² = |a| - 0.5 (当 δ=1)
    // mean = (1.5 + 2.5 + 3.5) / 3 = 2.5
    assert_abs_diff_eq!(loss_val, 2.5, epsilon = 1e-6);
}

/// 同时包含小误差和大误差的情况
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[0.5, 1.0, 1.5, 2.0]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// # = (0.125 + 0.5 + 1.0 + 1.5) / 4 = 0.78125
/// ```
#[test]
fn test_huber_forward_mixed_errors() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.5, 1.0, 1.5, 2.0], &[1, 4]))
        .unwrap();
    let loss = input.huber_loss(&target).unwrap();

    loss.forward().unwrap();

    let loss_val = loss.item().unwrap();
    assert_abs_diff_eq!(loss_val, 0.78125, epsilon = 1e-6);
}

/// 验证小误差时 Huber ≈ 0.5 * MSE
#[test]
fn test_huber_vs_mse_small_error() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.1, 0.2, 0.3], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.15, 0.25, 0.35], &[1, 3]))
        .unwrap();

    let mse = input.mse_loss(&target).unwrap();
    let huber = input.huber_loss(&target).unwrap();

    mse.forward().unwrap();
    huber.forward().unwrap();

    let mse_val = mse.item().unwrap();
    let huber_val = huber.item().unwrap();

    // Huber(小误差) = 0.5 * MSE
    assert_abs_diff_eq!(huber_val, 0.5 * mse_val, epsilon = 1e-6);
}

/// 验证大误差时 Huber ≈ MAE - 0.5*δ（当 δ=1）
#[test]
fn test_huber_vs_mae_large_error() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3]))
        .unwrap();

    let mae = input.mae_loss(&target).unwrap();
    let huber = input.huber_loss(&target).unwrap();

    mae.forward().unwrap();
    huber.forward().unwrap();

    let mae_val = mae.item().unwrap();
    let huber_val = huber.item().unwrap();

    // Huber(大误差) = MAE - 0.5*δ = MAE - 0.5（当 δ=1）
    assert_abs_diff_eq!(huber_val, mae_val - 0.5, epsilon = 1e-6);
}

/// Huber 损失节点不能直接设置值
#[test]
fn test_huber_cannot_set_value() {
    let graph = Graph::new();

    let input = graph
        .input(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    let loss = input.huber_loss(&target).unwrap();

    let err = loss.set_value(&Tensor::new(&[0.0], &[1, 1]));
    assert!(err.is_err(), "Huber 损失节点不应支持直接设值");
}

// ==================== 2. VJP 单元测试（底层 calc_grad_to_parent_index）====================
//
// 使用底层 API 创建节点，通过 calc_grad_to_parent_index 直接验证梯度计算公式。
// 小误差: dL/da = a, 大误差: dL/da = δ * sign(a)
// Mean reduction: / N

/// 小误差 VJP：梯度 = diff / N
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
/// target = torch.tensor([[0.15, 0.25, 0.35]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// loss.backward()
/// # grad = [-0.0167, -0.0167, -0.0167]
/// ```
#[test]
fn test_huber_vjp_small_error() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("target"))
        .unwrap();
    let huber = inner
        .borrow_mut()
        .create_huber_default_node(input.clone(), target.clone(), Some("huber"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[0.1, 0.2, 0.3], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[0.15, 0.25, 0.35], &[1, 3])))
        .unwrap();
    huber.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[1, 1]);
    let grad = huber
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // diff = [-0.05, -0.05, -0.05], N = 3
    // grad = diff / N = [-0.0167, -0.0167, -0.0167]
    let expected = Tensor::new(&[-0.016_666_67, -0.016_666_67, -0.016_666_67], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);

    Ok(())
}

/// 大误差 VJP：梯度 = δ * sign(diff) / N
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 3.0, 4.0]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=1.0)
/// loss.backward()
/// # grad = [-0.333, -0.333, -0.333]
/// ```
#[test]
fn test_huber_vjp_large_error() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("target"))
        .unwrap();
    let huber = inner
        .borrow_mut()
        .create_huber_default_node(input.clone(), target.clone(), Some("huber"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();
    huber.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[1, 1]);
    let grad = huber
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // diff = [-2, -3, -4], sign = [-1, -1, -1]
    // grad = δ * sign(diff) / N = 1 * [-1, -1, -1] / 3
    let expected = Tensor::new(&[-0.333_333_34, -0.333_333_34, -0.333_333_34], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);

    Ok(())
}

/// 混合误差 VJP：分段计算
///
/// diff = [-0.5, -1.0, -1.5, -2.0]
/// |diff| <= 1: grad_elem = diff       → [-0.5, -1.0]
/// |diff| > 1:  grad_elem = δ*sign(diff) → [-1, -1]
/// Mean: / 4 → [-0.125, -0.25, -0.25, -0.25]
#[test]
fn test_huber_vjp_mixed_error() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 4], Some("target"))
        .unwrap();
    let huber = inner
        .borrow_mut()
        .create_huber_default_node(input.clone(), target.clone(), Some("huber"))
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 4])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[0.5, 1.0, 1.5, 2.0], &[1, 4])))
        .unwrap();
    huber.forward_recursive(1, false).unwrap();

    let upstream = Tensor::ones(&[1, 1]);
    let grad = huber
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    let expected = Tensor::new(&[-0.125, -0.25, -0.25, -0.25], &[1, 4]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);

    Ok(())
}

/// Sum Reduction VJP：不除以 N
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[2.0, 3.0, 4.0]])
/// loss = F.smooth_l1_loss(input, target, reduction='sum', beta=1.0)
/// # loss = 7.5
/// loss.backward()
/// # grad = [-1, -1, -1]
/// ```
#[test]
fn test_huber_vjp_sum_reduction() -> Result<(), GraphError> {
    use crate::nn::nodes::raw_node::Reduction;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 3], Some("target"))
        .unwrap();
    let huber = inner
        .borrow_mut()
        .create_huber_node(
            input.clone(),
            target.clone(),
            Reduction::Sum,
            1.0,
            Some("huber"),
        )
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3])))
        .unwrap();
    huber.forward_recursive(1, false).unwrap();

    // 验证 Sum 前向值
    let loss_val = huber.value().unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], 7.5, epsilon = 1e-6);

    // VJP
    let upstream = Tensor::ones(&[1, 1]);
    let grad = huber
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // Sum reduction: 不除以 N
    // grad = δ * sign(diff) = [-1, -1, -1]
    let expected = Tensor::new(&[-1.0, -1.0, -1.0], &[1, 3]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-6);

    Ok(())
}

/// 自定义 delta=0.5 的 VJP
///
/// PyTorch 验证:
/// ```python
/// input = torch.tensor([[0.0, 0.0]], requires_grad=True)
/// target = torch.tensor([[0.3, 1.0]])
/// loss = F.smooth_l1_loss(input, target, reduction='mean', beta=0.5)
/// # loss = 0.21
/// ```
#[test]
fn test_huber_vjp_custom_delta() -> Result<(), GraphError> {
    use crate::nn::nodes::raw_node::Reduction;

    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[1, 2], Some("target"))
        .unwrap();
    let huber = inner
        .borrow_mut()
        .create_huber_node(
            input.clone(),
            target.clone(),
            Reduction::Mean,
            0.5,
            Some("huber"),
        )
        .unwrap();

    input
        .set_value(Some(&Tensor::new(&[0.0, 0.0], &[1, 2])))
        .unwrap();
    target
        .set_value(Some(&Tensor::new(&[0.3, 1.0], &[1, 2])))
        .unwrap();
    huber.forward_recursive(1, false).unwrap();

    // 验证前向值
    // |0.3| <= 0.5: 0.5 * 0.09 = 0.045
    // |1.0| > 0.5:  0.5 * 1.0 - 0.5 * 0.25 = 0.375
    // mean = (0.045 + 0.375) / 2 = 0.21
    let loss_val = huber.value().unwrap();
    assert_abs_diff_eq!(loss_val[[0, 0]], 0.21, epsilon = 1e-6);

    // VJP
    let upstream = Tensor::ones(&[1, 1]);
    let grad = huber
        .calc_grad_to_parent_index(0, &upstream)?
        .resolve(&upstream);

    // |0.3| <= 0.5: grad_elem = diff = -0.3    → / N=2 → -0.15
    // |1.0| > 0.5:  grad_elem = δ*sign = -0.5  → / N=2 → -0.25
    let expected = Tensor::new(&[-0.15, -0.25], &[1, 2]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);

    Ok(())
}

// ==================== 3. 端到端反向传播测试（高层 Graph + Var API）====================

/// 混合误差端到端反向传播
#[test]
fn test_huber_backward_e2e_mixed() {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 4], Init::Zeros, "input").unwrap();
    input
        .set_value(&Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 4]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[0.5, 1.0, 1.5, 2.0], &[1, 4]))
        .unwrap();
    let loss = input.huber_loss(&target).unwrap();

    graph.zero_grad().unwrap();
    loss.backward().unwrap();

    let grad = input.grad().unwrap().unwrap();

    // diff = [-0.5, -1.0, -1.5, -2.0]
    // |diff| <= 1: grad_elem = diff       → [-0.5, -1.0]
    // |diff| > 1:  grad_elem = δ*sign(diff) → [-1, -1]
    // Mean: / 4 → [-0.125, -0.25, -0.25, -0.25]
    let expected = Tensor::new(&[-0.125, -0.25, -0.25, -0.25], &[1, 4]);
    assert_abs_diff_eq!(&grad, &expected, epsilon = 1e-5);
}

/// 简化的回归训练测试
///
/// 验证 Huber Loss 能驱动参数向目标收敛
#[test]
fn test_huber_regression_training() {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input").unwrap();
    // input 从零开始，目标是 [2.0, 3.0, 4.0]
    let target = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3]))
        .unwrap();
    let loss = input.huber_loss(&target).unwrap();

    // 记录初始 loss
    loss.forward().unwrap();
    let initial_loss = loss.item().unwrap();

    let lr = 0.5_f32;

    // 训练 30 步
    for _ in 0..30 {
        graph.zero_grad().unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let val = input.value().unwrap().unwrap();
        let new_val = &val - lr * &grad;
        input.set_value(&new_val).unwrap();
    }

    // 验证 loss 显著下降
    loss.forward().unwrap();
    let final_loss = loss.item().unwrap();
    assert!(
        final_loss < initial_loss / 10.0,
        "Loss 应显著下降 (初始: {initial_loss}, 最终: {final_loss})"
    );

    // 验证参数趋近目标
    let learned = input.value().unwrap().unwrap();
    assert_abs_diff_eq!(learned[[0, 0]], 2.0, epsilon = 0.5);
    assert_abs_diff_eq!(learned[[0, 1]], 3.0, epsilon = 0.5);
    assert_abs_diff_eq!(learned[[0, 2]], 4.0, epsilon = 0.5);
}

// ==================== 4. 梯度累积测试 ====================

#[test]
fn test_huber_gradient_accumulation() {
    let graph = Graph::new();

    let input = graph.parameter(&[1, 3], Init::Zeros, "input").unwrap();
    input
        .set_value(&Tensor::new(&[0.0, 0.0, 0.0], &[1, 3]))
        .unwrap();
    let target = graph
        .input(&Tensor::new(&[2.0, 3.0, 4.0], &[1, 3]))
        .unwrap();
    let loss = input.huber_loss(&target).unwrap();

    // 第一次前向+反向
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
    let grad_first = input.grad().unwrap().unwrap().clone();

    // 第二次反向传播（梯度累积，不 zero_grad）
    loss.backward().unwrap();
    let grad_second = input.grad().unwrap().unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad().unwrap();
    loss.backward().unwrap();
    let grad_after_clear = input.grad().unwrap().unwrap();
    assert_eq!(&grad_after_clear, &grad_first);
}

// ==================== 5. 动态形状测试 ====================

/// 测试 Huber Loss 节点的动态形状（输出固定为标量 [1, 1]）
#[test]
fn test_huber_dynamic_shape_output_fixed() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[4, 8], Some("target"))
        .unwrap();
    let huber = inner
        .borrow_mut()
        .create_huber_default_node(input, target, Some("huber"))
        .unwrap();

    let dyn_shape = huber.dynamic_expected_shape();

    // Huber 输出形状固定
    assert!(!dyn_shape.is_dynamic(0), "Huber 输出维度 0 应固定");
    assert!(!dyn_shape.is_dynamic(1), "Huber 输出维度 1 应固定");
    assert_eq!(dyn_shape.dim(0), Some(1));
    assert_eq!(dyn_shape.dim(1), Some(1));

    Ok(())
}

/// 测试 Huber Loss 接受动态 batch 输入
#[test]
fn test_huber_dynamic_batch_forward() -> Result<(), GraphError> {
    let graph = Graph::new();
    let inner = graph.inner_rc();

    let input = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("input"))
        .unwrap();
    let target = inner
        .borrow_mut()
        .create_basic_input_node(&[2, 4], Some("target"))
        .unwrap();
    let huber = inner
        .borrow_mut()
        .create_huber_default_node(input.clone(), target.clone(), Some("huber"))
        .unwrap();

    // 第一次 forward：batch=2
    input.set_value(Some(&Tensor::ones(&[2, 4]))).unwrap();
    target.set_value(Some(&Tensor::zeros(&[2, 4]))).unwrap();
    huber.forward_recursive(1, false).unwrap();
    let loss_val1 = huber.value().unwrap();
    assert_eq!(loss_val1.shape(), &[1, 1], "Huber 输出应为标量");
    assert!(loss_val1[[0, 0]] > 0.0);

    // 第二次 forward：batch=6（不同 batch 大小）
    input.set_value(Some(&Tensor::ones(&[6, 4]))).unwrap();
    target.set_value(Some(&Tensor::zeros(&[6, 4]))).unwrap();
    huber.forward_recursive(2, false).unwrap();
    let loss_val2 = huber.value().unwrap();
    assert_eq!(loss_val2.shape(), &[1, 1], "Huber 输出应始终为标量");

    Ok(())
}

// ==================== 节点创建 API 测试 ====================

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

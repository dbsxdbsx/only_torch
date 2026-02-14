/*
 * @Author       : 老董
 * @Description  : Unsqueeze 便捷方法单元测试
 *
 * Unsqueeze 通过 reshape 在指定位置插入 size=1 维度，不创建新节点类型。
 *
 * 测试策略：
 * 1. 前向传播测试 → unsqueeze(0) / unsqueeze(1) / unsqueeze(2)
 * 2. 错误处理 → axis 越界
 * 3. squeeze-unsqueeze 互逆测试
 * 4. 端到端反向传播 → unsqueeze + MSE loss
 * 5. 梯度累积
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// unsqueeze(0): [3,4] → [1,3,4]
#[test]
fn test_unsqueeze_forward_axis0() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[3, 4])).unwrap();
    let result = x.unsqueeze(0).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3, 4]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2, 3]], 12.0, epsilon = 1e-6);
}

/// unsqueeze(1): [3,4] → [3,1,4]
#[test]
fn test_unsqueeze_forward_axis1() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[3, 4])).unwrap();
    let result = x.unsqueeze(1).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 1, 4]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 0, 3]], 12.0, epsilon = 1e-6);
}

/// unsqueeze(2): [3,4] → [3,4,1]
#[test]
fn test_unsqueeze_forward_axis2() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[3, 4])).unwrap();
    let result = x.unsqueeze(2).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 4, 1]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 3, 0]], 12.0, epsilon = 1e-6);
}

/// unsqueeze(ndim) 有效（末尾追加）：[3,4] → unsqueeze(2) → [3,4,1]
/// 同时验证 ndim=2 时 axis=2 是合法的边界情况
#[test]
fn test_unsqueeze_forward_append() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();
    let result = x.unsqueeze(2).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 2, 1]);
}

// ==================== 2. 错误处理测试 ====================

/// unsqueeze(4) 对 [3,4] 应失败（ndim=2, 合法范围 0..=2）
#[test]
fn test_unsqueeze_error_axis_out_of_bounds() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();
    let result = x.unsqueeze(4);

    assert!(result.is_err(), "unsqueeze axis 越界应报错");
}

/// unsqueeze(3) 对 [3,4] 应失败（ndim=2, 合法范围 0..=2）
#[test]
fn test_unsqueeze_error_axis3_on_2d() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]))
        .unwrap();
    let result = x.unsqueeze(3);

    assert!(result.is_err(), "unsqueeze axis=3 对 2D 张量应报错");
}

// ==================== 3. squeeze-unsqueeze 互逆测试 ====================

/// unsqueeze(0) → squeeze(Some(0)) 应恢复原形状
#[test]
fn test_unsqueeze_squeeze_inverse() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[3, 4])).unwrap();

    let expanded = x.unsqueeze(0).unwrap();
    let restored = expanded.squeeze(Some(0)).unwrap();

    restored.forward().unwrap();

    let output = restored.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 4], "squeeze(unsqueeze) 应恢复原形状");

    // 数据不变
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 3]], 12.0, epsilon = 1e-6);
}

/// squeeze(Some(0)) → unsqueeze(0) 应恢复原形状
#[test]
fn test_squeeze_unsqueeze_inverse() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[1, 3, 4])).unwrap();

    let squeezed = x.squeeze(Some(0)).unwrap();
    let restored = squeezed.unsqueeze(0).unwrap();

    restored.forward().unwrap();

    let output = restored.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3, 4], "unsqueeze(squeeze) 应恢复原形状");

    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2, 3]], 12.0, epsilon = 1e-6);
}

// ==================== 4. 端到端反向传播测试 ====================

/// unsqueeze(0) + MSE loss → backward
#[test]
fn test_unsqueeze_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // [3, 4] → unsqueeze(0) → [1, 3, 4] → MSE
    let x = graph.parameter(&[3, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[3, 4]))?;

    let expanded = x.unsqueeze(0)?;
    let target = graph.input(&Tensor::zeros(&[1, 3, 4]))?;
    let loss = expanded.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert!(loss_val > 0.0, "loss 应为正");
    assert!(loss_val.is_finite());

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[3, 4], "梯度形状应匹配原始输入");

    // 梯度非零
    assert!(x_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));

    Ok(())
}

/// unsqueeze(1) + MSE loss → backward
#[test]
fn test_unsqueeze_axis1_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // [3, 4] → unsqueeze(1) → [3, 1, 4] → MSE
    let x = graph.parameter(&[3, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[3, 4]))?;

    let expanded = x.unsqueeze(1)?;
    let target = graph.input(&Tensor::zeros(&[3, 1, 4]))?;
    let loss = expanded.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert!(loss_val > 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[3, 4], "梯度形状应匹配原始输入");

    Ok(())
}

// ==================== 5. 梯度累积测试 ====================

/// 测试 unsqueeze 梯度累积 + zero_grad
#[test]
fn test_unsqueeze_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[3, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[3, 4]))?;

    let expanded = x.unsqueeze(0)?;
    let target = graph.input(&Tensor::zeros(&[1, 3, 4]))?;
    let loss = expanded.mse_loss(&target)?;

    // 第 1 次 backward
    graph.zero_grad()?;
    loss.backward()?;
    let grad_first = x.grad()?.unwrap().clone();

    // 第 2 次 backward（梯度累积）
    loss.backward()?;
    let grad_second = x.grad()?.unwrap();
    assert_eq!(&grad_second, &(&grad_first * 2.0));

    // zero_grad 后重新计算
    graph.zero_grad()?;
    loss.backward()?;
    let grad_after_clear = x.grad()?.unwrap();
    assert_eq!(&grad_after_clear, &grad_first);

    Ok(())
}

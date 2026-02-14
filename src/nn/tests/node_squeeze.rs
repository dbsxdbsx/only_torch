/*
 * @Author       : 老董
 * @Description  : Squeeze 便捷方法单元测试
 *
 * Squeeze 通过 reshape 移除 size=1 的维度，不创建新节点类型。
 *
 * 测试策略：
 * 1. 前向传播测试 → squeeze(Some(axis)) / squeeze(None)
 * 2. 错误处理 → 非 size=1 维度 / 越界 axis
 * 3. 端到端反向传播 → squeeze + MSE loss
 * 4. 梯度累积
 */

use crate::nn::{Graph, GraphError, Init, VarLossOps, VarShapeOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== 1. 前向传播测试 ====================

/// squeeze(Some(0)): [1,3,1,4] → [3,1,4]
#[test]
fn test_squeeze_forward_axis0() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[1, 3, 1, 4])).unwrap();
    let result = x.squeeze(Some(0)).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 1, 4]);
    // 数据不变
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 0, 3]], 12.0, epsilon = 1e-6);
}

/// squeeze(Some(2)): [1,3,1,4] → [1,3,4]
#[test]
fn test_squeeze_forward_axis2() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[1, 3, 1, 4])).unwrap();
    let result = x.squeeze(Some(2)).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[1, 3, 4]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2, 3]], 12.0, epsilon = 1e-6);
}

/// squeeze(None): [1,3,1,4] → [3,4]（移除所有 size=1 维度）
#[test]
fn test_squeeze_forward_all() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[1, 3, 1, 4])).unwrap();
    let result = x.squeeze(None).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[3, 4]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 3]], 12.0, epsilon = 1e-6);
}

/// squeeze(None) 对无 size=1 维的张量不改变形状
#[test]
fn test_squeeze_none_no_ones() {
    let graph = Graph::new();

    let x = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]))
        .unwrap();
    let result = x.squeeze(None).unwrap();

    result.forward().unwrap();

    let output = result.value().unwrap().unwrap();
    assert_eq!(output.shape(), &[2, 3]);
}

// ==================== 2. 错误处理测试 ====================

/// squeeze(Some(1)) 对 [1,3,1,4] 应失败（dim 1 大小为 3 ≠ 1）
#[test]
fn test_squeeze_error_not_size_one() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[1, 3, 1, 4])).unwrap();
    let result = x.squeeze(Some(1));

    assert!(result.is_err(), "squeeze 非 size=1 维度应报错");
}

/// squeeze(Some(4)) 对 [1,3,1,4] 应失败（axis 越界）
#[test]
fn test_squeeze_error_axis_out_of_bounds() {
    let graph = Graph::new();

    let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[1, 3, 1, 4])).unwrap();
    let result = x.squeeze(Some(4));

    assert!(result.is_err(), "squeeze axis 越界应报错");
}

// ==================== 3. 端到端反向传播测试 ====================

/// squeeze + MSE loss → backward，验证梯度形状匹配原始输入
#[test]
fn test_squeeze_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // [1, 3, 1, 4] → squeeze(None) → [3, 4] → MSE
    let x = graph.parameter(&[1, 3, 1, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[1, 3, 1, 4]))?;

    let squeezed = x.squeeze(None)?;
    let target = graph.input(&Tensor::zeros(&[3, 4]))?;
    let loss = squeezed.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert!(loss_val > 0.0, "loss 应为正");
    assert!(loss_val.is_finite());

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[1, 3, 1, 4], "梯度形状应匹配原始输入");

    // 梯度非零
    assert!(x_grad.data_as_slice().iter().any(|&v| v.abs() > 1e-10));

    Ok(())
}

/// squeeze(Some(0)) + MSE loss → backward
#[test]
fn test_squeeze_axis_backward_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // [1, 3, 4] → squeeze(Some(0)) → [3, 4] → MSE
    let x = graph.parameter(&[1, 3, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[1, 3, 4]))?;

    let squeezed = x.squeeze(Some(0))?;
    let target = graph.input(&Tensor::zeros(&[3, 4]))?;
    let loss = squeezed.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert!(loss_val > 0.0);

    let x_grad = x.grad()?.expect("x 应有 grad");
    assert_eq!(x_grad.shape(), &[1, 3, 4], "梯度形状应匹配原始输入");

    Ok(())
}

// ==================== 4. 梯度累积测试 ====================

/// 测试 squeeze 梯度累积 + zero_grad
#[test]
fn test_squeeze_gradient_accumulation() -> Result<(), GraphError> {
    let graph = Graph::new();

    let x = graph.parameter(&[1, 3, 1, 4], Init::Zeros, "x")?;
    let data: Vec<f32> = (1..=12).map(|x| x as f32 * 0.1).collect();
    x.set_value(&Tensor::new(&data, &[1, 3, 1, 4]))?;

    let squeezed = x.squeeze(None)?;
    let target = graph.input(&Tensor::zeros(&[3, 4]))?;
    let loss = squeezed.mse_loss(&target)?;

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

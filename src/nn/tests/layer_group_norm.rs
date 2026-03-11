/*
 * @Author       : 老董
 * @Description  : GroupNorm 层单元测试
 */

use crate::nn::{Graph, GraphError, GroupNorm, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

/// GroupNorm 基本前向传播
#[test]
fn test_group_norm_forward() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 4 通道，2 组（每组 2 通道）
    let gn = GroupNorm::new(&graph, 2, 4, 1e-5, "gn")?;

    // [N=1, C=4, H=1, W=2]
    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        &[1, 4, 1, 2],
    ))?;

    let y = gn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 4, 1, 2]);

    Ok(())
}

/// GroupNorm 参数
#[test]
fn test_group_norm_parameters() -> Result<(), GraphError> {
    let graph = Graph::new();
    let gn = GroupNorm::new(&graph, 4, 16, 1e-5, "gn")?;

    let params = gn.parameters();
    assert_eq!(params.len(), 2); // gamma + beta

    Ok(())
}

/// GroupNorm(1, C) ≈ LayerNorm 行为
#[test]
fn test_group_norm_single_group() -> Result<(), GraphError> {
    let graph = Graph::new();

    // num_groups=1 → 所有通道一起归一化
    let gn = GroupNorm::new(&graph, 1, 4, 1e-5, "gn")?;

    let x = graph.input(&Tensor::new(
        &[1.0, 2.0, 3.0, 4.0],
        &[1, 4],
    ))?;
    let y = gn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[1, 4]);
    // 均值=2.5, 方差=1.25, std≈1.118
    // (1-2.5)/1.118 ≈ -1.3416
    assert_abs_diff_eq!(output[[0, 0]], -1.3416, epsilon = 1e-3);

    Ok(())
}

/// GroupNorm 4D 非对称维度前向传播 [N=2, C=6, H=3, W=4]
#[test]
fn test_group_norm_forward_4d_nonsquare() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 6 通道，3 组（每组 2 通道）
    let gn = GroupNorm::new(&graph, 3, 6, 1e-5, "gn")?;

    // [2, 6, 3, 4] = 144 元素
    let data: Vec<f32> = (1..=144).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 6, 3, 4]))?;

    let y = gn.forward(&x);
    y.forward()?;

    let output = y.value()?.unwrap();
    assert_eq!(output.shape(), &[2, 6, 3, 4]);

    Ok(())
}

/// GroupNorm 4D 反向传播测试，确认 gamma/beta 梯度正确
#[test]
fn test_group_norm_backward_4d() -> Result<(), GraphError> {
    let graph = Graph::new();

    let gn = GroupNorm::new(&graph, 3, 6, 1e-5, "gn")?;

    let data: Vec<f32> = (1..=144).map(|x| x as f32).collect();
    let x = graph.input(&Tensor::new(&data, &[2, 6, 3, 4]))?;

    let y = gn.forward(&x);
    let target = graph.input(&Tensor::zeros(&[2, 6, 3, 4]))?;
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    let loss_val = loss.backward()?;
    assert!(loss_val > 0.0);

    // gamma 和 beta 应有梯度
    let params = gn.parameters();
    let gamma_grad = params[0].grad()?.expect("gamma 应有 grad");
    assert_eq!(gamma_grad.shape(), &[1, 6]);
    let gamma_grad_sum: f32 = gamma_grad.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(gamma_grad_sum > 1e-6, "gamma_grad 不应全为 0");

    let beta_grad = params[1].grad()?.expect("beta 应有 grad");
    assert_eq!(beta_grad.shape(), &[1, 6]);
    let beta_grad_sum: f32 = beta_grad.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(beta_grad_sum > 1e-6, "beta_grad 不应全为 0");

    Ok(())
}

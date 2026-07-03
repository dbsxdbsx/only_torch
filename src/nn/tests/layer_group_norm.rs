/*
 * @Author       : 老董
 * @Description  : GroupNorm 层单元测试
 */

use crate::nn::{Graph, GraphError, GroupNorm, Init, Module, VarLossOps};
use crate::tensor::Tensor;
use approx::{assert_abs_diff_eq, assert_relative_eq};

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

    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]))?;
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

/// 回归测试：梯度必须回传到 GroupNorm 的**上游输入**（旧实现在图外算 x_hat
/// 再包回 input 节点，输入侧梯度被整体截断——gamma/beta 有梯度但上游永远学不到）。
///
/// 用 parameter 直接作为 GroupNorm 输入，反向后：
/// 1. 输入梯度存在且非零；
/// 2. 与中心差分数值梯度逐点吻合（可微性正确，非碰巧非零）。
#[test]
fn test_group_norm_gradient_flows_to_upstream_input() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(7);
    let gn = GroupNorm::new(&graph, 2, 4, 1e-5, "gn")?;

    // [N=2, C=4]，非对称初值（避免归一化对称性使梯度恰为 0）
    let w = graph.parameter(&[2, 4], Init::Kaiming, "w")?;
    let target = graph.input(&Tensor::new(
        &[0.3, -1.2, 0.8, 2.0, -0.5, 1.1, -2.0, 0.4],
        &[2, 4],
    ))?;

    let y = gn.forward(&w);
    let loss = y.mse_loss(&target)?;

    graph.zero_grad()?;
    loss.backward()?;
    let analytic = w.grad()?.expect("GroupNorm 上游输入应有梯度");
    let grad_abs_sum: f32 = analytic.data_as_slice().iter().map(|v| v.abs()).sum();
    assert!(
        grad_abs_sum > 1e-6,
        "上游输入梯度不应全为 0（梯度断流回归）"
    );

    // 中心差分校验前 3 个坐标
    let base = w.value()?.unwrap();
    let eps = 1e-2f32;
    for flat_idx in 0..3 {
        let (i, j) = (flat_idx / 4, flat_idx % 4);
        let mut plus = base.clone();
        plus[[i, j]] += eps;
        w.set_value(&plus)?;
        loss.forward()?;
        let l_plus = loss.value()?.unwrap()[[0, 0]];

        let mut minus = base.clone();
        minus[[i, j]] -= eps;
        w.set_value(&minus)?;
        loss.forward()?;
        let l_minus = loss.value()?.unwrap()[[0, 0]];

        let numeric = (l_plus - l_minus) / (2.0 * eps);
        assert_relative_eq!(
            analytic[[i, j]],
            numeric,
            max_relative = 5e-2,
            epsilon = 1e-3
        );
    }
    w.set_value(&base)?;

    Ok(())
}

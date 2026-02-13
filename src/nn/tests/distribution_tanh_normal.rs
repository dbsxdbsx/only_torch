/*
 * @Author       : 老董
 * @Description  : TanhNormal 分布（Squashed Gaussian）单元测试
 *
 * 测试策略：
 * 1. log_prob 前向值测试（与 PyTorch 对照）
 * 2. rsample 形状和 tanh 范围测试
 * 3. 梯度正确性测试
 * 4. Batch 支持测试
 * 5. SAC 风格端到端测试
 *
 * 参考脚本：tests/test_tanh_normal.py（PyTorch 对照值来源）
 */

use crate::nn::distributions::TanhNormal;
use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarReduceOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== log_prob 前向值测试 ====================

/// 标准正态 TanhNormal: u=0, mean=0, std=1
///
/// PyTorch 参考：log_prob = -0.91893947
/// 因为 tanh(0)=0，所以 correction = log(1-0+eps) ≈ 0，log_prob ≈ Normal.log_prob
#[test]
fn test_tanh_normal_log_prob_standard() {
    let graph = Graph::new();

    let mean = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let std = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let dist = TanhNormal::new(mean, std);

    let raw_action = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let lp = dist.log_prob(&raw_action);

    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[1, 1]);
    // tanh(0)=0, correction≈0, log_prob ≈ Normal(0,1).log_prob(0)
    assert_abs_diff_eq!(output[[0, 0]], -0.91893947, epsilon = 1e-4);
}

/// 一般 TanhNormal log_prob: mean=[0,1], std=[1,0.5], u=[0.5,-0.3]
///
/// PyTorch 参考：[-0.80371076, -3.51711059]
#[test]
fn test_tanh_normal_log_prob_general() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(&[0.0, 1.0], &[1, 2]))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[1.0, 0.5], &[1, 2]))
        .unwrap();
    let dist = TanhNormal::new(mean, std);

    let raw_action = graph
        .input(&Tensor::new(&[0.5, -0.3], &[1, 2]))
        .unwrap();
    let lp = dist.log_prob(&raw_action);

    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert_abs_diff_eq!(output[[0, 0]], -0.80371076, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[0, 1]], -3.51711059, epsilon = 1e-3);
}

/// Batch TanhNormal log_prob: batch=2, dim=2
///
/// PyTorch 参考：
///   [[-0.80371076, -3.51711059],
///    [-17.13345718, -0.40121090]]
#[test]
fn test_tanh_normal_log_prob_batch() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(&[0.0, 1.0, -1.0, 0.5], &[2, 2]))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[1.0, 0.5, 0.3, 2.0], &[2, 2]))
        .unwrap();
    let dist = TanhNormal::new(mean, std);

    let raw_action = graph
        .input(&Tensor::new(&[0.5, -0.3, 0.8, -1.5], &[2, 2]))
        .unwrap();
    let lp = dist.log_prob(&raw_action);

    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], -0.80371076, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[0, 1]], -3.51711059, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[1, 0]], -17.13345718, epsilon = 1e-1); // 极端值，精度放宽
    assert_abs_diff_eq!(output[[1, 1]], -0.40121090, epsilon = 1e-3);
}

// ==================== rsample 测试 ====================

/// rsample 输出形状和 tanh 范围
#[test]
fn test_tanh_normal_rsample_shape_and_range() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(&[0.0, 1.0, -1.0, 0.5], &[2, 2]))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[1.0, 0.5, 0.3, 2.0], &[2, 2]))
        .unwrap();
    let dist = TanhNormal::new(mean, std);

    let (squashed, raw) = dist.rsample();

    squashed.forward().unwrap();
    raw.forward().unwrap();

    let squashed_val = squashed.value().unwrap().unwrap();
    let raw_val = raw.value().unwrap().unwrap();

    // 形状检查
    assert_eq!(squashed_val.shape(), &[2, 2]);
    assert_eq!(raw_val.shape(), &[2, 2]);

    // squashed_action 应在 (-1, 1) 范围内
    for i in 0..2 {
        for j in 0..2 {
            let val = squashed_val[[i, j]];
            assert!(
                val > -1.0 && val < 1.0,
                "tanh 输出应在 (-1, 1) 范围内，实际 [{i},{j}]={val}"
            );
        }
    }
}

/// rsample_and_log_prob 便捷方法
#[test]
fn test_tanh_normal_rsample_and_log_prob() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(&[0.0, 1.0], &[1, 2]))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[1.0, 0.5], &[1, 2]))
        .unwrap();
    let dist = TanhNormal::new(mean, std);

    let (action, log_prob) = dist.rsample_and_log_prob();

    action.forward().unwrap();
    log_prob.forward().unwrap();

    let action_val = action.value().unwrap().unwrap();
    let lp_val = log_prob.value().unwrap().unwrap();

    assert_eq!(action_val.shape(), &[1, 2]);
    assert_eq!(lp_val.shape(), &[1, 2]);

    // log_prob 应为有限值
    assert!(lp_val[[0, 0]].is_finite());
    assert!(lp_val[[0, 1]].is_finite());
}

// ==================== 梯度测试 ====================

/// log_prob 的 mean/std 梯度（与 PyTorch 对照）
///
/// mean=[0,1], std=[1,0.5], u=[0.5,-0.3]
/// loss = sum(log_prob) = -4.32082129
/// mean.grad = [0.5, -5.2], std.grad = [-0.75, 11.52]
#[test]
fn test_tanh_normal_log_prob_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();

    let mean_param = graph.parameter(&[1, 2], Init::Zeros, "mean")?;
    mean_param.set_value(&Tensor::new(&[0.0, 1.0], &[1, 2]))?;

    let std_param = graph.parameter(&[1, 2], Init::Ones, "std")?;
    std_param.set_value(&Tensor::new(&[1.0, 0.5], &[1, 2]))?;

    let dist = TanhNormal::new(mean_param.clone(), std_param.clone());

    let raw_action = graph.input(&Tensor::new(&[0.5, -0.3], &[1, 2]))?;
    let lp = dist.log_prob(&raw_action);
    let loss = lp.sum();

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // 验证 loss 值
    assert_abs_diff_eq!(loss_val, -4.32082129, epsilon = 1e-2);

    // 验证 mean 梯度（与 PyTorch 对照）
    let mean_grad = mean_param.grad()?.expect("mean 应有梯度");
    assert_abs_diff_eq!(mean_grad[[0, 0]], 0.5, epsilon = 1e-2);
    assert_abs_diff_eq!(mean_grad[[0, 1]], -5.2, epsilon = 1e-1);

    // 验证 std 梯度（与 PyTorch 对照）
    let std_grad = std_param.grad()?.expect("std 应有梯度");
    assert_abs_diff_eq!(std_grad[[0, 0]], -0.75, epsilon = 1e-2);
    assert_abs_diff_eq!(std_grad[[0, 1]], 11.52, epsilon = 5e-1);

    Ok(())
}

// ==================== SAC 端到端测试 ====================

/// SAC-Continuous 风格：log_std → exp → std → TanhNormal → log_prob → loss
#[test]
fn test_tanh_normal_sac_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 模拟 Actor 网络输出
    let mean_param = graph.parameter(&[1, 2], Init::Zeros, "mean")?;
    mean_param.set_value(&Tensor::new(&[0.0, 0.5], &[1, 2]))?;

    let log_std_param = graph.parameter(&[1, 2], Init::Zeros, "log_std")?;
    log_std_param.set_value(&Tensor::new(&[0.0, -0.5], &[1, 2]))?;

    // std = exp(log_std)
    let std = log_std_param.exp();
    let dist = TanhNormal::new(mean_param.clone(), std);

    // 使用固定 raw_action（模拟 replay buffer）
    let raw_action = graph.input(&Tensor::new(&[0.3, -0.2], &[1, 2]))?;
    let lp = dist.log_prob(&raw_action);

    // SAC Actor Loss: -mean(log_prob)
    let neg_lp = -&lp;
    let loss = neg_lp.mean();

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // 验证 loss 有限
    assert!(loss_val.is_finite(), "loss 应为有限值");

    // 验证梯度存在且非零
    let mean_grad = mean_param.grad()?.expect("mean 应有梯度");
    let log_std_grad = log_std_param.grad()?.expect("log_std 应有梯度");

    let grad_sum: f32 = mean_grad[[0, 0]].abs()
        + mean_grad[[0, 1]].abs()
        + log_std_grad[[0, 0]].abs()
        + log_std_grad[[0, 1]].abs();
    assert!(grad_sum > 1e-6, "梯度不应全为零");

    Ok(())
}

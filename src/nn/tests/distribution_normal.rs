/*
 * @Author       : 老董
 * @Description  : Normal 分布单元测试
 *
 * 测试策略：
 * 1. log_prob 前向值测试（与 PyTorch 对照）
 * 2. entropy 前向值测试（与 PyTorch 对照）
 * 3. rsample 形状和梯度流测试
 * 4. 梯度正确性测试（log_prob / entropy 的反向传播）
 * 5. Batch 支持测试
 *
 * 参考脚本：tests/test_normal.py（PyTorch 对照值来源）
 */

use crate::nn::distributions::Normal;
use crate::nn::{Graph, GraphError, Init, VarActivationOps, VarReduceOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== log_prob 前向值测试 ====================

/// 标准正态分布 log_prob(0) = -0.5 * ln(2π) ≈ -0.9189
///
/// PyTorch 参考：Normal(0, 1).log_prob(0) = -0.91893852
#[test]
fn test_normal_log_prob_standard() {
    let graph = Graph::new();

    let mean = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let std = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let dist = Normal::new(mean, std);

    let value = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let lp = dist.log_prob(&value);

    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[1, 1]);
    assert_abs_diff_eq!(output[[0, 0]], -0.91893852, epsilon = 1e-4);
}

/// 一般正态分布 log_prob：μ=[1,2], σ=[0.5,1], x=[1.2,2.5]
///
/// PyTorch 参考：[-0.30579138, -1.04393852]
#[test]
fn test_normal_log_prob_general() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(&[1.0, 2.0], &[1, 2]))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[0.5, 1.0], &[1, 2]))
        .unwrap();
    let dist = Normal::new(mean, std);

    let value = graph
        .input(&Tensor::new(&[1.2, 2.5], &[1, 2]))
        .unwrap();
    let lp = dist.log_prob(&value);

    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert_abs_diff_eq!(output[[0, 0]], -0.30579138, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1]], -1.04393852, epsilon = 1e-4);
}

/// Batch log_prob 测试：batch=3, dim=2
///
/// PyTorch 参考：
///   [[-1.04393852, -0.30579138],
///    [-1.73708570,  0.88364756],
///    [-1.10385454, -1.37995923]]
#[test]
fn test_normal_log_prob_batch() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(
            &[0.0, 1.0, 2.0, 3.0, -1.0, 0.5],
            &[3, 2],
        ))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[1.0, 0.5, 2.0, 0.1, 0.3, 1.5], &[3, 2]))
        .unwrap();
    let dist = Normal::new(mean, std);

    let value = graph
        .input(&Tensor::new(
            &[0.5, 1.2, 1.0, 3.1, -0.5, 0.0],
            &[3, 2],
        ))
        .unwrap();
    let lp = dist.log_prob(&value);

    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    // Row 0: mean=[0,1], std=[1,0.5], value=[0.5,1.2]
    assert_abs_diff_eq!(output[[0, 0]], -1.04393852, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1]], -0.30579138, epsilon = 1e-4);
    // Row 1: mean=[2,3], std=[2,0.1], value=[1.0,3.1]
    assert_abs_diff_eq!(output[[1, 0]], -1.73708570, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 1]], 0.88364756, epsilon = 1e-3);
    // Row 2: mean=[-1,0.5], std=[0.3,1.5], value=[-0.5,0.0]
    assert_abs_diff_eq!(output[[2, 0]], -1.10385454, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[2, 1]], -1.37995923, epsilon = 1e-4);
}

// ==================== entropy 前向值测试 ====================

/// 标准正态分布 entropy = 0.5 + 0.5 * ln(2π) ≈ 1.4189
///
/// PyTorch 参考：Normal(0, 1).entropy() = 1.41893852
#[test]
fn test_normal_entropy_standard() {
    let graph = Graph::new();

    let mean = graph.input(&Tensor::new(&[0.0], &[1, 1])).unwrap();
    let std = graph.input(&Tensor::new(&[1.0], &[1, 1])).unwrap();
    let dist = Normal::new(mean, std);

    let ent = dist.entropy();
    ent.forward().unwrap();
    let output = ent.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[1, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 1.41893852, epsilon = 1e-4);
}

/// 一般正态分布 entropy：σ=[0.5, 1.0]
///
/// PyTorch 参考：[0.72579134, 1.41893852]
#[test]
fn test_normal_entropy_general() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(&[1.0, 2.0], &[1, 2]))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[0.5, 1.0], &[1, 2]))
        .unwrap();
    let dist = Normal::new(mean, std);

    let ent = dist.entropy();
    ent.forward().unwrap();
    let output = ent.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[1, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 0.72579134, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1]], 1.41893852, epsilon = 1e-4);
}

/// Batch entropy 测试：batch=3, dim=2
///
/// PyTorch 参考：
///   [[1.41893852,  0.72579134],
///    [2.11208582, -0.88364661],
///    [0.21496570,  1.82440364]]
#[test]
fn test_normal_entropy_batch() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(
            &[0.0, 1.0, 2.0, 3.0, -1.0, 0.5],
            &[3, 2],
        ))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[1.0, 0.5, 2.0, 0.1, 0.3, 1.5], &[3, 2]))
        .unwrap();
    let dist = Normal::new(mean, std);

    let ent = dist.entropy();
    ent.forward().unwrap();
    let output = ent.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[3, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.41893852, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1]], 0.72579134, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 0]], 2.11208582, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 1]], -0.88364661, epsilon = 1e-3);
    assert_abs_diff_eq!(output[[2, 0]], 0.21496570, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[2, 1]], 1.82440364, epsilon = 1e-4);
}

// ==================== rsample 测试 ====================

/// rsample 输出形状应与 mean/std 一致
#[test]
fn test_normal_rsample_shape() {
    let graph = Graph::new();

    let mean = graph
        .input(&Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))
        .unwrap();
    let std = graph
        .input(&Tensor::new(&[0.5, 0.5, 0.5, 0.5], &[2, 2]))
        .unwrap();
    let dist = Normal::new(mean, std);

    let sample = dist.rsample();
    sample.forward().unwrap();
    let output = sample.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[2, 2]);
}

/// rsample 梯度可通过 mean 和 std 传播
///
/// sample = mean + std * eps
/// ∂sample/∂mean = 1（全元素为 1）
/// ∂sample/∂std = eps（随机值，但应非零）
#[test]
fn test_normal_rsample_gradient_flow() -> Result<(), GraphError> {
    let graph = Graph::new();

    let mean_param = graph.parameter(&[1, 2], Init::Zeros, "mean")?;
    mean_param.set_value(&Tensor::new(&[1.0, 2.0], &[1, 2]))?;

    let std_param = graph.parameter(&[1, 2], Init::Ones, "std")?;
    std_param.set_value(&Tensor::new(&[0.5, 1.0], &[1, 2]))?;

    let dist = Normal::new(mean_param.clone(), std_param.clone());
    let sample = dist.rsample();

    // loss = sum(sample)
    let loss = sample.sum();

    graph.zero_grad()?;
    loss.backward()?;

    // mean 梯度应为 1（sum 对每个元素梯度为 1，sample 对 mean 梯度为 1）
    let mean_grad = mean_param.grad()?.expect("mean 应有梯度");
    assert_eq!(mean_grad.shape(), &[1, 2]);
    assert_abs_diff_eq!(mean_grad[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(mean_grad[[0, 1]], 1.0, epsilon = 1e-6);

    // std 梯度应非零（= eps，随机值）
    let std_grad = std_param.grad()?.expect("std 应有梯度");
    assert_eq!(std_grad.shape(), &[1, 2]);
    // eps 不为 0 的概率极高（正态分布精确为 0 的概率为 0）
    assert!(
        std_grad[[0, 0]].abs() > 1e-10 || std_grad[[0, 1]].abs() > 1e-10,
        "std 梯度不应全为零（概率论保证）"
    );

    Ok(())
}

// ==================== 梯度正确性测试 ====================

/// log_prob 的 mean/std 梯度与 PyTorch 对照
///
/// μ=[1,2], σ=[0.5,1], x=[1.2,2.5]
/// loss = sum(log_prob) = -1.34972990
/// ∂loss/∂μ = (x-μ)/σ² = [0.8, 0.5]
/// ∂loss/∂σ = ((x-μ)²/σ³ - 1/σ) = [-1.68, -0.75]
#[test]
fn test_normal_log_prob_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();

    let mean_param = graph.parameter(&[1, 2], Init::Zeros, "mean")?;
    mean_param.set_value(&Tensor::new(&[1.0, 2.0], &[1, 2]))?;

    let std_param = graph.parameter(&[1, 2], Init::Ones, "std")?;
    std_param.set_value(&Tensor::new(&[0.5, 1.0], &[1, 2]))?;

    let dist = Normal::new(mean_param.clone(), std_param.clone());

    let value = graph.input(&Tensor::new(&[1.2, 2.5], &[1, 2]))?;
    let lp = dist.log_prob(&value);
    let loss = lp.sum();

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // 验证 loss 值
    assert_abs_diff_eq!(loss_val, -1.34972990, epsilon = 1e-3);

    // 验证 mean 梯度
    let mean_grad = mean_param.grad()?.expect("mean 应有梯度");
    assert_abs_diff_eq!(mean_grad[[0, 0]], 0.8, epsilon = 1e-3);
    assert_abs_diff_eq!(mean_grad[[0, 1]], 0.5, epsilon = 1e-3);

    // 验证 std 梯度
    let std_grad = std_param.grad()?.expect("std 应有梯度");
    assert_abs_diff_eq!(std_grad[[0, 0]], -1.68, epsilon = 1e-2);
    assert_abs_diff_eq!(std_grad[[0, 1]], -0.75, epsilon = 1e-2);

    Ok(())
}

/// entropy 的 std 梯度与 PyTorch 对照
///
/// σ=[0.5, 1.0]
/// loss = sum(entropy) = 2.14472985
/// ∂loss/∂σ = 1/σ = [2.0, 1.0]
/// ∂loss/∂μ = 0（entropy 不依赖 mean）
#[test]
fn test_normal_entropy_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();

    let mean_param = graph.parameter(&[1, 2], Init::Zeros, "mean")?;
    mean_param.set_value(&Tensor::new(&[1.0, 2.0], &[1, 2]))?;

    let std_param = graph.parameter(&[1, 2], Init::Ones, "std")?;
    std_param.set_value(&Tensor::new(&[0.5, 1.0], &[1, 2]))?;

    let dist = Normal::new(mean_param.clone(), std_param.clone());

    let ent = dist.entropy();
    let loss = ent.sum();

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // 验证 loss 值
    assert_abs_diff_eq!(loss_val, 2.14472985, epsilon = 1e-3);

    // 验证 std 梯度：∂H/∂σ = 1/σ
    let std_grad = std_param.grad()?.expect("std 应有梯度");
    assert_abs_diff_eq!(std_grad[[0, 0]], 2.0, epsilon = 1e-3);
    assert_abs_diff_eq!(std_grad[[0, 1]], 1.0, epsilon = 1e-3);

    Ok(())
}

// ==================== 端到端测试 ====================

/// 模拟 SAC 场景：log_std → exp → std → Normal → log_prob(fixed_value) → loss
///
/// 验证 exp → std → Normal 的完整前向 + 反向传播链路
/// 注意：使用固定 value 而非 rsample，避免共享参数路径的梯度抵消
#[test]
fn test_normal_sac_style_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 模拟 Actor 网络输出
    let mean_param = graph.parameter(&[1, 2], Init::Zeros, "mean")?;
    mean_param.set_value(&Tensor::new(&[0.0, 0.5], &[1, 2]))?;

    let log_std_param = graph.parameter(&[1, 2], Init::Zeros, "log_std")?;
    log_std_param.set_value(&Tensor::new(&[0.0, -0.5], &[1, 2]))?;

    // std = exp(log_std) — 标准 SAC 模式
    let std = log_std_param.exp();

    // 构建分布
    let dist = Normal::new(mean_param.clone(), std);

    // 使用固定值计算 log_prob（模拟从 replay buffer 取出的动作）
    let action = graph.input(&Tensor::new(&[0.5, 1.0], &[1, 2]))?;
    let lp = dist.log_prob(&action);

    // loss = -mean(log_prob)
    let neg_lp = -&lp;
    let loss = neg_lp.mean();

    // 前向 + 反向
    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    // 验证 loss 是有限值
    assert!(loss_val.is_finite(), "loss 应为有限值，实际为 {loss_val}");

    // 验证 mean 梯度存在且非零（对于固定 value ≠ mean，梯度非零）
    let mean_grad = mean_param.grad()?.expect("mean 应有梯度");
    assert_eq!(mean_grad.shape(), &[1, 2]);
    let mean_grad_sum: f32 = mean_grad[[0, 0]].abs() + mean_grad[[0, 1]].abs();
    assert!(
        mean_grad_sum > 1e-6,
        "mean 梯度不应全为零"
    );

    // 验证 log_std 梯度存在且非零
    let log_std_grad = log_std_param.grad()?.expect("log_std 应有梯度");
    assert_eq!(log_std_grad.shape(), &[1, 2]);
    let log_std_grad_sum: f32 = log_std_grad[[0, 0]].abs() + log_std_grad[[0, 1]].abs();
    assert!(
        log_std_grad_sum > 1e-6,
        "log_std 梯度不应全为零"
    );

    Ok(())
}

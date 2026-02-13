/*
 * @Author       : 老董
 * @Description  : Categorical 分布单元测试
 *
 * 测试策略：
 * 1. log_prob 前向值测试（与 PyTorch 对照）
 * 2. entropy 前向值测试（与 PyTorch 对照）
 * 3. sample 形状和范围测试
 * 4. 梯度正确性测试
 * 5. Batch 支持测试
 *
 * 参考脚本：tests/test_categorical.py（PyTorch 对照值来源）
 */

use crate::nn::distributions::Categorical;
use crate::nn::{Graph, GraphError, Init, VarReduceOps};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

// ==================== log_prob 前向值测试 ====================

/// 基本 log_prob：logits=[1,2,0.5]
///
/// PyTorch 参考：
///   log_prob(a=0) = -1.46436882
///   log_prob(a=1) = -0.46436882
///   log_prob(a=2) = -1.96436882
#[test]
fn test_categorical_log_prob_basic() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .unwrap();
    let dist = Categorical::new(logits);

    // 测试每个动作的 log_prob
    for (action_idx, expected) in [(0, -1.46436882_f32), (1, -0.46436882), (2, -1.96436882)] {
        let action = Tensor::new(&[action_idx as f32], &[1, 1]);
        let lp = dist.log_prob(&action);
        lp.forward().unwrap();
        let output = lp.value().unwrap().unwrap();

        assert_eq!(output.shape(), &[1, 1]);
        assert_abs_diff_eq!(output[[0, 0]], expected, epsilon = 1e-4);
    }
}

/// 均匀分布 log_prob：logits=[1,1,1,1]
///
/// PyTorch 参考：log_prob(a=0) = -ln(4) ≈ -1.38629436
#[test]
fn test_categorical_log_prob_uniform() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let dist = Categorical::new(logits);

    let action = Tensor::new(&[0.0], &[1, 1]);
    let lp = dist.log_prob(&action);
    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_abs_diff_eq!(output[[0, 0]], -1.38629436, epsilon = 1e-4);
}

/// Batch log_prob：batch=3
///
/// PyTorch 参考：[-1.46436882, -1.09861231, -0.06588387]
#[test]
fn test_categorical_log_prob_batch() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 0.5, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0],
            &[3, 3],
        ))
        .unwrap();
    let dist = Categorical::new(logits);

    // actions: [0, 2, 1] → [batch, 1]
    let action = Tensor::new(&[0.0, 2.0, 1.0], &[3, 1]);
    let lp = dist.log_prob(&action);
    lp.forward().unwrap();
    let output = lp.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[3, 1]);
    assert_abs_diff_eq!(output[[0, 0]], -1.46436882, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 0]], -1.09861231, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[2, 0]], -0.06588387, epsilon = 1e-3);
}

// ==================== entropy 前向值测试 ====================

/// 基本 entropy：logits=[1,2,0.5]
///
/// PyTorch 参考：entropy = 0.90595925
#[test]
fn test_categorical_entropy_basic() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .unwrap();
    let dist = Categorical::new(logits);

    let ent = dist.entropy();
    ent.forward().unwrap();
    let output = ent.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[1, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 0.90595925, epsilon = 1e-4);
}

/// 均匀分布 entropy = ln(4)
///
/// PyTorch 参考：entropy = 1.38629436
#[test]
fn test_categorical_entropy_uniform() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[1, 4]))
        .unwrap();
    let dist = Categorical::new(logits);

    let ent = dist.entropy();
    ent.forward().unwrap();
    let output = ent.value().unwrap().unwrap();

    assert_abs_diff_eq!(output[[0, 0]], 1.38629436, epsilon = 1e-4);
}

/// Batch entropy：batch=3
///
/// PyTorch 参考：[0.90595925, 1.09861231, 0.27431303]
#[test]
fn test_categorical_entropy_batch() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(
            &[1.0, 2.0, 0.5, 0.0, 0.0, 0.0, -1.0, 3.0, 0.0],
            &[3, 3],
        ))
        .unwrap();
    let dist = Categorical::new(logits);

    let ent = dist.entropy();
    ent.forward().unwrap();
    let output = ent.value().unwrap().unwrap();

    assert_eq!(output.shape(), &[3, 1]);
    assert_abs_diff_eq!(output[[0, 0]], 0.90595925, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[1, 0]], 1.09861231, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[2, 0]], 0.27431303, epsilon = 1e-3);
}

// ==================== sample 测试 ====================

/// sample 输出形状和范围
#[test]
fn test_categorical_sample_shape_and_range() {
    let graph = Graph::new();

    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5, 0.0, 0.0, 0.0], &[2, 3]))
        .unwrap();
    let dist = Categorical::new(logits);

    let action = dist.sample();
    assert_eq!(action.shape(), &[2, 1]);

    // 索引应在 [0, num_classes) 范围内
    for b in 0..2 {
        let idx = action[[b, 0]] as usize;
        assert!(idx < 3, "采样索引应 < num_classes=3，实际 {idx}");
    }
}

/// 多次采样验证概率分布近似正确
#[test]
fn test_categorical_sample_distribution() {
    let graph = Graph::new();

    // 偏向 action=1 的分布（概率 ≈ 0.63）
    let logits = graph
        .input(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))
        .unwrap();
    let dist = Categorical::new(logits);

    let num_samples = 1000;
    let mut counts = [0usize; 3];
    for _ in 0..num_samples {
        let action = dist.sample();
        let idx = action[[0, 0]] as usize;
        counts[idx] += 1;
    }

    // action=1 应该被采样最多（概率 ≈ 0.63）
    assert!(
        counts[1] > counts[0] && counts[1] > counts[2],
        "action=1（概率最高）应被采样最多，counts={counts:?}"
    );

    // 粗略检查比例：action=1 占比应 > 40%（放宽阈值避免随机失败）
    let ratio = counts[1] as f32 / num_samples as f32;
    assert!(ratio > 0.4, "action=1 占比应 > 40%，实际 {ratio:.2}%");
}

// ==================== 梯度测试 ====================

/// log_prob 的 logits 梯度（与 PyTorch 对照）
///
/// logits=[1,2,0.5], action=1
/// logits.grad = [-0.23122388, 0.37146831, -0.14024438]
#[test]
fn test_categorical_log_prob_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();

    let logits_param = graph.parameter(&[1, 3], Init::Zeros, "logits")?;
    logits_param.set_value(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))?;

    let dist = Categorical::new(logits_param.clone());

    let action = Tensor::new(&[1.0], &[1, 1]);
    let lp = dist.log_prob(&action);
    let loss = lp.sum();

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert_abs_diff_eq!(loss_val, -0.46436882, epsilon = 1e-3);

    let grad = logits_param.grad()?.expect("logits 应有梯度");
    assert_abs_diff_eq!(grad[[0, 0]], -0.23122388, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 1]], 0.37146831, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 2]], -0.14024438, epsilon = 1e-3);

    Ok(())
}

/// entropy 的 logits 梯度（与 PyTorch 对照）
///
/// logits=[1,2,0.5]
/// logits.grad = [0.12911759, -0.27755362, 0.14843598]
#[test]
fn test_categorical_entropy_gradient() -> Result<(), GraphError> {
    let graph = Graph::new();

    let logits_param = graph.parameter(&[1, 3], Init::Zeros, "logits")?;
    logits_param.set_value(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))?;

    let dist = Categorical::new(logits_param.clone());

    let ent = dist.entropy();
    let loss = ent.sum(); // 虽然只有 1 行，但 sum 确保输出为标量

    graph.zero_grad()?;
    let loss_val = loss.backward()?;

    assert_abs_diff_eq!(loss_val, 0.90595925, epsilon = 1e-3);

    let grad = logits_param.grad()?.expect("logits 应有梯度");
    assert_abs_diff_eq!(grad[[0, 0]], 0.12911759, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 1]], -0.27755362, epsilon = 1e-3);
    assert_abs_diff_eq!(grad[[0, 2]], 0.14843598, epsilon = 1e-3);

    Ok(())
}

// ==================== SAC 端到端测试 ====================

/// SAC-Discrete 风格：logits → Categorical → sample + log_prob + entropy → loss
#[test]
fn test_categorical_sac_discrete_e2e() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 模拟 Actor 网络输出 logits
    let logits_param = graph.parameter(&[1, 3], Init::Zeros, "logits")?;
    logits_param.set_value(&Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]))?;

    let dist = Categorical::new(logits_param.clone());

    // 采样动作
    let action = dist.sample();
    assert_eq!(action.shape(), &[1, 1]);

    // 计算 log_prob（在计算图中）
    let log_prob = dist.log_prob(&action);

    // 计算 entropy
    let entropy = dist.entropy();

    // SAC Actor Loss: α * log_prob - Q + α * entropy（简化版）
    let alpha = Tensor::new(&[0.2], &[1, 1]);
    let fake_q = Tensor::new(&[1.5], &[1, 1]);
    let loss = &log_prob * &alpha - fake_q - &entropy * &alpha;
    let final_loss = loss.mean();

    graph.zero_grad()?;
    let loss_val = final_loss.backward()?;

    // 验证 loss 有限
    assert!(loss_val.is_finite(), "loss 应为有限值");

    // 验证 logits 有梯度
    let grad = logits_param.grad()?.expect("logits 应有梯度");
    assert_eq!(grad.shape(), &[1, 3]);

    Ok(())
}

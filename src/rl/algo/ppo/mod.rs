//! PPO（Proximal Policy Optimization）函数式 helper
//!
//! 提供 PPO 训练中可复用的纯函数，涵盖：
//! - [`compute_gae`]：GAE 优势估计（terminated/truncated 分离）
//! - [`clipped_policy_loss`] / [`value_loss`] / [`entropy_bonus`]：三损失构件
//! - [`rollout_to_batch`]：`&[RolloutStep]` 转批量 Tensor
//! - [`normalize_advantages`]：优势标准化
//!
//! # 设计边界
//! - **入库**：训练步骤的纯函数、GAE 计算
//! - **留示例**：actor-critic 网络结构、训练主循环

mod gae;
mod loss;

pub use gae::compute_gae;
pub use loss::{clipped_policy_loss, entropy_bonus, value_loss};

use crate::rl::RolloutStep;
use crate::tensor::Tensor;

/// 从 `&[RolloutStep]` 提取的批量 Tensor，供 PPO 训练步骤使用
pub struct PpoBatch {
    /// 观察 `[batch, obs_dim]`
    pub obs: Tensor,
    /// 动作 `[batch, action_dim]`
    pub actions: Tensor,
    /// 旧策略的 log π(a|s) `[batch, 1]`（常量，不反传）
    pub old_log_probs: Tensor,
    /// GAE 优势 `[batch, 1]`
    pub advantages: Tensor,
    /// GAE 回报 `[batch, 1]`（value target）
    pub returns: Tensor,
}

/// 将 rollout 步骤 + GAE 结果转为 [`PpoBatch`]
///
/// # Panics
/// `steps` 为空时 panic。
pub fn rollout_to_batch(
    steps: &[RolloutStep],
    advantages: &[f32],
    returns: &[f32],
    obs_dim: usize,
) -> PpoBatch {
    assert!(!steps.is_empty(), "rollout_to_batch: 空 batch");
    let bs = steps.len();
    let action_dim = steps[0].action.len();

    let obs_data: Vec<f32> = steps.iter().flat_map(|s| s.obs.iter().copied()).collect();
    let actions_data: Vec<f32> = steps.iter().flat_map(|s| s.action.iter().copied()).collect();
    let log_probs: Vec<f32> = steps.iter().map(|s| s.log_prob).collect();

    PpoBatch {
        obs: Tensor::new(&obs_data, &[bs, obs_dim]),
        actions: Tensor::new(&actions_data, &[bs, action_dim]),
        old_log_probs: Tensor::new(&log_probs, &[bs, 1]),
        advantages: Tensor::new(advantages, &[bs, 1]),
        returns: Tensor::new(returns, &[bs, 1]),
    }
}

/// 优势标准化（减均值除标准差）
///
/// 降低方差，稳定 PPO 训练。对空切片或零方差不处理（返回原值）。
pub fn normalize_advantages(advantages: &mut [f32]) {
    let n = advantages.len();
    if n < 2 {
        return;
    }
    let mean = advantages.iter().sum::<f32>() / n as f32;
    let var = advantages.iter().map(|&a| (a - mean).powi(2)).sum::<f32>() / n as f32;
    let std = var.sqrt();
    if std < 1e-8 {
        return;
    }
    for a in advantages.iter_mut() {
        *a = (*a - mean) / (std + 1e-8);
    }
}

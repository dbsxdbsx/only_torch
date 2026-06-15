//! SAC（Soft Actor-Critic）函数式 helper
//!
//! 提供 SAC 训练中可跨变体复用的纯函数，涵盖：
//! - [`transitions_to_batch`]：将 `&[Transition]` 一次性转为批量 Tensor
//! - [`compute_td_target`]：TD target `r + γ·(1 - terminated)·V(s')`
//! - [`compute_v_discrete`] / [`compute_v_continuous`] / [`compute_v_hybrid`]：软 V 值计算
//! - [`update_alpha`]：温度参数梯度步进
//!
//! # 设计边界
//! - **入库**：训练步骤的纯函数、通用数学操作
//! - **留示例**：`SacAgent`（有状态）、网络拓扑、训练主循环

mod target;
mod update;

pub use target::{compute_v_continuous, compute_v_discrete, compute_v_hybrid};
pub use update::{compute_td_target, update_alpha};

use crate::rl::Transition;
use crate::tensor::Tensor;

/// 从 `&[Transition]` 提取的批量 Tensor，供 SAC 训练步骤使用。
///
/// 单点转换，避免训练循环中散布 `flat_map` + `Tensor::new` 样板代码。
pub struct SacBatch {
    /// 观察 `[batch, obs_dim]`
    pub obs: Tensor,
    /// 动作 `[batch, action_dim]`（编码约定见 [`Transition`] doc）
    pub actions: Tensor,
    /// 奖励 `[batch, 1]`
    pub rewards: Tensor,
    /// 下一步观察 `[batch, obs_dim]`
    pub next_obs: Tensor,
    /// `1.0 - terminated as f32`，用于 TD target 的 bootstrap 掩码 `[batch, 1]`
    pub not_terminated: Tensor,
}

/// 将 `Transition` 切片转为 [`SacBatch`]。
///
/// `obs_dim` 显式传入以确保 shape 正确（`Transition.obs` 长度已由调用方保证一致）。
///
/// # Panics
/// `transitions` 为空时 panic（空 batch 无意义）。
pub fn transitions_to_batch(transitions: &[Transition], obs_dim: usize) -> SacBatch {
    assert!(!transitions.is_empty(), "transitions_to_batch: 空 batch");
    let bs = transitions.len();
    let action_dim = transitions[0].action.len();

    let obs_data: Vec<f32> = transitions
        .iter()
        .flat_map(|t| t.obs.iter().copied())
        .collect();
    let actions_data: Vec<f32> = transitions
        .iter()
        .flat_map(|t| t.action.iter().copied())
        .collect();
    let rewards: Vec<f32> = transitions.iter().map(|t| t.reward).collect();
    let next_obs_data: Vec<f32> = transitions
        .iter()
        .flat_map(|t| t.next_obs.iter().copied())
        .collect();
    let not_terminated: Vec<f32> = transitions
        .iter()
        .map(|t| if t.terminated { 0.0 } else { 1.0 })
        .collect();

    SacBatch {
        obs: Tensor::new(&obs_data, &[bs, obs_dim]),
        actions: Tensor::new(&actions_data, &[bs, action_dim]),
        rewards: Tensor::new(&rewards, &[bs, 1]),
        next_obs: Tensor::new(&next_obs_data, &[bs, obs_dim]),
        not_terminated: Tensor::new(&not_terminated, &[bs, 1]),
    }
}

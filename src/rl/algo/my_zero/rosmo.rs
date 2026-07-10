//! ROSMO 式一步 target 刷新（Xiao et al., ICLR 2023 · arXiv:2210.05980）。
//!
//! reanalyze 复活阶梯一（[收口规划 §2](../../../../.doc/design/rl_closure_plan.md)）：
//! 训练采样时用当前网络对旧数据**现算**改进 target，一步 look-ahead 替代全树 MCTS 重搜。
//!
//! # 与旧 [`reanalyze`](super::reanalyze)（全树重搜 + 写回）的三点差异
//! - **一步 look-ahead**：对每个候选动作只用 dynamics 推一步
//!   `q(s,a) = r_g + γ·v(s'_g)`，policy target `p(a|s) ∝ π_prior(a|s)·exp(adv)`，
//!   `adv = q − v(s)`——弱模型下误差暴露面最小（论文 Fig.1 的病理诊断）。
//! - **不写回 buffer**：target 是网络的派生量、非环境事实；现算用完即弃，
//!   永远新鲜（与官方 MuZero Reanalyse 的流式设计同构，旧实现的写回属自创偏差）。
//! - **value target 现算 bootstrap**：n-step 尾值用当前网络对 `s_{t+n}` 的评估，
//!   替代 buffer 里 stale 的 self-play root value（论文 Eq.8；本实现用 online 网络，
//!   target 网络接线留 Phase 3 条件项）。
//!
//! 另含**优势过滤行为正则**（论文 Eq.11）：`w_j = 1[adv(s_j, a_j) > 0]`，
//! 训练时对执行过的好动作做 BC（loss 侧见 `UnrollItem::bc_weights`）。
//!
//! CartPole 仅回归覆盖（条款二）；价值裁决在图像域（Pong A/B）。

use super::n_step::compute_n_step_target_indexed;
use crate::rl::SelfPlayStep;
use crate::rl::mcts::{ActionId, ActionPayload, Dynamics};

/// 一条训练样本 unroll 窗口的 ROSMO 现算 target（对齐 [`UnrollItem`](super::network::UnrollItem) 的槽位结构）。
#[derive(Debug, Clone)]
pub struct RosmoTargets {
    /// 各槽位 policy target（len = `actual_k + 1`；越界槽位 = uniform，同 legacy padding）
    pub policies: Vec<Vec<f32>>,
    /// 各槽位 value target（len = `actual_k + 1`；n-step + 现算 bootstrap，越界 = 0）
    pub values: Vec<f32>,
    /// 优势过滤 BC 权重（len = `actual_k`；`1.0` = 执行动作优势为正，`0.0` = 其余/越界）
    pub bc_weights: Vec<f32>,
}

/// 单状态一步 look-ahead：返回 `(policy_target, root_v, 各动作 adv)`。
///
/// `p(a|s) ∝ π_prior(a|s) · exp(adv(s,a))`，`adv = q − v(s)`，
/// `q(s,a) = r_g + γ·v(s'_g)`（terminal 边不 bootstrap）。
/// exp 前减 max(adv) 保数值稳定；退化（和为 0/非有限）时回退 uniform。
pub fn one_step_improved_policy<D: Dynamics>(
    model: &D,
    obs: &[f32],
    gamma: f32,
    action_dim: usize,
) -> (Vec<f32>, f32, Vec<f32>) {
    let actions: Vec<ActionPayload> = (0..action_dim).map(ActionPayload::Discrete).collect();
    one_step_improved_policy_with_actions(model, obs, gamma, &actions)
}

/// 结构化动作版本；`ActionId` 取 catalog 下标，payload 保持真实执行语义。
pub fn one_step_improved_policy_with_actions<D: Dynamics>(
    model: &D,
    obs: &[f32],
    gamma: f32,
    actions: &[ActionPayload],
) -> (Vec<f32>, f32, Vec<f32>) {
    let action_dim = actions.len();
    assert!(action_dim > 0, "ROSMO action catalog 不能为空");
    let (latent, prior, root_v) = model.initial_state(obs);
    assert_eq!(
        prior.len(),
        action_dim,
        "ROSMO prior 宽度 {} 与 action catalog {} 不一致",
        prior.len(),
        action_dim
    );

    let mut advs = Vec::with_capacity(action_dim);
    for (a, payload) in actions.iter().enumerate() {
        let out = model.recurrent_with_id(&latent, ActionId(a), payload);
        let q = if out.terminal {
            out.reward
        } else {
            out.reward + gamma * out.value
        };
        advs.push(q - root_v);
    }

    let max_adv = advs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut target: Vec<f32> = (0..action_dim)
        .map(|a| {
            let p = prior.get(a).copied().unwrap_or(0.0).max(0.0);
            p * (advs[a] - max_adv).exp()
        })
        .collect();
    let z: f32 = target.iter().sum();
    if z > 0.0 && z.is_finite() {
        for p in &mut target {
            *p /= z;
        }
    } else {
        target = vec![1.0 / action_dim as f32; action_dim];
    }
    (target, root_v, advs)
}

/// 对一条样本的 unroll 窗口 `[start, start + actual_k]` 现算 ROSMO target（不写回）。
///
/// - policy：各**真实**槽位一步 look-ahead 改进分布；越界槽位 uniform（同 legacy padding）。
/// - value：`n-step + 现算 bootstrap`（[`compute_n_step_target_indexed`]，
///   尾值 = 当前网络对 bootstrap 位置 obs 的评估，不读 stale `root_value`）。
/// - `bc_weights`：槽位 j 的执行动作 `a_{t+j}` 优势为正 → 1.0（优势过滤行为正则）。
///
/// `obs_at(pos)` 物化位置 `pos` 的模型输入 obs（向量直通；图像模式帧堆叠反量化）。
#[allow(clippy::too_many_arguments)]
pub fn rosmo_refresh_window<D: Dynamics>(
    model: &D,
    steps: &[SelfPlayStep],
    start: usize,
    actual_k: usize,
    td_steps: usize,
    gamma: f32,
    action_dim: usize,
    obs_at: &dyn Fn(usize) -> Vec<f32>,
) -> RosmoTargets {
    let actions: Vec<ActionPayload> = (0..action_dim).map(ActionPayload::Discrete).collect();
    rosmo_refresh_window_with_actions(
        model, steps, start, actual_k, td_steps, gamma, &actions, obs_at,
    )
}

/// 结构化动作 catalog 版本；其余 ROSMO 目标语义与历史函数相同。
#[allow(clippy::too_many_arguments)]
pub fn rosmo_refresh_window_with_actions<D: Dynamics>(
    model: &D,
    steps: &[SelfPlayStep],
    start: usize,
    actual_k: usize,
    td_steps: usize,
    gamma: f32,
    actions: &[ActionPayload],
    obs_at: &dyn Fn(usize) -> Vec<f32>,
) -> RosmoTargets {
    let action_dim = actions.len();
    let len = steps.len();
    let uniform = vec![1.0 / action_dim as f32; action_dim];

    // 现算 bootstrap 值缓存（同一 bootstrap 位置在相邻槽位间复用，省一次前向）
    let value_at = |pos: usize| -> f32 { model.initial_state(&obs_at(pos)).2 };

    let mut policies = Vec::with_capacity(actual_k + 1);
    let mut values = Vec::with_capacity(actual_k + 1);
    let mut bc_weights = Vec::with_capacity(actual_k);

    for j in 0..=actual_k {
        let pos = start + j;
        if pos < len {
            let (policy, _root_v, advs) =
                one_step_improved_policy_with_actions(model, &obs_at(pos), gamma, actions);
            policies.push(policy);
            values.push(compute_n_step_target_indexed(
                steps, pos, td_steps, gamma, value_at,
            ));
            if j < actual_k {
                let a = validated_replay_action_id(&steps[pos].action, action_dim);
                let w = if advs[a] > 0.0 { 1.0 } else { 0.0 };
                bc_weights.push(w);
            }
        } else {
            policies.push(uniform.clone());
            values.push(0.0);
            if j < actual_k {
                bc_weights.push(0.0);
            }
        }
    }

    RosmoTargets {
        policies,
        values,
        bc_weights,
    }
}

pub(crate) fn validated_replay_action_id(action: &[f32], action_dim: usize) -> usize {
    assert_eq!(
        action.len(),
        1,
        "ROSMO replay action 必须只存一个稳定 joint ActionId"
    );
    let raw = action[0];
    assert!(
        raw.is_finite() && raw >= 0.0 && raw.fract() == 0.0,
        "ROSMO replay ActionId 必须是有限非负整数，实际 {raw}"
    );
    let action_id = raw as usize;
    assert!(
        action_id < action_dim,
        "ROSMO replay ActionId {action_id} 越界 action catalog {action_dim}"
    );
    action_id
}

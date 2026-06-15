//! SAC 软状态值（V）计算

use crate::tensor::Tensor;

/// 离散动作的软 V 值。
///
/// `V = Σ_a π(a) × (Q(a) − α · log π(a))`
///
/// - `probs`: `[batch, action_dim]` — 动作概率
/// - `q_min`: `[batch, action_dim]` — min(Q1, Q2)
/// - `log_probs`: `[batch, action_dim]` — log π
/// - 返回 `[batch, 1]`
pub fn compute_v_discrete(
    probs: &Tensor,
    q_min: &Tensor,
    log_probs: &Tensor,
    alpha: f32,
) -> Tensor {
    (probs * &(q_min - &(log_probs * alpha))).sum_axis_keepdims(1)
}

/// 连续动作的软 V 值。
///
/// `V = Q(s, a) − α · log_prob`
///
/// - `q_min`: `[batch, 1]` — min(Q1, Q2)
/// - `log_prob_sum`: `[batch, 1]` — 各维 log prob 之和
/// - 返回 `[batch, 1]`
pub fn compute_v_continuous(q_min: &Tensor, log_prob_sum: &Tensor, alpha: f32) -> Tensor {
    q_min - &(log_prob_sum * alpha)
}

/// 混合动作的软 V 值（双温度）。
///
/// `V = Σ_d π(d) × (Q(d) − α_d · log π(d)) − α_c · log_prob_c`
///
/// - `probs`: `[batch, num_discrete]`
/// - `q_min`: `[batch, num_discrete]`
/// - `log_probs`: `[batch, num_discrete]`
/// - `cont_log_prob_sum`: `[batch, 1]` — 连续维 log prob 之和
/// - 返回 `[batch, 1]`
pub fn compute_v_hybrid(
    probs: &Tensor,
    q_min: &Tensor,
    log_probs: &Tensor,
    alpha_d: f32,
    cont_log_prob_sum: &Tensor,
    alpha_c: f32,
) -> Tensor {
    let discrete_v = (probs * &(q_min - &(log_probs * alpha_d))).sum_axis_keepdims(1);
    &discrete_v - &(cont_log_prob_sum * alpha_c)
}

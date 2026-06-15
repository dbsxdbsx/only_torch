//! SAC 训练更新 helper

use crate::tensor::Tensor;

/// TD target: `r + γ · (1 − terminated) · V(s')`
///
/// - `rewards`: `[batch, 1]`
/// - `v_next`: `[batch, 1]` — 下一步软 V 值
/// - `not_terminated`: `[batch, 1]` — `1.0` 表示未终止，`0.0` 表示已终止
/// - 返回 `[batch, 1]`
pub fn compute_td_target(
    rewards: &Tensor,
    v_next: &Tensor,
    not_terminated: &Tensor,
    gamma: f32,
) -> Tensor {
    rewards + &(not_terminated * &(v_next * gamma))
}

/// Alpha 梯度步进，返回更新后的 `log_alpha`（clamp 到 `[-20, 2]`）。
///
/// 梯度: `∂L/∂log_α = α · (current_entropy − target_entropy)`
///
/// - 实际熵 < 目标 → α 增大（鼓励探索）
/// - 实际熵 > 目标 → α 减小（允许利用）
pub fn update_alpha(
    log_alpha: f32,
    alpha_lr: f32,
    current_entropy: f32,
    target_entropy: f32,
) -> f32 {
    let alpha = log_alpha.exp();
    let grad = alpha * (current_entropy - target_entropy);
    (log_alpha - alpha_lr * grad).clamp(-20.0, 2.0)
}

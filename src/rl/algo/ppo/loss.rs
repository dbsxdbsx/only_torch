//! PPO 损失函数（返回 Var，接入计算图）

use crate::nn::{Var, VarActivationOps, VarLossOps, VarReduceOps};

/// PPO clipped surrogate 策略损失
///
/// L_clip = -min(ratio · A, clip(ratio, 1-ε, 1+ε) · A)
///
/// `old_log_probs` 必须 detach（行为策略常量，不反传）。
pub fn clipped_policy_loss(
    new_log_probs: &Var,
    old_log_probs: &Var,
    advantages: &Var,
    clip_eps: f32,
) -> Var {
    let ratio = (new_log_probs - old_log_probs).exp();
    let clipped_ratio = ratio.clip(1.0 - clip_eps, 1.0 + clip_eps);
    let surr1 = &ratio * advantages;
    let surr2 = &clipped_ratio * advantages;
    -(surr1.minimum(&surr2).expect("clipped_policy_loss: minimum shape 不匹配")).mean()
}

/// PPO value loss（MSE）
///
/// L_v = MSE(new_values, returns)
pub fn value_loss(new_values: &Var, returns: &Var) -> Var {
    new_values.mse_loss(returns).expect("value_loss: shape 不匹配")
}

/// 熵 bonus（鼓励探索）
///
/// L_ent = -mean(entropy)，取负号使梯度下降时最大化熵
pub fn entropy_bonus(entropy: &Var) -> Var {
    -(entropy.mean())
}

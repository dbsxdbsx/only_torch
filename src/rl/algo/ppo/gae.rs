//! GAE（Generalized Advantage Estimation）计算
//!
//! 核心正确性约定：只 mask `terminated`，`truncated` 仍 bootstrap。
//! 延续项目从 `Transition` 起就坚持的 terminated/truncated 分离。

/// 计算 GAE 优势和回报
///
/// # 参数
/// - `rewards`：每步奖励 r_t
/// - `values`：每步 V(s_t)（critic 估计）
/// - `terminated`：MDP 真终止（杆倒了 / 到目标）→ 不 bootstrap
/// - `truncated`：外部截断（时间上限）→ 仍 bootstrap
/// - `next_values`：每步的 V(s_{t+1})（解决跨 episode 边界问题：
///   truncated 时应为被截断状态的后继 V，而非 reset 后的 V）
/// - `last_value`：rollout 末状态的 V(s_{T})，用于最后一步 bootstrap
/// - `gamma`：折扣因子
/// - `lambda`：GAE 衰减因子
///
/// # 返回
/// `(advantages, returns)` 两个 Vec，长度均等于步数
///
/// # 公式
/// δ_t = r_t + γ · V(s_{t+1}) · (1 - terminated_t) - V(s_t)
/// A_t = Σ_{l=0}^{T-t-1} (γλ)^l · δ_{t+l}
/// R_t = A_t + V(s_t)
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    terminated: &[bool],
    truncated: &[bool],
    next_values: &[f32],
    last_value: f32,
    gamma: f32,
    lambda: f32,
) -> (Vec<f32>, Vec<f32>) {
    let n = rewards.len();
    assert_eq!(n, values.len());
    assert_eq!(n, terminated.len());
    assert_eq!(n, truncated.len());
    assert_eq!(n, next_values.len());

    let mut advantages = vec![0.0f32; n];
    let mut last_gae = 0.0f32;

    for t in (0..n).rev() {
        let next_v = if t + 1 < n { next_values[t] } else { last_value };
        let not_terminated = if terminated[t] { 0.0 } else { 1.0 };
        let not_done = if terminated[t] || truncated[t] {
            0.0
        } else {
            1.0
        };

        let delta = rewards[t] + gamma * next_v * not_terminated - values[t];
        last_gae = delta + gamma * lambda * not_done * last_gae;
        advantages[t] = last_gae;
    }

    let returns: Vec<f32> = advantages
        .iter()
        .zip(values.iter())
        .map(|(&a, &v)| a + v)
        .collect();

    (advantages, returns)
}

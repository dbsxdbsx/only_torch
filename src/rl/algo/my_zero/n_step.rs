//! MyZero n-step value target 计算（区分 terminated / truncated）
//!
//! `V_target(t) = Σ_{k=0..n-1} (∏_{i=0}^{k-1} d_{t+i}) · r_{t+k}
//!              + (∏_{i=0}^{n-1} d_{t+i}) · root_value_{t+n}`
//!
//! 其中 `d_t = γ · continuation_t`。
//!
//! # terminated vs truncated（关键正确性）
//! - **terminated**（杆倒）：终止后无后续回报，边界不 bootstrap（`V(s_end)=0`）。
//! - **truncated**（撞步数上限）：人为截断，**仍应 bootstrap**（夹到最后一个有 value 的 step）。

use crate::rl::SelfPlayStep;

/// 从一局 self-play 数据中计算 n-step value target（区分 terminated / truncated）。
pub fn compute_n_step_target(steps: &[SelfPlayStep], start: usize, n: usize, gamma: f32) -> f32 {
    compute_n_step_target_with(steps, start, n, gamma, |s| s.root_value.unwrap_or(0.0))
}

/// 与 [`compute_n_step_target`] 同口径，但 bootstrap 尾值由 `bootstrap_value` 闭包提供，
/// 而非读 `steps[b].root_value`。
///
/// 用于 target network：bootstrap 尾值用「稳定的 target 网络当前评估」，而非 buffer 里
/// stale 的 self-play root value。
pub fn compute_n_step_target_with<F>(
    steps: &[SelfPlayStep],
    start: usize,
    n: usize,
    gamma: f32,
    bootstrap_value: F,
) -> f32
where
    F: Fn(&SelfPlayStep) -> f32,
{
    compute_n_step_target_indexed(steps, start, n, gamma, |idx| bootstrap_value(&steps[idx]))
}

/// 与 [`compute_n_step_target_with`] 同口径，但闭包收到 bootstrap 位置**下标**而非引用。
///
/// 用于图像 obs 等需要按位置组装模型输入（帧堆叠）的现算 bootstrap（ROSMO 一步刷新）。
pub fn compute_n_step_target_indexed<F>(
    steps: &[SelfPlayStep],
    start: usize,
    n: usize,
    gamma: f32,
    bootstrap_value: F,
) -> f32
where
    F: Fn(usize) -> f32,
{
    let len = steps.len();
    if len == 0 || start >= len {
        return 0.0;
    }

    let truncated_end = steps[len - 1].truncated && !steps[len - 1].terminated;
    let max_bootstrap = if truncated_end { len - 1 } else { len };

    let bootstrap = (start + n).min(max_bootstrap);

    let mut target = 0.0;
    let mut discount_prod = 1.0;
    for step in &steps[start..bootstrap] {
        target += discount_prod * step.reward;
        discount_prod *= gamma * step.continuation.clamp(0.0, 1.0);
        if discount_prod <= 0.0 {
            return target;
        }
    }

    if bootstrap < len {
        target += discount_prod * bootstrap_value(bootstrap);
    }

    target
}

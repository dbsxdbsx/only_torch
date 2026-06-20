//! MyZero n-step value target 计算（区分 terminated / truncated）
//!
//! `V_target(t) = Σ_{k=0..n-1} γ^k · r_{t+k} + γ^n · root_value_{t+n}`
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
    let len = steps.len();
    if len == 0 || start >= len {
        return 0.0;
    }

    let truncated_end = !steps[len - 1].terminated;
    let max_bootstrap = if truncated_end { len - 1 } else { len };

    let bootstrap = (start + n).min(max_bootstrap);

    let mut target = 0.0;
    for (offset, step) in steps[start..bootstrap].iter().enumerate() {
        target += gamma.powi(offset as i32) * step.reward;
    }

    if bootstrap < len {
        target += gamma.powi((bootstrap - start) as i32) * bootstrap_value(&steps[bootstrap]);
    }

    target
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::SelfPlayStep;

    fn step(reward: f32, root_value: Option<f32>) -> SelfPlayStep {
        SelfPlayStep {
            obs: vec![],
            action: vec![],
            policy_target: vec![],
            player: 0,
            reward,
            root_value,
            terminated: false,
            extras: Default::default(),
        }
    }

    fn terminated(mut steps: Vec<SelfPlayStep>) -> Vec<SelfPlayStep> {
        steps.last_mut().unwrap().terminated = true;
        steps
    }

    #[test]
    fn basic_n_step() {
        let steps = terminated(vec![
            step(1.0, Some(10.0)),
            step(1.0, Some(10.0)),
            step(1.0, Some(10.0)),
        ]);
        let gamma = 0.99;

        let t = compute_n_step_target(&steps, 0, 1, gamma);
        assert!((t - 10.9).abs() < 1e-5, "n=1: {t}");

        let t = compute_n_step_target(&steps, 0, 3, gamma);
        let expected = 1.0 + 0.99 + 0.99_f32.powi(2);
        assert!((t - expected).abs() < 1e-5, "n=3 无 bootstrap: {t}");
    }

    #[test]
    fn truncation_bootstraps_at_last_value() {
        let steps = vec![
            step(1.0, Some(10.0)),
            step(1.0, Some(20.0)),
            step(1.0, Some(30.0)),
        ];
        let gamma = 0.99;

        let t = compute_n_step_target(&steps, 0, 5, gamma);
        let expected = 1.0 + 0.99 + 0.99_f32.powi(2) * 30.0;
        assert!((t - expected).abs() < 1e-4, "truncation 应 bootstrap: {t}");

        let term = terminated(steps.clone());
        let t_term = compute_n_step_target(&term, 0, 5, gamma);
        let expected_term = 1.0 + 0.99 + 0.99_f32.powi(2);
        assert!(
            (t_term - expected_term).abs() < 1e-4,
            "terminated 不 bootstrap: {t_term}"
        );
    }

    #[test]
    fn no_root_value_no_bootstrap() {
        let steps = terminated(vec![step(2.0, None), step(3.0, None)]);
        let t = compute_n_step_target(&steps, 0, 1, 0.99);
        assert!((t - 2.0).abs() < 1e-5);
    }
}

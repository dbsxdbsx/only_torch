//! MuZero n-step value target 计算
//!
//! 从一局 SelfPlayGame 中计算 n-step bootstrapped return：
//! `V_target(t) = Σ_{k=0..n-1} γ^k · r_{t+k} + γ^n · root_value_{t+n}`

use crate::rl::SelfPlayStep;

/// 从一局 self-play 数据中计算 n-step value target
///
/// `target(t) = Σ_{k=0..n-1} γ^k · reward_{t+k} + γ^n · root_value_{t+n}`
///
/// 若 `t+n` 超出 episode 长度，bootstrap 项为 0（episode 自然结束）。
pub fn compute_n_step_target(
    steps: &[SelfPlayStep],
    start: usize,
    n: usize,
    gamma: f32,
) -> f32 {
    let mut target = 0.0;
    let end = (start + n).min(steps.len());

    for i in start..end {
        target += gamma.powi((i - start) as i32) * steps[i].reward;
    }

    // bootstrap
    if end < steps.len() {
        if let Some(root_v) = steps[end].root_value {
            target += gamma.powi((end - start) as i32) * root_v;
        }
    }

    target
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::SelfPlayStep;

    fn make_step(reward: f32, root_value: Option<f32>) -> SelfPlayStep {
        SelfPlayStep {
            obs: vec![],
            action: vec![],
            policy_target: vec![],
            player: 0,
            reward,
            root_value,
        }
    }

    #[test]
    fn basic_n_step() {
        // 3 步 episode：rewards = [1, 1, 1]，root_values = [10, 10, 10]
        let steps = vec![
            make_step(1.0, Some(10.0)),
            make_step(1.0, Some(10.0)),
            make_step(1.0, Some(10.0)),
        ];
        let gamma = 0.99;

        // n=1, start=0: 1.0 + 0.99 * 10.0 = 10.9
        let t = compute_n_step_target(&steps, 0, 1, gamma);
        assert!((t - 10.9).abs() < 1e-5, "n=1: {t}");

        // n=3, start=0: 1 + 0.99 + 0.99^2，无 bootstrap（end=3 == len）
        let t = compute_n_step_target(&steps, 0, 3, gamma);
        let expected = 1.0 + 0.99 + 0.99_f32.powi(2);
        assert!((t - expected).abs() < 1e-5, "n=3 无 bootstrap: {t}");
    }

    #[test]
    fn bootstrap_at_boundary() {
        let steps = vec![
            make_step(1.0, Some(5.0)),
            make_step(1.0, Some(8.0)),
        ];
        // n=1, start=1: 1.0，end=2 == len，无 bootstrap
        let t = compute_n_step_target(&steps, 1, 1, 0.99);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn n_larger_than_remaining() {
        let steps = vec![make_step(1.0, Some(5.0))];
        // n=10, start=0: 只有 1 步 reward=1.0，无 bootstrap
        let t = compute_n_step_target(&steps, 0, 10, 0.99);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn no_root_value_no_bootstrap() {
        let steps = vec![
            make_step(2.0, None),
            make_step(3.0, None),
        ];
        // n=1, start=0: 2.0 + bootstrap(steps[1].root_value=None) = 2.0
        let t = compute_n_step_target(&steps, 0, 1, 0.99);
        assert!((t - 2.0).abs() < 1e-5);
    }
}

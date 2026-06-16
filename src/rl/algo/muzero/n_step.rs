//! MuZero n-step value target 计算
//!
//! 从一局 SelfPlayGame 中计算 n-step bootstrapped return：
//! `V_target(t) = Σ_{k=0..n-1} γ^k · r_{t+k} + γ^n · root_value_{t+n}`
//!
//! # terminated vs truncated（关键正确性）
//! 镜像 Gymnasium 语义：episode 末端要区分「MDP 真终止」与「步数上限截断」。
//! - **terminated**（杆倒）：终止后无后续回报，边界不 bootstrap（`V(s_end)=0`）。
//! - **truncated**（CartPole 撞 200 步）：是人为截断，**仍应 bootstrap**——否则会把每个
//!   满分局末端 ~n 个位置的 value 目标系统性低估（杀伤「跨到 ≥195」这一段）。
//!
//! truncation 时无法取到截断点之后的状态 value，故 bootstrap 索引夹到**最后一个有 value 的
//! step**（`len-1`），等价于对尾部位置改用更短的 n-step return（无双计、理论合法）。

use crate::rl::SelfPlayStep;

/// 从一局 self-play 数据中计算 n-step value target（区分 terminated / truncated）
///
/// `target(t) = Σ_{k=0..b-1-t} γ^k · reward_{t+k} + γ^(b-t) · root_value_b`
/// 其中 bootstrap 索引 `b = min(t+n, max_bootstrap)`：
/// - episode 以 **terminated** 收尾：`max_bootstrap = len`（到末尾即停，不 bootstrap）；
/// - episode 以 **truncated** 收尾：`max_bootstrap = len-1`（夹到最后有 value 的 step，仍 bootstrap）。
///
/// `b < len` 且 `root_value_b` 存在时才加 bootstrap 项。
pub fn compute_n_step_target(steps: &[SelfPlayStep], start: usize, n: usize, gamma: f32) -> f32 {
    let len = steps.len();
    if len == 0 || start >= len {
        return 0.0;
    }

    // 末步是否 truncation 收尾（非 MDP 真终止）→ 决定边界能否 bootstrap
    let truncated_end = !steps[len - 1].terminated;
    let max_bootstrap = if truncated_end { len - 1 } else { len };

    // bootstrap 索引：正常 = start+n；越界时夹到 max_bootstrap
    let bootstrap = (start + n).min(max_bootstrap);

    let mut target = 0.0;
    for i in start..bootstrap {
        target += gamma.powi((i - start) as i32) * steps[i].reward;
    }

    if bootstrap < len {
        if let Some(root_v) = steps[bootstrap].root_value {
            target += gamma.powi((bootstrap - start) as i32) * root_v;
        }
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

    /// 把一串 step 标记为「terminated 收尾」（末步 terminated=true）
    fn terminated(mut steps: Vec<SelfPlayStep>) -> Vec<SelfPlayStep> {
        steps.last_mut().unwrap().terminated = true;
        steps
    }

    #[test]
    fn basic_n_step() {
        // 3 步 terminated episode：rewards = [1, 1, 1]，root_values = [10, 10, 10]
        let steps = terminated(vec![
            step(1.0, Some(10.0)),
            step(1.0, Some(10.0)),
            step(1.0, Some(10.0)),
        ]);
        let gamma = 0.99;

        // n=1, start=0: 1.0 + 0.99 * 10.0 = 10.9（bootstrap 索引 1 < len）
        let t = compute_n_step_target(&steps, 0, 1, gamma);
        assert!((t - 10.9).abs() < 1e-5, "n=1: {t}");

        // n=3, start=0: 1 + 0.99 + 0.99^2，terminated 收尾无 bootstrap
        let t = compute_n_step_target(&steps, 0, 3, gamma);
        let expected = 1.0 + 0.99 + 0.99_f32.powi(2);
        assert!((t - expected).abs() < 1e-5, "n=3 无 bootstrap: {t}");
    }

    #[test]
    fn bootstrap_at_boundary() {
        let steps = terminated(vec![step(1.0, Some(5.0)), step(1.0, Some(8.0))]);
        // n=1, start=1: 1.0，terminated 收尾，无 bootstrap
        let t = compute_n_step_target(&steps, 1, 1, 0.99);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn n_larger_than_remaining() {
        let steps = terminated(vec![step(1.0, Some(5.0))]);
        // n=10, start=0: 只有 1 步 reward=1.0，terminated 收尾无 bootstrap
        let t = compute_n_step_target(&steps, 0, 10, 0.99);
        assert!((t - 1.0).abs() < 1e-5);
    }

    #[test]
    fn no_root_value_no_bootstrap() {
        let steps = terminated(vec![step(2.0, None), step(3.0, None)]);
        // n=1, start=0: 2.0 + bootstrap(steps[1].root_value=None) = 2.0
        let t = compute_n_step_target(&steps, 0, 1, 0.99);
        assert!((t - 2.0).abs() < 1e-5);
    }

    #[test]
    fn truncation_bootstraps_at_last_value() {
        // truncated 收尾（末步 terminated=false）：尾部应 bootstrap，而非当作终止
        let steps = vec![
            step(1.0, Some(10.0)),
            step(1.0, Some(20.0)),
            step(1.0, Some(30.0)),
        ];
        let gamma = 0.99;

        // n=5（越界），start=0：truncated → max_bootstrap=len-1=2
        // target = r0 + γ r1 + γ^2 · V2 = 1 + 0.99 + 0.9801*30 = 31.393
        let t = compute_n_step_target(&steps, 0, 5, gamma);
        let expected = 1.0 + 0.99 + 0.99_f32.powi(2) * 30.0;
        assert!(
            (t - expected).abs() < 1e-4,
            "truncation 应 bootstrap: got {t}, expected {expected}"
        );

        // 对照：若同样数据是 terminated 收尾，则无 bootstrap
        let term = terminated(steps.clone());
        let t_term = compute_n_step_target(&term, 0, 5, gamma);
        let expected_term = 1.0 + 0.99 + 0.99_f32.powi(2);
        assert!(
            (t_term - expected_term).abs() < 1e-4,
            "terminated 不 bootstrap: got {t_term}"
        );
    }
}

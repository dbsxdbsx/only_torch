//! MyZero value prefix 目标（LSTM 累计 reward 前缀，Ye et al. 2021）。
//!
//! 把「逐步精确预测 reward」改为「预测从子根到第 k 步的累计 reward 前缀和」，规避 reward
//! 落点的 state-aliasing，使监督更稳。
//!
//! 本模块提供**训练期累计前缀目标**的纯函数（[`reward_prefix_targets`]）+ 增量还原校验
//! （[`prefix_to_delta`]）；LSTM value-prefix 头与搜索期 hidden 穿树属网络结构，在
//! [`super::network`] 实现。

/// 计算 value prefix 训练目标：逐步累计 reward 前缀和。
///
/// `prefix[k] = Σ_{i=0..=k} rewards[i]`（不带额外折扣）。返回与 `rewards` 等长的前缀序列。
pub fn reward_prefix_targets(rewards: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(rewards.len());
    let mut running = 0.0;
    for &r in rewards {
        running += r;
        out.push(running);
    }
    out
}

/// 从累计前缀序列还原单步增量：`delta[k] = prefix[k] − prefix[k-1]`（`prefix[-1] = 0`）。
///
/// 用于校验「搜索期 reward 取 prefix 增量」与训练期累计前缀口径一致。
pub fn prefix_to_delta(prefix: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(prefix.len());
    let mut prev = 0.0;
    for &p in prefix {
        out.push(p - prev);
        prev = p;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_accumulates_unit_rewards() {
        let r = [1.0, 1.0, 1.0, 1.0];
        assert_eq!(reward_prefix_targets(&r), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn prefix_handles_varied_rewards() {
        let r = [0.5, -0.25, 2.0];
        let p = reward_prefix_targets(&r);
        assert!((p[0] - 0.5).abs() < 1e-6);
        assert!((p[1] - 0.25).abs() < 1e-6);
        assert!((p[2] - 2.25).abs() < 1e-6);
    }

    #[test]
    fn empty_rewards_give_empty_prefix() {
        assert!(reward_prefix_targets(&[]).is_empty());
    }

    /// 关键契约：prefix 的单步增量必须还原原始 reward。
    #[test]
    fn delta_recovers_step_reward() {
        let r = [1.0, 2.0, -0.5, 3.0];
        let prefix = reward_prefix_targets(&r);
        let delta = prefix_to_delta(&prefix);
        for (a, b) in delta.iter().zip(r.iter()) {
            assert!((a - b).abs() < 1e-6, "prefix 增量应还原单步 reward：{a} vs {b}");
        }
    }
}

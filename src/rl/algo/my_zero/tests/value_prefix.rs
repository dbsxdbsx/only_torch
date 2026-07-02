//! `value_prefix.rs` 前缀目标测试：累计前缀 + 增量还原契约

use super::super::value_prefix::{prefix_to_delta, reward_prefix_targets};

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
        assert!(
            (a - b).abs() < 1e-6,
            "prefix 增量应还原单步 reward：{a} vs {b}"
        );
    }
}

//! self-play 温度调度单测（显式常数调度，与 `max_episodes` 预算解耦）。

use crate::rl::algo::my_zero::runner::self_play_temperature;

/// hold 段恒 1.0；退火段线性降；之后恒 0.25。
#[test]
fn schedule_piecewise_shape() {
    let (hold, decay) = (1000, 1000);
    assert_eq!(self_play_temperature(0, hold, decay), 1.0);
    assert_eq!(self_play_temperature(999, hold, decay), 1.0);
    // 退火段起点 = 1.0（progress=0）
    assert_eq!(self_play_temperature(1000, hold, decay), 1.0);
    // 中点 = 0.625
    let mid = self_play_temperature(1500, hold, decay);
    assert!((mid - 0.625).abs() < 1e-6, "mid={mid}");
    // 退火段结束后恒 0.25
    assert_eq!(self_play_temperature(2000, hold, decay), 0.25);
    assert_eq!(self_play_temperature(999_999, hold, decay), 0.25);
}

/// 与旧实现（`ep / max_episodes` 比例退火，max=2000）在官方 CartPole 口径下逐点等价。
#[test]
fn schedule_matches_legacy_cartpole_2000() {
    let max_episodes = 2000usize;
    for ep in 0..max_episodes {
        let progress = ep as f32 / max_episodes as f32;
        let legacy = if progress < 0.5 {
            1.0
        } else {
            1.0 - (progress - 0.5) * 2.0 * 0.75
        };
        let now = self_play_temperature(ep, 1000, 1000);
        assert!(
            (legacy - now).abs() < 1e-6,
            "ep={ep} legacy={legacy} now={now}"
        );
    }
}

/// decay=0 时 hold 结束直接跳 0.25（无除零）。
#[test]
fn zero_decay_jumps_to_final() {
    assert_eq!(self_play_temperature(9, 10, 0), 1.0);
    assert_eq!(self_play_temperature(10, 10, 0), 0.25);
}

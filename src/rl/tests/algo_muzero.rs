//! MuZero helper 单元测试

use crate::rl::algo::muzero::{compute_n_step_target, value_transform, value_transform_inv};
use crate::rl::algo::muzero::loss;
use crate::rl::SelfPlayStep;

// ============================================================================
// value_transform 测试
// ============================================================================

#[test]
fn value_transform_zero_identity() {
    assert!((value_transform(0.0)).abs() < 1e-7);
    assert!((value_transform_inv(0.0)).abs() < 1e-7);
}

#[test]
fn value_transform_monotonic() {
    let xs: Vec<f32> = (-100..=100).map(|i| i as f32 * 2.0).collect();
    for w in xs.windows(2) {
        assert!(
            value_transform(w[1]) > value_transform(w[0]),
            "h({}) 应 > h({})",
            w[1],
            w[0]
        );
    }
}

#[test]
fn value_transform_compresses_cartpole_range() {
    let h200 = value_transform(200.0);
    assert!(h200 < 15.0, "h(200)={h200}，CartPole 最大 value 应被压缩");
    assert!(h200 > 10.0, "h(200)={h200}，不应退化到 0");

    let h1 = value_transform(1.0);
    assert!(h1 > 0.0 && h1 < 1.5, "h(1)={h1}，小 value 近似恒等");
}

#[test]
fn value_transform_round_trip() {
    let test_values = [
        0.0, 1.0, -1.0, 5.0, -5.0, 10.0, -10.0, 50.0, -50.0, 100.0, -100.0, 200.0, -200.0,
    ];
    for &x in &test_values {
        let y = value_transform(x);
        let x_back = value_transform_inv(y);
        let err = (x_back - x).abs();
        assert!(
            err < 0.5,
            "round-trip: x={x}, h(x)={y}, h⁻¹(h(x))={x_back}, err={err}"
        );
    }
}

#[test]
fn value_transform_negative_symmetry() {
    for &x in &[1.0, 10.0, 100.0, 200.0] {
        let hp = value_transform(x);
        let hn = value_transform(-x);
        assert!(
            (hp + hn).abs() < 1e-5,
            "h({x}) + h(-{x}) = {} 应为 0（奇函数）",
            hp + hn
        );
    }
}

// ============================================================================
// n_step_target 测试
// ============================================================================

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
fn n_step_target_basic() {
    let steps = vec![
        make_step(1.0, Some(10.0)),
        make_step(1.0, Some(10.0)),
        make_step(1.0, Some(10.0)),
    ];
    let gamma = 0.99;

    // n=1, start=0: r_0 + γ·V_1 = 1.0 + 0.99*10.0 = 10.9
    let t = compute_n_step_target(&steps, 0, 1, gamma);
    assert!((t - 10.9).abs() < 1e-4, "n=1: got {t}");
}

#[test]
fn n_step_target_no_bootstrap_at_end() {
    let steps = vec![
        make_step(1.0, Some(10.0)),
        make_step(1.0, Some(10.0)),
    ];
    // n=2, start=0: 两步用完，end=2==len，无 bootstrap
    let t = compute_n_step_target(&steps, 0, 2, 0.99);
    let expected = 1.0 + 0.99;
    assert!((t - expected).abs() < 1e-4, "got {t}, expected {expected}");
}

#[test]
fn n_step_target_oversized_n() {
    let steps = vec![make_step(2.0, Some(999.0))];
    // n=100 但只有 1 步，end=1==len，无 bootstrap
    let t = compute_n_step_target(&steps, 0, 100, 0.99);
    assert!((t - 2.0).abs() < 1e-4);
}

// ============================================================================
// loss 常量测试
// ============================================================================

#[test]
fn loss_constants_valid() {
    assert!(loss::VALUE_LOSS_COEF > 0.0 && loss::VALUE_LOSS_COEF <= 1.0);
    assert!(loss::REWARD_LOSS_COEF > 0.0 && loss::REWARD_LOSS_COEF <= 1.0);
    assert!(loss::DYNAMICS_GRADIENT_SCALE > 0.0 && loss::DYNAMICS_GRADIENT_SCALE <= 1.0);
}

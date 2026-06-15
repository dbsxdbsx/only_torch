//! MuZero helper 单元测试

use crate::rl::SelfPlayStep;
use crate::rl::algo::muzero::loss;
use crate::rl::algo::muzero::{
    SupportConfig, compute_n_step_target, scalar_to_two_hot, two_hot_to_scalar, value_transform,
    value_transform_inv,
};

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
        terminated: false,
    }
}

/// 把一串 step 标记为「terminated 收尾」（末步 terminated=true）
fn terminated(mut steps: Vec<SelfPlayStep>) -> Vec<SelfPlayStep> {
    steps.last_mut().unwrap().terminated = true;
    steps
}

#[test]
fn n_step_target_basic() {
    let steps = terminated(vec![
        make_step(1.0, Some(10.0)),
        make_step(1.0, Some(10.0)),
        make_step(1.0, Some(10.0)),
    ]);
    let gamma = 0.99;

    // n=1, start=0: r_0 + γ·V_1 = 1.0 + 0.99*10.0 = 10.9
    let t = compute_n_step_target(&steps, 0, 1, gamma);
    assert!((t - 10.9).abs() < 1e-4, "n=1: got {t}");
}

#[test]
fn n_step_target_no_bootstrap_at_end() {
    let steps = terminated(vec![make_step(1.0, Some(10.0)), make_step(1.0, Some(10.0))]);
    // n=2, start=0: 两步用完，terminated 收尾，无 bootstrap
    let t = compute_n_step_target(&steps, 0, 2, 0.99);
    let expected = 1.0 + 0.99;
    assert!((t - expected).abs() < 1e-4, "got {t}, expected {expected}");
}

#[test]
fn n_step_target_oversized_n() {
    let steps = terminated(vec![make_step(2.0, Some(999.0))]);
    // n=100 但只有 1 步，terminated 收尾，无 bootstrap
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

// ============================================================================
// categorical support / two-hot 测试
// ============================================================================

#[test]
fn support_config_size_and_atoms() {
    let cfg = SupportConfig::new(20);
    assert_eq!(cfg.size(), 41, "size = 2*half+1");
    assert_eq!(cfg.half_size(), 20);
    // 原子值 = i - half
    assert!((cfg.atom(0) - (-20.0)).abs() < 1e-6);
    assert!((cfg.atom(20) - 0.0).abs() < 1e-6, "中心原子应为 0");
    assert!((cfg.atom(40) - 20.0).abs() < 1e-6);
}

#[test]
fn two_hot_sums_to_one() {
    let cfg = SupportConfig::new(20);
    for &x in &[
        0.0, 1.0, -1.0, 5.0, 42.0, -42.0, 150.0, 333.0, -333.0, 0.37, -0.37,
    ] {
        let th = scalar_to_two_hot(x, &cfg);
        assert_eq!(th.len(), cfg.size());
        let s: f32 = th.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "two-hot({x}) 和应为 1，实际 {s}");
        for &p in &th {
            assert!(p >= -1e-7, "two-hot 概率不应为负: {p}");
        }
        // 至多两个非零分量
        let nonzero = th.iter().filter(|&&p| p > 1e-7).count();
        assert!(nonzero <= 2, "two-hot({x}) 非零分量应 ≤ 2，实际 {nonzero}");
    }
}

#[test]
fn two_hot_round_trip_in_range() {
    // half=20 覆盖变换域 [-20,20] → value 域约 ±420，足够覆盖 CartPole bootstrap 值（~333）
    let cfg = SupportConfig::new(20);
    let test_values = [
        0.0, 1.0, -1.0, 5.0, -5.0, 50.0, -50.0, 150.0, -150.0, 333.0, -333.0,
    ];
    for &x in &test_values {
        let th = scalar_to_two_hot(x, &cfg);
        let back = two_hot_to_scalar(&th, &cfg);
        let err = (back - x).abs();
        // 容差随量级放宽：two-hot 在变换域线性插值，大 value 经 h⁻¹ 后绝对误差略增
        let tol = 1.0 + x.abs() * 0.02;
        assert!(
            err < tol,
            "round-trip: x={x}, back={back}, err={err}, tol={tol}"
        );
    }
}

#[test]
fn two_hot_clamps_out_of_range() {
    let cfg = SupportConfig::new(20);
    // 远超 support 覆盖范围的 value：编码应饱和到最高原子，解码回来被 clamp
    let th = scalar_to_two_hot(1.0e6, &cfg);
    assert!(
        (th[cfg.size() - 1] - 1.0).abs() < 1e-5,
        "极大值应全压最高原子"
    );
    let th_neg = scalar_to_two_hot(-1.0e6, &cfg);
    assert!((th_neg[0] - 1.0).abs() < 1e-5, "极小值应全压最低原子");
}

#[test]
fn two_hot_reward_small_value() {
    // CartPole reward 恒 +1，验证小标量编解码精确
    let cfg = SupportConfig::new(20);
    let th = scalar_to_two_hot(1.0, &cfg);
    let back = two_hot_to_scalar(&th, &cfg);
    assert!((back - 1.0).abs() < 0.05, "reward=1 round-trip: {back}");

    // reward=0 应精确落在中心原子
    let th0 = scalar_to_two_hot(0.0, &cfg);
    let back0 = two_hot_to_scalar(&th0, &cfg);
    assert!(back0.abs() < 1e-4, "reward=0 round-trip: {back0}");
}

//! PPO helper 单元测试（纯 Rust，无 pyo3）

use crate::rl::algo::ppo::{compute_gae, normalize_advantages};

// ============================================================================
// GAE 基础：3 步手算
// ============================================================================

#[test]
fn test_gae_3_steps_no_termination() {
    // 3 步，无终止/截断，gamma=0.99, lambda=0.95
    // V = [1.0, 2.0, 3.0], r = [0.5, 1.0, 0.5], last_value = 4.0
    //
    // δ_2 = 0.5 + 0.99*4.0 - 3.0 = 0.5 + 3.96 - 3.0 = 1.46
    // δ_1 = 1.0 + 0.99*3.0 - 2.0 = 1.0 + 2.97 - 2.0 = 1.97
    // δ_0 = 0.5 + 0.99*2.0 - 1.0 = 0.5 + 1.98 - 1.0 = 1.48
    //
    // A_2 = 1.46
    // A_1 = 1.97 + 0.99*0.95*1.46 = 1.97 + 1.37337 = 3.34337
    // A_0 = 1.48 + 0.99*0.95*3.34337 = 1.48 + 3.14347... = 4.62347...

    let rewards = [0.5, 1.0, 0.5];
    let values = [1.0, 2.0, 3.0];
    let terminated = [false, false, false];
    let truncated = [false, false, false];

    let next_values = [values[1], values[2], 4.0];
    let (adv, ret) = compute_gae(&rewards, &values, &terminated, &truncated, &next_values, 4.0, 0.99, 0.95);

    assert_eq!(adv.len(), 3);
    assert!((adv[2] - 1.46).abs() < 1e-4, "A_2={}", adv[2]);
    assert!((adv[1] - 3.34337).abs() < 1e-3, "A_1={}", adv[1]);
    assert!((adv[0] - 4.62347).abs() < 1e-2, "A_0={}", adv[0]);

    for i in 0..3 {
        assert!(
            (ret[i] - (adv[i] + values[i])).abs() < 1e-6,
            "returns[{i}] 应等于 adv + value"
        );
    }
}

// ============================================================================
// GAE terminated 路径：真终止不 bootstrap
// ============================================================================

#[test]
fn test_gae_terminated_no_bootstrap() {
    // 步 0: 正常; 步 1: terminated → δ_1 不 bootstrap V(s_2)
    let rewards = [1.0, 1.0];
    let values = [0.5, 0.5];
    let terminated = [false, true];
    let truncated = [false, false];

    let next_values = [values[1], 10.0]; // terminated 时 next_value 不影响（被 mask）
    let (adv, _) = compute_gae(&rewards, &values, &terminated, &truncated, &next_values, 10.0, 0.99, 0.95);

    // δ_1 = 1.0 + 0.99 * 10.0 * 0.0 - 0.5 = 0.5 (terminated → not_terminated=0)
    // A_1 = 0.5
    assert!(
        (adv[1] - 0.5).abs() < 1e-6,
        "terminated 时不应 bootstrap: A_1={}",
        adv[1]
    );
}

// ============================================================================
// GAE truncated 路径：截断仍 bootstrap value，但不延续 GAE 链
// ============================================================================

#[test]
fn test_gae_truncated_bootstraps_value() {
    // 步 0: 正常; 步 1: truncated → δ_1 仍 bootstrap V(s_2)，但 GAE 链断开
    let rewards = [1.0, 1.0];
    let values = [0.5, 0.5];
    let terminated = [false, false];
    let truncated = [false, true];

    // next_values[1] 是被截断状态的真实后继 V（不是 reset 后的）
    let next_values = [values[1], 10.0];
    let (adv, _) = compute_gae(&rewards, &values, &terminated, &truncated, &next_values, 10.0, 0.99, 0.95);

    // δ_1 = 1.0 + 0.99 * 10.0 * 1.0 - 0.5 = 10.4 (truncated → not_terminated=1, bootstrap 有效)
    // A_1 = 10.4
    assert!(
        (adv[1] - 10.4).abs() < 1e-4,
        "truncated 应仍 bootstrap value: A_1={}",
        adv[1]
    );

    // δ_0 = 1.0 + 0.99 * 0.5 - 0.5 = 0.995
    // A_0 = 0.995 + 0.99 * 0.95 * 1.0 * 10.4 ≈ 10.776
    assert!(
        (adv[0] - 10.776).abs() < 0.01,
        "同 episode 内应携带 advantage: A_0={}",
        adv[0]
    );
}

// ============================================================================
// 优势标准化
// ============================================================================

#[test]
fn test_normalize_advantages() {
    let mut adv = vec![1.0, 3.0, 5.0, 7.0];
    normalize_advantages(&mut adv);

    let mean: f32 = adv.iter().sum::<f32>() / adv.len() as f32;
    assert!(mean.abs() < 1e-5, "标准化后均值应近零: {mean}");

    let var: f32 = adv.iter().map(|&a| a * a).sum::<f32>() / adv.len() as f32;
    assert!(
        (var - 1.0).abs() < 0.1,
        "标准化后方差应近 1: {var}"
    );
}

#[test]
fn test_normalize_single_element() {
    let mut adv = vec![5.0];
    normalize_advantages(&mut adv);
    assert!((adv[0] - 5.0).abs() < 1e-6, "单元素不标准化");
}

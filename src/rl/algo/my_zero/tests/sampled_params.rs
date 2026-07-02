//! `sampled_params.rs` B/N/K 解析测试：团队公式 K_cfg = min(max(5, N/2), floor(sims×2/3))

use super::super::config::ActionPlan;
use super::super::sampled_params::{
    compute_sampled_k_cfg, resolve_sampled_params, sampled_k_effective,
};

#[test]
fn cartpole_n2_sims20() {
    let p = resolve_sampled_params(ActionPlan::Auto, 2, 20);
    assert_eq!(p.b, None);
    assert_eq!(p.n, 2);
    assert_eq!(p.k_cfg, 5);
    assert_eq!(p.k_effective, 2);
}

#[test]
fn discrete_a20_sims20() {
    let p = resolve_sampled_params(ActionPlan::Auto, 20, 20);
    assert_eq!(p.k_cfg, 10);
    assert_eq!(p.k_effective, 10);
}

#[test]
fn pendulum_1d_b7_sims20() {
    let p = resolve_sampled_params(ActionPlan::Discretize { buckets: 7 }, 7, 20);
    assert_eq!(p.b, Some(7));
    assert_eq!(p.n, 7);
    assert_eq!(p.k_cfg, 5);
    assert_eq!(p.k_effective, 5);
}

#[test]
fn pendulum_1d_b10_sims20() {
    let p = resolve_sampled_params(ActionPlan::Discretize { buckets: 10 }, 10, 20);
    assert_eq!(p.b, Some(10));
    assert_eq!(p.k_cfg, 5);
    assert_eq!(p.k_effective, 5);
}

#[test]
fn hybrid_joint147_sims20() {
    // Platform 类：|A_d|=3 × B=7 × B=7
    let p = resolve_sampled_params(ActionPlan::Auto, 147, 20);
    assert_eq!(p.k_cfg, 13);
    assert_eq!(p.k_effective, 13);
}

#[test]
fn continuous_2d_b7_sims20() {
    let p = resolve_sampled_params(ActionPlan::Discretize { buckets: 7 }, 49, 20);
    assert_eq!(p.b, Some(7));
    assert_eq!(p.n, 49);
    assert_eq!(p.k_cfg, 13);
    assert_eq!(p.k_effective, 13);
}

#[test]
fn small_n_clamps_k_eff() {
    assert_eq!(sampled_k_effective(5, 3), 3);
}

#[test]
fn sim_cap_for_large_n() {
    assert_eq!(compute_sampled_k_cfg(1000, 20), 13);
}

//! `n_step.rs` value target 测试：terminated/truncated 边界、continuation 折扣连乘

use super::super::n_step::compute_n_step_target;
use crate::rl::SelfPlayStep;

fn step(reward: f32, root_value: Option<f32>) -> SelfPlayStep {
    SelfPlayStep {
        obs: vec![].into(),
        action: vec![],
        policy_target: vec![],
        player: 0,
        reward,
        root_value,
        terminated: false,
        truncated: false,
        continuation: 1.0,
        extras: Default::default(),
    }
}

fn terminated(mut steps: Vec<SelfPlayStep>) -> Vec<SelfPlayStep> {
    steps.last_mut().unwrap().terminated = true;
    steps.last_mut().unwrap().continuation = 0.0;
    steps
}

fn truncated(mut steps: Vec<SelfPlayStep>) -> Vec<SelfPlayStep> {
    steps.last_mut().unwrap().truncated = true;
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
    let steps = truncated(vec![
        step(1.0, Some(10.0)),
        step(1.0, Some(20.0)),
        step(1.0, Some(30.0)),
    ]);
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

#[test]
fn continuation_zero_stops_future_value() {
    let mut steps = vec![
        step(2.0, Some(10.0)),
        step(3.0, Some(20.0)),
        step(4.0, Some(30.0)),
    ];
    steps[1].continuation = 0.0;
    let t = compute_n_step_target(&steps, 0, 3, 0.5);
    let expected = 2.0 + 0.5 * 3.0;
    assert!(
        (t - expected).abs() < 1e-5,
        "continuation=0 后不应再累积 reward/bootstrap: {t}"
    );
}

#[test]
fn variable_continuation_discount_product() {
    let mut steps = truncated(vec![
        step(1.0, Some(10.0)),
        step(2.0, Some(20.0)),
        step(3.0, Some(30.0)),
    ]);
    steps[0].continuation = 0.5;
    steps[1].continuation = 0.25;
    let t = compute_n_step_target(&steps, 0, 5, 0.8);
    let expected = 1.0 + (0.8 * 0.5) * 2.0 + (0.8 * 0.5) * (0.8 * 0.25) * 30.0;
    assert!(
        (t - expected).abs() < 1e-5,
        "应按每步 transition discount 连乘: {t}"
    );
}

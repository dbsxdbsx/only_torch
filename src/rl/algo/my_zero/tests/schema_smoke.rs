//! 新 observation/action schema 的最小训练管线 smoke。

use crate::rl::algo::my_zero::MyZero;
use serial_test::serial;

fn run_smoke(builder: super::super::builder::MyZeroBuilder) {
    let model = builder
        .solved(f32::INFINITY)
        .max_episodes(3)
        .num_simulations(2)
        .train_batch_size(2)
        .smoke()
        .train()
        .expect("schema smoke 应跑通");
    let report = model.train_report().expect("smoke 应生成训练报告");
    assert!(report.wall_secs.is_finite() && report.final_greedy.is_finite());
}

#[test]
#[serial]
fn multidiscrete_myzero_smoke() {
    run_smoke(MyZero::new("MyZero-MultiDiscrete-v0"));
}

#[test]
#[serial]
fn continuous_2d_myzero_smoke() {
    run_smoke(MyZero::new("MyZero-Continuous2D-v0").discretize(3));
}

#[test]
#[serial]
fn image_dense_myzero_smoke() {
    run_smoke(MyZero::new("MyZero-ImageDense-v0"));
}

#[test]
#[serial]
fn token_myzero_smoke() {
    run_smoke(MyZero::new("MyZero-Token-v0").token_observation(6, 32, 8));
}

#[test]
#[serial]
#[ignore = "manual: Platform-v0 Hybrid MyZero smoke（需 hybrid-platform）"]
fn platform_hybrid_myzero_smoke() {
    run_smoke(MyZero::new("Platform-v0").discretize(7));
}

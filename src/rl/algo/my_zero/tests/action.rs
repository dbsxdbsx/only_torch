//! `action.rs` 连续离散化测试：bin 中点语义（Sampled MuZero Appendix 对齐）

use super::super::action::{ActionAdapter, idx_to_continuous};
use super::super::config::ActionPlan;
use super::super::schema::ActionSchema;
use crate::rl::GymEnv;
use approx::assert_abs_diff_eq;
use pyo3::Python;
use serial_test::serial;

#[test]
fn discretize_uses_bin_centers() {
    // [0, 10] × 10 档 → 中点 0.5, 1.5, …, 9.5
    assert!((idx_to_continuous(0, 0.0, 10.0, 10) - 0.5).abs() < 1e-5);
    assert!((idx_to_continuous(9, 0.0, 10.0, 10) - 9.5).abs() < 1e-5);
    assert!((idx_to_continuous(4, 0.0, 10.0, 10) - 4.5).abs() < 1e-5);
}

#[test]
fn discretize_single_bucket_is_midpoint() {
    assert!((idx_to_continuous(0, -2.0, 2.0, 1) - 0.0).abs() < 1e-5);
}

#[test]
#[serial]
fn resolves_multidiscrete_toy_environment() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "MyZero-MultiDiscrete-v0");
        let adapter = ActionAdapter::resolve(&env, ActionPlan::Auto);
        assert_eq!(
            adapter.schema(),
            &ActionSchema::MultiDiscrete {
                factors: vec![4, 4, 16]
            }
        );
        assert_eq!(adapter.action_dim(), 256);
        let id = adapter.schema().encode_joint(&[1, 2, 3]);
        assert_eq!(adapter.to_env(id), vec![1.0, 2.0, 3.0]);
        let _ = env.reset(Some(42));
        let (_obs, _reward, _terminated, _truncated) = env.step(&adapter.to_env(id));
        env.close();
    });
}

#[test]
#[serial]
fn resolves_two_dimensional_continuous_toy() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "MyZero-Continuous2D-v0");
        let adapter = ActionAdapter::resolve(&env, ActionPlan::Discretize { buckets: 3 });
        assert_eq!(adapter.action_dim(), 9);
        assert!(matches!(
            adapter.schema(),
            ActionSchema::ContinuousBins { ranges, buckets: 3 } if ranges.len() == 2
        ));
        let center = adapter.schema().encode_joint(&[1, 1]);
        let env_action = adapter.to_env(center);
        assert_abs_diff_eq!(env_action[0], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(env_action[1], 0.0, epsilon = 1e-6);
        env.close();
    });
}

#[test]
#[serial]
fn resolves_platform_fixed_hybrid_tuple() {
    Python::attach(|py| {
        let env = GymEnv::new(py, "Platform-v0");
        let adapter = ActionAdapter::resolve(&env, ActionPlan::Discretize { buckets: 7 });
        assert!(matches!(
            adapter.schema(),
            ActionSchema::HybridBins {
                discrete,
                continuous,
                buckets: 7,
            } if discrete == &[3] && continuous.len() == 3
        ));
        assert_eq!(adapter.action_dim(), 3 * 7 * 7 * 7);
        let center = adapter.schema().encode_joint(&[1, 3, 3, 3]);
        let env_action = adapter.to_env(center);
        assert_eq!(env_action.len(), 4);
        let _ = env.reset(Some(42));
        let (_obs, _reward, _terminated, _truncated) = env.step(&env_action);
        env.close();
    });
}

//! `recipe.rs` 环境配方测试：CartPole promoted 栈、Pendulum 诊断栈、未知 env 退回 base

use super::super::component::Components;
use super::super::config::ActionPlan;
use super::super::recipe::{action_plan_for, components_for};
use super::super::sampled_params::DEFAULT_CONTINUOUS_BUCKETS;

#[test]
fn cartpole_has_consistency_and_reconstruction() {
    let c = components_for("CartPole-v1");
    assert!(c.consistency);
    assert!(c.reconstruction);
    assert!(c.sampled);
    assert!(!c.reanalyze);
    assert!(!c.value_prefix);
    assert!(!c.target_net);
    assert!(!c.sve_enabled());
    assert!(!c.gumbel);
    assert!(!c.completed_q_target);
}

#[test]
fn pendulum_has_same_stack_as_cartpole() {
    let cp = components_for("CartPole-v1");
    let pe = components_for("Pendulum-v1");
    assert_eq!(cp, pe);
    assert!(pe.consistency);
    assert!(pe.reconstruction);
    assert!(pe.sampled);
    assert!(!pe.reanalyze);
}

#[test]
fn pendulum_default_action_is_b7() {
    assert_eq!(
        action_plan_for("Pendulum-v1"),
        ActionPlan::Discretize {
            buckets: DEFAULT_CONTINUOUS_BUCKETS
        }
    );
}

#[test]
fn unknown_env_is_base() {
    let c = components_for("LunarLander-v3");
    assert_eq!(c, Components::base());
}

#[test]
fn large_structured_action_recipes_enable_sampled_only() {
    for env in ["MyZero-MultiDiscrete-v0", "Platform-v0"] {
        let c = components_for(env);
        assert!(c.sampled, "env={env} 应避免展开完整 joint catalog");
        let mut expected = Components::base();
        expected.sampled = true;
        assert_eq!(c, expected, "env={env}");
    }
}

/// 棋盘栈 = base 全关（M4 收口定型；M3 九臂消融无组件过 promote 线）。
#[test]
fn gomoku_is_base() {
    for env in ["Gomoku-selfplay-v0", "Gomoku-random-v0"] {
        assert_eq!(components_for(env), Components::base(), "env={env}");
    }
}

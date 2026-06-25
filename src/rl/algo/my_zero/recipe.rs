//! 按环境内置算法配方（内部；用户 API 不暴露组件开关）。
//!
//! 团队 promote 组件时只改此模块；公开入口 [`MyZero::new`](super::my_zero::MyZero::new) 自动套用。

use super::component::Components;
use super::config::ActionPlan;
use super::sampled_params::DEFAULT_CONTINUOUS_BUCKETS;

/// CartPole 已验收的 MyZero 组件栈（consistency + reconstruction + Sampled）。
fn cartpole_stack() -> Components {
    let mut c = Components::base();
    c.consistency = true;
    c.reconstruction = true;
    c.sampled = true;
    // K 由 sampled_params 按 N、sims 自动算
    // reanalyze / completedQ / Gumbel：CartPole promote 暂缓 → .issue/items/
    c.reanalyze = false;
    c
}

/// Pendulum 当前诊断栈：先复用 CartPole 已验收组件，但不把 Pendulum 裁决标记为已通过。
fn pendulum_diagnostic_stack() -> Components {
    cartpole_stack()
}

/// 给定 Gymnasium `env_id` 返回当前内置组件组合。
pub(crate) fn components_for(env_id: &str) -> Components {
    match env_id {
        "CartPole-v1" => cartpole_stack(),
        "Pendulum-v1" => pendulum_diagnostic_stack(),
        _ => Components::base(),
    }
}

/// 给定 `env_id` 返回默认动作方案（连续 env 用 Sampled MuZero 默认 B=7；离散 env 用 Auto）。
pub(crate) fn action_plan_for(env_id: &str) -> ActionPlan {
    match env_id {
        "Pendulum-v1" => ActionPlan::Discretize {
            buckets: DEFAULT_CONTINUOUS_BUCKETS,
        },
        _ => ActionPlan::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}

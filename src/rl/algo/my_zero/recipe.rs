//! 按环境内置算法配方（内部；用户 API 不暴露组件开关）。
//!
//! 团队 promote 组件时只改此模块；公开入口 [`MyZero::new`](super::my_zero::MyZero::new) 自动套用。

use super::component::Components;

/// 给定 Gymnasium `env_id` 返回已验收的组件组合（未 promote 的组件保持关）。
pub(crate) fn components_for(env_id: &str) -> Components {
    match env_id {
        "CartPole-v1" => {
            let mut c = Components::base();
            c.consistency = true;
            // reanalyze 写回已实现；CartPole promote 暂缓 → .issue/items/my_zero_reanalyze_cartpole_regression.md
            c.reanalyze = false;
            c
        }
        _ => Components::base(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cartpole_has_consistency_only() {
        let c = components_for("CartPole-v1");
        assert!(c.consistency);
        assert!(!c.reanalyze);
        assert!(!c.value_prefix);
        assert!(!c.target_net);
        assert!(!c.sve_enabled());
        assert!(!c.gumbel);
        assert!(!c.completed_q_target);
    }

    #[test]
    fn unknown_env_is_base() {
        let c = components_for("Pendulum-v1");
        assert_eq!(c, Components::base());
    }
}

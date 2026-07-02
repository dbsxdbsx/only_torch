//! `config.rs` 默认配置测试：base 全关默认、CartPole 友好训练参数、recipe 注入

use super::super::component::Components;
use super::super::config::{ActionPlan, DEFAULT_SEED, EvalSettings, MyZeroConfig, TrainSettings};
use super::super::my_zero::MyZero;
use super::super::sampled_params::DEFAULT_CONTINUOUS_BUCKETS;

#[test]
fn default_is_base() {
    let cfg = MyZeroConfig::default();
    assert_eq!(cfg.components, Components::base(), "默认组件全关 = base");
    assert_eq!(cfg.train, TrainSettings::default());
    assert_eq!(cfg.env.env_id, "CartPole-v1");
    assert!((cfg.env.reward_scale - 1.0).abs() < 1e-6);
    assert_eq!(cfg.env.action, ActionPlan::Auto);
    assert_eq!(cfg.model.latent_dim, 64);
    assert!(!cfg.eval.checkpoint.enabled, "默认不落盘");
}

#[test]
fn train_default_is_cartpole_friendly() {
    let t = TrainSettings::default();
    assert_eq!(t.num_simulations, 20);
    assert_eq!(t.k_unroll, 5);
    assert!((t.gamma - 0.997).abs() < 1e-6);
    assert_eq!(t.train_batch_size, 8);
}

#[test]
fn eval_default_seed() {
    let e = EvalSettings::default();
    assert_eq!(e.seed, DEFAULT_SEED);
    assert_eq!(e.seed_runs, 1);
}

#[test]
fn component_toggle() {
    let cfg = MyZeroConfig {
        components: Components {
            consistency: true,
            ..Components::default()
        },
        ..MyZeroConfig::default()
    };
    assert!(cfg.components.consistency);
    assert!(!cfg.components.value_prefix, "只开 consistency，其余不变");
}

#[test]
fn new_cartpole_applies_recipe() {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(100)
        .build()
        .unwrap();
    assert!(cfg.components.consistency);
    assert!(cfg.components.reconstruction);
    assert!(cfg.components.sampled);
    assert!(!cfg.components.reanalyze);
    assert!(!cfg.components.completed_q_target);
    assert_eq!(cfg.env.action, ActionPlan::Auto);
}

#[test]
fn new_pendulum_applies_recipe() {
    let cfg = MyZero::new("Pendulum-v1")
        .solved(-200.0)
        .max_episodes(100)
        .build()
        .unwrap();
    assert!(cfg.components.consistency);
    assert!(cfg.components.reconstruction);
    assert!(cfg.components.sampled);
    assert_eq!(
        cfg.env.action,
        ActionPlan::Discretize {
            buckets: DEFAULT_CONTINUOUS_BUCKETS
        }
    );
}

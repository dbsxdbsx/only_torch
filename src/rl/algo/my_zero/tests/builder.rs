//! `builder.rs` 链式配置测试：recipe 注入、必填校验、组件冲突守卫

use super::super::config::{ActionPlan, ObservationPlan};
use super::super::my_zero::MyZero;
use crate::nn::GraphError;

#[test]
fn discrete_env_defaults_to_auto_action() {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .build()
        .unwrap();
    assert_eq!(cfg.env.action, ActionPlan::Auto);
}

#[test]
fn cartpole_recipe_has_consistency_and_reconstruction() {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .build()
        .unwrap();
    assert!(cfg.components.consistency);
    assert!(cfg.components.reconstruction);
    assert!(cfg.components.sampled);
    assert!(!cfg.components.reanalyze);
    assert!(!cfg.components.completed_q_target);
}

#[test]
fn pendulum_recipe_matches_cartpole_stack() {
    let cfg = MyZero::new("Pendulum-v1")
        .reward_scale(0.1)
        .solved(-200.0)
        .max_episodes(600)
        .build()
        .unwrap();
    assert!(cfg.components.consistency);
    assert!(cfg.components.reconstruction);
    assert!(cfg.components.sampled);
    assert!((cfg.env.reward_scale - 0.1).abs() < 1e-6);
}

#[test]
fn td_steps_override_clamps_to_at_least_one() {
    let cfg = MyZero::new("CartPole-v1")
        .td_steps(0)
        .solved(475.0)
        .max_episodes(2000)
        .build()
        .unwrap();
    assert_eq!(cfg.train.td_steps, 1);
}

#[test]
fn save_model_when_eval_sets_path() {
    let path = std::path::PathBuf::from("models/my_zero/CartPole-v1/seed_42/best");
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .save_model_when_eval(&path)
        .build()
        .unwrap();
    assert!(cfg.eval.checkpoint.enabled);
    assert_eq!(cfg.eval.checkpoint.best_base.as_ref(), Some(&path));
}

#[test]
fn seed_sets_config() {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(100)
        .seed(99)
        .build()
        .unwrap();
    assert_eq!(cfg.eval.seed, 99);
}

#[test]
fn missing_solved_is_error_on_train() {
    let r = MyZero::new("CartPole-v1").max_episodes(2000).train();
    assert!(matches!(r, Err(GraphError::InvalidOperation(_))));
}

#[test]
fn missing_max_episodes_is_error_on_train() {
    let r = MyZero::new("CartPole-v1").solved(475.0).train();
    assert!(matches!(r, Err(GraphError::InvalidOperation(_))));
}

#[test]
fn gumbel_disables_sampled_by_default() {
    let cfg = MyZero::new("CartPole-v1")
        .gumbel(true)
        .solved(475.0)
        .max_episodes(2000)
        .build()
        .unwrap();
    assert!(cfg.components.gumbel);
    assert!(!cfg.components.sampled);
}

#[test]
fn gumbel_and_sampled_conflict_is_error() {
    let r = MyZero::new("CartPole-v1")
        .gumbel(true)
        .sampled(true)
        .solved(475.0)
        .max_episodes(2000)
        .build();
    assert!(matches!(r, Err(GraphError::InvalidOperation(msg)) if msg.contains("Gumbel root")));
}

#[test]
fn observation_builders_record_schema_choices() {
    let image = MyZero::new("CartPole-v1")
        .image_observation(144, 256, 3)
        .solved(1.0)
        .max_episodes(1)
        .build()
        .unwrap();
    assert_eq!(
        image.env.observation,
        ObservationPlan::Image {
            height: 144,
            width: 256,
            history: 3,
        }
    );

    let tokens = MyZero::new("CartPole-v1")
        .token_observation_with_padding(16, 1024, 32, 7)
        .solved(1.0)
        .max_episodes(1)
        .build()
        .unwrap();
    assert_eq!(
        tokens.env.observation,
        ObservationPlan::Tokens {
            length: 16,
            vocab_size: 1024,
            embed_dim: 32,
            pad_id: 7,
        }
    );
}

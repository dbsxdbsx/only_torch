//! 阶段 ②：随机轨迹算术 → hint → 再进 [`route_components`]。
//!
//! 不碰 Gym / Python；用人造轨迹锁谓词。`!` 组件必须保持关。

use super::super::component::Components;
use super::super::probe::{
    OBS_SCALE_ORDER_RATIO, ProbeStats, REWARD_SCALE_COMPRESSED, REWARD_SCALE_DEFAULT,
    REWARD_SCALE_DOWN_MEAN_ABS, probe_trajectories,
};
use super::super::route::{EnvStructure, route_components};
use super::super::schema::{ActionSchema, ObservationSchema};

fn cartpole_schema() -> EnvStructure {
    EnvStructure::new(ObservationSchema::Flat(4), ActionSchema::discrete(2))
}

fn cartpole_obs() -> Vec<Vec<f32>> {
    vec![
        vec![0.12, -0.40, 0.03, 0.55],
        vec![-0.20, 1.10, -0.05, -0.80],
        vec![0.05, 0.20, 0.01, 0.15],
        vec![-0.08, -0.90, 0.04, 1.20],
    ]
}

fn cartpole_rewards() -> Vec<f32> {
    vec![1.0, 1.0, 1.0, 1.0]
}

fn assert_bang_and_unrouted_off(c: &Components) {
    assert!(!c.reanalyze);
    assert!(!c.rosmo);
    assert!(!c.target_net);
    assert!(!c.sve_enabled());
    assert!(!c.value_prefix);
    assert!(!c.completed_q_target);
    assert!(!c.gumbel);
    assert!(!c.hl_gauss);
    assert!(!c.recurrent_posterior);
}

fn route_after_probe(
    mut env: EnvStructure,
    observations: &[Vec<f32>],
    rewards: &[f32],
) -> (ProbeStats, Components) {
    let stats = probe_trajectories(env.observation, observations, rewards);
    env.apply_probe(&stats);
    let c = route_components(&env);
    (stats, c)
}

#[test]
fn cartpole_like_probe_keeps_symlog_off_and_reward_scale_one() {
    let (stats, c) = route_after_probe(cartpole_schema(), &cartpole_obs(), &cartpole_rewards());
    assert!(
        !stats.obs_scale_spans_orders,
        "CartPole 各维都在 O(1)，不跨数量级"
    );
    assert_eq!(stats.reward_scale, REWARD_SCALE_DEFAULT);
    assert!(c.reconstruction);
    assert!(!c.consistency);
    assert!(!c.sampled);
    assert!(!c.obs_symlog);
    assert_eq!(c.num_chance_outcomes, 8);
    assert_bang_and_unrouted_off(&c);
}

#[test]
fn wide_flat_probe_opens_symlog() {
    let obs = vec![vec![0.01, 1.0, 0.5, -0.2], vec![2.0, 1.1, 0.4, 0.3]];
    // 维 0：2.0 / 0.01 = 200 ≥ 100
    let (stats, c) = route_after_probe(cartpole_schema(), &obs, &cartpole_rewards());
    assert!(stats.obs_scale_spans_orders);
    assert!(c.obs_symlog);
    assert!(c.reconstruction);
    assert_bang_and_unrouted_off(&c);
}

#[test]
fn scale_ratio_threshold_is_inclusive() {
    let just_below = vec![vec![1.0, 1.0], vec![OBS_SCALE_ORDER_RATIO - 1.0, 1.0]];
    let at_ratio = vec![vec![1.0, 1.0], vec![OBS_SCALE_ORDER_RATIO, 1.0]];
    let env = EnvStructure::new(ObservationSchema::Flat(2), ActionSchema::discrete(2));
    assert!(!probe_trajectories(env.observation, &just_below, &[]).obs_scale_spans_orders);
    assert!(probe_trajectories(env.observation, &at_ratio, &[]).obs_scale_spans_orders);
}

#[test]
fn image_probe_never_sets_scale_hint() {
    let env = EnvStructure::new(
        ObservationSchema::Image {
            channels: 1,
            side: 2,
        },
        ActionSchema::discrete(3),
    );
    let obs = vec![vec![0.01, 0.02, 0.03, 2.0], vec![3.0, 0.01, 0.02, 0.03]];
    let (stats, c) = route_after_probe(env, &obs, &[0.0, 1.0]);
    assert!(
        !stats.obs_scale_spans_orders,
        "非 Flat 即使数值跨数量级也不填 hint"
    );
    assert!(!c.obs_symlog);
    assert!(c.consistency);
}

#[test]
fn board_probe_never_sets_scale_hint() {
    let env = EnvStructure::new(
        ObservationSchema::Board {
            channels: 1,
            side: 2,
        },
        ActionSchema::discrete(4),
    );
    let obs = vec![vec![0.0, 1.0, 0.0, 1.0]];
    let stats = probe_trajectories(env.observation, &obs, &[0.0]);
    assert!(!stats.obs_scale_spans_orders);
}

#[test]
fn empty_or_nonfinite_probe_is_conservative() {
    let obs_schema = ObservationSchema::Flat(2);
    let empty = probe_trajectories(obs_schema, &[], &[]);
    assert!(!empty.obs_scale_spans_orders);
    assert_eq!(empty.reward_scale, REWARD_SCALE_DEFAULT);

    let nan_obs = vec![vec![f32::NAN, f32::INFINITY], vec![0.0, 0.0]];
    let nan_r = vec![f32::NAN, f32::NEG_INFINITY];
    let stats = probe_trajectories(obs_schema, &nan_obs, &nan_r);
    assert!(!stats.obs_scale_spans_orders);
    assert_eq!(stats.reward_scale, REWARD_SCALE_DEFAULT);
}

#[test]
fn pendulum_like_rewards_suggest_compressed_scale() {
    let rewards = vec![-8.2, -7.1, -6.5, -9.0, -5.5];
    let stats = probe_trajectories(ObservationSchema::Flat(3), &[], &rewards);
    assert_eq!(stats.reward_scale, REWARD_SCALE_COMPRESSED);
}

#[test]
fn reward_mean_at_threshold_stays_default() {
    let rewards = vec![REWARD_SCALE_DOWN_MEAN_ABS; 4];
    let stats = probe_trajectories(ObservationSchema::Flat(1), &[], &rewards);
    assert_eq!(stats.reward_scale, REWARD_SCALE_DEFAULT);
}

#[test]
fn probe_does_not_invent_pomdp_or_value_noise() {
    let mut env = cartpole_schema();
    let stats = probe_trajectories(env.observation, &cartpole_obs(), &cartpole_rewards());
    env.apply_probe(&stats);
    assert!(!env.partially_observable);
    assert!(!env.value_support_noisy);
    let c = route_components(&env);
    assert!(!c.recurrent_posterior);
    assert!(!c.hl_gauss);
}

#[test]
fn tokens_probe_does_not_open_symlog() {
    let env = EnvStructure::new(
        ObservationSchema::Tokens {
            length: 4,
            vocab_size: 16,
            embed_dim: 8,
            pad_id: 0,
        },
        ActionSchema::discrete(3),
    );
    let obs = vec![vec![1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0]];
    let (stats, c) = route_after_probe(env, &obs, &[1.0]);
    assert!(!stats.obs_scale_spans_orders);
    assert!(!c.obs_symlog);
    assert!(!c.consistency);
    assert!(!c.reconstruction);
}

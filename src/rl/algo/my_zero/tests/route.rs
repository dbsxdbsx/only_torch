//! 结构路由：不同观测 / 动作特征应自动开或关哪些组件。
//!
//! 测的是 [`route_components`](super::super::route::route_components)，不是 `env_id` recipe。
//! CartPole 等已 promote 栈可以比这里更「满」（例如小离散也开 Sampled），那是 overlay，不在本文件断言。

use super::super::component::Components;
use super::super::route::{EnvStructure, FLAT_RECON_MAX_DIM, route_components};
use super::super::sampled_params::AUTO_SAMPLED_MIN_JOINT_ACTIONS;
use super::super::schema::{ActionSchema, ObservationSchema};

fn cartpole_like() -> EnvStructure {
    EnvStructure::new(ObservationSchema::Flat(4), ActionSchema::discrete(2))
}

fn pendulum_like() -> EnvStructure {
    EnvStructure::new(
        ObservationSchema::Flat(3),
        ActionSchema::ContinuousBins {
            ranges: vec![(-2.0, 2.0)],
            buckets: 7,
        },
    )
}

fn minatar_like() -> EnvStructure {
    EnvStructure::new(
        ObservationSchema::Board {
            channels: 4,
            side: 10,
        },
        ActionSchema::discrete(3),
    )
}

fn atari_like() -> EnvStructure {
    EnvStructure::new(
        ObservationSchema::Image {
            channels: 4,
            side: 84,
        },
        ActionSchema::discrete(6),
    )
}

fn gomoku_like() -> EnvStructure {
    EnvStructure::new(
        ObservationSchema::Board {
            channels: 3,
            side: 15,
        },
        ActionSchema::discrete(225),
    )
}

fn platform_like() -> EnvStructure {
    EnvStructure::new(
        ObservationSchema::Flat(10),
        ActionSchema::HybridBins {
            discrete: vec![3],
            continuous: vec![(-1.0, 1.0)],
            buckets: 7,
        },
    )
}

/// 没有结构谓词的旋钮必须保持关（含 KL / PER 这类过程旋钮：根本不在 Components 里开）。
fn assert_unrouted_knobs_off(c: &Components) {
    assert!(!c.reanalyze, "reanalyze 无结构谓词");
    assert!(!c.rosmo, "ROSMO 无结构谓词");
    assert!(!c.target_net, "target_net 无结构谓词");
    assert!(!c.sve_enabled(), "SVE 无结构谓词");
    assert!(!c.value_prefix, "value_prefix 断链，路由不得开");
    assert!(
        !c.completed_q_target,
        "completedQ 取决于 sims，不在 schema 里"
    );
    assert!(!c.gumbel, "Gumbel 与 Sampled 暂不兼容，结构路由不开");
}

#[test]
fn stochastic_k8_is_on_for_every_structure() {
    for env in [
        cartpole_like(),
        pendulum_like(),
        minatar_like(),
        atari_like(),
        gomoku_like(),
        platform_like(),
    ] {
        let c = route_components(&env);
        assert_eq!(c.num_chance_outcomes, 8, "K=8 always-on");
        assert_unrouted_knobs_off(&c);
    }
}

#[test]
fn cartpole_like_opens_recon_not_spatial_or_sampled() {
    let c = route_components(&cartpole_like());
    assert!(c.reconstruction, "低维向量：obs 比特任务相关");
    assert!(!c.consistency, "非空间观测：consistency 不是结构必开");
    assert!(
        !c.sampled,
        "N=2 < {AUTO_SAMPLED_MIN_JOINT_ACTIONS}，Sampled 只是簿记"
    );
    assert!(!c.recurrent_posterior);
    assert!(c.obs_symlog, "连续观测 always-on");
    assert!(!c.hl_gauss);
}

#[test]
fn pendulum_like_opens_recon_and_sampled_for_continuous() {
    let c = route_components(&pendulum_like());
    assert!(c.reconstruction);
    assert!(c.sampled, "连续动作必须 Sampled，不能展开无界 catalog");
    assert!(!c.consistency);
    assert!(!c.gumbel);
}

#[test]
fn minatar_like_opens_consistency_not_recon() {
    let c = route_components(&minatar_like());
    assert!(c.consistency, "棋盘 / 网格是空间观测");
    assert!(!c.reconstruction, "网格通道不是低维语义向量");
    assert!(!c.sampled);
    assert!(!c.hl_gauss, "图像噪声未声明时不开 HL-Gauss");
}

#[test]
fn atari_like_matches_minatar_spatial_rules() {
    let c = route_components(&atari_like());
    assert!(c.consistency);
    assert!(!c.reconstruction);
    assert!(!c.sampled);
    assert!(c.obs_symlog, "连续观测 always-on，含像素");
}

#[test]
fn gomoku_like_opens_consistency_and_sampled_for_large_board() {
    let c = route_components(&gomoku_like());
    assert!(c.consistency);
    assert!(c.sampled, "15×15=225 ≥ {AUTO_SAMPLED_MIN_JOINT_ACTIONS}");
    assert!(!c.reconstruction);
}

#[test]
fn small_board_does_not_open_sampled() {
    let env = EnvStructure::new(
        ObservationSchema::Board {
            channels: 2,
            side: 9,
        },
        ActionSchema::discrete(81),
    );
    let c = route_components(&env);
    assert!(c.consistency);
    assert!(!c.sampled, "81 < {AUTO_SAMPLED_MIN_JOINT_ACTIONS}");
}

#[test]
fn platform_hybrid_opens_sampled() {
    let c = route_components(&platform_like());
    assert!(c.sampled);
    assert!(c.reconstruction, "Flat(10) ≤ {FLAT_RECON_MAX_DIM}");
    assert!(!c.consistency);
}

#[test]
fn multidiscrete_joint_threshold_controls_sampled() {
    let small = EnvStructure::new(
        ObservationSchema::Flat(6),
        ActionSchema::MultiDiscrete {
            factors: vec![3, 5],
        },
    );
    let large = EnvStructure::new(
        ObservationSchema::Flat(6),
        ActionSchema::MultiDiscrete {
            factors: vec![16, 16],
        },
    );
    assert!(!route_components(&small).sampled, "3×5=15");
    assert!(route_components(&large).sampled, "16×16=256");
}

#[test]
fn posterior_follows_partial_observability_flag() {
    let mut env = cartpole_like();
    assert!(!route_components(&env).recurrent_posterior);
    env.partially_observable = true;
    assert!(route_components(&env).recurrent_posterior);
}

#[test]
fn obs_symlog_on_for_all_continuous_schemas() {
    for env in [
        cartpole_like(),
        pendulum_like(),
        minatar_like(),
        atari_like(),
        gomoku_like(),
        platform_like(),
    ] {
        assert!(
            route_components(&env).obs_symlog,
            "连续观测不读 Box、一律 symlog"
        );
    }
}

#[test]
fn hl_gauss_only_when_value_noise_hint_set() {
    assert!(!route_components(&atari_like()).hl_gauss);
    let mut env = atari_like();
    env.value_support_noisy = true;
    assert!(route_components(&env).hl_gauss);
}

#[test]
fn high_dim_flat_does_not_open_reconstruction() {
    let env = EnvStructure::new(
        ObservationSchema::Flat(FLAT_RECON_MAX_DIM + 1),
        ActionSchema::discrete(4),
    );
    assert!(!route_components(&env).reconstruction);
}

#[test]
fn image_rect_counts_as_spatial() {
    let env = EnvStructure::new(
        ObservationSchema::ImageRect {
            channels: 3,
            height: 12,
            width: 16,
        },
        ActionSchema::discrete(4),
    );
    let c = route_components(&env);
    assert!(c.consistency);
    assert!(!c.reconstruction);
    assert!(c.obs_symlog);
}

#[test]
fn empty_flat_does_not_open_reconstruction() {
    let env = EnvStructure::new(ObservationSchema::Flat(0), ActionSchema::discrete(2));
    assert!(!route_components(&env).reconstruction);
}

#[test]
fn tokens_are_neither_spatial_nor_flat_recon() {
    let env = EnvStructure::new(
        ObservationSchema::Tokens {
            length: 8,
            vocab_size: 64,
            embed_dim: 16,
            pad_id: 0,
        },
        ActionSchema::discrete(4),
    );
    let c = route_components(&env);
    assert!(!c.consistency);
    assert!(!c.reconstruction);
    assert!(!c.obs_symlog, "token ID 查表前不能 symlog");
}

#[test]
fn image_dense_counts_as_spatial() {
    let env = EnvStructure::new(
        ObservationSchema::ImageDense {
            channels: 3,
            height: 16,
            width: 16,
            aux_dim: 4,
        },
        ActionSchema::discrete(5),
    );
    let c = route_components(&env);
    assert!(c.consistency);
    assert!(!c.reconstruction);
}

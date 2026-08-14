//! 结构路由：不同观测 / 动作特征应自动开或关哪些组件。
//!
//! 测的是 [`route_components`](super::super::route::route_components)。
//! CartPole 生产路径已与本函数对齐；Pendulum 等仍可有更「满」的 recipe overlay。

use super::super::component::Components;
use super::super::route::{EnvStructure, FLAT_RECON_MAX_DIM, route_components};
use super::super::schema::{ActionSchema, ObservationSchema};

/// 结构路由单测默认搜索预算；与 CartPole / Pendulum recipe 的 sims=20 对齐。
const TEST_SIMS: u32 = 20;

fn route(env: &EnvStructure) -> Components {
    route_components(env, TEST_SIMS)
}

fn cartpole_like() -> EnvStructure {
    super::super::route::cartpole_structure()
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
        let c = route(&env);
        assert_eq!(c.num_chance_outcomes, 8, "K=8 always-on");
        assert_unrouted_knobs_off(&c);
    }
}

#[test]
fn cartpole_like_opens_recon_not_spatial_or_sampled() {
    let c = route(&cartpole_like());
    assert!(c.reconstruction, "低维向量：obs 比特任务相关");
    assert!(!c.consistency, "非空间观测：consistency 不是结构必开");
    assert!(!c.sampled, "N=2，K_eff=2，全展开");
    assert!(!c.recurrent_posterior);
    assert!(c.obs_symlog, "连续观测 always-on");
    assert!(!c.hl_gauss);
}

#[test]
fn tiny_continuous_bins_can_full_enum() {
    let env = EnvStructure::new(
        ObservationSchema::Flat(3),
        ActionSchema::ContinuousBins {
            ranges: vec![(-2.0, 2.0)],
            buckets: 3,
        },
    );
    assert!(!route(&env).sampled, "连续先算 N=3，K_eff=3，不是类型必开");
}

#[test]
fn pendulum_like_opens_recon_and_sampled_for_continuous() {
    let c = route(&pendulum_like());
    assert!(c.reconstruction);
    assert!(c.sampled, "N=7，sims=20 → K_eff=5 < 7");
    assert!(!c.consistency);
    assert!(!c.gumbel);
}

#[test]
fn minatar_like_opens_consistency_not_recon() {
    let c = route(&minatar_like());
    assert!(c.consistency, "棋盘 / 网格是空间观测");
    assert!(!c.reconstruction, "网格通道不是低维语义向量");
    assert!(!c.sampled, "N=3，K_eff=3，全展开");
    assert!(!c.hl_gauss, "图像噪声未声明时不开 HL-Gauss");
}

#[test]
fn atari_like_matches_minatar_spatial_rules() {
    let c = route(&atari_like());
    assert!(c.consistency);
    assert!(!c.reconstruction);
    assert!(c.sampled, "N=6，sims=20 → K_eff=5 < 6");
    assert!(c.obs_symlog, "连续观测 always-on，含像素");
}

#[test]
fn gomoku_like_opens_consistency_and_sampled_for_large_board() {
    let c = route(&gomoku_like());
    assert!(c.consistency);
    assert!(c.sampled, "N=225，sims=20 → K_eff=13 < 225");
    assert!(!c.reconstruction);
}

#[test]
fn nine_by_nine_opens_sampled() {
    let env = EnvStructure::new(
        ObservationSchema::Board {
            channels: 2,
            side: 9,
        },
        ActionSchema::discrete(81),
    );
    let c = route(&env);
    assert!(c.consistency);
    assert!(c.sampled, "N=81，sims=20 → K_eff=13 < 81");
}

#[test]
fn platform_hybrid_opens_sampled() {
    let c = route(&platform_like());
    assert!(c.sampled);
    assert!(c.reconstruction, "Flat(10) ≤ {FLAT_RECON_MAX_DIM}");
    assert!(!c.consistency);
}

#[test]
fn multidiscrete_follows_k_eff_lt_n() {
    let full_enum = EnvStructure::new(
        ObservationSchema::Flat(6),
        ActionSchema::MultiDiscrete {
            factors: vec![2, 2],
        },
    );
    let subsample = EnvStructure::new(
        ObservationSchema::Flat(6),
        ActionSchema::MultiDiscrete {
            factors: vec![3, 2],
        },
    );
    assert!(!route(&full_enum).sampled, "2×2=4，K_eff=4");
    assert!(route(&subsample).sampled, "3×2=6，K_eff=5 < 6");
}

#[test]
fn sampled_reads_sims_at_n5_boundary() {
    let env = EnvStructure::new(ObservationSchema::Flat(4), ActionSchema::discrete(5));
    assert!(
        !route_components(&env, 20).sampled,
        "N=5，sims=20 → K_eff=5"
    );
    assert!(
        route_components(&env, 3).sampled,
        "N=5，sims=3 → K_eff=2 < 5"
    );
}

#[test]
fn posterior_follows_partial_observability_flag() {
    let mut env = cartpole_like();
    assert!(!route(&env).recurrent_posterior);
    env.partially_observable = true;
    assert!(route(&env).recurrent_posterior);
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
        assert!(route(&env).obs_symlog, "连续观测不读 Box、一律 symlog");
    }
}

#[test]
fn hl_gauss_only_when_value_noise_hint_set() {
    assert!(!route(&atari_like()).hl_gauss);
    let mut env = atari_like();
    env.value_support_noisy = true;
    assert!(route(&env).hl_gauss);
}

#[test]
fn high_dim_flat_does_not_open_reconstruction() {
    let env = EnvStructure::new(
        ObservationSchema::Flat(FLAT_RECON_MAX_DIM + 1),
        ActionSchema::discrete(4),
    );
    assert!(!route(&env).reconstruction);
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
    let c = route(&env);
    assert!(c.consistency);
    assert!(!c.reconstruction);
    assert!(c.obs_symlog);
}

#[test]
fn empty_flat_does_not_open_reconstruction() {
    let env = EnvStructure::new(ObservationSchema::Flat(0), ActionSchema::discrete(2));
    assert!(!route(&env).reconstruction);
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
    let c = route(&env);
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
    let c = route(&env);
    assert!(c.consistency);
    assert!(!c.reconstruction);
}

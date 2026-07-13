//! Phase 3A0 手动档：冻结 checkpoint 上审计 model-error proxy 的任务相关性与可学习性。
//!
//! 三个预算档使用同一训练 seed、温度与 recipe；同 seed 的 early/middle/late 是同一
//! 确定性训练轨迹的前缀 checkpoint。审计 episode 使用独立固定 seed lane，不进 replay。
//!
//! ```bash
//! M3_SEEDS=52 cargo test --release --features blas-mkl gomoku_p3a0_proxy_early \
//!   -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::GymEnv;
use crate::rl::algo::my_zero::board::{
    BoardTrainConfig, train_board_on_games, train_board_with_checkpoints, train_board_with_model,
};
use crate::rl::algo::my_zero::component::Components;
use crate::rl::algo::my_zero::model_error_audit::{
    BoardAuditTransition, BoardProxyAuditReport, audit_gomoku_proxy, collect_gomoku_audit_block,
    collect_gomoku_audit_data, score_gomoku_audit_block,
};
use std::collections::HashSet;

const DISCOVERY_SEEDS: &[u64] = &[52, 53, 54];
const AUDIT_EPISODES: usize = 20;
const INTERVENTION_UPDATES: usize = 500;

fn filter_seeds_by_env(all: &[u64]) -> Vec<u64> {
    match std::env::var("M3_SEEDS") {
        Ok(raw) => {
            let wanted: Vec<u64> = raw
                .split(',')
                .filter_map(|token| token.trim().parse().ok())
                .collect();
            let filtered: Vec<u64> = all
                .iter()
                .copied()
                .filter(|seed| wanted.contains(seed))
                .collect();
            assert!(
                !filtered.is_empty(),
                "M3_SEEDS='{raw}' 与 3A0 discovery seeds {all:?} 无交集"
            );
            filtered
        }
        Err(_) => all.to_vec(),
    }
}

fn p3a0_cfg(seed: u64, max_episodes: usize) -> BoardTrainConfig {
    BoardTrainConfig {
        seed,
        true_rules_tree: false,
        cnn_repr: true,
        augment: true,
        max_episodes,
        temp_hold_episodes: 5000,
        temp_decay_episodes: 0,
        buffer_capacity: 500,
        snapshot_at_episode: None,
        eval_every: usize::MAX,
        gate_games: 0,
        naive_ladder: false,
        early_stop: false,
        components: Components::base(),
        ..Default::default()
    }
}

fn run_proxy_audit(stage: &str, max_episodes: usize) -> Result<(), GraphError> {
    for seed in filter_seeds_by_env(DISCOVERY_SEEDS) {
        println!("\n--- Phase 3A0 stage={stage} seed={seed} train_episodes={max_episodes} ---");
        let cfg = p3a0_cfg(seed, max_episodes);
        let (train_report, model) = train_board_with_model(&cfg)?;
        let audit = pyo3::Python::attach(|py| {
            let env = GymEnv::new(py, cfg.env_id);
            let report = audit_gomoku_proxy(
                &env,
                &model,
                &cfg.components,
                cfg.num_simulations,
                AUDIT_EPISODES,
                seed.wrapping_add(30_000_000),
            );
            env.close();
            report
        });
        print_audit(stage, seed, &train_report, &audit);
        assert!(!audit.records.is_empty(), "3A0 audit 不应为空");
        assert!(
            audit
                .records
                .iter()
                .all(|record| record.error.is_finite_nonnegative()),
            "3A0 所有误差分量必须有限非负"
        );
    }
    Ok(())
}

fn print_audit(
    stage: &str,
    seed: u64,
    train: &super::super::board::BoardTrainReport,
    audit: &BoardProxyAuditReport,
) {
    println!(
        "[3A0] stage={stage} seed={seed} train_env_steps={} audit_transitions={} tactical_positions={}",
        train.total_env_steps,
        audit.records.len(),
        audit.tactical_positions,
    );
    print_component_rows(audit);
}

fn print_component_rows(audit: &BoardProxyAuditReport) {
    for component in &audit.components {
        println!(
            "  {:>20}: n={} mean={:.6} rho_policy={} tactical_position_top10_lift={}",
            component.name,
            component.count,
            component.mean,
            format_optional(component.spearman_to_reference_policy),
            format_optional(component.tactical_position_top_decile_lift),
        );
    }
}

fn format_optional(value: Option<f32>) -> String {
    value.map_or_else(|| "n/a".to_owned(), |value| format!("{value:.4}"))
}

fn transition_key(transition: &BoardAuditTransition) -> Vec<u32> {
    let mut key: Vec<u32> = transition.obs.iter().map(|value| value.to_bits()).collect();
    key.push(transition.action.index() as u32);
    key
}

fn run_proxy_intervention() -> Result<(), GraphError> {
    for seed in filter_seeds_by_env(DISCOVERY_SEEDS) {
        let cfg = p3a0_cfg(seed, 2000);
        let (_report, model) = train_board_with_model(&cfg)?;
        let (train_data, mut holdout) = pyo3::Python::attach(|py| {
            let env = GymEnv::new(py, cfg.env_id);
            let train = collect_gomoku_audit_data(
                &env,
                &model,
                &cfg.components,
                cfg.num_simulations,
                30,
                seed.wrapping_add(70_000_000),
            );
            let holdout = collect_gomoku_audit_block(
                &env,
                &model,
                &cfg.components,
                cfg.num_simulations,
                20,
                seed.wrapping_add(80_000_000),
            );
            env.close();
            (train, holdout)
        });
        let train_keys: HashSet<Vec<u32>> =
            train_data.transitions.iter().map(transition_key).collect();
        holdout.retain(|transition| !train_keys.contains(&transition_key(transition)));
        assert!(!holdout.is_empty(), "去重后 intervention holdout 不能为空");

        let before = score_gomoku_audit_block(&model, &holdout);
        let losses = train_board_on_games(
            &model,
            &train_data.games,
            &cfg,
            INTERVENTION_UPDATES,
            seed ^ 0x3A00_7001,
        )?;
        let after = score_gomoku_audit_block(&model, &holdout);
        println!(
            "[3A0-intervention] seed={seed} train_games={} holdout_transitions={} updates={} final_loss={:.6}",
            train_data.games.len(),
            holdout.len(),
            INTERVENTION_UPDATES,
            losses.last().copied().unwrap_or(0.0),
        );
        for (before_component, after_component) in before.components.iter().zip(&after.components) {
            assert_eq!(before_component.name, after_component.name);
            let relative = if before_component.mean > 1e-8 {
                (after_component.mean / before_component.mean - 1.0) * 100.0
            } else {
                0.0
            };
            println!(
                "  {:>20}: before={:.6} after={:.6} relative={relative:+.2}%",
                before_component.name, before_component.mean, after_component.mean,
            );
        }
        assert!(losses.iter().all(|loss| loss.is_finite()));
    }
    Ok(())
}

/// 同一 early future block 在后续 revision 上重评分，隔离 on-policy 分布漂移。
fn run_fixed_block_reducibility() -> Result<(), GraphError> {
    for seed in filter_seeds_by_env(DISCOVERY_SEEDS) {
        let cfg = p3a0_cfg(seed, 5000);
        let (_train_report, late_model, snapshots) =
            train_board_with_checkpoints(&cfg, &[400, 2000])?;
        let early_model = snapshots
            .iter()
            .find_map(|(episode, model)| (*episode == 400).then_some(model))
            .expect("应捕获 early checkpoint");
        let middle_model = snapshots
            .iter()
            .find_map(|(episode, model)| (*episode == 2000).then_some(model))
            .expect("应捕获 middle checkpoint");
        let fixed_block = pyo3::Python::attach(|py| {
            let env = GymEnv::new(py, cfg.env_id);
            let block = collect_gomoku_audit_block(
                &env,
                early_model,
                &cfg.components,
                cfg.num_simulations,
                AUDIT_EPISODES,
                seed.wrapping_add(40_000_000),
            );
            env.close();
            block
        });
        for (stage, episode, model) in [
            ("early", 400, early_model),
            ("middle", 2000, middle_model),
            ("late", 5000, &late_model),
        ] {
            let audit = score_gomoku_audit_block(model, &fixed_block);
            println!(
                "[3A0-fixed] stage={stage} seed={seed} train_episodes={episode} fixed_transitions={} tactical_positions={}",
                audit.records.len(),
                audit.tactical_positions,
            );
            print_component_rows(&audit);
        }
    }
    Ok(())
}

#[test]
#[ignore = "manual: Phase 3A0 early checkpoint（3 seeds × 400 局 + 未来 20 局审计）"]
fn gomoku_p3a0_proxy_early() -> Result<(), GraphError> {
    run_proxy_audit("early", 400)
}

#[test]
#[ignore = "manual: Phase 3A0 middle checkpoint（3 seeds × 2000 局 + 未来 20 局审计）"]
fn gomoku_p3a0_proxy_middle() -> Result<(), GraphError> {
    run_proxy_audit("middle", 2000)
}

#[test]
#[ignore = "manual: Phase 3A0 late checkpoint（3 seeds × 5000 局 + 未来 20 局审计）"]
fn gomoku_p3a0_proxy_late() -> Result<(), GraphError> {
    run_proxy_audit("late", 5000)
}

#[test]
#[ignore = "manual: Phase 3A0 固定 early future block 的 400→2000→5000 reducibility 复核"]
fn gomoku_p3a0_fixed_block_reducibility() -> Result<(), GraphError> {
    run_fixed_block_reducibility()
}

#[test]
#[ignore = "manual: Phase 3A0 新增真实 game 干预后未见 block 误差是否下降（3 seeds × 2000 局）"]
fn gomoku_p3a0_proxy_intervention() -> Result<(), GraphError> {
    run_proxy_intervention()
}

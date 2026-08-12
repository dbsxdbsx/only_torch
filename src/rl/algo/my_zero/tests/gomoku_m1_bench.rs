//! Gomoku M1 训练闭环（手动档）：9×9 最小基线，验收「vs random 胜率明确爬升」。
//!
//! **冒烟/闭环口径**（定性闸门，非预注册门槛；历史标签 M1）：
//! - base 组件全关（canonical MuZero）+ Flat MLP 编码器 + negamax MC target；
//! - 评测 vs `Gomoku-random-v0`（20 局 greedy），信号 = 胜率从 ~0.5 明确上升；
//! - M2 才立预注册门槛（vs random ≥95% + vs 旧 checkpoint ≥55%）。
//!
//! ```bash
//! cargo test --release --features blas-mkl gomoku_m1_train -- --ignored --nocapture --test-threads=1
//! # 快速冒烟（3 局 self-play + 训练 + eval 各路径打通）：
//! cargo test --release --features blas-mkl gomoku_m1_smoke -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::board::{BoardTrainConfig, train_board};

/// 闭环冒烟：3 局 self-play + 训练 + eval + M2 机件（快照 / gating 对弈 / naive 梯队）
/// 全路径打通、loss 有限。
#[test]
#[ignore = "manual: Gomoku M1 冒烟（需 Python + gym_env）"]
fn gomoku_m1_smoke() -> Result<(), GraphError> {
    let cfg = BoardTrainConfig {
        num_simulations: 16,
        start_training_after: 2,
        trains_per_episode: 1,
        max_episodes: 3,
        eval_every: 3,
        eval_episodes: 2,
        temp_hold_episodes: 3,
        temp_decay_episodes: 0,
        early_stop: false,
        snapshot_at_episode: Some(2),
        gate_games: 2,
        naive_ladder: true,
        ..Default::default()
    };
    let report = train_board(&cfg)?;
    assert!(report.total_env_steps > 0, "冒烟应有 env-steps");
    assert!(report.gate_vs_random.is_some(), "终局闸门应已跑");
    assert!(report.gate_vs_checkpoint.is_some(), "gating 对弈应已跑");
    assert_eq!(report.naive_win_rates.len(), 4, "naive 梯队应 4 档");
    println!("[SMOKE] gomoku M1+M2 机件通过");
    Ok(())
}

/// M1 正跑：默认口径（sims=100 / 400 局上限 / 每 25 局 eval 20 局）。
#[test]
#[ignore = "manual: Gomoku M1 训练闭环（约 1–2 小时）"]
fn gomoku_m1_train() -> Result<(), GraphError> {
    let report = train_board(&BoardTrainConfig::default())?;
    println!(
        "[M1] final_win_rate={:.2} best={:.2} solved_at={:?} total_env_steps={} wall={:.0}s",
        report.final_win_rate,
        report.best_win_rate,
        report.solved_at_steps,
        report.total_env_steps,
        report.wall_secs,
    );
    Ok(())
}

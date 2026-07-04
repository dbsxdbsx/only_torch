//! Gomoku M2 预注册基准（手动档）：棋盘支柱立柱正裁。
//!
//! # 预注册协议（跑前定稿，2026-07-04；收口规划 §3 棋盘账本口径）
//! - **载体**：9×9 base 栈（组件全关 + Flat MLP + negamax MC target），sims=100，
//!   满预算 400 局（**不因 vs random 达标提前收工**），seeds 42/43/44。
//! - **双硬门槛**（每 seed 皆须通过，3/3 判达标）：
//!   1. vs `Gomoku-random-v0` greedy 40 局胜率 **≥ 0.95**；
//!   2. 终局模型 vs **半程快照**（ep=200 权重）gating 对弈 40 局（黑白各半，
//!      胜 1 / 平 0.5）比赛分 **≥ 0.55**——证明后半程仍在实质变强。
//! - **观察梯队**（记录不设门槛）：naive0–3 各 20 局胜率，为 M3 消融备尺。
//! - **判读**：双门槛 3/3 → 棋盘支柱立柱、数字进账本；任一失败 → 记 issue、
//!   按嫌疑（预算 / lr / 表征）诊断，不 silent 调参重跑。
//!
//! # 协议修订（2026-07-04，首裁后、公开记录）
//! 首裁发现 gating 对弈为纯贪心确定性对局——40 局退化为每色 1 盘独立棋重复
//! （分数只能取 {0, 0.5, 1}，测量仪器缺陷而非臂结果）。修订：**随机开局 2 步 +
//! 同开局黑白镜像成对**（AlphaZero 系评测标准做法），门槛数值不变，整体重跑。
//!
//! ```bash
//! cargo test --release --features blas-mkl gomoku_m2_3seed -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::board::{BoardTrainConfig, BoardTrainReport, train_board};

fn m2_cfg(seed: u64) -> BoardTrainConfig {
    BoardTrainConfig {
        seed,
        max_episodes: 400,
        early_stop: false,
        snapshot_at_episode: Some(200),
        gate_games: 40,
        naive_ladder: true,
        eval_every: 50,
        ..Default::default()
    }
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// M2 正裁：3-seed 双硬门槛 + naive 梯队。
#[test]
#[ignore = "manual: Gomoku M2 预注册基准（3 seeds，约 10–20 分钟）"]
fn gomoku_m2_3seed() -> Result<(), GraphError> {
    let seeds = [42u64, 43, 44];
    let mut reports: Vec<(u64, BoardTrainReport)> = Vec::new();
    for &seed in &seeds {
        println!("\n--- seed {seed} ---");
        reports.push((seed, train_board(&m2_cfg(seed))?));
    }

    println!("\n--- M2 汇总（门槛：vs random ≥0.95 · vs 快照 ≥0.55）---");
    let mut pass_count = 0;
    for (seed, r) in &reports {
        let g1 = r.gate_vs_random.unwrap_or(0.0);
        let g2 = r.gate_vs_checkpoint.unwrap_or(0.0);
        let pass = g1 >= 0.95 && g2 >= 0.55;
        pass_count += usize::from(pass);
        let ladder: Vec<String> = r
            .naive_win_rates
            .iter()
            .map(|(n, w)| format!("{n}={w:.2}"))
            .collect();
        println!(
            "  seed={seed} vs_random={g1:.3} vs_snapshot={g2:.3} {} | 梯队 {} | env_steps={} wall={:.0}s",
            if pass { "✅" } else { "❌" },
            ladder.join(" "),
            r.total_env_steps,
            r.wall_secs,
        );
    }
    let med_g1 = median(
        reports
            .iter()
            .map(|(_, r)| r.gate_vs_random.unwrap_or(0.0))
            .collect(),
    );
    let med_g2 = median(
        reports
            .iter()
            .map(|(_, r)| r.gate_vs_checkpoint.unwrap_or(0.0))
            .collect(),
    );
    println!(
        "  [M2-verdict] {pass_count}/3 达标 | 中位 vs_random={med_g1:.3} vs_snapshot={med_g2:.3}"
    );
    Ok(())
}

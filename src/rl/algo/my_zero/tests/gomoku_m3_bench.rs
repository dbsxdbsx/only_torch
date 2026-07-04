//! Gomoku M3 组件消融（手动档）：一次一臂，对照 M2 base 账本。
//!
//! # 预注册协议（跑前定稿，2026-07-04；收口规划 §3）
//! - **base 对照** = M2 正裁三 seed（组件全关，`.bench/gomoku_m2_3seed_rerun_20260704.log`）：
//!   vs random 1.000/0.950/1.000 · vs 快照 0.975/0.925/0.950 · naive0 ≈ 0.00–0.15。
//! - **口径**：与 M2 完全同载体（400 局满预算 / sims=100 / 半程快照 / 终局 40 局双闸门
//!   / naive 梯队 20 局），3-seed（42/43/44），一次只动一个开关。
//! - **臂**（按序，前臂出结果才开下臂）：
//!   1. `gumbel_standard`：Gumbel-root + completedQ（|A|=81 ≫ sims=100 的 native 复裁；
//!      前置双修复已落地——greedy 去噪 + tree-level q_range + negamax 视角翻转）。
//!   2. 后续臂（CNN 2D 表征 / 8 重对称增广）另行开载体。
//! - **判读**：主指标 = naive0/1 胜率（base 在 random 上已饱和、快照分接近满分，
//!   naive 档是当前唯一有上升空间的尺子）；vs random / vs 快照为回归护栏
//!   （臂上如果掉下 0.95 / 0.55 即为负结果的强信号）。数字进棋盘账本。
//!
//! ```bash
//! cargo test --release --features blas-mkl gomoku_m3_gumbel -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::board::{BoardTrainConfig, BoardTrainReport, train_board};
use crate::rl::algo::my_zero::component::Components;

fn m3_cfg(seed: u64, components: Components, num_simulations: u32) -> BoardTrainConfig {
    BoardTrainConfig {
        seed,
        num_simulations,
        max_episodes: 400,
        early_stop: false,
        snapshot_at_episode: Some(200),
        gate_games: 40,
        naive_ladder: true,
        eval_every: 50,
        components,
        ..Default::default()
    }
}

fn run_arm(name: &str, components: Components, num_simulations: u32) -> Result<(), GraphError> {
    run_arm_with(name, components, num_simulations, |c| c)
}

fn run_arm_with(
    name: &str,
    components: Components,
    num_simulations: u32,
    tweak: impl Fn(BoardTrainConfig) -> BoardTrainConfig,
) -> Result<(), GraphError> {
    let seeds = [42u64, 43, 44];
    let mut reports: Vec<(u64, BoardTrainReport)> = Vec::new();
    for &seed in &seeds {
        println!("\n--- arm={name} seed={seed} sims={num_simulations} ---");
        reports.push((
            seed,
            train_board(&tweak(m3_cfg(seed, components.clone(), num_simulations)))?,
        ));
    }
    println!("\n--- M3 arm={name} 汇总（base 对照见 bench 头注释）---");
    for (seed, r) in &reports {
        let ladder: Vec<String> = r
            .naive_win_rates
            .iter()
            .map(|(n, w)| format!("{n}={w:.2}"))
            .collect();
        println!(
            "  seed={seed} vs_random={:.3} vs_snapshot={:.3} | 梯队 {} | env_steps={} wall={:.0}s",
            r.gate_vs_random.unwrap_or(0.0),
            r.gate_vs_checkpoint.unwrap_or(0.0),
            ladder.join(" "),
            r.total_env_steps,
            r.wall_secs,
        );
    }
    Ok(())
}

/// 臂 1：Gumbel-root + completedQ（论文标准 bundle 的棋盘域复裁，sims=100）。
///
/// **口径注记（2026-07-04 首跑后补）**：sims=100 > |A|=81，本臂实为 n>|A| regime
/// （与 CartPole 负结果同域），不构成 issue 关闭条件的 |A|≫sims 复裁；
/// 决定性对照在下方 s16 双臂（sims=16 ≪ 81，Gumbel native 场景）。
#[test]
#[ignore = "manual: Gomoku M3 Gumbel+completedQ 消融（3 seeds，约 10–20 分钟）"]
fn gomoku_m3_gumbel() -> Result<(), GraphError> {
    let mut c = Components::base();
    c.gumbel = true;
    c.completed_q_target = true;
    run_arm("gumbel_standard", c, 100)
}

/// 臂 3：consistency（SimSiam 自监督，CartPole promoted 组件的棋盘列裁决；
/// coef 用默认 2.0）。排序依据：表征塑形信号与 CNN 攻同一靶（naive0 瓶颈），
/// 零架构改动先测（2026-07-04 与用户定序：cons → recon → CNN）。
#[test]
#[ignore = "manual: Gomoku M3 consistency 臂（3 seeds）"]
fn gomoku_m3_cons() -> Result<(), GraphError> {
    let mut c = Components::base();
    c.consistency = true;
    run_arm("cons", c, 100)
}

/// 臂 4：reconstruction（CartPole 最大单组件杠杆的棋盘列裁决）。
///
/// coef 用 16.0 = CartPole 标定值（loss.rs 注明"临时值，跨域须复验"——棋盘 obs
/// 为 0/1 平面、量纲与 CartPole 同属小量级，预注册按 16 单臂；若臂结果对系数
/// 敏感的形态出现，再开 {4,16} pilot，不 silent 调参）。
#[test]
#[ignore = "manual: Gomoku M3 reconstruction 臂（3 seeds）"]
fn gomoku_m3_recon() -> Result<(), GraphError> {
    let mut c = Components::base();
    c.reconstruction = true;
    run_arm("recon", c, 100)
}

/// 臂 5：CNN 2D 表征（stride-1 棋盘卷积塔替代 Flat MLP 编码器）。
///
/// 预注册预期：naive0 瓶颈假设下（"一步胜必走/必挡" = 平移不变局部模式检测），
/// 本臂应显著抬升 naive0 胜率；若仍 ≈0，嫌疑转向训练预算（→ 预算标定臂）。
#[test]
#[ignore = "manual: Gomoku M3 CNN 表征臂（3 seeds）"]
fn gomoku_m3_cnn() -> Result<(), GraphError> {
    run_arm_with("cnn_repr", Components::base(), 100, |mut cfg| {
        cfg.cnn_repr = true;
        cfg
    })
}

/// 臂 6：训练预算标定（base 组件全关、唯一变量 = 400 → 2000 局）。
///
/// 触发条件（已满足，2026-07-04）：Gumbel/cons/recon/CNN 四臂 naive0 全平 ≈ 0，
/// 组件与表征嫌疑排除 → 按预注册分岔转向预算嫌疑（400 局 ≈ 1.8 万 env-steps）。
/// 温度调度维持 hold=100/decay=100 绝对局数不变（严格单变量；代价是长跑段
/// 恒 0.25 低温，若本臂有效、调度缩放另立一臂）。快照/闸门等比后移。
#[test]
#[ignore = "manual: Gomoku M3 预算标定臂（3 seeds × 2000 局，约 30 分钟）"]
fn gomoku_m3_budget() -> Result<(), GraphError> {
    run_arm_with("budget_2000", Components::base(), 100, |mut cfg| {
        cfg.max_episodes = 2000;
        cfg.snapshot_at_episode = Some(1000);
        cfg.eval_every = 200;
        cfg
    })
}

/// 臂 7：replay ratio（唯一变量 = trains_per_episode 4 → 32，400 局预算不变）。
///
/// 触发（2026-07-04）：预算臂证伪"交互太少"（2000 局 naive0 仍 ≈0.1、后半程
/// gating 分 ≈0.5 = 学习早已平台化）→ 参照 junxiaosong/AlphaZero_Gomoku 的
/// 每局训练强度（≈ 我们的 40–50×），验证"每份数据榨得太浅"假设。
/// 棋盘 value target = 终局事实永不过期，天然耐高 replay ratio；policy target
/// 过期毒性若显现（gating 分崩），ROSMO 刷新臂为对症后续。
#[test]
#[ignore = "manual: Gomoku M3 replay-ratio 臂（3 seeds，约 15 分钟）"]
fn gomoku_m3_replay32() -> Result<(), GraphError> {
    run_arm_with("replay32", Components::base(), 100, |mut cfg| {
        cfg.trains_per_episode = 32;
        cfg
    })
}

/// 臂 8：lr 重标（唯一变量 = 0.02 → 0.003；对齐参照实现 2e-3 量级）。
///
/// 触发（2026-07-05）：六个单杠杆臂全平后转向"配方合力"假设——lr 是与参照
/// 配方差距中最廉价可测的一项，且可解释 replay32 走低（高 RR × 高 lr 过热）。
#[test]
#[ignore = "manual: Gomoku M3 lr 臂（3 seeds）"]
fn gomoku_m3_lr3e3() -> Result<(), GraphError> {
    run_arm_with("lr3e3", Components::base(), 100, |mut cfg| {
        cfg.lr = 0.003;
        cfg
    })
}

/// 臂 9：8 重对称增广（唯一变量 = D4 采样时随机变换；参照实现标配部件）。
#[test]
#[ignore = "manual: Gomoku M3 增广臂（3 seeds）"]
fn gomoku_m3_augment() -> Result<(), GraphError> {
    run_arm_with("augment", Components::base(), 100, |mut cfg| {
        cfg.augment = true;
        cfg
    })
}

/// 臂 10：参照配方组合臂（增广 + replay×8 + lr 3e-3 一起上）。
///
/// 单杠杆全平后的配方合力终审（junxiaosong 参照 = 高 RR × 增广 × 低 lr 同时成立；
/// 若本臂显著抬升 naive0 而单臂全平，则裁决"合力生效"，M4 recipe 按组合定型）。
#[test]
#[ignore = "manual: Gomoku M3 参照配方组合臂（3 seeds，约 20 分钟）"]
fn gomoku_m3_combo() -> Result<(), GraphError> {
    run_arm_with(
        "combo_aug_rr32_lr3e3",
        Components::base(),
        100,
        |mut cfg| {
            cfg.augment = true;
            cfg.trains_per_episode = 32;
            cfg.lr = 0.003;
            cfg
        },
    )
}

/// 臂 2a：少 sim 对照的 PUCT 基线（sims=16 ≪ |A|=81；兼 P2 少 sim acting 复测）。
#[test]
#[ignore = "manual: Gomoku M3 少 sim PUCT 基线（3 seeds）"]
fn gomoku_m3_base_s16() -> Result<(), GraphError> {
    run_arm("base_s16", Components::base(), 16)
}

/// 臂 2b：少 sim Gumbel+completedQ（|A|≫sims 的 native 复裁，issue 关闭条件所指场景）。
#[test]
#[ignore = "manual: Gomoku M3 少 sim Gumbel 臂（3 seeds）"]
fn gomoku_m3_gumbel_s16() -> Result<(), GraphError> {
    let mut c = Components::base();
    c.gumbel = true;
    c.completed_q_target = true;
    run_arm("gumbel_s16", c, 16)
}

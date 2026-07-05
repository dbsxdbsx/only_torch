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

/// 同 `run_arm_with` 但自定 seeds（seed 加密复核用；协议修订须公开记录于臂注释）。
fn run_arm_seeds(
    name: &str,
    components: Components,
    num_simulations: u32,
    seeds: &[u64],
    tweak: impl Fn(BoardTrainConfig) -> BoardTrainConfig,
) -> Result<(), GraphError> {
    let mut reports: Vec<(u64, BoardTrainReport)> = Vec::new();
    for &seed in seeds {
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

/// 臂 11（M4 后 · naive0 战术墙 issue §四-①，2026-07-05 预注册）：sims 100→400。
///
/// 触发：M3 九臂收口后，naive0 墙根因假设 2 =「搜索预算差距」（参照实现 400 playouts
/// vs 我们 100 sims）的直接裁决——唯一变量 = num_simulations，载体/判读与 M3 完全同口径。
/// 预注册判读：主指标 naive0 显著抬升（中位 ≥0.3 视为搜索预算是主因）；仍 ≈0.1 →
/// 嫌疑收敛到假设 1「MuZero 规则学习税」，进入 ② 树内真规则诊断臂。
#[test]
#[ignore = "manual: Gomoku naive0-① sims=400 臂（3 seeds，约 20–40 分钟）"]
fn gomoku_naive0_sims400() -> Result<(), GraphError> {
    run_arm("sims400", Components::base(), 400)
}

/// 臂 12（M4 后 · naive0 战术墙 issue §四-②，2026-07-05 预注册）：树内真规则诊断。
///
/// 触发：① sims400 臂平（搜索预算嫌疑排除）→ 头号假设「MuZero 规则学习税」直接裁决。
/// 唯一变量 = `true_rules_tree`（树内转移/终局/候选 mask 走 `RulesBoard` 真规则，
/// 叶子先验/价值仍由网络给、训练侧不变），sims=100 同 base 口径。
/// 预注册判读：naive0 显著抬升（中位 ≥0.5 = 规则学习税坐实，升级战略分叉讨论——
/// 规则已知棋类是否松开万金油铁律；0.3–0.5 = 部分坐实）；仍 ≈0.1 → 假设 1 亦排除，
/// 嫌疑转向训练侧（value/policy 信号本身），③ replay32×ROSMO 臂顺位。
#[test]
#[ignore = "manual: Gomoku naive0-② 树内真规则诊断臂（3 seeds）"]
fn gomoku_naive0_true_rules() -> Result<(), GraphError> {
    run_arm_with("true_rules", Components::base(), 100, |mut cfg| {
        cfg.true_rules_tree = true;
        cfg
    })
}

/// 臂 12-ext（2026-07-05 首跑后追加，协议修订公开记录）：真规则臂 seed 扩展（45/46）。
///
/// 触发：首跑 42/43/44 = naive0 0.10/0.00/0.45——中位 0.10 未达预注册线，但 seed 44
/// 的 0.45 + naive1 0.15 是全批次消融的历史最强单 seed 信号，方差形态与「中位判读」
/// 冲突 → 按 CartPole 哨兵复裁先例扩到 5-seed 再下终局裁决（不改判读线、不改载体）。
#[test]
#[ignore = "manual: Gomoku naive0-② 真规则臂 seed 扩展（45/46）"]
fn gomoku_naive0_true_rules_seedext() -> Result<(), GraphError> {
    run_arm_seeds(
        "true_rules_ext",
        Components::base(),
        100,
        &[45, 46],
        |mut cfg| {
            cfg.true_rules_tree = true;
            cfg
        },
    )
}

/// 臂 13（M4 后 · naive0 战术墙 issue §四-③，2026-07-05 预注册）：replay×8 × ROSMO 刷新。
///
/// 触发：①②臂后根因修订「训练信号瓶颈」上位——本臂对症 M3 replay32 偏害的
/// policy target 过期嫌疑：trains_per_episode 4→32 复刻 replay32 载体，唯一新增
/// = `rosmo_refresh`（采样时现算一步 look-ahead 改进 policy target，top-16 剪枝、
/// 不写回；value 维持 negamax MC）。兼 ROSMO 棋盘域价值裁决。
/// 预注册判读（对照 = M3 replay32 臂 vs random 0.875 / gating 0.825 / naive0 0.00–0.05）：
/// naive0 中位显著抬升（≥0.2）或 replay32 的护栏回落消失 = ROSMO 解毒生效；
/// 与 replay32 同形态 = policy 过期非主因，训练信号嫌疑再收敛（value 方差 / 探索覆盖）。
#[test]
#[ignore = "manual: Gomoku naive0-③ replay32×ROSMO 刷新臂（3 seeds，约 20–30 分钟）"]
fn gomoku_naive0_rr32_rosmo() -> Result<(), GraphError> {
    run_arm_with("rr32_rosmo", Components::base(), 100, |mut cfg| {
        cfg.trains_per_episode = 32;
        cfg.rosmo_refresh = true;
        cfg
    })
}

/// 臂 14（M4 后 · naive0 战术墙 issue §四-④，2026-07-05 预注册）：batch 16→256。
///
/// 触发：①②③臂后剩余头号嫌疑 =「negamax MC ±1 高方差 → 梯度噪声」。唯一变量 =
/// `train_batch_size` 16→256（梯度噪声方差 ∝ 1/B 降 16 倍；每局样本消耗 64→1024，
/// 与参照实现 batch 512 × 5 epoch 同量级）。与已测臂正交：预算×5 = 数据量、
/// replay×8 = 更新次数、本臂 = 每步梯度噪声水平。
/// 预注册判读：naive0 中位 ≥0.3 = 部分坐实（lr 联动/KL 自适应臂顺位）；
/// 仍 ≈0.1 且护栏正常 = 排除、嫌疑收敛探索覆盖；vs random <0.9 = lr 失配改判重测。
#[test]
#[ignore = "manual: Gomoku naive0-④ batch256 臂（3 seeds，约 15–30 分钟）"]
fn gomoku_naive0_batch256() -> Result<(), GraphError> {
    run_arm_with("batch256", Components::base(), 100, |mut cfg| {
        cfg.train_batch_size = 256;
        cfg
    })
}

/// 臂 15（naive0 issue §四-⑤，2026-07-05 预注册）：G1 上限标定组合臂（诊断性破例）。
///
/// 触发：①–④ 单变量臂全平/弱阳后改判「参照配方免费部分的合力能否翻墙」——
/// true_rules × 2000 局 × D4 增广 × 温度全程 1.0 × buffer 300（≈参照 1 万局面等比）。
/// lr/batch/trains 不动（训练强度包 = G2 后续）。
/// 预注册判读（对照 ②臂 0.20 / 预算臂 0.10–0.15 / 增广臂 0.15）：
/// naive0 中位 ≥0.5 = 翻墙（战略分叉正式化 + G2 + G3 四档验收）；
/// 0.3–0.5 = 方向确认进 G2；<0.3 = 免费配方不足，战术开局课程优先。
#[test]
#[ignore = "manual: Gomoku naive0-⑤ G1 组合臂（3 seeds × 2000 局，约 20–35 分钟）"]
fn gomoku_naive0_combo_ceiling() -> Result<(), GraphError> {
    run_arm_with("combo_ceiling", Components::base(), 100, |mut cfg| {
        cfg.true_rules_tree = true;
        cfg.augment = true;
        cfg.max_episodes = 2000;
        cfg.temp_hold_episodes = 2000; // 全程温度 1.0（参照实现口径）
        cfg.buffer_capacity = 300;
        cfg.snapshot_at_episode = Some(1000);
        cfg.eval_every = 200;
        cfg
    })
}

/// 臂 16（naive0 issue §四-⑥，2026-07-05 预注册）：定向战术开局课程。
///
/// 触发：G1 组合臂未翻墙 + seed 方差三度复现（同配方 0.00 vs 0.45）坐实
/// 「战术局面 = self-play 抽签」→ 把必挡局面确定性注入训练分布（机制通用：
/// Leela 开局库 / KataGo forced openings 同族；内容领域特定：一步胜威胁生成器）。
/// 唯一新变量 = `tactical_opening_fraction=0.25`，底座 = ②臂口径（true_rules，
/// 400 局）——真规则树让注入局面的「必挡」在树内结构性可见，课程信号才可转化。
/// 预注册判读（对照 ②臂 5-seed 中位 0.20 / 首跑 3-seed 0.10/0.00/0.45）：
/// naive0 中位 ≥0.5 = 翻墙（课程 promote + 上 2000 局终局配方 + G3 四档验收）；
/// 0.3–0.5 = 强正信号（扩 2000 局复核后定 promote）；与 ②臂持平 = 排除
/// （嫌疑转 G2 训练强度包 / bootstrap value target）。
#[test]
#[ignore = "manual: Gomoku naive0-⑥ 战术开局课程臂（3 seeds，约 3–5 分钟）"]
fn gomoku_naive0_tactical_openings() -> Result<(), GraphError> {
    run_arm_with("tactical_openings", Components::base(), 100, |mut cfg| {
        cfg.true_rules_tree = true;
        cfg.tactical_opening_fraction = 0.25;
        cfg
    })
}

/// 臂 17（naive0 issue §四-⑦，2026-07-05 预注册）：终局配方复核（G1 组合 × 课程）。
///
/// 触发：⑥ 臂强正信号（400 局中位 0.40 = ②臂 2×）→ 按其预注册升级路径，
/// 在 G1 组合底座（true_rules × 2000 局 × 增广 × 温度 1.0 × buffer 300）上
/// 叠加 `tactical_opening_fraction=0.25`，唯一对照 = G1（同底座无课程，中位 0.20）。
/// 预注册判读：naive0 中位 ≥0.5 = 翻墙坐实（课程 promote 进棋盘 recipe 讨论 +
/// G3 四档验收开启）；0.3–0.5 = 课程有效但预算无增益（400 局版已够，取短版）；
/// <0.3 = ⑥ 臂信号不稳（回 5-seed 复裁）。
#[test]
#[ignore = "manual: Gomoku naive0-⑦ 终局配方复核臂（3 seeds × 2000 局，约 15–25 分钟）"]
fn gomoku_naive0_final_recipe() -> Result<(), GraphError> {
    run_arm_with("final_recipe", Components::base(), 100, |mut cfg| {
        cfg.true_rules_tree = true;
        cfg.augment = true;
        cfg.max_episodes = 2000;
        cfg.temp_hold_episodes = 2000;
        cfg.buffer_capacity = 300;
        cfg.snapshot_at_episode = Some(1000);
        cfg.eval_every = 200;
        cfg.tactical_opening_fraction = 0.25;
        cfg
    })
}

/// 效率探针（非裁决臂，2026-07-05）：batch 512/1024 吞吐与内存观察。
///
/// 定位：④ 臂已裁决「梯度噪声排除」，本探针只测**效率曲线**（墙钟随 batch 的
/// 摊薄红利还剩多少、内存是否可忽略），单 seed 观察性数据，喂给性能台账候选 #8
/// 「自动 batch size」的微基准方案做第一份实测锚点。对照：base(b16) ~90s /
/// batch256 seed42 = 67s。
#[test]
#[ignore = "manual: Gomoku batch 512/1024 效率探针（单 seed，约 3–5 分钟）"]
fn gomoku_naive0_batch_scale_probe() -> Result<(), GraphError> {
    for &bs in &[512usize, 1024] {
        run_arm_seeds(
            &format!("batch{bs}_probe"),
            Components::base(),
            100,
            &[42],
            |mut cfg| {
                cfg.train_batch_size = bs;
                cfg
            },
        )?;
    }
    Ok(())
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

//! v0.26 Phase 1 §S3：图像域（ALE/Pong-v5）native 复裁 A/B（手动档）。
//!
//! 预注册协议见 `.doc/design/rl_phase1_image_plan.md` §S3 与
//! `examples/my_zero/pong/README.md`（唯一账本）：
//!
//! - **base 臂**：image base 栈（consistency ON + recon OFF + two-hot + raw obs），
//!   即 3-seed 基准本身，不在此重跑。
//! - **A/B 预算**：基准半额 = 75 局/seed（≈ 60k env-steps），seeds 42/43/44。
//! - **臂**（一次一项）：
//!   1. `recon`：reconstruction ON——先 1-seed pilot 挑 coef（`RECON_COEF` 环境变量,
//!      {1,4,16} 各跑一次），再 3-seed 定裁。目标重建 = 4 帧堆叠 obs（28224 维）。
//!   2. `cons_off`：consistency OFF（图像是 cons 的 native 场景，验证其增量）。
//!   3. `hl_gauss`：HL-Gauss 编码 ON（CartPole 负结果的图像域复测，Simulus A1 复裁条款）。
//! - **判读**：各臂 vs 基准的中位 best greedy + 曲线形态；数字进 Pong README 账本。
//!
//! ```bash
//! # pilot（1 seed）：
//! RECON_COEF=4 cargo test --release --features blas-mkl pong_ablation_recon_pilot -- --ignored --nocapture --test-threads=1
//! # 3-seed 正裁：
//! RECON_COEF=4 cargo test --release --features blas-mkl pong_ablation_recon_3seed -- --ignored --nocapture --test-threads=1
//! cargo test --release --features blas-mkl pong_ablation_cons_off -- --ignored --nocapture --test-threads=1
//! cargo test --release --features blas-mkl pong_ablation_hl_gauss -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::runner::train_all_seeds;
use crate::rl::algo::my_zero::{MyZero, MyZeroConfig, TrainSettings};

/// Pong 预注册训练口径（与 `examples/my_zero/pong/main.rs` 保持一致）+ A/B 半额预算。
fn pong_ablation_cfg(seeds: u64) -> Result<MyZeroConfig, GraphError> {
    MyZero::new("ALE/Pong-v5")
        .solved(-18.0)
        .max_episodes(75) // 基准 150 局的半额
        .train_settings(TrainSettings {
            gamma: 0.997,
            k_unroll: 5,
            td_steps: 5,
            num_simulations: 20,
            lr: 0.003,
            train_batch_size: 16,
            trains_per_episode: 64,
            buffer_capacity: 32,
            start_training_after: 2,
            // 与基准（150 局）同一显式温度口径；旧「按预算比例退火」会让半额臂
            // 静默用不同探索调度（预算耦合 bug），现统一为常数调度
            temp_hold_episodes: 75,
            temp_decay_episodes: 75,
            kl_adaptive_lr: false,
            ..TrainSettings::default()
        })
        .eval_every(10)
        .seeds(seeds)
        .build()
}

fn recon_coef_from_env() -> f32 {
    std::env::var("RECON_COEF")
        .expect("请设置 RECON_COEF（如 1 / 4 / 16）")
        .parse()
        .expect("RECON_COEF 必须是数字")
}

/// 臂 1 pilot：reconstruction ON（单 seed，coef 由 RECON_COEF 指定）。
#[test]
#[ignore = "manual: v0.26 Phase 1 图像域 recon pilot（1 seed）"]
fn pong_ablation_recon_pilot() -> Result<(), GraphError> {
    let coef = recon_coef_from_env();
    let mut cfg = pong_ablation_cfg(1)?;
    cfg.components.reconstruction = true;
    cfg.components.reconstruction_coef = coef;
    println!("[pong-ablation] arm=recon-pilot coef={coef} seeds=1");
    train_all_seeds(cfg)?;
    Ok(())
}

/// 臂 1 正裁：reconstruction ON（3 seeds，coef 取 pilot 最优）。
#[test]
#[ignore = "manual: v0.26 Phase 1 图像域 recon 3-seed"]
fn pong_ablation_recon_3seed() -> Result<(), GraphError> {
    let coef = recon_coef_from_env();
    let mut cfg = pong_ablation_cfg(3)?;
    cfg.components.reconstruction = true;
    cfg.components.reconstruction_coef = coef;
    println!("[pong-ablation] arm=recon coef={coef} seeds=3");
    train_all_seeds(cfg)?;
    Ok(())
}

/// 臂 2：consistency OFF（3 seeds）。
#[test]
#[ignore = "manual: v0.26 Phase 1 图像域 consistency 消融"]
fn pong_ablation_cons_off() -> Result<(), GraphError> {
    let mut cfg = pong_ablation_cfg(3)?;
    cfg.components.consistency = false;
    println!("[pong-ablation] arm=cons-off seeds=3");
    train_all_seeds(cfg)?;
    Ok(())
}

/// 臂 3：HL-Gauss 编码 ON（3 seeds；CartPole 负结果的图像域复测）。
#[test]
#[ignore = "manual: v0.26 Phase 1 图像域 HL-Gauss 复测"]
fn pong_ablation_hl_gauss() -> Result<(), GraphError> {
    let mut cfg = pong_ablation_cfg(3)?;
    cfg.components.hl_gauss = true;
    println!("[pong-ablation] arm=hl-gauss seeds=3");
    train_all_seeds(cfg)?;
    Ok(())
}

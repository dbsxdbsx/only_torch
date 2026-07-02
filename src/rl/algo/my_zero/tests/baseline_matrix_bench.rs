//! CartPole 增量链基线矩阵（发版重定基线用；不纳入 CI）。
//!
//! 口径（v0.25 起官方）：**release（thin LTO）+ 自动检测 BLAS + `SEEDS=3`（seed 42/43/44）**，
//! 取中位 env-steps 与达标率；autograd `upstream_grad` 修复 + batch-native 训练后的诚实基线。
//! 实测结果回填 `examples/my_zero/cartpole/README.md`（唯一基准账本）。
//!
//! ```bash
//! # 逐档运行（每档 3 seeds；建议按 t3 → t0 顺序，先拿最重要的 promoted 数字）
//! cargo test --release --features blas-mkl cartpole_baseline_t3_promoted -- --ignored --nocapture
//! cargo test --release --features blas-mkl cartpole_baseline_t2_cons_recon -- --ignored --nocapture
//! cargo test --release --features blas-mkl cartpole_baseline_t1_cons -- --ignored --nocapture
//! cargo test --release --features blas-mkl cartpole_baseline_t0_base -- --ignored --nocapture
//! ```

use super::super::component::Components;
use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

/// 跑一档增量配置（3 seeds，打印多 seed 汇总；不落盘模型）。
fn run_tier(label: &str, components: Components, max_episodes: usize) -> Result<(), GraphError> {
    println!("[baseline] CartPole-v1 tier={label} · SEEDS=3 · max_episodes={max_episodes}");
    let mut cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(max_episodes)
        .seeds(3)
        .build()?;
    cfg.components = components;
    train_all_seeds(cfg)?;
    Ok(())
}

/// t0：组件全关（canonical MuZero base）。上限 1000 局以约束 wall-clock；
/// 预期不达标——证据链的"负对照"。
#[test]
#[ignore = "manual: 发版基线 t0 base（组件全关）"]
fn cartpole_baseline_t0_base() -> Result<(), GraphError> {
    run_tier("base", Components::base(), 1000)
}

/// t1：仅 +consistency。
#[test]
#[ignore = "manual: 发版基线 t1 +consistency"]
fn cartpole_baseline_t1_cons() -> Result<(), GraphError> {
    let mut c = Components::base();
    c.consistency = true;
    run_tier("+consistency", c, 2000)
}

/// t2：+consistency +reconstruction（无 Sampled）。
#[test]
#[ignore = "manual: 发版基线 t2 +cons+recon"]
fn cartpole_baseline_t2_cons_recon() -> Result<(), GraphError> {
    let mut c = Components::base();
    c.consistency = true;
    c.reconstruction = true;
    run_tier("+cons+recon", c, 2000)
}

/// t3：当前 recipe（cons+recon+Sampled）= promoted 栈，等价 `SEEDS=3` 跑示例。
#[test]
#[ignore = "manual: 发版基线 t3 promoted（当前 recipe）"]
fn cartpole_baseline_t3_promoted() -> Result<(), GraphError> {
    let cfg = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .seeds(3)
        .build()?;
    println!("[baseline] CartPole-v1 tier=promoted(recipe) · SEEDS=3 · max_episodes=2000");
    train_all_seeds(cfg)?;
    Ok(())
}

//! Phase 6 跨轴验收矩阵 benchmark（手动档）。
//!
//! 验证 Stochastic MuZero K=8 作为万金油基底，跨现有环境的统计 benchmark。
//! 每个测试 = 矩阵的一个格子（输入 × 动作 × transition），3-seed 统计口径。
//!
//! 已有统计 benchmark 的格子（不在此文件重复）：
//! - vector × discrete × deterministic = CartPole（`baseline_matrix_bench.rs`）
//! - vector × discrete × stochastic = StochasticCartPole（`stochastic` 验收日志）
//! - board × discrete × deterministic = Gomoku（`gomoku_m2_bench.rs`）
//!
//! ```bash
//! # Gomoku + K=8 stochastic 回归验证（约 15 分钟）
//! cargo test --release --features blas-mkl gomoku_stochastic_k8 -- --ignored --nocapture --test-threads=1
//!
//! # Platform-v0 混合动作 3-seed（约 10–20 分钟）
//! cargo test --release --features blas-mkl platform_hybrid_3seed -- --ignored --nocapture --test-threads=1
//!
//! # Pendulum 连续动作 3-seed（约 5–10 分钟，预期仍在失败区间）
//! cargo test --release --features blas-mkl pendulum_continuous_3seed -- --ignored --nocapture --test-threads=1
//!
//! # POMDP CartPole 3-seed（约 15 分钟，验证 posterior+stochastic 组合）
//! cargo test --release --features blas-mkl pomdp_stochastic_3seed -- --ignored --nocapture --test-threads=1
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

// ============================================================================
// vector × hybrid × deterministic — Platform-v0
// ============================================================================

/// Platform-v0 混合动作空间 3-seed benchmark。
///
/// 环境：Discrete(3) + 3 维连续 → 离散化 B=7 → 联合 1029 动作（Sampled 自动启用）。
/// 门禁：greedy eval return 趋势上升（非硬性阈值；证明学得动即可）。
/// K=8 stochastic 由 recipe 默认继承。
#[test]
#[ignore = "manual: Phase 6 — Platform-v0 hybrid 3-seed（约 10–20 分钟）"]
fn platform_hybrid_3seed() -> Result<(), GraphError> {
    println!("[Phase6] Platform-v0 · vector × hybrid × deterministic · SEEDS=3");
    let cfg = MyZero::new("Platform-v0")
        .discretize(7)
        .solved(f32::INFINITY) // 不设硬性门槛，观察学习曲线
        .max_episodes(300)
        .eval_every(50)
        .seeds(3)
        .build()?;
    train_all_seeds(cfg)?;
    Ok(())
}

// ============================================================================
// vector × continuous × deterministic — Pendulum-v1
// ============================================================================

/// Pendulum 连续动作 3-seed benchmark（当前诊断栈 + stochastic K=8）。
///
/// 门禁 ≥ −200（当前预期不达标；记录最新 stochastic K=8 下的行为）。
/// B=7 连续候选由 recipe 注入。
#[test]
#[ignore = "manual: Phase 6 — Pendulum 连续 3-seed（约 5–10 分钟）"]
fn pendulum_continuous_3seed() -> Result<(), GraphError> {
    println!("[Phase6] Pendulum-v1 · vector × continuous × deterministic · SEEDS=3");
    let cfg = MyZero::new("Pendulum-v1")
        .solved(-200.0)
        .max_episodes(600)
        .eval_every(50)
        .seeds(3)
        .build()?;
    train_all_seeds(cfg)?;
    Ok(())
}

// ============================================================================
// vector × discrete × POMDP — CartPole velocity-masked + posterior + stochastic
// ============================================================================

/// CartPole POMDP-lite 3-seed：posterior=ON + stochastic K=8 组合验证。
///
/// obs_mask=[1,3]（遮蔽速度），recurrent_posterior=true。
/// 门禁 ≥ 400（此前负裁 ~45，本轮验证 stochastic 叠加是否改善）。
#[test]
#[ignore = "manual: Phase 6 — CartPole POMDP 3-seed（约 15 分钟）"]
fn pomdp_stochastic_3seed() -> Result<(), GraphError> {
    println!("[Phase6] CartPole POMDP-lite · vector × discrete × POMDP · posterior+stochastic · SEEDS=3");
    let cfg = MyZero::new("CartPole-v1")
        .solved(400.0)
        .max_episodes(3000)
        .eval_every(100)
        .obs_mask(&[1, 3])
        .recurrent_posterior(true)
        .seeds(3)
        .build()?;
    train_all_seeds(cfg)?;
    Ok(())
}

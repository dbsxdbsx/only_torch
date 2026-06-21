//! CartPole completedQ / Gumbel 消融（不纳入 CI；CartPole 均不 promote）。
//!
//! | sims | PUCT visit | +completedQ | Gumbel visit |
//! |------|------------|-------------|--------------|
//! | 50   | 11,682 @ ep275 | 34,490 @ ep575 ❌ | — |
//! | 20   | 12,186 @ ep250 ✅ | 30,409 @ ep450 ❌ | 峰值 greedy 123 @ ep1725 ❌ |
//! | 15   | 26,306 @ ep500 | （待测） | — |
//! | 10   | 16,152 @ ep875 ✅ | （待测） | 峰值 greedy 154 @ ep1800+ ❌ |
//!
//! 详见 `.issue/items/my_zero_gumbel_completedq_cartpole_negative.md`
//!
//! ```bash
//! # sim=20 基线
//! cargo test --release cartpole_bench_s20_visit --features blas-mkl -- --ignored --nocapture
//! # sim=20 + completedQ
//! cargo test --release cartpole_bench_s20_completed_q --features blas-mkl -- --ignored --nocapture
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

fn run_cartpole_bench(
    sims: u32,
    gumbel: bool,
    completed_q: bool,
    save_suffix: &str,
) -> Result<(), GraphError> {
    let tag = if completed_q { "completedQ" } else { "visit" };
    let gtag = if gumbel { "Gumbel" } else { "PUCT" };
    println!("[bench] CartPole cons+recon sims={sims} search={gtag} target={tag}");
    let mut builder = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .num_simulations(sims)
        .save_model_when_eval(format!(
            "models/my_zero/CartPole-v1/seed_42/bench_s{sims}_{save_suffix}"
        ));
    if gumbel {
        builder = builder.gumbel(true);
    }
    if completed_q {
        builder = builder.completed_q_target(true);
    }
    let cfg = builder.build()?;
    assert!(cfg.components.consistency);
    assert!(cfg.components.reconstruction);
    assert_eq!(cfg.components.gumbel, gumbel);
    assert_eq!(cfg.components.completed_q_target, completed_q);
    assert_eq!(cfg.train.num_simulations, sims);
    train_all_seeds(cfg)?;
    Ok(())
}

fn run_cartpole_bench_visit(sims: u32, save_suffix: &str) -> Result<(), GraphError> {
    run_cartpole_bench(sims, false, false, save_suffix)
}

#[test]
#[ignore = "manual: sim=50 visit 基线；已有 ~11.7k steps 可跳过"]
fn cartpole_bench_s50_visit() -> Result<(), GraphError> {
    run_cartpole_bench_visit(50, "visit")
}

#[test]
#[ignore = "manual: sim=50 + completedQ；已有 ~34.5k steps 可跳过"]
fn cartpole_bench_s50_completed_q() -> Result<(), GraphError> {
    run_cartpole_bench(50, false, true, "completed_q")
}

#[test]
#[ignore = "manual: sim=10 visit 扫参"]
fn cartpole_bench_s10_visit() -> Result<(), GraphError> {
    run_cartpole_bench_visit(10, "visit")
}

#[test]
#[ignore = "manual: sim=15 visit 扫参"]
fn cartpole_bench_s15_visit() -> Result<(), GraphError> {
    run_cartpole_bench_visit(15, "visit")
}

#[test]
#[ignore = "manual: sim=20 visit 基线对照"]
fn cartpole_bench_s20_visit() -> Result<(), GraphError> {
    run_cartpole_bench_visit(20, "visit")
}

#[test]
#[ignore = "manual: sim=20 + completedQ 关键对照"]
fn cartpole_bench_s20_completed_q() -> Result<(), GraphError> {
    run_cartpole_bench(20, false, true, "completed_q")
}

#[test]
#[ignore = "manual: sim=10 Gumbel-root only（低 sim 对照）"]
fn cartpole_bench_s10_gumbel_visit() -> Result<(), GraphError> {
    run_cartpole_bench(10, true, false, "gumbel_visit")
}

#[test]
#[ignore = "manual: sim=10 Gumbel 标准 bundle（低 sim 对照）"]
fn cartpole_bench_s10_gumbel_standard() -> Result<(), GraphError> {
    run_cartpole_bench(10, true, true, "gumbel_standard")
}

#[test]
#[ignore = "manual: sim=20 Gumbel-root only（阶段 A）"]
fn cartpole_bench_s20_gumbel_visit() -> Result<(), GraphError> {
    run_cartpole_bench(20, true, false, "gumbel_visit")
}

#[test]
#[ignore = "manual: sim=20 Gumbel 标准 bundle（阶段 B）"]
fn cartpole_bench_s20_gumbel_standard() -> Result<(), GraphError> {
    run_cartpole_bench(20, true, true, "gumbel_standard")
}

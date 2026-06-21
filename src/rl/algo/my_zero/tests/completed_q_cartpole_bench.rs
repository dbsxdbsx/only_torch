//! CartPole completedQ 2×2 消融（不纳入 CI）。
//!
//! | sims | visit（基线） | +completedQ |
//! |------|--------------|-------------|
//! | 50   | 11,682 @ ep275 | 34,490 @ ep575 |
//! | 20   | 12,186 @ ep250 | 30,409 @ ep450 |
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

fn run_cartpole_bench(sims: u32, completed_q: bool, save_suffix: &str) -> Result<(), GraphError> {
    let tag = if completed_q { "completedQ" } else { "visit" };
    println!("[bench] CartPole cons+recon sims={sims} target={tag}");
    let mut builder = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(2000)
        .num_simulations(sims)
        .save_model_when_eval(format!(
            "models/my_zero/CartPole-v1/seed_42/bench_s{sims}_{save_suffix}"
        ));
    if completed_q {
        builder = builder.completed_q_target(true);
    }
    let cfg = builder.build()?;
    assert!(cfg.components.consistency);
    assert!(cfg.components.reconstruction);
    assert_eq!(cfg.components.completed_q_target, completed_q);
    assert_eq!(cfg.train.num_simulations, sims);
    train_all_seeds(cfg)?;
    Ok(())
}

#[test]
#[ignore = "manual: sim=50 visit 基线；已有 ~11.7k steps 可跳过"]
fn cartpole_bench_s50_visit() -> Result<(), GraphError> {
    run_cartpole_bench(50, false, "visit")
}

#[test]
#[ignore = "manual: sim=50 + completedQ；已有 ~34.5k steps 可跳过"]
fn cartpole_bench_s50_completed_q() -> Result<(), GraphError> {
    run_cartpole_bench(50, true, "completed_q")
}

#[test]
#[ignore = "manual: sim=20 visit 基线对照"]
fn cartpole_bench_s20_visit() -> Result<(), GraphError> {
    run_cartpole_bench(20, false, "visit")
}

#[test]
#[ignore = "manual: sim=20 + completedQ 关键对照"]
fn cartpole_bench_s20_completed_q() -> Result<(), GraphError> {
    run_cartpole_bench(20, true, "completed_q")
}

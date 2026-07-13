//! POMDP-lite CartPole velocity-masked 三组对照 bench（手动触发）。
//!
//! 验证 recurrent posterior 在 observation-aliasing 环境下的收敛性：
//! - 正组（posterior=true + masked）应收敛
//! - 负组（posterior=false + masked）应失败
//!
//! ```bash
//! # 正组（约 20-40 分钟）
//! cargo test --release --features blas-mkl pomdp_cartpole_posterior_on -- --ignored --nocapture
//!
//! # 负组（约 10-20 分钟，预期提前放弃）
//! cargo test --release --features blas-mkl pomdp_cartpole_posterior_off -- --ignored --nocapture
//! ```

use crate::nn::GraphError;
use crate::rl::algo::my_zero::MyZero;
use crate::rl::algo::my_zero::runner::train_all_seeds;

fn run_pomdp(label: &str, use_posterior: bool) -> Result<(), GraphError> {
    println!("[POMDP-lite] CartPole masked [{label}] · SEEDS=3 · posterior={use_posterior}");
    let cfg = MyZero::new("CartPole-v1")
        .solved(if use_posterior { 400.0 } else { 200.0 })
        .max_episodes(3000)
        .obs_mask(&[1, 3])
        .recurrent_posterior(use_posterior)
        .seeds(3)
        .build()?;
    train_all_seeds(cfg)?;
    Ok(())
}

/// 正组：velocity-masked + posterior=true（预期 3/3 达标 greedy>=400）
#[test]
#[ignore = "manual: POMDP-lite CartPole posterior=ON（约 20-40 分钟）"]
fn pomdp_cartpole_posterior_on() -> Result<(), GraphError> {
    run_pomdp("posterior=ON", true)
}

/// 负组：velocity-masked + posterior=false（预期 0/3 或极低分）
#[test]
#[ignore = "manual: POMDP-lite CartPole posterior=OFF 负对照（约 10-20 分钟）"]
fn pomdp_cartpole_posterior_off() -> Result<(), GraphError> {
    run_pomdp("posterior=OFF (负对照)", false)
}

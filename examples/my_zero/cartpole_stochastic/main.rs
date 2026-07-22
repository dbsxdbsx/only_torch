//! Stochastic CartPole：Stochastic MuZero 验证。
//!
//! 默认使用 StochasticCartPole-v0（CartPole + 速度噪声），验证 K>1 的 stochastic
//! 搜索在随机环境中是否优于 K=1 确定性路径。
//!
//! ```bash
//! # K=8 stochastic（默认）
//! SEEDS=3 cargo run --example my_zero_cartpole_stochastic --release
//!
//! # K=1 对照（同一随机环境，但 dynamics 确定性）
//! SEEDS=3 CHANCE_K=1 cargo run --example my_zero_cartpole_stochastic --release
//!
//! # 管线验证
//! SMOKE=1 cargo run --example my_zero_cartpole_stochastic
//! ```

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    // PYTHONPATH 须包含 stochastic_cartpole.py 所在目录
    let k: usize = std::env::var("CHANCE_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    let mut builder = MyZero::new("StochasticCartPole-v0")
        .solved(if smoke { 50.0 } else { 300.0 })
        .max_episodes(if smoke { 3 } else { 3000 })
        .num_chance_outcomes(k);

    if let Ok(v) = std::env::var("SEEDS") {
        builder = builder.seeds(v.parse().expect("SEEDS 必须是正整数"));
    }
    if smoke {
        builder = builder.smoke();
    }

    let _mz = builder.train()?;
    Ok(())
}

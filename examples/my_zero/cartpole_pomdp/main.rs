//! CartPole velocity-masked POMDP-lite 验证
//!
//! 遮蔽 obs[1]（cart_velocity）和 obs[3]（pole_velocity），使单帧信息不完整。
//! 默认开启 recurrent posterior（GRU），使 agent 可从 history 推断速度。
//!
//! # 用法
//! ```bash
//! # 正组：masked + posterior=true（应收敛）
//! cargo run --example my_zero_cartpole_pomdp --release
//!
//! # 负对照：masked + posterior=false（应失败）
//! POSTERIOR=0 cargo run --example my_zero_cartpole_pomdp --release
//!
//! # Smoke（管线验证，3 局）
//! SMOKE=1 cargo run --example my_zero_cartpole_pomdp
//!
//! # 多 seed 统计
//! SEEDS=3 cargo run --example my_zero_cartpole_pomdp --release
//! ```

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let use_posterior = std::env::var("POSTERIOR")
        .map(|v| v != "0")
        .unwrap_or(true);

    let mut builder = MyZero::new("CartPole-v1")
        .solved(if use_posterior { 400.0 } else { 200.0 })
        .max_episodes(if smoke { 3 } else { 3000 })
        .obs_mask(&[1, 3])
        .recurrent_posterior(use_posterior);

    if let Ok(v) = std::env::var("SEEDS") {
        builder = builder.seeds(v.parse().expect("SEEDS 必须是正整数"));
    }
    if smoke {
        builder = builder.smoke();
    }

    let tag = if use_posterior {
        "posterior=ON"
    } else {
        "posterior=OFF (负对照)"
    };
    println!("=== CartPole POMDP-lite [{tag}] obs_mask=[1,3] ===");

    let mz = builder.train()?;

    if !smoke {
        mz.eval(5)?.run(Some(1))?;
    }
    Ok(())
}

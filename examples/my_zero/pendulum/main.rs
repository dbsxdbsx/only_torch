//! MyZero · Pendulum-v1（纯连续 → 离散候选）。
//!
//! ```bash
//! cargo run --example my_zero_pendulum --release
//! SMOKE=1 cargo run --example my_zero_pendulum  # 管线验证
//! ```
//!
//! 运行命令、诊断与 benchmark 见同目录 [`README.md`](README.md)。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    let mut builder = MyZero::new("Pendulum-v1")
        .discretize(9)
        .reward_scale(0.1)
        .solved(-200.0)
        .max_episodes(if smoke { 3 } else { 600 });
    if smoke {
        builder = builder.smoke();
    }

    builder.train()?;
    Ok(())
}

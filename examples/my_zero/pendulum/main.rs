//! MyZero · Pendulum-v1（纯连续 1 维 → 离散候选）
//!
//! ```bash
//! cargo run --example my_zero_pendulum --release
//! CONSISTENCY=1 cargo run --example my_zero_pendulum --release
//! SEEDS=3 cargo run --example my_zero_pendulum --release
//! SMOKE=1 cargo run --example my_zero_pendulum
//! DIAG=1 cargo run --example my_zero_pendulum --release
//! ```
//!
//! 旋钮：`CONSISTENCY` `CQ` `SIMS` `SEEDS` `SMOKE` `DIAG` `GAMMA` `MAX_EP` `LR` `NUM_ACTIONS` `RSCALE` `SOLVED`

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

fn main() -> Result<(), GraphError> {
    let _ = MyZero::new("Pendulum-v1")
        .discretize(9) // 连续 env：须声明 MCTS 离散近似（档数；区间由库从 env 读取）
        .reward_scale(0.1)
        .solved(-200.0)
        .max_episodes(600)
        .train()?;
    Ok(())
}

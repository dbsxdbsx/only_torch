//! MyZero · Pendulum-v1（纯连续 → 离散候选）。
//!
//! 运行命令、诊断与 benchmark 见同目录 [`README.md`](README.md)。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

fn main() -> Result<(), GraphError> {
    let _ = MyZero::new("Pendulum-v1")
        .discretize(9)
        .reward_scale(0.1)
        .solved(-200.0)
        .max_episodes(600)
        .train()?;
    Ok(())
}

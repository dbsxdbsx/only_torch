//! MyZero · CartPole-v1（离散 2 动作，回归哨兵）。
//!
//! 运行命令、消融与 benchmark 见同目录 [`README.md`](README.md)。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

const BEST: &str = "models/my_zero/CartPole-v1/seed_42/best";

fn main() -> Result<(), GraphError> {
    let mz = MyZero::new("CartPole-v1")
        .consistency()
        .solved(475.0)
        .max_episodes(2000)
        .save_model_when_eval(BEST)
        .train()?;

    mz.load_model_if_exists(BEST)?.eval(10)?.run(Some(1))?;
    Ok(())
}

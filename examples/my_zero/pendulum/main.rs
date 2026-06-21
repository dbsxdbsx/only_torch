//! MyZero · Pendulum-v1（纯连续 → Sampled MuZero B=7，recipe 内置）。
//!
//! ```bash
//! cargo run --example my_zero_pendulum --release
//! SMOKE=1 cargo run --example my_zero_pendulum  # 管线验证
//! ```
//!
//! 运行命令、诊断与 benchmark 见同目录 [`README.md`](README.md)。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

const BEST: &str = "models/my_zero/Pendulum-v1/seed_42/best";

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    let mut builder = MyZero::new("Pendulum-v1")
        // Pendulum 专属：reward 缩放到 categorical support 域（组件栈见 recipe.rs）
        .reward_scale(0.1)
        .solved(-200.0)
        .max_episodes(if smoke { 3 } else { 600 })
        .save_model_when_eval(BEST);
    if smoke {
        builder = builder.smoke();
    }

    let mz = builder.train()?;

    if !smoke {
        mz.load_model_if_exists(BEST)?.eval(10)?.run(Some(1))?;
    }
    Ok(())
}

//! MyZero · Pendulum-v1（纯连续 → Sampled MuZero B=7，recipe 内置）。
//!
//! ```bash
//! cargo run --example my_zero_pendulum --release
//! SMOKE=1 cargo run --example my_zero_pendulum  # 管线验证
//! ```
//!
//! 运行命令、诊断与 benchmark 见同目录 [`README.md`](README.md)。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::{MyZero, TrainSettings};

// best 基名；多 seed 时库内自动插入 `seed_{seed}/` 子目录（见 checkpoint.rs）
const BEST: &str = "models/my_zero/Pendulum-v1/best";

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let diagnose = std::env::var("DIAG").is_ok();

    let mut builder = MyZero::new("Pendulum-v1")
        // Pendulum 专属：reward 缩放到 categorical support 域（组件栈见 recipe.rs）
        .reward_scale(0.1)
        .solved(-200.0)
        .max_episodes(if smoke { 3 } else { 600 })
        // 与旧「按 600 局预算前 50% 恒温」行为等价的显式常数（退火解耦预算后需自带）
        .train_settings(TrainSettings {
            temp_hold_episodes: 300,
            temp_decay_episodes: 300,
            ..TrainSettings::default()
        })
        .save_model_when_eval(BEST);
    if let Ok(v) = std::env::var("TD_STEPS") {
        builder = builder.td_steps(v.parse().expect("TD_STEPS 必须是正整数"));
    }
    if smoke {
        builder = builder.smoke();
    }
    if diagnose {
        builder = builder.diagnose();
    }

    let mz = builder.train()?;

    if !smoke {
        // 用本次训练实际落盘的 best 路径，避免多 seed 时加载错位
        let best = mz.train_report().and_then(|r| r.model_path.clone());
        let mz = match best {
            Some(p) => mz.load_model_if_exists(p)?,
            None => mz,
        };
        mz.eval(10)?.run(Some(1))?;
    }
    Ok(())
}

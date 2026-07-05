//! MyZero · CartPole-v1（离散 2 动作，回归哨兵）。
//!
//! ```bash
//! cargo run --example my_zero_cartpole --release
//! SMOKE=1 cargo run --example my_zero_cartpole  # 管线验证
//! ```
//!
//! 运行命令、消融与 benchmark 见同目录 [`README.md`](README.md)。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

// best 基名；多 seed 时库内自动插入 `seed_{seed}/` 子目录（见 checkpoint.rs）
const BEST: &str = "models/my_zero/CartPole-v1/best";

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();
    let diagnose = std::env::var("DIAG").is_ok();

    let mut builder = MyZero::new("CartPole-v1")
        .solved(475.0)
        .max_episodes(if smoke { 3 } else { 2000 })
        .save_model_when_eval(BEST);
    if let Ok(v) = std::env::var("TD_STEPS") {
        builder = builder.td_steps(v.parse().expect("TD_STEPS 必须是正整数"));
    }
    // SEEDS=N：多 seed 回归（统计口径哨兵，打印中位 env-steps）。
    if let Ok(v) = std::env::var("SEEDS") {
        builder = builder.seeds(v.parse().expect("SEEDS 必须是正整数"));
    }
    // KL_LR=1：KL 自适应 lr 消融开关（默认关；闸门与诊断用，勿入哨兵口径）。
    if std::env::var("KL_LR").is_ok() {
        builder = builder.kl_adaptive_lr(true);
    }
    if smoke {
        builder = builder.smoke();
    }
    if diagnose {
        builder = builder.diagnose();
    }

    let mz = builder.train()?;

    if !smoke {
        // 用本次训练实际落盘的 best 路径（多 seed 时为最后一个 seed 的 seed_{k}/best），
        // 而非固定路径——避免加载到旧文件或其他 seed 的模型。
        let best = mz.train_report().and_then(|r| r.model_path.clone());
        let mz = match best {
            Some(p) => mz.load_model_if_exists(p)?,
            None => mz,
        };
        mz.eval(10)?.run(Some(1))?;
    }
    Ok(())
}

//! MyZero · ALE/Pong-v5（图像离散环境基准账本）。
//!
//! **预注册口径**（见 `.doc/design/rl_myzero_status.md` §3 与本目录 README，跑前定稿）：
//! - obs：210×160×3 → 灰度 84² × 4 帧堆叠（库内图像管线自动接管）
//! - 门槛：3-seed 中位 best greedy(10 局) ≥ **−18**（随机 ≈ −20.7；非 SOTA 口径）
//! - 预算：每 seed ≤ 150 局（≈ 120k env-steps）或 wall-clock 24h 先到为准
//! - 组件：image base 栈 = consistency ON + reconstruction OFF + two-hot + raw obs（recipe 注入）
//!
//! ```bash
//! cargo run --example my_zero_pong --release --features blas-mkl
//! SMOKE=1 cargo run --example my_zero_pong        # 管线验证（3 局）
//! SEEDS=3 cargo run --example my_zero_pong --release --features blas-mkl  # 官方 3-seed
//! ```

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::{MyZero, TrainSettings};

// best 基名；多 seed 时库内自动插入 `seed_{seed}/` 子目录（见 checkpoint.rs）
const BEST: &str = "models/my_zero/Pong-v5/best";

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    // 预注册训练口径（见本目录 README；改动须走消融纪律）
    let train = TrainSettings {
        gamma: 0.997,
        k_unroll: 5,
        td_steps: 5,
        num_simulations: 20,
        lr: 0.003,              // CNN + Adam，比 CartPole MLP 的 0.02 保守
        train_batch_size: 16,   // 提高样本吞吐（CartPole 为 8）
        trains_per_episode: 64, // Pong 单局 ~900 步，按局训练需更高 replay ratio
        buffer_capacity: 32,    // 按局计;图像帧内存纪律（单帧存储 ≈ 28KB/步）
        start_training_after: 2,
        // 与旧「按 150 局预算前 50% 恒温」行为等价的显式常数（退火解耦预算后需自带）
        temp_hold_episodes: 75,
        temp_decay_episodes: 75,
        kl_adaptive_lr: false,
        ..TrainSettings::default()
    };

    let max_ep: usize = std::env::var("MAX_EP")
        .ok()
        .map(|v| v.parse().expect("MAX_EP 必须是正整数"))
        .unwrap_or(150);

    let mut builder = MyZero::new("ALE/Pong-v5")
        .solved(-18.0)
        .max_episodes(if smoke { 3 } else { max_ep })
        .train_settings(train)
        .eval_every(10)
        .save_model_when_eval(BEST);
    if let Ok(v) = std::env::var("SEEDS") {
        builder = builder.seeds(v.parse().expect("SEEDS 必须是正整数"));
    }
    if smoke {
        builder = builder.smoke();
    }

    let mz = builder.train()?;

    if !smoke {
        // 用本次训练实际落盘的 best 路径，避免多 seed 时加载错位
        let best = mz.train_report().and_then(|r| r.model_path.clone());
        let mz = match best {
            Some(p) => mz.load_model_if_exists(p)?,
            None => mz,
        };
        mz.eval(10)?;
    }
    Ok(())
}

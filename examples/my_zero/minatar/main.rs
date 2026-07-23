//! MyZero · MinAtar/Breakout-v1（第三支柱裁决场，v0.26）。
//!
//! **预注册口径**（`examples/my_zero/minatar/README.md`）：
//! - obs：10×10×4 bool → Board adapter（HWC→CHW 转置，不做灰度/缩放/堆叠）
//! - 网络：BoardConvRepresentationNet stride-1 3×3（4→32→64）→ fc → latent_dim=64
//! - 门槛：3-seed 中位 best greedy(10 局) ≥ 8（脱离随机 ~1-2）
//! - 预算：每 seed 200k env-steps 或 wall-clock 2h 先到为准
//! - 组件：minatar base 栈 = consistency ON + reconstruction OFF（recipe 注入）
//!
//! ```bash
//! cargo run --example my_zero_minatar --release --features blas-mkl
//! SMOKE=1 cargo run --example my_zero_minatar         # 管线验证（3 局）
//! SEEDS=3 cargo run --example my_zero_minatar --release --features blas-mkl  # 官方 3-seed
//! GAME=Freeway SEEDS=3 cargo run --example my_zero_minatar --release --features blas-mkl
//! ```

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::{MyZero, TrainSettings};

const BEST_PREFIX: &str = "models/my_zero/MinAtar";

fn main() -> Result<(), GraphError> {
    let smoke = std::env::var("SMOKE").is_ok();

    // 默认 Breakout；可通过 GAME=Freeway 切换
    let game = std::env::var("GAME").unwrap_or_else(|_| "Breakout".to_string());
    let env_id: &'static str = match game.as_str() {
        "Breakout" => "MinAtar/Breakout-v1",
        "Freeway" => "MinAtar/Freeway-v1",
        "Asterix" => "MinAtar/Asterix-v1",
        "Seaquest" => "MinAtar/Seaquest-v1",
        "SpaceInvaders" => "MinAtar/SpaceInvaders-v1",
        other => {
            eprintln!("未知 MinAtar 游戏: {other}，支持: Breakout/Freeway/Asterix/Seaquest/SpaceInvaders");
            std::process::exit(1);
        }
    };

    let best_path = format!("{BEST_PREFIX}/{game}/best");
    let best: &'static str = Box::leak(best_path.into_boxed_str());

    // 预注册训练口径（MinAtar 计划；改动须走消融纪律）
    let train = TrainSettings {
        gamma: 0.99,
        k_unroll: 5,
        td_steps: 10,
        num_simulations: 25,
        lr: 0.005,
        train_batch_size: 32,
        trains_per_episode: 16,
        buffer_capacity: 128,
        start_training_after: 4,
        temp_hold_episodes: 100,
        temp_decay_episodes: 100,
        kl_adaptive_lr: false,
        ..TrainSettings::default()
    };

    // 200k env-steps budget；MinAtar Breakout 平均 ~25 步/局 → ~8000 局
    // 但 MCTS 搜索+训练开销让有效吞吐远低于此，实际约 500-2000 局
    let max_ep: usize = std::env::var("MAX_EP")
        .ok()
        .map(|v| v.parse().expect("MAX_EP 必须是正整数"))
        .unwrap_or(2000);

    let mut builder = MyZero::new(env_id)
        .solved(if game == "Freeway" { 20.0 } else { 8.0 })
        .max_episodes(if smoke { 3 } else { max_ep })
        .train_settings(train)
        .eval_every(50)
        .save_model_when_eval(best);

    if let Ok(v) = std::env::var("SEEDS") {
        builder = builder.seeds(v.parse().expect("SEEDS 必须是正整数"));
    }
    if smoke {
        builder = builder.smoke();
    }

    let mz = builder.train()?;

    if !smoke {
        let best_model = mz.train_report().and_then(|r| r.model_path.clone());
        let mz = match best_model {
            Some(p) => mz.load_model_if_exists(p)?,
            None => mz,
        };
        mz.eval(10)?;
    }
    Ok(())
}

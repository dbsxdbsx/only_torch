//! MyZero · CartPole-v1（离散 2 动作）
//!
//! ```bash
//! cargo run --example my_zero_cartpole --release
//! CONSISTENCY=1 CQ=1 SIMS=16 cargo run --example my_zero_cartpole --release
//! SEEDS=3 CONSISTENCY=1 cargo run --example my_zero_cartpole --release
//! SMOKE=1 cargo run --example my_zero_cartpole
//! ```

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::MyZero;

fn main() -> Result<(), GraphError> {
    let mz = MyZero::new("CartPole-v1")
        .solved(475.0) // Gymnasium greedy eval solved
        .max_episodes(2000)
        .train()?;

    if let Some(r) = mz.train_report() {
        println!(
            "训练完成: best greedy={:.1} @ ep {:?}, model={:?}",
            r.best_greedy, r.best_at_episode, r.model_path,
        );
    }

    // 同一实例训后直接 eval：使用 latest 权重；要看 best 请先 load_model(model_path)
    mz.eval(10)?.run(Some(1))?;
    Ok(())
}

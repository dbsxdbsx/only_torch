//! MyZero · CartPole-v1（离散 2 动作）训练示例
//!
//! 算法主体全在库 `only_torch::rl::algo::my_zero`；本示例只填 config + 调 `run`。
//! 动作空间（离散 2 档）由库从 env 自动推断，无需在此声明。
//!
//! ```bash
//! cargo run --example my_zero_cartpole --release
//! CONSISTENCY=1 CQ=1 SIMS=16 cargo run --example my_zero_cartpole --release  # 最快达标 ~40s
//! SEEDS=3 CONSISTENCY=1 cargo run --example my_zero_cartpole --release        # 多 seed 中位数
//! SMOKE=1 cargo run --example my_zero_cartpole                            # 管线验证 ~30s
//! ```
//!
//! 达标：greedy(temp=0) eval 10 局（固定 seed）均值 ≥ 475（Gymnasium 官方 solved）。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::{
    ActionPlan, EnvConfig, MyZeroConfig, RunConfig, TrainConfig, run,
};

fn main() -> Result<(), GraphError> {
    let mut cfg = MyZeroConfig {
        env_config: EnvConfig {
            env_id: "CartPole-v1",
            reward_scale: 1.0,
            action: ActionPlan::Auto, // 离散：库自动按 env 动作数枚举
        },
        train_config: TrainConfig {
            gamma: 0.997,
            ..TrainConfig::default()
        },
        run_config: RunConfig {
            solved: 475.0,
            max_episodes: 2000,
            ..RunConfig::default()
        },
        ..MyZeroConfig::default()
    };
    // 消融 / 调参 / 跑法旋钮统一从环境变量覆盖
    cfg.apply_env_overrides();
    run(&cfg)
}

//! MyZero · Pendulum-v1（纯连续 1 维 → 离散候选）训练示例
//!
//! 算法主体全在库 `only_torch::rl::algo::my_zero`；本示例只填 config + 调 `run`。
//! 唯一的建模选择是「连续怎么近似」——这里离散成 9 档（env 的连续区间 [-2,2] 由库自动读）。
//! 忠实 Gumbel 连续搜索留作后续——**仅当离散化触顶（达不到 SAC 水平）时才需要**。
//!
//! ```bash
//! cargo run --example my_zero_pendulum --release
//! CONSISTENCY=1 cargo run --example my_zero_pendulum --release    # +consistency
//! SEEDS=3 cargo run --example my_zero_pendulum --release      # 多 seed 中位数
//! SMOKE=1 cargo run --example my_zero_pendulum                # 管线验证
//! DIAG=1 cargo run --example my_zero_pendulum --release       # dynamics 诊断
//! ```
//!
//! 达标：greedy(temp=0) eval return ≥ -200。
//! 支持旋钮：`CONSISTENCY / VALUE_PREFIX / TARGET_NET / SVE / CQ / CQ_SCALE / CQ_VISIT /
//! SIMS / SEEDS / SMOKE / DIAG / GAMMA / MAX_EP / LR / NUM_ACTIONS / RSCALE / SOLVED`。

use only_torch::nn::GraphError;
use only_torch::rl::algo::my_zero::{
    ActionPlan, EnvConfig, MyZeroConfig, RunConfig, TrainConfig, run,
};

fn main() -> Result<(), GraphError> {
    let mut cfg = MyZeroConfig {
        env_config: EnvConfig {
            env_id: "Pendulum-v1",
            reward_scale: 0.1, // 累计 value 落入 categorical support 域
            action: ActionPlan::Discretize { buckets: 9 },
        },
        train_config: TrainConfig {
            // 200 步密集奖励连续控制：短折扣 0.99 通常比 0.997 更稳更快
            gamma: 0.99,
            ..TrainConfig::default()
        },
        run_config: RunConfig {
            solved: -200.0,
            max_episodes: 600,
            ..RunConfig::default()
        },
        ..MyZeroConfig::default()
    };
    cfg.apply_env_overrides();
    run(&cfg)
}

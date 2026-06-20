//! MyZero 统一 model-based RL 算法
//!
//! only_torch 唯一的 `*Zero` 算法——从 canonical MuZero 出发，以消融实验方式逐增量叠加
//! 组件（consistency / value prefix / target / SVE / Gumbel），覆盖全动作空间与全状态类型。
//!
//! 核心哲学：**奥卡姆剃刀**——每叠一个组件必须用消融证明其价值，保证不回归。
//!
//! # 用法（示例侧只需 config + run）
//!
//! ```ignore
//! use only_torch::rl::algo::my_zero::{run, ActionPlan, EnvConfig, MyZeroConfig, RunConfig, TrainConfig};
//!
//! let mut cfg = MyZeroConfig {
//!     env_config: EnvConfig { env_id: "CartPole-v1", reward_scale: 1.0, action: ActionPlan::Auto },
//!     train_config: TrainConfig { gamma: 0.997, ..TrainConfig::default() },
//!     run_config: RunConfig { solved: 475.0, ..RunConfig::default() },
//!     ..MyZeroConfig::default()
//! };
//! cfg.apply_env_overrides(); // CONSISTENCY / CQ / SIMS / SEEDS / SMOKE ...
//! run(&cfg).unwrap();
//! ```
//!
//! # 代码组织
//!
//! 框架：
//! - [`config`]：5 层统一配置（env / model / train / component / run）+ env 变量覆盖
//! - [`component`]：消融组件开关
//! - [`network`]：三网络模型（repr / dyn / pred + value-prefix LSTM + SimSiam 分支）
//! - [`action`]：动作适配（从 env 推断离散/连续 + idx→env 映射）
//! - [`runner`]：统一 `run()`（self-play + 训练 + eval + 多 seed + SMOKE + DIAG）
//!
//! 算法组件（**均为 MyZero 自身实现，自包含**；MyZero 是项目唯一的 `*Zero` 实现）：
//! - [`value_encoding`] / [`value_transform`]：categorical value/reward 编解码 + h(x) 变换
//! - [`n_step`]：n-step bootstrap value target（区分 terminated / truncated）
//! - [`reanalyze`]：用最新网络重跑 MCTS 刷新旧轨迹目标
//! - [`loss`]：loss 系数与梯度缩放常量
//! - [`consistency`]：自监督 consistency loss（SimSiam）
//! - [`value_prefix`]：value prefix 累计 reward 前缀目标
//! - [`target`]：completedQ 改进策略目标
//! - [`target_net`]：target network 同步（可用组件，接线属后续消融）
//! - [`sve`]：search-based value estimation blend（可用组件，接线属后续消融）

pub mod action;
pub mod component;
pub mod config;
pub mod consistency;
pub mod loss;
pub mod n_step;
pub mod network;
pub mod reanalyze;
pub mod runner;
pub mod sve;
pub mod value_encoding;
pub mod target;
pub mod target_net;
pub mod value_prefix;
pub mod value_transform;

#[cfg(test)]
mod tests;

pub use action::ActionAdapter;
pub use component::ComponentConfig;
pub use config::{ActionPlan, EnvConfig, ModelConfig, MyZeroConfig, RunConfig, TrainConfig};
pub use consistency::negative_cosine_similarity;
pub use n_step::{compute_n_step_target, compute_n_step_target_with};
pub use network::MyZeroModel;
pub use reanalyze::reanalyze_game;
pub use runner::run;
pub use sve::sve_blend;
pub use value_encoding::{SupportConfig, scalar_to_two_hot, two_hot_to_scalar};
pub use target::completed_q_policy_target;
pub use target_net::{TargetConfig, ema_update, hard_update, is_hard_sync_step, sync_target};
pub use value_prefix::{prefix_to_delta, reward_prefix_targets};
pub use value_transform::{value_transform, value_transform_inv};

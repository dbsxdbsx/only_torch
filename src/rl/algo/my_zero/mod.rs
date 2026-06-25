//! MyZero 统一 model-based RL 算法
//!
//! only_torch 唯一的 `*Zero` 算法——从 canonical MuZero 出发，以消融实验方式逐增量叠加
//! 组件（consistency / value prefix / target / SVE / Gumbel），覆盖全动作空间与全状态类型。
//!
//! 核心哲学：**奥卡姆剃刀**——每叠一个组件必须用消融证明其价值，保证不回归。
//!
//! # 用法
//!
//! ```ignore
//! use only_torch::rl::algo::my_zero::MyZero;
//!
//! // 训练（返回 latest）；eval 创新高时落盘须 .save_model_when_eval(path)
//! let best = "models/my_zero/CartPole-v1/seed_42/best";
//! let mz = MyZero::new("CartPole-v1")
//!     .solved(475.0)
//!     .max_episodes(2000)
//!     .save_model_when_eval(best)
//!     .train()?;
//!
//! mz.load_model_if_exists(best)?.eval(10)?;
//!
//! // 冷启动推理
//! MyZero::new("CartPole-v1")
//!     .load_model_if_exists("models/my_zero/CartPole-v1/seed_42/best")?
//!     .run(Some(10))?;
//! ```
//!
//! # 代码组织
//!
//! 框架：
//! - [`config`]：配置（env / model / train / components / eval）
//! - [`builder`]：链式 builder（`MyZero::new(env_id)` 为唯一入口）
//! - [`my_zero`]：运行体（`train` 返回 **latest**；`load_model_if_exists` 加载磁盘 best；`eval` / `run` 用当前实例权重）
//! - [`model_io`]：`.otm` 持久化（内部契约校验）
//! - [`checkpoint`]：eval 创新高时落盘（须显式 `.save_model_when_eval(path)`）
//! - [`report`]：train / eval / run 分数报告
//! - [`component`]：内部组件开关（[`recipe`] 按 env 注入）
//! - [`recipe`]：环境内置算法配方（团队维护，用户不可见）
//! - [`network`]：三网络模型（repr / dyn / pred + continuation head + value-prefix LSTM + SimSiam / reconstruction 分支）
//! - [`action`]：动作适配（从 env 推断离散/连续 + idx→env 映射）
//! - [`runner`]：训练循环 + greedy eval 内部实现
//!
//! 算法组件（**均为 MyZero 自身实现，自包含**；MyZero 是项目唯一的 `*Zero` 实现）：
//! - [`value_encoding`] / [`value_transform`]：categorical value/reward 编解码 + h(x) 变换
//! - [`n_step`]：n-step bootstrap value target（区分 terminated / truncated，并使用 transition continuation）
//! - [`reanalyze`]：position 级 MCTS 重搜 + train 后写回（`Components.reanalyze`；CartPole 暂不 promote）
//! - [`loss`]：loss 系数与梯度缩放常量
//! - [`consistency`]：自监督 consistency loss（SimSiam）
//! - [`reconstruction`]：自监督 reconstruction loss（Scholz et al. 2021 · arXiv:2102.05599）
//! - [`value_prefix`]：value prefix 累计 reward 前缀目标
//! - [`target`]：completedQ 改进策略目标
//! - [`target_net`]：target network 同步（可用组件，接线属后续消融）
//! - [`sve`]：search-based value estimation blend（可用组件，接线属后续消融）

pub mod action;
pub mod builder;
pub mod checkpoint;
pub mod component;
pub mod config;
pub mod consistency;
pub mod loss;
pub mod model_io;
pub mod my_zero;
pub mod n_step;
pub mod network;
pub mod reanalyze;
pub mod recipe;
pub mod reconstruction;
pub mod report;
pub mod runner;
pub(crate) mod sampled_params;
pub(crate) mod search_policy;
pub mod sve;
pub mod target;
pub mod target_net;
pub mod value_encoding;
pub mod value_prefix;
pub mod value_transform;

#[cfg(test)]
mod tests;

pub use action::ActionAdapter;
pub use builder::MyZeroBuilder;
pub use config::{
    ActionPlan, CheckpointSettings, DEFAULT_EVAL_SEED, DEFAULT_ROLLOUT_SEED, DEFAULT_SEED,
    DEFAULT_TRAIN_SEED, EnvSettings, EvalSettings, ModelSettings, MyZeroConfig, TrainSettings,
};
pub use consistency::negative_cosine_similarity;
pub use my_zero::MyZero;
pub use n_step::{compute_n_step_target, compute_n_step_target_with};
pub use network::MyZeroModel;
pub use reanalyze::{reanalyze_game, reanalyze_step, reanalyze_unroll_window};
pub use report::{EvalReport, RunReport, TrainReport};
pub use sve::sve_blend;
pub use target::completed_q_policy_target;
pub use target_net::{TargetConfig, ema_update, hard_update, is_hard_sync_step, sync_target};
pub use value_encoding::{SupportConfig, scalar_to_two_hot, two_hot_to_scalar};
pub use value_prefix::{prefix_to_delta, reward_prefix_targets};
pub use value_transform::{value_transform, value_transform_inv};

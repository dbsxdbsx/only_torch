//! EfficientZero V2 函数式 helper + 配置（v0.24）
//!
//! 复用 `muzero/` 的 `support` / `value_transform` / `n_step` / `reanalyze`；EZ 专属增量分模块：
//! - [`config`]：[`EfficientZeroConfig`]（组合 base/search/gumbel/reanalyze/target/loss）
//! - [`consistency`]：自监督 consistency loss（SimSiam stop-grad）—— Phase 1 实现
//! - [`value_prefix`]：value prefix（LSTM 累计 reward 前缀）目标 —— Phase 1 实现（忠实版）
//! - [`target`]：target network（hard / EMA 更新 + 同步间隔）—— Phase 1 实现
//! - [`sve`]：search-based value estimation —— Phase 1 实现
//!
//! # 设计边界
//! - **入库**：配置组合、consistency / value-prefix 目标、target 更新、SVE blend（纯函数）
//! - **留示例**：三网络结构（repr/dyn/pred）+ LSTM value-prefix 头 + SimSiam projector、
//!   K 步 unroll 训练循环、self-play 主流程
//! - **复用 mcts/**：`mcts_search`（RNG 注入）、`SearchPolicy` / `RootScheduler`（Gumbel）、
//!   `ActionSampler`（连续/混合候选）、`MctsModel::State` 不透明契约（value prefix hidden 穿树）

pub mod config;
pub mod consistency;
pub mod sve;
pub mod target;
pub mod value_prefix;

pub use config::{EfficientZeroConfig, EzLossConfig, GumbelConfig, ReanalyzeConfig, TargetConfig};
pub use consistency::negative_cosine_similarity;
pub use sve::sve_blend;
pub use target::{ema_update, hard_update, is_hard_sync_step, sync_target};
pub use value_prefix::{prefix_to_delta, reward_prefix_targets};

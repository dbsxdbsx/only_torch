//! MuZero 函数式 helper
//!
//! 提供 MuZero 训练中可复用的纯函数 + 配置，涵盖：
//! - [`MuZeroConfig`]：训练/搜索超参容器（`num_simulations` 等按环境配置）
//! - [`value_transform`] / [`value_transform_inv`]：标量 value/reward 变换 `h(x)`
//! - [`compute_n_step_target`]：n-step bootstrapped return 计算（区分 terminated/truncated）
//! - [`reanalyze_game`]：用最新网络重跑 MCTS 刷新旧轨迹的 policy/value 目标
//! - [`support`] 模块：categorical value/reward 的 support + two-hot 编解码
//! - [`loss`] 模块：标准 loss 系数常量
//!
//! # 设计边界
//! - **入库**：超参容器、数学变换、n-step target、reanalyze、categorical 编解码、loss 系数
//! - **留示例**：三网络结构（repr/dyn/pred）、K 步 unroll 训练循环、self-play 主流程
//! - **推迟 v0.24 EZ-V2**：value prefix、SVE、Gumbel 搜索、target network

pub mod config;
pub mod loss;
mod n_step;
pub mod reanalyze;
pub mod support;
mod value_transform;

pub use config::MuZeroConfig;
pub use n_step::compute_n_step_target;
pub use reanalyze::reanalyze_game;
pub use support::{SupportConfig, scalar_to_two_hot, two_hot_to_scalar};
pub use value_transform::{value_transform, value_transform_inv};

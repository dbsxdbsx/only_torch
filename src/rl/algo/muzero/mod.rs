//! MuZero 函数式 helper
//!
//! 提供 MuZero 训练中可复用的纯函数，涵盖：
//! - [`value_transform`] / [`value_transform_inv`]：标量 value/reward 变换 `h(x)`
//! - [`compute_n_step_target`]：n-step bootstrapped return 计算（区分 terminated/truncated）
//! - [`support`] 模块：categorical value/reward 的 support + two-hot 编解码
//! - [`loss`] 模块：标准 loss 系数常量
//!
//! # 设计边界
//! - **入库**：数学变换、n-step target、categorical 编解码、loss 系数
//! - **留示例**：三网络结构（repr/dyn/pred）、K 步 unroll 训练循环、self-play 主流程
//! - **推迟 v0.24 EZ-V2**：reanalyze、value prefix、SVE、Gumbel 搜索

pub mod loss;
mod n_step;
pub mod support;
mod value_transform;

pub use n_step::compute_n_step_target;
pub use support::{scalar_to_two_hot, two_hot_to_scalar, SupportConfig};
pub use value_transform::{value_transform, value_transform_inv};

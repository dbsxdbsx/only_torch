//! MuZero 函数式 helper
//!
//! 提供 MuZero 训练中可复用的纯函数，涵盖：
//! - [`value_transform`] / [`value_transform_inv`]：标量 value/reward 变换（稳定 MSE 训练）
//! - [`compute_n_step_target`]：n-step bootstrapped return 计算
//! - [`loss`] 模块：标准 loss 系数常量
//!
//! # 设计边界
//! - **入库**：数学变换、n-step target、loss 系数
//! - **留示例**：三网络结构（repr/dyn/pred）、K 步 unroll 训练循环、self-play 主流程
//! - **推迟 v0.24 EZ-V2**：categorical value/reward 表示、reanalyze、value prefix

pub mod loss;
mod n_step;
mod value_transform;

pub use n_step::compute_n_step_target;
pub use value_transform::{value_transform, value_transform_inv};

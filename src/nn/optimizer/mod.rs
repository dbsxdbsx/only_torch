/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:00:00
 * @LastEditors  : 老董
 * @LastEditTime : 2025-07-24 16:00:00
 * @Description  : 优化器模块，实现各种梯度优化算法
 */

mod adam;
mod base;
mod sgd;

pub use adam::Adam;
pub use base::Optimizer;
pub use sgd::SGD;

// 内部实现，仅 crate 内可见（用于单元测试）
#[cfg(test)]
pub(crate) use base::{GradientAccumulator, OptimizerState};

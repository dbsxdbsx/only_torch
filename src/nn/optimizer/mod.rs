/*
 * @Author       : 老董
 * @Date         : 2025-07-24 16:00:00
 * @LastEditors  : 老董
 * @LastEditTime : 2026-01-17
 * @Description  : 优化器模块，实现各种梯度优化算法
 */

mod adam;
mod base;
mod optimizer_v2;
mod sgd;

// V1 API (旧版，需要 &mut Graph)
pub use adam::Adam;
pub use base::Optimizer;
pub use sgd::SGD;

// V2 API (新版，PyTorch 风格)
pub use optimizer_v2::{Adamv2, OptimizerV2, SGDv2};

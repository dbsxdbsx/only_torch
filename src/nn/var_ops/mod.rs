/*
 * @Author       : 老董
 * @Date         : 2026-01-09
 * @Description  : Var 扩展 trait 模块
 *
 * 按功能领域组织 Var 的扩展方法，用户按需 import。
 * 设计依据：architecture_v2_design.md §4.2.1.3
 *
 * # 模块结构
 * - `activation`: 激活函数（relu, sigmoid, tanh, leaky_relu, softplus, step, sign）
 * - `loss`: 损失函数（cross_entropy, mse_loss）
 * - `matrix`: 矩阵运算（matmul）
 * - `shape`: 形状变换（reshape, flatten）
 *
 * # 使用示例
 * ```ignore
 * use only_torch::nn::var::{Var, VarActivationOps, VarLossOps, VarMatrixOps, VarShapeOps};
 *
 * let h = x.relu().sigmoid();
 * let y = h.matmul(&w)?;
 * let loss = y.cross_entropy(&labels)?;
 * let flat = h.flatten()?;
 * ```
 */

mod activation;
mod loss;
mod matrix;
mod shape;

pub use activation::VarActivationOps;
pub use loss::VarLossOps;
pub use matrix::VarMatrixOps;
pub use shape::VarShapeOps;

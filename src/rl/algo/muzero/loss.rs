//! MuZero loss 系数与梯度缩放常量
//!
//! 论文（Schrittwieser et al., 2020）附录 G 中的标准配置。

/// Value loss 系数（论文默认 0.25）
///
/// 经 value_transform 后 value MSE 量级已大幅降低，
/// 0.25 进一步平衡 policy CE 和 value/reward MSE 的梯度贡献。
pub const VALUE_LOSS_COEF: f32 = 0.25;

/// Reward loss 系数
///
/// 经 value_transform 后 reward MSE 量级与 policy CE 接近，
/// 使用 1.0（不额外缩放）。
pub const REWARD_LOSS_COEF: f32 = 1.0;

/// K 步 unroll 中 dynamics 边界的梯度缩放因子
///
/// 原论文在每个 dynamics step 边界乘以 0.5，
/// 防止 K 步反传的梯度指数增长。
pub const DYNAMICS_GRADIENT_SCALE: f32 = 0.5;

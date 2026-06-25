//! MyZero loss 系数与梯度缩放常量（Schrittwieser et al. 2020 附录 G）。

/// Value loss 系数（论文默认 0.25）。
pub const VALUE_LOSS_COEF: f32 = 0.25;

/// Reward loss 系数（经 value_transform 后量级与 policy CE 接近，1.0 不额外缩放）。
pub const REWARD_LOSS_COEF: f32 = 1.0;

/// Continuation loss 系数（单标量 MSE；保持基础语义监督但不主导 policy/value）。
pub const CONTINUATION_LOSS_COEF: f32 = 1.0;

/// Reconstruction loss 系数（Scholz et al. 2021 默认 lg 权重 1.0）。
pub const RECONSTRUCTION_LOSS_COEF: f32 = 1.0;

/// K 步 unroll 中 dynamics 边界的梯度缩放因子（每个 dynamics step 边界乘 0.5，
/// 防 K 步反传梯度指数增长）。
pub const DYNAMICS_GRADIENT_SCALE: f32 = 0.5;

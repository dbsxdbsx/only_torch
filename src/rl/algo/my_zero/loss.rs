//! MyZero loss 系数与梯度缩放常量（Schrittwieser et al. 2020 附录 G）。

/// Value loss 系数（论文默认 0.25）。
pub const VALUE_LOSS_COEF: f32 = 0.25;

/// Reward loss 系数（经 value_transform 后量级与 policy CE 接近，1.0 不额外缩放）。
pub const REWARD_LOSS_COEF: f32 = 1.0;

/// Continuation loss 系数（单标量 MSE；保持基础语义监督但不主导 policy/value）。
pub const CONTINUATION_LOSS_COEF: f32 = 1.0;

/// Reconstruction loss 系数。
///
/// 论文（Scholz et al. 2021）lg 权重 1.0；**v0.26 P0 重标定为 16.0**（CartPole 对数网格
/// {1,4,16,64} 消融，账本 examples/my_zero/cartpole/README.md）：autograd `upstream_grad`
/// 修复前该 loss 反向丢 `step_scale(1/K)` 与 batch 均值 `(1/B)`，等效隐式放大 ≈ K×B=40，
/// 旧基线的样本效率部分来源于此；修复后 coef=1 实测有害（5-seed 4/5 达标、中位 55.7k），
/// 16 为网格平台点（4 偏弱、64 过冲达标率跌 2/3）。
/// **临时值**：单环境证据；图像线 obs 归一化后量纲变化，须复验后定稿。
pub const RECONSTRUCTION_LOSS_COEF: f32 = 16.0;

/// Consistency loss 系数（对齐 EfficientZero 自监督权重 2.0）。
pub const CONSISTENCY_LOSS_COEF: f32 = 2.0;

/// K 步 unroll 中 dynamics 边界的梯度缩放因子（每个 dynamics step 边界乘 0.5，
/// 防 K 步反传梯度指数增长）。
pub const DYNAMICS_GRADIENT_SCALE: f32 = 0.5;

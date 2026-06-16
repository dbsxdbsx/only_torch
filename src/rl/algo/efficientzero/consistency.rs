//! 自监督 consistency loss（SimSiam stop-grad）—— v0.24 Phase 1 `+consistency` 实现。
//!
//! 目标：让 dynamics 预测的 `next_latent` 与 `repr(next_obs)`（stop-grad）对齐，给 dynamics
//! 一个稠密的自监督信号（EZ 样本效率的关键之一）。
//!
//! 需在 `Var`（autograd）层实现负余弦相似度 + stop-gradient（复用 `detach`），与示例 model 的
//! projector / predictor 头耦合，故随 Phase 1 EZ 示例一起落地；此处先占位，保持模块结构稳定。

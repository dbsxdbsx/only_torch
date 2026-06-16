//! Target network 更新（hard / EMA + 同步间隔）—— v0.24 Phase 1 `+target` 实现。
//!
//! Target network 是 EZ 的稳定性增强（base MuZero 不需要，reanalyze 也可改用 target net）。
//! 配置见 [`crate::rl::algo::efficientzero::TargetConfig`]：`sync_interval > 0` 走 hard copy，
//! `== 0` 走 EMA（用 `tau`）。
//!
//! 参数同步操作于 `Tensor` / `Var` 参数列表，与示例 model 的两份网络实例耦合，故随 Phase 1
//! 落地；此处先占位，保持模块结构稳定。

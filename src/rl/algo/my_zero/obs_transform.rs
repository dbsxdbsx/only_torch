//! obs 无量纲化变换（v0.26 Phase 0，收口规划 §1）。
//!
//! `symlog(x) = sign(x)·ln(1+|x|)`（DreamerV3 口径）：无状态、可逆、原点附近 ≈ 恒等、
//! 长尾对数压缩。用于把 obs（含 reconstruction 解码目标）压到统一量纲，使 recon MSE
//! 与其系数跨环境可迁移——与 value/reward 侧的 h 变换（ε-symlog 同族）补全对称。
//!
//! **边界铁律**：变换属于模型（[`MyZeroModel`](super::network::MyZeroModel) obs 入口单点，
//! 按 `Components.obs_symlog` 注入）；replay buffer / env I/O / 账本一律存 **raw obs**。

use std::borrow::Cow;

/// `symlog(x) = sign(x)·ln(1+|x|)`
pub(crate) fn symlog(x: f32) -> f32 {
    x.signum() * x.abs().ln_1p()
}

/// `symexp(y) = sign(y)·(e^{|y|}−1)`（symlog 逆变换；目前仅测试用）
#[cfg(test)]
pub(crate) fn symexp(y: f32) -> f32 {
    y.signum() * (y.abs().exp() - 1.0)
}

/// 按开关对 obs 切片做 symlog：关 → 原样借用（零拷贝零行为变化），开 → 逐元素变换。
pub(crate) fn maybe_symlog(on: bool, obs: &[f32]) -> Cow<'_, [f32]> {
    if on {
        Cow::Owned(obs.iter().copied().map(symlog).collect())
    } else {
        Cow::Borrowed(obs)
    }
}

/// 按开关对已展平的 `[G × dim]` 数据做原位 symlog（batch 路径用）。
pub(crate) fn maybe_symlog_in_place(on: bool, flat: &mut [f32]) {
    if on {
        for v in flat.iter_mut() {
            *v = symlog(*v);
        }
    }
}

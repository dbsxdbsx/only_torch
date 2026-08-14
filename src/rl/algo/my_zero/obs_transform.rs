//! obs 无量纲化变换（symlog；连续观测 always-on）。
//!
//! `symlog(x) = sign(x)·ln(1+|x|)`（DreamerV3 口径）：无状态、可逆、原点附近 ≈ 恒等、
//! 长尾对数压缩。与 value/reward 侧共用同一公式，但观测**不进** 41 原子 support。
//!
//! 不读 Box、不比维间倍数：门槛会因「A 比 B 十倍、B 比 C 十倍」叠成百倍而站不住。
//! token ID 除外（查表前必须是整数）。
//!
//! **边界铁律**：变换属于模型（[`MyZeroModel`](super::network::MyZeroModel) obs 入口单点）；
//! replay buffer / env I/O / 账本一律存 **raw obs**。

use super::value_transform::value_transform;
#[cfg(test)]
use super::value_transform::value_transform_inv;
use std::borrow::Cow;

/// 与 value 编码同一公式；观测不进 support 原子。
pub(crate) fn symlog(x: f32) -> f32 {
    value_transform(x)
}

/// `symexp`（目前仅测试用）
#[cfg(test)]
pub(crate) fn symexp(y: f32) -> f32 {
    value_transform_inv(y)
}

/// 运行时门闩：关 → 原样借用（零拷贝），开 → 逐元素 `symlog`。
/// 生产路径对连续观测恒开；token 与单测关路径走零拷贝。
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

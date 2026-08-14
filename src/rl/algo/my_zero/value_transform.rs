//! MyZero 标量 value/reward 变换：always-on `symlog`。
//!
//! `symlog(x) = sign(x)·ln(1+|x|)`，逆变换 `symexp(y) = sign(y)·(e^{|y|}-1)`。
//! 小处近线性、大处对数压缩、无水平墙、可逆。配合 support `[-20,20]` 罩住未知量纲。
//!
//! 观测不走这套码本：观测不进 41 个原子。连续观测另有入口 `symlog`，见 [`obs_transform`](super::obs_transform)。

/// value/reward 正向：`symlog`。
pub fn value_transform(x: f32) -> f32 {
    x.signum() * x.abs().ln_1p()
}

/// value/reward 逆向：`symexp`。搜索 / 解码时还原标量。
pub fn value_transform_inv(y: f32) -> f32 {
    y.signum() * (y.abs().exp() - 1.0)
}

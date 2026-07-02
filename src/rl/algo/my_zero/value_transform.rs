//! MyZero 标量 value/reward 变换（Schrittwieser et al. 2020 附录 F）
//!
//! `h(x) = sign(x)(sqrt(|x|+1) - 1) + εx`
//!
//! 作用：压缩大 value 的梯度，使 loss 在不同量级的 value 上保持稳定。
//! CartPole value 范围 0~200，经变换后压缩到 ~0-13.4。

const EPS: f32 = 0.001;

/// value/reward 正向变换 h(x) = sign(x)(sqrt(|x|+1) - 1) + εx
///
/// 单调递增、连续、可微。x=0 处 h(0)=0；大 |x| 时 h(x) ≈ sign(x)·sqrt(|x|)。
pub fn value_transform(x: f32) -> f32 {
    x.signum() * ((x.abs() + 1.0).sqrt() - 1.0) + EPS * x
}

/// value/reward 逆变换 h⁻¹(y)
///
/// 解析解：对 h(x)=y 关于 x 求解。搜索推理时将网络输出还原为真实标量。
pub fn value_transform_inv(y: f32) -> f32 {
    // 论文（附录 F）给出的闭合形式（对绝对值部分）：
    //   |x| = ((sqrt(1 + 4ε(|y| + 1 + ε)) - 1) / (2ε))² - 1
    let abs_y = y.abs();
    let inner = (1.0 + 4.0 * EPS * (abs_y + 1.0 + EPS)).sqrt();
    let abs_x = ((inner - 1.0) / (2.0 * EPS)).powi(2) - 1.0;
    // 防止浮点误差导致负数
    let abs_x = abs_x.max(0.0);
    y.signum() * abs_x
}

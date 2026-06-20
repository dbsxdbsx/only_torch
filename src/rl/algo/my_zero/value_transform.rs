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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_maps_to_zero() {
        assert!((value_transform(0.0)).abs() < 1e-7);
        assert!((value_transform_inv(0.0)).abs() < 1e-7);
    }

    #[test]
    fn monotonically_increasing() {
        let xs: Vec<f32> = (-50..=50).map(|i| i as f32 * 4.0).collect();
        for w in xs.windows(2) {
            assert!(value_transform(w[1]) > value_transform(w[0]));
        }
    }

    #[test]
    fn compresses_large_values() {
        let h200 = value_transform(200.0);
        assert!(h200 < 15.0, "h(200) = {h200}，应显著小于 200");
        assert!(h200 > 10.0, "h(200) = {h200}，应大于 10（非退化）");
    }

    #[test]
    fn round_trip_precision() {
        let test_values = [
            0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 200.0, -200.0, 0.5, -0.001,
        ];
        for &x in &test_values {
            let y = value_transform(x);
            let x_back = value_transform_inv(y);
            assert!((x_back - x).abs() < 0.1, "round-trip 失败：x={x}");
        }
    }

    #[test]
    fn negative_values_symmetric() {
        for &x in &[1.0, 10.0, 100.0] {
            let hp = value_transform(x);
            let hn = value_transform(-x);
            assert!((hp + hn).abs() < 1e-6);
        }
    }
}

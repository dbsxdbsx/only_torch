//! `value_transform.rs` h(x)/h⁻¹ 测试：单调性、大值压缩、round-trip 精度、对称性

use super::super::value_transform::{value_transform, value_transform_inv};

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

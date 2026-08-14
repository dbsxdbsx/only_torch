//! always-on `symlog` / `symexp`：单调、可逆、比 tanh 敏感、大数仍在 support 内。

use super::super::value_transform::{value_transform, value_transform_inv};
use approx::{assert_abs_diff_eq, assert_relative_eq};

#[test]
fn zero_maps_to_zero() {
    assert_abs_diff_eq!(value_transform(0.0), 0.0, epsilon = 1e-7);
    assert_abs_diff_eq!(value_transform_inv(0.0), 0.0, epsilon = 1e-7);
}

#[test]
fn monotonically_increasing() {
    let xs: Vec<f32> = (-50..=50).map(|i| i as f32 * 4.0).collect();
    for w in xs.windows(2) {
        assert!(value_transform(w[1]) > value_transform(w[0]));
    }
}

#[test]
fn small_values_near_linear() {
    assert_abs_diff_eq!(value_transform(0.1), 0.1, epsilon = 0.01);
}

#[test]
fn compresses_large_values_without_wall() {
    let y200 = value_transform(200.0);
    assert!(y200 < 6.0, "symlog(200)={y200} 应远小于 200");
    assert!(y200 > 5.0, "symlog(200)={y200} 不应退化");
    assert!(
        value_transform(1e6) < 20.0,
        "1e6 应落在 [-20,20] 内，不必线性预缩放"
    );
}

#[test]
fn more_sensitive_than_tanh_on_large_gap() {
    let ds = (value_transform(100.0) - value_transform(5.0)).abs();
    let dt = 100.0f32.tanh() - 5.0f32.tanh();
    assert!(
        ds > dt,
        "symlog 对 +5 vs +100 的差应大于 tanh（{ds} vs {dt}）"
    );
}

#[test]
fn opposite_signs_stay_far_apart() {
    let gap = value_transform(80.0) - value_transform(-90.0);
    assert!(gap > 8.0, "+80 与 -90 应分在正负半轴，gap={gap}");
}

#[test]
fn round_trip_precision() {
    let test_values = [
        0.0, 1.0, -1.0, 10.0, -10.0, 100.0, -100.0, 200.0, -200.0, 0.5, -0.001, 1e6,
    ];
    for &x in &test_values {
        let y = value_transform(x);
        let x_back = value_transform_inv(y);
        assert_relative_eq!(x_back, x, epsilon = 1e-3, max_relative = 1e-3);
    }
}

#[test]
fn negative_values_symmetric() {
    for &x in &[1.0, 10.0, 100.0] {
        assert_abs_diff_eq!(
            value_transform(x) + value_transform(-x),
            0.0,
            epsilon = 1e-6
        );
    }
}

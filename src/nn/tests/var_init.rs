/*
 * Init 枚举单元测试（参数初始化策略）
 *
 * 测试 Zeros、Ones、Kaiming、Xavier 等初始化方式的正确性。
 */

use crate::nn::Init;

#[test]
fn test_init_zeros() {
    let tensor = Init::Zeros.generate(&[2, 3]);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert!(tensor.data_as_slice().iter().all(|&x| x == 0.0));
}

#[test]
fn test_init_ones() {
    let tensor = Init::Ones.generate(&[2, 3]);
    assert_eq!(tensor.shape(), &[2, 3]);
    assert!(tensor.data_as_slice().iter().all(|&x| x == 1.0));
}

#[test]
fn test_init_kaiming_linear() {
    // Linear: [in_features=100, out_features=50] → fan_in = 100
    let tensor = Init::Kaiming.generate(&[100, 50]);
    assert_eq!(tensor.shape(), &[100, 50]);
    // Kaiming: std = sqrt(2/fan_in) = sqrt(2/100) ≈ 0.1414
    let expected_std = (2.0 / 100.0_f32).sqrt();
    let data = tensor.data_as_slice();
    let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
    let actual_std = actual_std.sqrt();
    assert!(
        (actual_std - expected_std).abs() < 0.05,
        "Kaiming Linear: expected std ≈ {expected_std:.4}, got {actual_std:.4}"
    );
}

#[test]
fn test_init_kaiming_conv2d() {
    // Conv2d: [C_out=32, C_in=16, kH=3, kW=3] → fan_in = 16 * 3 * 3 = 144
    let tensor = Init::Kaiming.generate(&[32, 16, 3, 3]);
    assert_eq!(tensor.shape(), &[32, 16, 3, 3]);
    let fan_in = 16 * 3 * 3; // = 144
    let expected_std = (2.0 / fan_in as f32).sqrt(); // ≈ 0.1178
    let data = tensor.data_as_slice();
    let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
    let actual_std = actual_std.sqrt();
    assert!(
        (actual_std - expected_std).abs() < 0.02,
        "Kaiming Conv2d: expected std ≈ {expected_std:.4}, got {actual_std:.4}"
    );
}

#[test]
fn test_init_xavier_linear() {
    // Linear: [in=100, out=50] → fan_in=100, fan_out=50
    let tensor = Init::Xavier.generate(&[100, 50]);
    assert_eq!(tensor.shape(), &[100, 50]);
    // Xavier: std = sqrt(2/(fan_in + fan_out)) = sqrt(2/150) ≈ 0.1155
    let expected_std = (2.0 / 150.0_f32).sqrt();
    let data = tensor.data_as_slice();
    let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
    let actual_std = actual_std.sqrt();
    assert!(
        (actual_std - expected_std).abs() < 0.05,
        "Xavier Linear: expected std ≈ {expected_std:.4}, got {actual_std:.4}"
    );
}

#[test]
fn test_init_xavier_conv2d() {
    // Conv2d: [C_out=32, C_in=16, kH=3, kW=3]
    //   fan_in = 16 * 9 = 144, fan_out = 32 * 9 = 288
    let tensor = Init::Xavier.generate(&[32, 16, 3, 3]);
    assert_eq!(tensor.shape(), &[32, 16, 3, 3]);
    let fan_in = 16 * 3 * 3; // = 144
    let fan_out = 32 * 3 * 3; // = 288
    let expected_std = (2.0 / (fan_in + fan_out) as f32).sqrt(); // ≈ 0.0681
    let data = tensor.data_as_slice();
    let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
    let actual_std = actual_std.sqrt();
    assert!(
        (actual_std - expected_std).abs() < 0.01,
        "Xavier Conv2d: expected std ≈ {expected_std:.4}, got {actual_std:.4}"
    );
}

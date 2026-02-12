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
fn test_init_kaiming() {
    let tensor = Init::Kaiming.generate(&[100, 50]);
    assert_eq!(tensor.shape(), &[100, 50]);
    // Kaiming: std = sqrt(2/fan_in) = sqrt(2/100) ≈ 0.1414
    let expected_std = (2.0 / 100.0_f32).sqrt();
    let data = tensor.data_as_slice();
    let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
    let actual_std = actual_std.sqrt();
    assert!((actual_std - expected_std).abs() < 0.05);
}

#[test]
fn test_init_xavier() {
    let tensor = Init::Xavier.generate(&[100, 50]);
    assert_eq!(tensor.shape(), &[100, 50]);
    // Xavier: std = sqrt(2/(fan_in + fan_out)) = sqrt(2/150) ≈ 0.1155
    let expected_std = (2.0 / 150.0_f32).sqrt();
    let data = tensor.data_as_slice();
    let actual_std = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
    let actual_std = actual_std.sqrt();
    assert!((actual_std - expected_std).abs() < 0.05);
}

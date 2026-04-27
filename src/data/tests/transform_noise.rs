//! GaussianNoise 变换测试

use crate::data::transforms::{GaussianNoise, Transform};
use crate::tensor::Tensor;

#[test]
fn test_noise_preserves_shape() {
    let noise = GaussianNoise::new(0.0, 0.1);
    let input = Tensor::new(&[1.0; 12], &[3, 2, 2]);
    let output = noise.apply(&input);

    assert_eq!(output.shape(), &[3, 2, 2]);
}

#[test]
fn test_noise_changes_values() {
    let noise = GaussianNoise::new(0.0, 1.0);
    let input = Tensor::new(&[0.0; 100], &[100]);

    let output = noise.apply(&input);
    let flat = output.flatten_view();

    // 不应全为 0
    let non_zero_count = flat.iter().filter(|&&v| v.abs() > 1e-6).count();
    assert!(
        non_zero_count > 50,
        "大部分值应被噪声改变，非零数: {non_zero_count}"
    );
}

#[test]
fn test_noise_statistics() {
    // 验证噪声的统计特性（均值和标准差）
    let target_mean = 0.0;
    let target_std = 0.5;
    let noise = GaussianNoise::new(target_mean, target_std);
    let input = Tensor::zeros(&[10000]);

    let output = noise.apply(&input);
    let flat = output.flatten_view();

    // 计算实际均值
    let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;
    // 计算实际标准差
    let var: f32 = flat.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
    let std = var.sqrt();

    // 宽松的统计检验
    assert!(mean.abs() < 0.05, "噪声均值 {mean:.4} 应接近 0");
    assert!(
        (std - target_std as f32).abs() < 0.1,
        "噪声标准差 {std:.4} 应接近 {target_std}"
    );
}

#[test]
fn test_noise_with_nonzero_mean() {
    let noise = GaussianNoise::new(5.0, 0.1);
    let input = Tensor::zeros(&[1000]);

    let output = noise.apply(&input);
    let flat = output.flatten_view();
    let mean: f32 = flat.iter().sum::<f32>() / flat.len() as f32;

    // 均值应接近 5.0
    assert!((mean - 5.0).abs() < 0.1, "噪声均值 {mean:.4} 应接近 5.0");
}

#[test]
fn test_noise_different_each_call() {
    let noise = GaussianNoise::new(0.0, 1.0);
    let input = Tensor::new(&[0.0; 10], &[10]);

    let output1 = noise.apply(&input);
    let output2 = noise.apply(&input);

    // 两次调用结果应不同
    let flat1 = output1.flatten_view();
    let flat2 = output2.flatten_view();
    let diff: f32 = flat1
        .iter()
        .zip(flat2.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum();

    assert!(diff > 0.1, "两次噪声调用结果应不同");
}

#[test]
#[should_panic(expected = "std 必须大于 0")]
fn test_noise_zero_std() {
    GaussianNoise::new(0.0, 0.0);
}

#[test]
fn test_noise_1d() {
    // 1D 输入也应工作（通用变换）
    let noise = GaussianNoise::new(0.0, 0.1);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = noise.apply(&input);

    assert_eq!(output.shape(), &[3]);
}

//! Normalize 变换测试

use crate::data::transforms::{Normalize, Transform};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_normalize_single_channel() {
    // [C=1, H=2, W=2]，mean=0.5, std=0.5
    let input = Tensor::new(&[0.0, 0.5, 1.0, 0.25], &[1, 2, 2]);
    let norm = Normalize::new(vec![0.5], vec![0.5]);
    let output = norm.apply(&input);

    assert_eq!(output.shape(), &[1, 2, 2]);
    // (0.0-0.5)/0.5 = -1.0
    assert_abs_diff_eq!(output[[0, 0, 0]], -1.0, epsilon = 1e-6);
    // (0.5-0.5)/0.5 = 0.0
    assert_abs_diff_eq!(output[[0, 0, 1]], 0.0, epsilon = 1e-6);
    // (1.0-0.5)/0.5 = 1.0
    assert_abs_diff_eq!(output[[0, 1, 0]], 1.0, epsilon = 1e-6);
    // (0.25-0.5)/0.5 = -0.5
    assert_abs_diff_eq!(output[[0, 1, 1]], -0.5, epsilon = 1e-6);
}

#[test]
fn test_normalize_three_channels() {
    // [C=3, H=1, W=2]
    // R: [0.4, 0.6], G: [0.3, 0.7], B: [0.2, 0.8]
    let input = Tensor::new(&[0.4, 0.6, 0.3, 0.7, 0.2, 0.8], &[3, 1, 2]);
    let norm = Normalize::new(vec![0.5, 0.5, 0.5], vec![0.1, 0.2, 0.4]);
    let output = norm.apply(&input);

    assert_eq!(output.shape(), &[3, 1, 2]);
    // R: (0.4-0.5)/0.1 = -1.0, (0.6-0.5)/0.1 = 1.0
    assert_abs_diff_eq!(output[[0, 0, 0]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 1.0, epsilon = 1e-6);
    // G: (0.3-0.5)/0.2 = -1.0, (0.7-0.5)/0.2 = 1.0
    assert_abs_diff_eq!(output[[1, 0, 0]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 1]], 1.0, epsilon = 1e-6);
    // B: (0.2-0.5)/0.4 = -0.75, (0.8-0.5)/0.4 = 0.75
    assert_abs_diff_eq!(output[[2, 0, 0]], -0.75, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[2, 0, 1]], 0.75, epsilon = 1e-6);
}

#[test]
fn test_normalize_2d() {
    // [C=2, W=3] — 2D 也应工作
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let norm = Normalize::new(vec![2.0, 5.0], vec![1.0, 1.0]);
    let output = norm.apply(&input);

    assert_eq!(output.shape(), &[2, 3]);
    assert_abs_diff_eq!(output[[0, 0]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], -1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 2]], 1.0, epsilon = 1e-6);
}

#[test]
#[should_panic(expected = "mean 和 std 长度必须一致")]
fn test_normalize_mismatched_lengths() {
    Normalize::new(vec![0.5, 0.5], vec![0.5]);
}

#[test]
#[should_panic(expected = "std[0] 必须大于 0")]
fn test_normalize_zero_std() {
    Normalize::new(vec![0.5], vec![0.0]);
}

#[test]
#[should_panic(expected = "通道数 2 与 mean 长度 1 不匹配")]
fn test_normalize_channel_mismatch() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let norm = Normalize::new(vec![0.5], vec![0.5]);
    norm.apply(&input);
}

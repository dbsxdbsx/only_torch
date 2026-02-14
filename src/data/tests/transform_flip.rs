//! RandomHorizontalFlip 变换测试

use crate::data::transforms::random_flip::flip_horizontal;
use crate::data::transforms::{RandomHorizontalFlip, Transform};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_flip_2d() {
    // [H=2, W=3]
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let output = flip_horizontal(&input);

    assert_eq!(output.shape(), &[2, 3]);
    // 第一行: [1,2,3] -> [3,2,1]
    assert_abs_diff_eq!(output[[0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 2]], 1.0, epsilon = 1e-6);
    // 第二行: [4,5,6] -> [6,5,4]
    assert_abs_diff_eq!(output[[1, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 2]], 4.0, epsilon = 1e-6);
}

#[test]
fn test_flip_3d() {
    // [C=1, H=2, W=3]
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]);
    let output = flip_horizontal(&input);

    assert_eq!(output.shape(), &[1, 2, 3]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 2]], 4.0, epsilon = 1e-6);
}

#[test]
fn test_flip_double_is_identity() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let output = flip_horizontal(&flip_horizontal(&input));

    for i in 0..2 {
        for j in 0..3 {
            assert_abs_diff_eq!(output[[i, j]], input[[i, j]], epsilon = 1e-6);
        }
    }
}

#[test]
fn test_random_flip_always() {
    // p=1.0 → 总是翻转
    let flip = RandomHorizontalFlip::new(1.0);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3]);

    for _ in 0..10 {
        let output = flip.apply(&input);
        assert_abs_diff_eq!(output[[0, 0, 0]], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[0, 0, 2]], 1.0, epsilon = 1e-6);
    }
}

#[test]
fn test_random_flip_never() {
    // p=0.0 → 永不翻转
    let flip = RandomHorizontalFlip::new(0.0);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[1, 1, 3]);

    for _ in 0..10 {
        let output = flip.apply(&input);
        assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(output[[0, 0, 2]], 3.0, epsilon = 1e-6);
    }
}

#[test]
fn test_random_flip_multi_channel() {
    // [C=2, H=1, W=3]
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 1, 3]);
    let output = flip_horizontal(&input);

    // 通道 0: [1,2,3] -> [3,2,1]
    assert_abs_diff_eq!(output[[0, 0, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 2]], 1.0, epsilon = 1e-6);
    // 通道 1: [4,5,6] -> [6,5,4]
    assert_abs_diff_eq!(output[[1, 0, 0]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0, 2]], 4.0, epsilon = 1e-6);
}

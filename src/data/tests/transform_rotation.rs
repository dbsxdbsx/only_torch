//! RandomRotation 变换测试

use crate::data::transforms::random_rotation::rotate;
use crate::data::transforms::{RandomRotation, Transform};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_rotate_zero_degrees() {
    // 0° 旋转 → 原样
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let output = rotate(&input, 0.0, 0.0);

    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 0, 1]], 2.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 1, 0]], 3.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[0, 1, 1]], 4.0, epsilon = 1e-5);
}

#[test]
fn test_rotate_180_degrees() {
    // 180° 旋转（近似，因为 bilinear interpolation 在整数像素位置上是精确的）
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[1, 3, 3]);
    let output = rotate(&input, 180.0, 0.0);

    assert_eq!(output.shape(), &[1, 3, 3]);
    // 180° 旋转应将 [1,2,3;4,5,6;7,8,9] → [9,8,7;6,5,4;3,2,1]
    assert_abs_diff_eq!(output[[0, 0, 0]], 9.0, epsilon = 1e-4);
    assert_abs_diff_eq!(output[[0, 1, 1]], 5.0, epsilon = 1e-4); // 中心不变
    assert_abs_diff_eq!(output[[0, 2, 2]], 1.0, epsilon = 1e-4);
}

#[test]
fn test_rotate_preserves_shape() {
    let input = Tensor::new(&[1.0; 12], &[1, 3, 4]);
    let output = rotate(&input, 30.0, 0.0);

    assert_eq!(output.shape(), &[1, 3, 4]);
}

#[test]
fn test_rotate_2d() {
    // [H=3, W=3] — 2D 输入
    let input = Tensor::new(&[1.0; 9], &[3, 3]);
    let output = rotate(&input, 45.0, 0.0);

    assert_eq!(output.shape(), &[3, 3]);
}

#[test]
fn test_rotate_fill_value() {
    // 旋转后边角应用 fill_value
    let input = Tensor::new(&[1.0; 25], &[1, 5, 5]);
    let output = rotate(&input, 45.0, -1.0);

    assert_eq!(output.shape(), &[1, 5, 5]);
    // 角落像素映射到输入外部时使用 fill_value
    // 某些角落应该是 -1.0
    let flat = output.flatten_view();
    let has_fill = flat.iter().any(|&v| (v - (-1.0)).abs() < 1e-6);
    assert!(has_fill, "旋转后应有像素使用 fill_value");
}

#[test]
fn test_random_rotation_zero_range() {
    // degrees=0 → 原样
    let rot = RandomRotation::new(0.0);
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);

    for _ in 0..10 {
        let output = rot.apply(&input);
        assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-5);
        assert_abs_diff_eq!(output[[0, 1, 1]], 4.0, epsilon = 1e-5);
    }
}

#[test]
fn test_rotate_multi_channel() {
    // [C=2, H=3, W=3]
    let mut data = vec![0.0f32; 18];
    // 通道 0 全 1，通道 1 全 2
    for i in 0..9 {
        data[i] = 1.0;
    }
    for i in 9..18 {
        data[i] = 2.0;
    }
    let input = Tensor::new(&data, &[2, 3, 3]);
    let output = rotate(&input, 0.0, 0.0);

    // 0° 旋转，值应保持
    assert_abs_diff_eq!(output[[0, 1, 1]], 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(output[[1, 1, 1]], 2.0, epsilon = 1e-5);
}

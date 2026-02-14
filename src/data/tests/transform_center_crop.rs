//! CenterCrop 变换测试

use crate::data::transforms::{CenterCrop, Transform};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_center_crop_3d() {
    // [C=1, H=4, W=4] → 中心裁切到 [1, 2, 2]
    #[rustfmt::skip]
    let input = Tensor::new(&[
        0.0,  1.0,  2.0,  3.0,
        4.0,  5.0,  6.0,  7.0,
        8.0,  9.0,  10.0, 11.0,
        12.0, 13.0, 14.0, 15.0,
    ], &[1, 4, 4]);

    let crop = CenterCrop::new(2, 2);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[1, 2, 2]);
    // 中心 2x2: 行 1-2, 列 1-2
    assert_abs_diff_eq!(output[[0, 0, 0]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 6.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 0]], 9.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 1]], 10.0, epsilon = 1e-6);
}

#[test]
fn test_center_crop_2d() {
    // [H=4, W=6] → 中心裁切到 [2, 4]
    #[rustfmt::skip]
    let input = Tensor::new(&[
        0.0,  1.0,  2.0,  3.0,  4.0,  5.0,
        6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
        12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
        18.0, 19.0, 20.0, 21.0, 22.0, 23.0,
    ], &[4, 6]);

    let crop = CenterCrop::new(2, 4);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[2, 4]);
    // 中心 2x4: 行 1-2, 列 1-4
    assert_abs_diff_eq!(output[[0, 0]], 7.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 3]], 10.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 0]], 13.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 3]], 16.0, epsilon = 1e-6);
}

#[test]
fn test_center_crop_same_size() {
    // 输入与目标尺寸相同
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let crop = CenterCrop::new(2, 2);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[2, 2]);
    assert_abs_diff_eq!(output[[0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[1, 1]], 4.0, epsilon = 1e-6);
}

#[test]
fn test_center_crop_multi_channel() {
    // [C=2, H=3, W=3] → [2, 1, 1]
    #[rustfmt::skip]
    let input = Tensor::new(&[
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,  // 通道 0
        10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,  // 通道 1
    ], &[2, 3, 3]);

    let crop = CenterCrop::new(1, 1);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[2, 1, 1]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 5.0, epsilon = 1e-6);  // 通道 0 中心
    assert_abs_diff_eq!(output[[1, 0, 0]], 14.0, epsilon = 1e-6); // 通道 1 中心
}

#[test]
#[should_panic(expected = "输入尺寸")]
fn test_center_crop_too_large() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let crop = CenterCrop::new(3, 3);
    crop.apply(&input);
}

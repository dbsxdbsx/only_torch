//! RandomCrop 变换测试

use crate::data::transforms::{RandomCrop, Transform};
use crate::tensor::Tensor;
use approx::assert_abs_diff_eq;

#[test]
fn test_random_crop_exact_size() {
    // 输入与目标尺寸相同 → 原样返回
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let crop = RandomCrop::new(2, 2);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[1, 2, 2]);
    assert_abs_diff_eq!(output[[0, 0, 0]], 1.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 1]], 2.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 0]], 3.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 1, 1]], 4.0, epsilon = 1e-6);
}

#[test]
fn test_random_crop_larger_input() {
    // [C=1, H=4, W=4] → 裁切到 [1, 2, 2]
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let input = Tensor::new(&data, &[1, 4, 4]);
    let crop = RandomCrop::new(2, 2);

    for _ in 0..20 {
        let output = crop.apply(&input);
        assert_eq!(output.shape(), &[1, 2, 2]);
        // 所有值应在 [0, 15] 范围内
        let flat = output.flatten_view();
        for &v in flat.iter() {
            assert!((0.0..=15.0).contains(&v));
        }
    }
}

#[test]
fn test_random_crop_with_padding() {
    // [C=1, H=2, W=2]，padding=1 → 变成 [1, 4, 4] 再裁切到 [1, 3, 3]
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let crop = RandomCrop::new(3, 3).padding(1);

    for _ in 0..20 {
        let output = crop.apply(&input);
        assert_eq!(output.shape(), &[1, 3, 3]);
    }
}

#[test]
fn test_random_crop_2d() {
    // [H=4, W=4] → 裁切到 [2, 3]
    let data: Vec<f32> = (0..16).map(|x| x as f32).collect();
    let input = Tensor::new(&data, &[4, 4]);
    let crop = RandomCrop::new(2, 3);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[2, 3]);
}

#[test]
fn test_random_crop_with_fill_value() {
    // [C=1, H=1, W=1]，padding=1，fill=99 → [1, 3, 3]
    let input = Tensor::new(&[5.0], &[1, 1, 1]);
    let crop = RandomCrop::new(3, 3).padding(1).fill_value(99.0);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[1, 3, 3]);
    // 中心应该是 5.0，边缘应该是 99.0
    assert_abs_diff_eq!(output[[0, 1, 1]], 5.0, epsilon = 1e-6);
    assert_abs_diff_eq!(output[[0, 0, 0]], 99.0, epsilon = 1e-6);
}

#[test]
#[should_panic(expected = "填充后尺寸")]
fn test_random_crop_too_large() {
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
    let crop = RandomCrop::new(5, 5);
    crop.apply(&input);
}

//! RandomResizedCrop 变换测试

use crate::data::transforms::{RandomResizedCrop, Transform};
use crate::tensor::Tensor;

#[test]
fn output_shape_3d() {
    let crop = RandomResizedCrop::new(8, 8);
    let input = Tensor::new(&vec![1.0; 3 * 16 * 16], &[3, 16, 16]);
    let output = crop.apply(&input);
    assert_eq!(output.shape(), &[3, 8, 8]);
}

#[test]
fn output_shape_2d() {
    let crop = RandomResizedCrop::new(5, 5);
    let input = Tensor::new(&vec![1.0; 10 * 10], &[10, 10]);
    let output = crop.apply(&input);
    assert_eq!(output.shape(), &[5, 5]);
}

#[test]
fn output_shape_upscale() {
    // 裁切后放大：输入 8x8，输出 16x16
    let crop = RandomResizedCrop::new(16, 16).scale(0.5, 1.0);
    let input = Tensor::new(&vec![0.5; 3 * 8 * 8], &[3, 8, 8]);
    let output = crop.apply(&input);
    assert_eq!(output.shape(), &[3, 16, 16]);
}

#[test]
fn identity_crop_preserves_values() {
    // scale=(1.0, 1.0) + ratio=(1.0, 1.0) + same size → 应接近恒等变换
    let crop = RandomResizedCrop::new(4, 4).scale(1.0, 1.0).ratio(1.0, 1.0);
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let input = Tensor::new(&data, &[3, 4, 4]);
    let output = crop.apply(&input);

    assert_eq!(output.shape(), &[3, 4, 4]);

    // 全图裁切 + 相同尺寸 resize，结果应与输入非常接近
    let in_flat = input.flatten_view();
    let out_flat = output.flatten_view();
    for i in 0..48 {
        assert!(
            (in_flat[i] - out_flat[i]).abs() < 0.5,
            "pixel {i}: expected ~{}, got {}",
            in_flat[i],
            out_flat[i]
        );
    }
}

#[test]
fn uniform_input_stays_uniform() {
    // 均匀值图像经过任意裁切+resize 后仍应是均匀值
    let crop = RandomResizedCrop::new(10, 10);
    let val = 0.42;
    let input = Tensor::new(&vec![val; 3 * 20 * 20], &[3, 20, 20]);

    for _ in 0..10 {
        let output = crop.apply(&input);
        let flat = output.flatten_view();
        for (i, &v) in flat.iter().enumerate() {
            assert!(
                (v - val).abs() < 1e-5,
                "pixel {i}: expected {val}, got {v}"
            );
        }
    }
}

#[test]
fn values_in_range() {
    // 输入值在 [0, 1] → 双线性插值后输出也应在 [0, 1]
    let crop = RandomResizedCrop::new(8, 8);
    let data: Vec<f32> = (0..3 * 16 * 16)
        .map(|i| (i as f32 / (3.0 * 16.0 * 16.0)))
        .collect();
    let input = Tensor::new(&data, &[3, 16, 16]);

    for _ in 0..20 {
        let output = crop.apply(&input);
        let flat = output.flatten_view();
        for &v in flat.iter() {
            assert!(v >= -1e-5 && v <= 1.0 + 1e-5, "value {v} out of [0, 1]");
        }
    }
}

#[test]
fn custom_scale_and_ratio() {
    let crop = RandomResizedCrop::new(6, 6)
        .scale(0.5, 0.8)
        .ratio(0.8, 1.2);
    let input = Tensor::new(&vec![1.0; 3 * 12 * 12], &[3, 12, 12]);
    let output = crop.apply(&input);
    assert_eq!(output.shape(), &[3, 6, 6]);
}

#[test]
fn deterministic_with_full_scale() {
    // scale=(1.0, 1.0) 强制裁切全图，多次调用结果应一致
    let crop = RandomResizedCrop::new(4, 4).scale(1.0, 1.0).ratio(1.0, 1.0);
    let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = Tensor::new(&data, &[4, 4]);

    let out1 = crop.apply(&input);
    let out2 = crop.apply(&input);

    let f1 = out1.flatten_view();
    let f2 = out2.flatten_view();
    for i in 0..16 {
        assert!(
            (f1[i] - f2[i]).abs() < 1e-5,
            "full-scale crop should be deterministic"
        );
    }
}

//! RandomAffine 变换测试

use crate::data::transforms::{RandomAffine, Transform};
use crate::tensor::Tensor;

#[test]
fn identity_preserves_values() {
    // degrees=0, no translate/scale/shear → 恒等变换
    let t = RandomAffine::new(0.0);
    let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
    let input = Tensor::new(&data, &[3, 4, 4]);
    let output = t.apply(&input);

    let in_flat = input.flatten_view();
    let out_flat = output.flatten_view();
    for i in 0..48 {
        assert!(
            (in_flat[i] - out_flat[i]).abs() < 1e-5,
            "identity: pixel {i} mismatch: {} vs {}",
            in_flat[i],
            out_flat[i]
        );
    }
}

#[test]
fn output_shape_3d() {
    let t = RandomAffine::new(15.0).translate(0.1, 0.1).scale(0.8, 1.2);
    let input = Tensor::new(&vec![1.0; 3 * 16 * 16], &[3, 16, 16]);
    let output = t.apply(&input);
    assert_eq!(output.shape(), &[3, 16, 16]);
}

#[test]
fn output_shape_2d() {
    let t = RandomAffine::new(10.0);
    let input = Tensor::new(&vec![1.0; 8 * 8], &[8, 8]);
    let output = t.apply(&input);
    assert_eq!(output.shape(), &[8, 8]);
}

#[test]
fn uniform_input_stays_uniform() {
    // 均匀值图像经过任何仿射变换后，非填充区域仍为原值
    let val = 0.7;
    let t = RandomAffine::new(30.0)
        .translate(0.2, 0.2)
        .scale(0.5, 1.5)
        .shear(15.0)
        .fill_value(val); // fill = val，所以整个输出都应是 val
    let input = Tensor::new(&vec![val; 3 * 12 * 12], &[3, 12, 12]);

    for _ in 0..10 {
        let output = t.apply(&input);
        let flat = output.flatten_view();
        for (i, &v) in flat.iter().enumerate() {
            assert!((v - val).abs() < 1e-4, "pixel {i}: expected {val}, got {v}");
        }
    }
}

#[test]
fn rotation_only_center_pixel_stable() {
    // 旋转围绕中心进行，中心像素应不变
    let t = RandomAffine::new(45.0);
    let size = 9; // 奇数，中心在 (4, 4)
    let data: Vec<f32> = (0..(3 * size * size)).map(|i| i as f32).collect();
    let input = Tensor::new(&data, &[3, size, size]);

    for _ in 0..20 {
        let output = t.apply(&input);
        let in_flat = input.flatten_view();
        let out_flat = output.flatten_view();
        let center = 4 * size + 4; // center pixel in each channel
        for ch in 0..3 {
            let idx = ch * size * size + center;
            assert!(
                (in_flat[idx] - out_flat[idx]).abs() < 0.5,
                "center pixel ch={ch}: {} vs {}",
                in_flat[idx],
                out_flat[idx]
            );
        }
    }
}

#[test]
fn translate_only_shifts_content() {
    // 纯平移（无旋转/缩放/剪切），fill_value=-1
    // 验证输出中至少有一些原始值和一些填充值
    let t = RandomAffine::new(0.0).translate(0.3, 0.3).fill_value(-1.0);
    let input = Tensor::new(&vec![1.0; 3 * 10 * 10], &[3, 10, 10]);

    let mut found_fill = false;
    for _ in 0..50 {
        let output = t.apply(&input);
        let flat = output.flatten_view();
        if flat.iter().any(|&v| (v + 1.0).abs() < 1e-6) {
            found_fill = true;
            // 也应该有原始值
            assert!(
                flat.iter().any(|&v| (v - 1.0).abs() < 1e-4),
                "should have original pixels too"
            );
            break;
        }
    }
    assert!(
        found_fill,
        "translate should introduce fill pixels in at least one trial"
    );
}

#[test]
fn scale_down_introduces_fill() {
    // 缩小（scale < 1），边缘应出现填充值
    let t = RandomAffine::new(0.0).scale(0.3, 0.3).fill_value(-1.0);
    let input = Tensor::new(&vec![1.0; 10 * 10], &[10, 10]);
    let output = t.apply(&input);
    let flat = output.flatten_view();

    // 角落像素应为填充值
    assert_eq!(flat[0], -1.0, "top-left corner should be fill");
    assert_eq!(flat[9], -1.0, "top-right corner should be fill");
    // 中心像素应保留
    assert!(
        (flat[5 * 10 + 5] - 1.0).abs() < 1e-4,
        "center should be original"
    );
}

#[test]
fn shear_only_preserves_shape() {
    let t = RandomAffine::new(0.0).shear(20.0);
    let input = Tensor::new(&vec![1.0; 3 * 8 * 8], &[3, 8, 8]);
    let output = t.apply(&input);
    assert_eq!(output.shape(), &[3, 8, 8]);
}

#[test]
fn full_params_no_panic() {
    let t = RandomAffine::new(30.0)
        .translate(0.2, 0.2)
        .scale(0.5, 1.5)
        .shear(15.0)
        .fill_value(0.0);
    let input = Tensor::new(&vec![0.5; 3 * 16 * 16], &[3, 16, 16]);

    // 多次运行不应 panic
    for _ in 0..20 {
        let output = t.apply(&input);
        assert_eq!(output.shape(), &[3, 16, 16]);
    }
}

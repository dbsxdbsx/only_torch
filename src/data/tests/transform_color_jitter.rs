//! ColorJitter 变换测试

use crate::data::transforms::{ColorJitter, Transform};
use crate::tensor::Tensor;

#[test]
fn test_color_jitter_no_change() {
    // 所有参数为 0 → 近似原样（浮点运算可能有微小误差）
    let jitter = ColorJitter::new(0.0, 0.0, 0.0);
    let input = Tensor::new(&[0.5; 12], &[3, 2, 2]);

    for _ in 0..10 {
        let output = jitter.apply(&input);
        assert_eq!(output.shape(), &[3, 2, 2]);
        let flat = output.flatten_view();
        for &v in flat.iter() {
            assert!((v - 0.5).abs() < 1e-5, "参数为 0 时不应改变值，得到 {v}");
        }
    }
}

#[test]
fn test_color_jitter_brightness_only() {
    let jitter = ColorJitter::new(0.5, 0.0, 0.0);
    let input = Tensor::new(&[1.0; 12], &[3, 2, 2]);

    // 运行多次验证亮度在 [0.5, 1.5] 范围内
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    for _ in 0..100 {
        let output = jitter.apply(&input);
        let flat = output.flatten_view();
        for &v in flat.iter() {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
    }
    // 所有像素是 1.0 * factor，factor 在 [0.5, 1.5]
    assert!(min_val >= 0.49, "最小亮度 {min_val} 应 >= 0.5");
    assert!(max_val <= 1.51, "最大亮度 {max_val} 应 <= 1.5");
    // 应有变化
    assert!(max_val - min_val > 0.1, "亮度应有变化");
}

#[test]
fn test_color_jitter_preserves_shape() {
    let jitter = ColorJitter::new(0.2, 0.2, 0.2);
    let input = Tensor::new(&[0.5; 48], &[3, 4, 4]);
    let output = jitter.apply(&input);

    assert_eq!(output.shape(), &[3, 4, 4]);
}

#[test]
fn test_color_jitter_saturation_grayscale_input() {
    // 单通道输入不应用饱和度调整
    // 但 ColorJitter 要求 3D，所以用 [C=1, H=2, W=2]
    // 饱和度仅在 C==3 时生效
    let jitter = ColorJitter::new(0.0, 0.0, 0.5);
    let input = Tensor::new(&[0.5; 4], &[1, 2, 2]);

    // C != 3 时饱和度不生效
    for _ in 0..10 {
        let output = jitter.apply(&input);
        let flat = output.flatten_view();
        for &v in flat.iter() {
            assert!((v - 0.5).abs() < 1e-5, "单通道饱和度不应改变值");
        }
    }
}

#[test]
fn test_color_jitter_output_clamped() {
    // brightness=0.5 时 factor 最大 1.5，输入 1.0 会溢出到 1.5
    // 修复后应裁剪到 [0, 1]
    let jitter = ColorJitter::new(0.5, 0.5, 0.5);
    let input = Tensor::new(&[1.0; 12], &[3, 2, 2]);

    for _ in 0..100 {
        let output = jitter.apply(&input);
        let flat = output.flatten_view();
        for &v in flat.iter() {
            assert!(v >= 0.0 && v <= 1.0, "输出应在 [0, 1] 范围内，得到 {v}");
        }
    }

    // 同样验证接近 0 的输入不会变成负数
    let dark = Tensor::new(&[0.05; 12], &[3, 2, 2]);
    for _ in 0..100 {
        let output = jitter.apply(&dark);
        let flat = output.flatten_view();
        for &v in flat.iter() {
            assert!(
                v >= 0.0 && v <= 1.0,
                "暗像素输出应在 [0, 1] 范围内，得到 {v}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "brightness 必须 >= 0")]
fn test_color_jitter_negative_brightness() {
    ColorJitter::new(-0.1, 0.0, 0.0);
}

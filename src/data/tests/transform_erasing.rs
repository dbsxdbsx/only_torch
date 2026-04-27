//! RandomErasing 变换测试

use crate::data::transforms::{RandomErasing, Transform};
use crate::tensor::Tensor;

#[test]
fn test_erasing_never() {
    // p=0.0 → 永不擦除
    let erasing = RandomErasing::new(0.0);
    let input = Tensor::new(&[1.0; 27], &[3, 3, 3]);

    for _ in 0..20 {
        let output = erasing.apply(&input);
        let flat = output.flatten_view();
        assert!(flat.iter().all(|&v| v == 1.0), "p=0 不应擦除任何像素");
    }
}

#[test]
fn test_erasing_always() {
    // p=1.0 → 总是尝试擦除
    let erasing = RandomErasing::new(1.0).value(-1.0);
    let input = Tensor::new(&[1.0; 48], &[3, 4, 4]);

    let mut ever_erased = false;
    for _ in 0..50 {
        let output = erasing.apply(&input);
        let flat = output.flatten_view();
        if flat.iter().any(|&v| v == -1.0) {
            ever_erased = true;
            break;
        }
    }
    assert!(ever_erased, "p=1.0 应至少有一次成功擦除");
}

#[test]
fn test_erasing_preserves_shape() {
    let erasing = RandomErasing::new(0.5);
    let input = Tensor::new(&[1.0; 48], &[3, 4, 4]);
    let output = erasing.apply(&input);

    assert_eq!(output.shape(), &[3, 4, 4]);
}

#[test]
fn test_erasing_value() {
    // 使用自定义擦除值
    let erasing = RandomErasing::new(1.0).value(99.0);
    let input = Tensor::new(&[0.0; 75], &[3, 5, 5]);

    let mut found_value = false;
    for _ in 0..50 {
        let output = erasing.apply(&input);
        let flat = output.flatten_view();
        if flat.iter().any(|&v| v == 99.0) {
            found_value = true;
            break;
        }
    }
    assert!(found_value, "擦除值应为指定的 99.0");
}

#[test]
fn test_erasing_rectangular() {
    // 验证擦除区域是矩形（所有通道相同位置）
    let erasing = RandomErasing::new(1.0).value(-1.0).scale(0.1, 0.3);
    let input = Tensor::new(&[1.0; 75], &[3, 5, 5]);

    for _ in 0..50 {
        let output = erasing.apply(&input);
        let flat = output.flatten_view();

        // 检查通道 0 中被擦除的位置
        let erased_positions: Vec<usize> = (0..25).filter(|&i| flat[i] == -1.0).collect();

        if erased_positions.is_empty() {
            continue;
        }

        // 同样位置在其他通道也应被擦除
        for &pos in &erased_positions {
            assert_eq!(flat[25 + pos], -1.0, "通道 1 相同位置应被擦除");
            assert_eq!(flat[50 + pos], -1.0, "通道 2 相同位置应被擦除");
        }
        return; // 成功验证一次即可
    }
}

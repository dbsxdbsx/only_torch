use crate::tensor::Tensor;
use crate::vision::mask::{argmax_to_class_map, foreground_from_multiclass, mask_to_ascii_lines};

#[test]
fn test_argmax_to_class_map_picks_largest_channel() {
    // 构造 [1, 3, 2, 2]: channel 0/1/2 各占一种最大值布局
    // 每个像素位置上选最大概率的 class
    let probs = Tensor::new(
        &[
            // class 0 (background) 概率
            0.9, 0.1, 0.2, 0.1, // class 1 概率
            0.05, 0.7, 0.6, 0.2, // class 2 概率
            0.05, 0.2, 0.2, 0.7,
        ],
        &[1, 3, 2, 2],
    );

    let class_map = argmax_to_class_map(&probs);

    assert_eq!(class_map.shape(), &[1, 2, 2]);
    assert_eq!(class_map[[0, 0, 0]], 0.0);
    assert_eq!(class_map[[0, 0, 1]], 1.0);
    assert_eq!(class_map[[0, 1, 0]], 1.0);
    assert_eq!(class_map[[0, 1, 1]], 2.0);
}

#[test]
fn test_foreground_from_multiclass_skips_background_channel() {
    // [1, 3, 1, 2]: 两个像素，每个像素的前景概率应等于 max(channel 1, channel 2)
    let probs = Tensor::new(
        &[
            // background
            0.9, 0.1, // foreground class 1
            0.2, 0.6, // foreground class 2
            0.05, 0.7,
        ],
        &[1, 3, 1, 2],
    );

    let fg = foreground_from_multiclass(&probs);

    assert_eq!(fg.shape(), &[1, 1, 1, 2]);
    // 像素 0：max(0.2, 0.05) = 0.2
    // 像素 1：max(0.6, 0.7) = 0.7
    assert!((fg[[0, 0, 0, 0]] - 0.2).abs() < 1e-6);
    assert!((fg[[0, 0, 0, 1]] - 0.7).abs() < 1e-6);
}

#[test]
fn test_mask_to_ascii_lines_uses_threshold() {
    // [1, 1, 2, 3]: 单 sample / 单 channel / 2 行 3 列
    let mask = Tensor::new(&[0.6, 0.3, 0.9, 0.1, 0.5, 0.7], &[1, 1, 2, 3]);

    let lines = mask_to_ascii_lines(&mask, 0, 0, 0.5, '#', '.');

    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0], "#.#");
    assert_eq!(lines[1], ".##");
}

#[test]
fn test_mask_to_ascii_lines_supports_multi_channel() {
    // [1, 2, 1, 2]: 两个 channel 用同一个 sample，验证 channel 参数
    let mask = Tensor::new(&[0.9, 0.1, 0.1, 0.9], &[1, 2, 1, 2]);

    let ch0 = mask_to_ascii_lines(&mask, 0, 0, 0.5, '#', '.');
    let ch1 = mask_to_ascii_lines(&mask, 0, 1, 0.5, '#', '.');

    assert_eq!(ch0[0], "#.");
    assert_eq!(ch1[0], ".#");
}

#[test]
#[should_panic(expected = "argmax_to_class_map")]
fn test_argmax_to_class_map_panics_on_wrong_shape() {
    let bad = Tensor::new(&[1.0, 2.0], &[2]);
    let _ = argmax_to_class_map(&bad);
}

#[test]
#[should_panic(expected = "foreground_from_multiclass")]
fn test_foreground_from_multiclass_panics_on_single_channel() {
    let single = Tensor::new(&[0.5], &[1, 1, 1, 1]);
    let _ = foreground_from_multiclass(&single);
}

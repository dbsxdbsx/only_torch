//! 像素级 mask 处理工具。
//!
//! 这里专门处理"模型输出 mask"与"标签 mask"之间的常见转换：argmax 解码、
//! 多类 → 前景二值、mask 文本化等。`vision` 下其它模块各有侧重：
//!
//! - [`crate::vision::draw`]：在原图上绘制 bbox / 矩形 / 圆等几何
//! - [`crate::vision::viz`]：展示画布工具（调色板、像素放大、alpha 混合）
//! - [`crate::vision::geom`]：图像几何变换（resize / crop）
//! - [`crate::vision::mask`]：本模块，**只动 mask 张量本身**
//!
//! 所有函数都接收 / 返回 `Tensor`，不依赖 `image::DynamicImage`。

use crate::tensor::Tensor;

/// 把 `[N, C, H, W]` 的逐像素分类张量按 channel 维 argmax 还原成
/// `[N, H, W]` 类别索引（仍以 `f32` 存储，值是类别索引）。
///
/// 常用于多类语义分割可视化，或对照 `metrics::segmentation` 内部 argmax 行为
/// 的调试。空间维 `H * W * N == 0` 时返回对应形状的空 `Tensor`。
pub fn argmax_to_class_map(probs: &Tensor) -> Tensor {
    let shape = probs.shape();
    assert!(
        shape.len() == 4 && shape[1] > 0,
        "argmax_to_class_map: 期望 shape=[N, C, H, W] 且 C>0，实际 {shape:?}"
    );
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let mut data = Vec::with_capacity(n * h * w);
    for sample in 0..n {
        for y in 0..h {
            for x in 0..w {
                let mut best_class = 0usize;
                let mut best_value = probs[[sample, 0, y, x]];
                for class_idx in 1..c {
                    let value = probs[[sample, class_idx, y, x]];
                    if value > best_value {
                        best_value = value;
                        best_class = class_idx;
                    }
                }
                data.push(best_class as f32);
            }
        }
    }
    Tensor::new(&data, &[n, h, w])
}

/// 把多类语义分割张量 `[N, C, H, W]` 缩成"前景概率" `[N, 1, H, W]`。
///
/// 约定 channel 0 是 background；输出 `[n, 0, y, x] = max(c=1..C, probs[n, c, y, x])`，
/// 即"任一前景类的最大概率"。常用于 background 是单独类的多类分割任务，
/// 把它降维成一个二值 IoU / Dice 友好的 mask。
///
/// 至少需要 2 个 channel（背景 + 1 个前景类），否则 panic。
pub fn foreground_from_multiclass(probs: &Tensor) -> Tensor {
    let shape = probs.shape();
    assert!(
        shape.len() == 4 && shape[1] >= 2,
        "foreground_from_multiclass: 期望 shape=[N, C, H, W] 且 C>=2，实际 {shape:?}"
    );
    let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let mut data = Vec::with_capacity(n * h * w);
    for sample in 0..n {
        for y in 0..h {
            for x in 0..w {
                let mut value = 0.0f32;
                for class_idx in 1..c {
                    value = value.max(probs[[sample, class_idx, y, x]]);
                }
                data.push(value);
            }
        }
    }
    Tensor::new(&data, &[n, 1, h, w])
}

/// 把 `[N, C, H, W]` 张量的某个 channel 转成可在终端打印的字符行。
///
/// 用 `>= threshold` 二值化，正样本输出 `fg` 字符，其它输出 `bg` 字符；
/// 返回 `H` 个长度为 `W` 的字符串，调用方直接 `for line in lines { println!("{line}") }` 即可。
///
/// 典型用法：用 `'#'` / `'.'` 分别表示前景 / 背景，做 toy 模型的 mask 调试。
pub fn mask_to_ascii_lines(
    mask: &Tensor,
    sample: usize,
    channel: usize,
    threshold: f32,
    fg: char,
    bg: char,
) -> Vec<String> {
    let shape = mask.shape();
    assert!(
        shape.len() == 4,
        "mask_to_ascii_lines: 期望 shape=[N, C, H, W]，实际 {shape:?}"
    );
    assert!(
        sample < shape[0] && channel < shape[1],
        "mask_to_ascii_lines: sample={sample} / channel={channel} 越界，shape={shape:?}"
    );
    let (h, w) = (shape[2], shape[3]);
    (0..h)
        .map(|y| {
            (0..w)
                .map(|x| {
                    if mask[[sample, channel, y, x]] >= threshold {
                        fg
                    } else {
                        bg
                    }
                })
                .collect()
        })
        .collect()
}

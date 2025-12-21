//! 数据变换函数
//!
//! 提供常用的数据预处理操作，如归一化、one-hot 编码等。

use crate::tensor::Tensor;

/// 将 0-255 像素值归一化到 0-1
///
/// # 参数
/// - `tensor`: 输入 Tensor，值范围 [0, 255]
///
/// # 返回
/// 归一化后的 Tensor，值范围 [0, 1]
pub fn normalize_pixels(tensor: &Tensor) -> Tensor {
    tensor / 255.0
}

/// 将类别索引转换为 one-hot 编码
///
/// # 参数
/// - `labels`: 类别索引 Tensor，形状 [N] 或 [N, 1]，值为 0..num_classes
/// - `num_classes`: 类别总数
///
/// # 返回
/// one-hot 编码 Tensor，形状 [N, num_classes]
///
/// # 示例
/// ```ignore
/// let labels = Tensor::new(&[0.0, 2.0, 1.0], &[3]);
/// let one_hot = one_hot(&labels, 3);
/// // 结果: [[1,0,0], [0,0,1], [0,1,0]]
/// ```
pub fn one_hot(labels: &Tensor, num_classes: usize) -> Tensor {
    let flat = labels.flatten();
    let n = flat.size();

    let mut data = vec![0.0; n * num_classes];
    for i in 0..n {
        let class_idx = flat[[i]] as usize;
        if class_idx < num_classes {
            data[i * num_classes + class_idx] = 1.0;
        }
    }

    Tensor::new(&data, &[n, num_classes])
}

/// 展平图像
///
/// # 参数
/// - `tensor`: 输入 Tensor
///   - 形状 [N, C, H, W] → 输出 [N, C*H*W]
///   - 形状 [C, H, W] → 输出 [C*H*W]
///   - 形状 [N, H, W] → 输出 [N, H*W]
///
/// # 返回
/// 展平后的 Tensor
pub fn flatten_images(tensor: &Tensor) -> Tensor {
    let shape = tensor.shape();
    match shape.len() {
        4 => {
            // [N, C, H, W] -> [N, C*H*W]
            let n = shape[0];
            let flat_size = shape[1] * shape[2] * shape[3];
            tensor.reshape(&[n, flat_size])
        }
        3 => {
            // [C, H, W] -> [C*H*W] 或 [N, H, W] -> [N, H*W]
            // 假设第一维是 batch，展平后两维
            let n = shape[0];
            let flat_size = shape[1] * shape[2];
            tensor.reshape(&[n, flat_size])
        }
        _ => tensor.flatten(),
    }
}

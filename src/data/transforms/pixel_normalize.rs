//! 像素值归一化（迁移自原 transforms.rs）

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

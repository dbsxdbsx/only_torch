//! 图像展平（迁移自原 transforms.rs）

use crate::tensor::Tensor;

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

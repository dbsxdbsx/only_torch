//! 按通道均值/标准差归一化
//!
//! 类似 PyTorch `transforms.Normalize(mean, std)`，
//! 对每个通道执行 `(x - mean) / std`。

use super::Transform;
use crate::tensor::Tensor;

/// 按通道归一化
///
/// 对输入张量 [C, H, W] 的每个通道 c 执行：
/// `output[c] = (input[c] - mean[c]) / std[c]`
///
/// # 示例
///
/// ```ignore
/// // ImageNet 常用均值/标准差
/// let norm = Normalize::new(
///     vec![0.485, 0.456, 0.406],
///     vec![0.229, 0.224, 0.225],
/// );
/// let output = norm.apply(&image_tensor);
/// ```
pub struct Normalize {
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl Normalize {
    /// 创建通道归一化变换
    ///
    /// # 参数
    /// - `mean`: 每个通道的均值
    /// - `std`: 每个通道的标准差（必须 > 0）
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(
            mean.len(),
            std.len(),
            "Normalize: mean 和 std 长度必须一致，得到 {} vs {}",
            mean.len(),
            std.len()
        );
        for (i, &s) in std.iter().enumerate() {
            assert!(
                s > 0.0,
                "Normalize: std[{i}] 必须大于 0，得到 {s}"
            );
        }
        Self { mean, std }
    }
}

impl Transform for Normalize {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        assert!(
            shape.len() >= 2,
            "Normalize: 输入至少为 2D（[C, ...] 或 [C, H, W]），得到 {}D",
            shape.len()
        );

        let c = shape[0];
        assert_eq!(
            c,
            self.mean.len(),
            "Normalize: 通道数 {c} 与 mean 长度 {} 不匹配",
            self.mean.len()
        );

        let spatial_size: usize = shape[1..].iter().product();
        let flat = tensor.flatten_view();
        let mut data = vec![0.0f32; tensor.size()];

        for ch in 0..c {
            let offset = ch * spatial_size;
            let inv_std = 1.0 / self.std[ch];
            let mean = self.mean[ch];
            for i in 0..spatial_size {
                data[offset + i] = (flat[offset + i] - mean) * inv_std;
            }
        }

        Tensor::new(&data, shape)
    }
}

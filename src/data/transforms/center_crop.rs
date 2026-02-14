//! 中心裁切
//!
//! 从图像中心裁切到目标尺寸。

use super::Transform;
use crate::tensor::Tensor;

/// 中心裁切
///
/// 对输入张量 [C, H, W] 或 [H, W] 从中心裁切到 `(target_h, target_w)`。
///
/// # 示例
///
/// ```ignore
/// let crop = CenterCrop::new(224, 224);
/// let output = crop.apply(&image_tensor);
/// ```
pub struct CenterCrop {
    target_h: usize,
    target_w: usize,
}

impl CenterCrop {
    /// 创建中心裁切变换
    ///
    /// # 参数
    /// - `target_h`: 目标高度
    /// - `target_w`: 目标宽度
    pub fn new(target_h: usize, target_w: usize) -> Self {
        Self { target_h, target_w }
    }
}

impl Transform for CenterCrop {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        let ndim = shape.len();
        assert!(
            ndim == 2 || ndim == 3,
            "CenterCrop: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
        );

        let (h, w) = if ndim == 2 {
            (shape[0], shape[1])
        } else {
            (shape[1], shape[2])
        };

        assert!(
            h >= self.target_h && w >= self.target_w,
            "CenterCrop: 输入尺寸 ({h}x{w}) 必须 >= 目标尺寸 ({}x{})",
            self.target_h,
            self.target_w
        );

        let top = (h - self.target_h) / 2;
        let left = (w - self.target_w) / 2;

        if ndim == 2 {
            tensor
                .narrow(0, top, self.target_h)
                .narrow(1, left, self.target_w)
        } else {
            tensor
                .narrow(1, top, self.target_h)
                .narrow(2, left, self.target_w)
        }
    }
}

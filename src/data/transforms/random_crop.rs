//! 随机裁切
//!
//! 可选填充后随机裁切到目标尺寸。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 随机裁切
///
/// 对输入张量 [C, H, W] 或 [H, W]，可选填充后随机裁切到 `(target_h, target_w)`。
///
/// # 示例
///
/// ```ignore
/// // 先填充 4 像素再裁切到 32x32
/// let crop = RandomCrop::new(32, 32).padding(4);
/// let output = crop.apply(&image_tensor);
/// ```
pub struct RandomCrop {
    target_h: usize,
    target_w: usize,
    padding: usize,
    fill_value: f32,
}

impl RandomCrop {
    /// 创建随机裁切变换
    ///
    /// # 参数
    /// - `target_h`: 目标高度
    /// - `target_w`: 目标宽度
    pub fn new(target_h: usize, target_w: usize) -> Self {
        Self {
            target_h,
            target_w,
            padding: 0,
            fill_value: 0.0,
        }
    }

    /// 设置填充量（四周等量填充）
    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// 设置填充值
    pub fn fill_value(mut self, value: f32) -> Self {
        self.fill_value = value;
        self
    }
}

impl Transform for RandomCrop {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        let ndim = shape.len();
        assert!(
            ndim == 2 || ndim == 3,
            "RandomCrop: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {ndim}D"
        );

        // 填充（如果需要）
        let padded = if self.padding > 0 {
            let p = self.padding;
            if ndim == 2 {
                tensor.pad(&[(p, p), (p, p)], self.fill_value)
            } else {
                // [C, H, W] — 通道维不填充
                tensor.pad(&[(0, 0), (p, p), (p, p)], self.fill_value)
            }
        } else {
            tensor.clone()
        };

        let padded_shape = padded.shape().to_vec();
        let (h, w) = if ndim == 2 {
            (padded_shape[0], padded_shape[1])
        } else {
            (padded_shape[1], padded_shape[2])
        };

        assert!(
            h >= self.target_h && w >= self.target_w,
            "RandomCrop: 填充后尺寸 ({h}x{w}) 必须 >= 目标尺寸 ({}x{})",
            self.target_h,
            self.target_w
        );

        // 随机起始位置
        let mut rng = rand::thread_rng();
        let top = if h == self.target_h {
            0
        } else {
            rng.gen_range(0..=h - self.target_h)
        };
        let left = if w == self.target_w {
            0
        } else {
            rng.gen_range(0..=w - self.target_w)
        };

        // 裁切（利用 narrow）
        if ndim == 2 {
            padded
                .narrow(0, top, self.target_h)
                .narrow(1, left, self.target_w)
        } else {
            padded
                .narrow(1, top, self.target_h)
                .narrow(2, left, self.target_w)
        }
    }
}

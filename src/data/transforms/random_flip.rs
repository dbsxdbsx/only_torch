//! 随机水平翻转
//!
//! 以概率 p 对图像沿宽度轴进行水平翻转。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 随机水平翻转
///
/// 对输入张量 [C, H, W] 或 [H, W] 以概率 `p` 进行水平翻转。
///
/// # 示例
///
/// ```ignore
/// let flip = RandomHorizontalFlip::new(0.5);
/// let output = flip.apply(&image_tensor);
/// ```
pub struct RandomHorizontalFlip {
    p: f64,
}

impl RandomHorizontalFlip {
    /// 创建随机水平翻转变换
    ///
    /// # 参数
    /// - `p`: 翻转概率，范围 [0, 1]
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "RandomHorizontalFlip: 概率 p 必须在 [0, 1] 范围内，得到 {p}"
        );
        Self { p }
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut rng = rand::thread_rng();
        if rng.gen_range(0.0_f64..1.0) >= self.p {
            return tensor.clone();
        }

        flip_horizontal(tensor)
    }
}

/// 水平翻转（确定性版本，供内部和测试使用）
pub(crate) fn flip_horizontal(tensor: &Tensor) -> Tensor {
    let shape = tensor.shape();
    let flat = tensor.flatten_view();

    match shape.len() {
        2 => {
            // [H, W]
            let (h, w) = (shape[0], shape[1]);
            let mut data = vec![0.0f32; h * w];
            for row in 0..h {
                for col in 0..w {
                    data[row * w + (w - 1 - col)] = flat[row * w + col];
                }
            }
            Tensor::new(&data, shape)
        }
        3 => {
            // [C, H, W]
            let (c, h, w) = (shape[0], shape[1], shape[2]);
            let mut data = vec![0.0f32; c * h * w];
            for ch in 0..c {
                for row in 0..h {
                    for col in 0..w {
                        let src = ch * h * w + row * w + col;
                        let dst = ch * h * w + row * w + (w - 1 - col);
                        data[dst] = flat[src];
                    }
                }
            }
            Tensor::new(&data, shape)
        }
        _ => panic!(
            "RandomHorizontalFlip: 输入应为 2D [H, W] 或 3D [C, H, W]，得到 {}D",
            shape.len()
        ),
    }
}

//! 随机区域擦除
//!
//! 以概率 p 在图像上随机选择一个矩形区域，用指定值填充。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 随机区域擦除
///
/// 以概率 `p` 在输入张量 [C, H, W] 上随机选取一个矩形区域并擦除。
///
/// # 参数说明
/// - `scale`: 擦除区域面积与原图面积之比的范围
/// - `ratio`: 擦除区域宽高比的范围
/// - `value`: 擦除值
///
/// # 示例
///
/// ```ignore
/// let erasing = RandomErasing::new(0.5);
/// let output = erasing.apply(&image_tensor);
/// ```
pub struct RandomErasing {
    p: f64,
    scale: (f64, f64),
    ratio: (f64, f64),
    value: f32,
}

impl RandomErasing {
    /// 创建随机擦除变换
    ///
    /// # 参数
    /// - `p`: 擦除概率，范围 [0, 1]
    pub fn new(p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "RandomErasing: 概率 p 必须在 [0, 1] 范围内，得到 {p}"
        );
        Self {
            p,
            scale: (0.02, 0.33),
            ratio: (0.3, 3.3),
            value: 0.0,
        }
    }

    /// 设置擦除区域面积比范围
    pub fn scale(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max && max <= 1.0);
        self.scale = (min, max);
        self
    }

    /// 设置擦除区域宽高比范围
    pub fn ratio(mut self, min: f64, max: f64) -> Self {
        assert!(0.0 < min && min <= max);
        self.ratio = (min, max);
        self
    }

    /// 设置擦除值
    pub fn value(mut self, value: f32) -> Self {
        self.value = value;
        self
    }
}

impl Transform for RandomErasing {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut rng = rand::thread_rng();
        if rng.gen_range(0.0_f64..1.0) >= self.p {
            return tensor.clone();
        }

        let shape = tensor.shape();
        assert!(
            shape.len() == 3,
            "RandomErasing: 输入应为 3D [C, H, W]，得到 {}D",
            shape.len()
        );

        let (c, h, w) = (shape[0], shape[1], shape[2]);
        let area = (h * w) as f64;

        // 尝试多次找到合适的擦除区域
        let max_attempts = 10;
        for _ in 0..max_attempts {
            let target_area = rng.gen_range(self.scale.0..=self.scale.1) * area;
            let log_ratio_min = self.ratio.0.ln();
            let log_ratio_max = self.ratio.1.ln();
            let aspect_ratio = (rng.gen_range(log_ratio_min..=log_ratio_max)).exp();

            let erase_w = (target_area * aspect_ratio).sqrt() as usize;
            let erase_h = (target_area / aspect_ratio).sqrt() as usize;

            if erase_w == 0 || erase_h == 0 || erase_w >= w || erase_h >= h {
                continue;
            }

            let top = rng.gen_range(0..h - erase_h);
            let left = rng.gen_range(0..w - erase_w);

            // 擦除
            let flat = tensor.flatten_view();
            let mut data: Vec<f32> = flat.to_vec();
            for ch in 0..c {
                let ch_offset = ch * h * w;
                for row in top..top + erase_h {
                    for col in left..left + erase_w {
                        data[ch_offset + row * w + col] = self.value;
                    }
                }
            }

            return Tensor::new(&data, shape);
        }

        // 所有尝试失败则不擦除
        tensor.clone()
    }
}

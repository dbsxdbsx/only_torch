//! 高斯噪声
//!
//! 向输入添加高斯噪声。通用变换，适用于图像/表格/序列。

use super::Transform;
use crate::tensor::Tensor;
use rand::Rng;

/// 高斯噪声变换
///
/// 向输入的每个元素添加 `N(mean, std²)` 高斯噪声。
///
/// # 示例
///
/// ```ignore
/// let noise = GaussianNoise::new(0.0, 0.1);
/// let output = noise.apply(&input);
/// ```
pub struct GaussianNoise {
    mean: f64,
    std: f64,
}

impl GaussianNoise {
    /// 创建高斯噪声变换
    ///
    /// # 参数
    /// - `mean`: 噪声均值
    /// - `std`: 噪声标准差（必须 > 0）
    pub fn new(mean: f64, std: f64) -> Self {
        assert!(std > 0.0, "GaussianNoise: std 必须大于 0，得到 {std}");
        Self { mean, std }
    }
}

impl Transform for GaussianNoise {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut rng = rand::thread_rng();
        let flat = tensor.flatten_view();
        let mut data = Vec::with_capacity(tensor.size());

        for &val in flat.iter() {
            let noise = box_muller(&mut rng, self.mean, self.std);
            data.push(val + noise as f32);
        }

        Tensor::new(&data, tensor.shape())
    }
}

/// Box-Muller 变换生成标准正态分布随机数
///
/// 避免额外引入 rand_distr 依赖
fn box_muller(rng: &mut impl Rng, mean: f64, std: f64) -> f64 {
    let u1: f64 = rng.gen_range(0.0_f64..1.0);
    let u2: f64 = rng.gen_range(0.0_f64..1.0);
    // 避免 ln(0)
    let u1 = u1.max(f64::EPSILON);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    mean + std * z
}

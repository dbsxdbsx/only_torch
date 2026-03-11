/*
 * @Author       : 老董
 * @Date         : 2025-01-21
 * @LastModified : 2025-02-15
 * @Description  : 数据变换模块
 *
 * 提供 Transform trait 和常用数据变换实现。
 *
 * # 主要组件
 *
 * ## 基础设施
 * - [`Transform`] trait: 数据变换抽象接口
 * - [`Compose`]: 链式组合多个变换
 * - [`RandomApply`]: 按概率应用变换
 *
 * ## 图像变换
 * - [`Normalize`]: 按通道均值/标准差归一化
 * - [`RandomHorizontalFlip`]: 随机水平翻转
 * - [`RandomCrop`]: 随机裁切（含可选填充）
 * - [`CenterCrop`]: 中心裁切
 * - [`RandomRotation`]: 随机旋转（双线性插值）
 * - [`ColorJitter`]: 随机色彩扰动（亮度/对比度/饱和度）
 * - [`RandomErasing`]: 随机区域擦除
 * - [`GaussianNoise`]: 高斯噪声（通用，适用于图像/表格/序列）
 *
 * ## 工具函数
 * - [`normalize_pixels`]: 像素值 [0,255] → [0,1]
 * - [`one_hot`]: 类别索引 → one-hot 编码
 * - [`flatten_images`]: 多维图像展平
 */

mod center_crop;
mod color_jitter;
mod flatten;
mod random_affine;
mod gaussian_noise;
mod normalize;
mod one_hot;
mod pixel_normalize;
mod random_crop;
mod random_erasing;
mod random_resized_crop;
pub(crate) mod random_flip;
pub(crate) mod random_rotation;

use crate::tensor::Tensor;

// 迁移的工具函数（向后兼容）
pub use flatten::flatten_images;
pub use one_hot::one_hot;
pub use pixel_normalize::normalize_pixels;

// 图像变换
pub use center_crop::CenterCrop;
pub use color_jitter::ColorJitter;
pub use random_affine::RandomAffine;
pub use gaussian_noise::GaussianNoise;
pub use normalize::Normalize;
pub use random_crop::RandomCrop;
pub use random_erasing::RandomErasing;
pub use random_flip::RandomHorizontalFlip;
pub use random_resized_crop::RandomResizedCrop;
pub use random_rotation::RandomRotation;

// ═══════════════════════════════════════════════════════════════
// Transform trait
// ═══════════════════════════════════════════════════════════════

/// 数据变换 trait
///
/// 对单个样本（非 batch 维度）应用变换。
///
/// # 示例
///
/// ```ignore
/// use only_torch::data::transforms::{Transform, Normalize, Compose};
///
/// let transform = Compose::new(vec![
///     Box::new(Normalize::new(vec![0.485], vec![0.229])),
/// ]);
/// let output = transform.apply(&input);
/// ```
pub trait Transform: Send + Sync {
    /// 对单个样本应用变换
    ///
    /// # 参数
    /// - `tensor`: 输入张量（通常为 [C, H, W] 或更低维度）
    ///
    /// # 返回
    /// 变换后的张量
    fn apply(&self, tensor: &Tensor) -> Tensor;
}

// ═══════════════════════════════════════════════════════════════
// Compose - 链式组合
// ═══════════════════════════════════════════════════════════════

/// 链式组合多个变换
///
/// 按顺序依次应用所有变换。
///
/// # 示例
///
/// ```ignore
/// let transform = Compose::new(vec![
///     Box::new(Normalize::new(vec![0.5], vec![0.5])),
///     Box::new(RandomHorizontalFlip::new(0.5)),
/// ]);
/// ```
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    /// 创建链式变换
    ///
    /// # 参数
    /// - `transforms`: 变换列表，按顺序依次应用
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut result = tensor.clone();
        for t in &self.transforms {
            result = t.apply(&result);
        }
        result
    }
}

// ═══════════════════════════════════════════════════════════════
// RandomApply - 按概率应用
// ═══════════════════════════════════════════════════════════════

/// 按概率应用变换
///
/// 以概率 `p` 应用内部变换，否则直接返回原始输入。
///
/// # 示例
///
/// ```ignore
/// let transform = RandomApply::new(
///     Box::new(GaussianNoise::new(0.0, 0.1)),
///     0.3,  // 30% 概率应用
/// );
/// ```
pub struct RandomApply {
    transform: Box<dyn Transform>,
    p: f64,
}

impl RandomApply {
    /// 创建按概率应用的变换
    ///
    /// # 参数
    /// - `transform`: 内部变换
    /// - `p`: 应用概率，范围 [0, 1]
    pub fn new(transform: Box<dyn Transform>, p: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&p),
            "RandomApply: 概率 p 必须在 [0, 1] 范围内，得到 {p}"
        );
        Self { transform, p }
    }
}

impl Transform for RandomApply {
    fn apply(&self, tensor: &Tensor) -> Tensor {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        if rng.gen_range(0.0_f64..1.0) < self.p {
            self.transform.apply(tensor)
        } else {
            tensor.clone()
        }
    }
}

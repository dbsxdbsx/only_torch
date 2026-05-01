/*
 * @Author       : 老董
 * @Date         : 2025-01-21
 * @LastModified : 2026-05-01
 * @Description  : 数据变换模块
 *
 * 提供两套正交的变换契约，外加常用工具函数。
 *
 * # 两套 trait
 *
 * - [`Transform`]：image-only，输入 `&Tensor` → `Tensor`，适用于分类 /
 *   自监督预训练等单图流水线。
 * - [`SampleTransform<S>`]：sample-level（owned），输入一个 `Sample` →
 *   `Sample`，适用于 detection / segmentation 等需要 image 与 label 几何
 *   同步的场景。Sample 类型见 [`crate::data::sample`]。
 *
 * # SampleTransform 实现矩阵
 *
 * | Transform | Classification | Detection | Segmentation |
 * |---|---|---|---|
 * | `RandomHorizontalFlip` | ✓ | ✓ 同步 bbox | ✓ 同步 mask |
 * | `CenterCrop`           | ✓ | ✓ 同步 bbox + 可选 `label_filter` | ✓ 同步 mask |
 * | `RandomCrop`           | ✓ | ✓ 同步 bbox + 可选 `label_filter` | ✓ 同步 mask |
 * | `RandomRotation`       | ✓ | ✓ 同步 bbox（4 角 → AABB）         | ✓ 同步 mask（nearest） |
 * | `RandomAffine`         | ✓ | ✓ 同步 bbox（4 角 → AABB）         | ✓ 同步 mask（nearest） |
 * | `RandomErasing`        | ✓ | ✓ 只擦图，labels 保留（A 方案）    | ✓ 只擦图，mask 保留（A 方案） |
 *
 * mask 之所以使用 **nearest** 插值，是因为它是离散类别——bilinear 会混出
 * 非法中间值。`RandomErasing` 的 A 方案对齐 torchvision v2：擦图的意图是
 * 训练抗遮挡能力，同步擦 label 会反向抵消。
 *
 * # 基础组合器
 *
 * - [`Compose`]: 链式组合多个 `Transform`
 * - [`RandomApply`]: 按概率应用 `Transform`
 *
 * # 纯图像变换（只实现 `Transform`）
 *
 * - [`Normalize`]: 按通道均值/标准差归一化
 * - [`ColorJitter`]: 随机色彩扰动（亮度/对比度/饱和度）
 * - [`RandomResizedCrop`]: 随机区域裁切 + resize
 * - [`GaussianNoise`]: 高斯噪声（通用，适用于图像/表格/序列）
 *
 * # 工具函数
 *
 * - [`normalize_pixels`]: 像素值 [0,255] → [0,1]
 * - [`one_hot`]: 类别索引 → one-hot 编码
 * - [`flatten_images`]: 多维图像展平
 */

pub(crate) mod affine_kernel;
mod center_crop;
mod color_jitter;
mod crop_helpers;
mod flatten;
mod gaussian_noise;
mod normalize;
mod one_hot;
mod pixel_normalize;
mod random_affine;
mod random_crop;
pub(crate) mod random_erasing;
pub(crate) mod random_flip;
mod random_resized_crop;
pub(crate) mod random_rotation;
mod sample_transform;

use crate::tensor::Tensor;

// 迁移的工具函数（向后兼容）
pub use flatten::flatten_images;
pub use one_hot::one_hot;
pub use pixel_normalize::normalize_pixels;

// 图像变换
pub use center_crop::CenterCrop;
pub use color_jitter::ColorJitter;
pub use gaussian_noise::GaussianNoise;
pub use normalize::Normalize;
pub use random_affine::RandomAffine;
pub use random_crop::RandomCrop;
pub use random_erasing::RandomErasing;
pub use random_flip::RandomHorizontalFlip;
pub use random_resized_crop::RandomResizedCrop;
pub use random_rotation::RandomRotation;
pub use sample_transform::SampleTransform;

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

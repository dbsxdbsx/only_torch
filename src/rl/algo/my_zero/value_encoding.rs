//! MyZero value/reward 的**分类(distributional)编码层**
//!
//! 不直接回归标量 value/reward，而是把它们表示成**一组固定标量锚点上的概率分布**
//! （distributional RL，同 C51）。这组锚点称 **support**（概率分布的支撑集），位于
//! `value_transform` `h(x)` 之后的变换域，如 `{-20,…,0,…,20}` 共 41 个原子。
//!
//! 层级：夹在「标量 target（`n_step` 算出）」与「网络的 categorical value/reward 头」之间，
//! 是一个纯**编解码工具**（与 `value_transform` 配套）：
//! - 编码 [`scalar_to_two_hot`]：`x → h(x) → clamp → 相邻两原子线性插值`（训练 soft target）。
//! - 解码 [`two_hot_to_scalar`]：`Σ pᵢ·atomᵢ（变换域期望）→ h⁻¹`（搜索/推理还原标量）。
//! - [`SupportConfig`]：配置 support（半宽 → 原子数）；value 与 reward 共用同一 support。
//!
//! 对齐 canonical MuZero（Schrittwieser et al., 2020，附录 F）：相比标量 MSE，交叉熵梯度更
//! 稳定、对大 value 的噪声不放大。

use super::value_transform::{value_transform, value_transform_inv};

/// Categorical value/reward 的 support 配置（原子个数 = `2 * half_size + 1`）。
///
/// "support" = 概率分布的**支撑集**（固定标量锚点集），本类型只配置它的形状。
#[derive(Debug, Clone, Copy)]
pub struct SupportConfig {
    half_size: usize,
}

impl SupportConfig {
    /// 用半宽 `half_size` 构造（覆盖变换域 `[-half_size, half_size]`）
    ///
    /// # Panics
    /// `half_size == 0` 时 panic（退化为单原子无意义）。
    pub const fn new(half_size: usize) -> Self {
        assert!(half_size > 0, "SupportConfig: half_size 必须 > 0");
        Self { half_size }
    }

    /// support 原子个数 = `2 * half_size + 1`
    pub const fn size(&self) -> usize {
        2 * self.half_size + 1
    }

    /// 半宽
    pub const fn half_size(&self) -> usize {
        self.half_size
    }

    /// 第 `i` 个原子在变换域的标量值 = `i - half_size`
    pub fn atom(&self, i: usize) -> f32 {
        i as f32 - self.half_size as f32
    }
}

/// 标量 → two-hot 概率分布
///
/// 流程：`x → h(x) → clamp 到 [-half, half] → 相邻两原子线性插值`。
/// 返回长度 = `cfg.size()` 的概率向量（至多两个相邻原子非零，和恒为 1）。
pub fn scalar_to_two_hot(x: f32, cfg: &SupportConfig) -> Vec<f32> {
    let size = cfg.size();
    let mut out = vec![0.0f32; size];

    let half = cfg.half_size as f32;
    let y = value_transform(x).clamp(-half, half);
    let pos = y + half;

    let lower = pos.floor() as usize;
    let upper = (lower + 1).min(size - 1);
    let upper_weight = pos - lower as f32;

    out[lower] = 1.0 - upper_weight;
    out[upper] += upper_weight;
    out
}

/// two-hot 概率分布 → 标量
///
/// 流程：`Σ pᵢ·atomᵢ（变换域期望）→ h⁻¹ → 标量`。
/// `probs` 应为已归一化的概率（如 `softmax(logits)`），长度须等于 `cfg.size()`。
/// 解码与编码方式无关（two-hot / HL-Gauss 共用：都取期望再 h⁻¹）。
pub fn two_hot_to_scalar(probs: &[f32], cfg: &SupportConfig) -> f32 {
    let mut y = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        y += p * cfg.atom(i);
    }
    value_transform_inv(y)
}

/// HL-Gauss 软标签的高斯宽度 σ（变换域；bin 宽 = 1，取 0.75 × bin 宽）。
///
/// Farebrother et al. 2024（*Stop Regressing*）与 Simulus（arXiv 2502.11537）实证口径：
/// σ/bin ≈ 0.75 使高斯質量跨约 ±3 个原子，标签平滑但不糊化。**非用户旋钮**。
pub const HL_GAUSS_SIGMA: f32 = 0.75;

/// 标量 → HL-Gauss 概率分布（histogram loss with Gaussian smoothing）
///
/// 流程：`x → h(x) → clamp 到 [-half, half] → 高斯 CDF 按 bin 边界差分 → 截断归一化`。
/// 每个原子 `zᵢ` 视为宽 1 的 bin（边界 `zᵢ ± 0.5`）：
/// `pᵢ ∝ Φ((zᵢ+0.5−y)/σ) − Φ((zᵢ−0.5−y)/σ)`，再对 support 全域截断归一（Σpᵢ = 1）。
///
/// 与 [`scalar_to_two_hot`] 的差异：质量摊到 ~±3 个原子（软标签更平滑，CE 梯度对
/// target 邻域误差不敏感），期望仍 ≈ y（内点处离散化偏差 < 1e-3）。
/// 解码端不变（[`two_hot_to_scalar`] 取期望 → h⁻¹）。
pub fn scalar_to_hl_gauss(x: f32, cfg: &SupportConfig) -> Vec<f32> {
    let size = cfg.size();
    let half = cfg.half_size as f32;
    let y = value_transform(x).clamp(-half, half);

    // bin 边界：b_i = atom(i) − 0.5，i ∈ 0..=size（共 size+1 条）
    let cdf_at = |border: f32| std_normal_cdf((border - y) / HL_GAUSS_SIGMA);
    let total = cdf_at(half + 0.5) - cdf_at(-half - 0.5);

    let mut out = vec![0.0f32; size];
    let mut lower_cdf = cdf_at(-half - 0.5);
    for (i, slot) in out.iter_mut().enumerate() {
        let upper_cdf = cdf_at(cfg.atom(i) + 0.5);
        *slot = (upper_cdf - lower_cdf) / total;
        lower_cdf = upper_cdf;
    }
    out
}

/// 标准正态 CDF：`Φ(t) = 0.5·(1 + erf(t/√2))`。
fn std_normal_cdf(t: f32) -> f32 {
    0.5 * (1.0 + erf(t / std::f32::consts::SQRT_2))
}

/// 误差函数近似（Abramowitz & Stegun 7.1.26，|误差| ≤ 1.5e-7，f32 足够）。
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f32 = 0.254_829_6;
    const A2: f32 = -0.284_496_74;
    const A3: f32 = 1.421_413_7;
    const A4: f32 = -1.453_152;
    const A5: f32 = 1.061_405_4;
    const P: f32 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    sign * (1.0 - poly * (-x * x).exp())
}

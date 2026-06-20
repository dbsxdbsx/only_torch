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
pub fn two_hot_to_scalar(probs: &[f32], cfg: &SupportConfig) -> f32 {
    let mut y = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        y += p * cfg.atom(i);
    }
    value_transform_inv(y)
}

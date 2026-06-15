//! MuZero categorical value/reward 表示（support + two-hot 编解码）
//!
//! 对齐 canonical MuZero（Schrittwieser et al., 2020，附录 F）：value/reward 不做
//! 标量回归，而是表示为一组**固定标量锚点（support）上的概率分布**，用交叉熵学习，
//! 取期望还原标量。相比标量 MSE，交叉熵梯度更稳定、对大 value 的噪声不放大。
//!
//! # 关键设计
//! - support 为 `2*half_size + 1` 个**整数原子** `{-half_size, …, 0, …, half_size}`，
//!   位于 `value_transform` `h(x)` 之后的**变换域**（与论文一致：在 h 域均匀切分，
//!   等价于在原始 value 域近 0 处更密、远处更疏）。
//! - 编码：`x → h(x) → clamp 到 [-half, half] → two-hot`（相邻两原子线性插值，和为 1）。
//! - 解码：`probs → Σ pᵢ·atomᵢ（变换域期望）→ h⁻¹ → 标量`。
//! - value 与 reward **共用同一 support**（与论文一致）。
//!
//! # 与 `Categorical` 分布的区别
//! `nn::distributions::Categorical` 是「离散动作上的分布」的通用积木（类别是纯索引）；
//! 本模块是把该思想用到 **value/reward 这个连续标量** 上——额外需要 support（每个类别
//! 绑定一个具体标量）+ two-hot 编解码，这是标量 ↔ 分布的编解码层，`Categorical` 不含。

use super::value_transform::{value_transform, value_transform_inv};

/// Categorical value/reward 的 support 配置
///
/// support 原子个数 = `2 * half_size + 1`，第 `i` 个原子在**变换域**的标量值
/// 为 `i as f32 - half_size`（即 `{-half_size, …, half_size}`）。
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
///
/// 用作 value/reward head 交叉熵损失的 soft target。
pub fn scalar_to_two_hot(x: f32, cfg: &SupportConfig) -> Vec<f32> {
    let size = cfg.size();
    let mut out = vec![0.0f32; size];

    let half = cfg.half_size as f32;
    // 变换并 clamp 到 support 覆盖的变换域
    let y = value_transform(x).clamp(-half, half);
    // 原子坐标：变换域值 y 映射到索引轴 [0, 2*half]
    let pos = y + half;

    let lower = pos.floor() as usize;
    // 上界裁剪到最后一个原子，处理 pos 恰为 2*half 的边界
    let upper = (lower + 1).min(size - 1);
    let upper_weight = pos - lower as f32; // 落在 upper 的权重 ∈ [0, 1)

    out[lower] = 1.0 - upper_weight;
    // 用 += 兼容 lower == upper（pos 恰在最高原子）的退化情形
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

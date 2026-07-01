//! MCTS 树内 Q 值 min-max 归一化
//!
//! MuZero 等 value 无界环境（CartPole reward 0~200）中，
//! 需要将 Q 值归一化到 [0,1] 后才代入 PUCT 公式，
//! 否则 exploration 项被 Q 值压死。
//! AlphaZero（value ∈ [-1,1]）下 min-max 近似恒等，无害。

/// 追踪搜索树中 Q 值的极值，提供归一化
#[derive(Debug, Clone)]
pub struct MinMaxStats {
    min: f32,
    max: f32,
}

impl MinMaxStats {
    pub fn new() -> Self {
        Self {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
        }
    }

    /// 用新的 Q 值更新极值
    pub fn update(&mut self, q: f32) {
        if q < self.min {
            self.min = q;
        }
        if q > self.max {
            self.max = q;
        }
    }

    /// 将 Q 值归一化到 [0, 1]
    ///
    /// 若尚无有效 range（未更新或 min==max），返回 0.5（中性值），
    /// 防止 raw Q 压死 PUCT exploration 项。
    pub fn normalize(&self, q: f32) -> f32 {
        let range = self.max - self.min;
        if range > f32::EPSILON {
            (q - self.min) / range
        } else {
            0.5
        }
    }

    /// 返回 tree-level Q 极值 `(min, max)`；尚无有效 range（未更新）时返回 `None`。
    ///
    /// completedQ 的 σ 归一化用此**全局**范围替代局部 over-children min-max：
    /// `|A|=2` 时局部 min-max 恒把两动作拉成 `{0,1}`，σ 退化为与 Q 差无关的符号开关；
    /// 改用整棵搜索树的 Q 范围后，根动作的 `norm_q` 才反映其在全局分布里的真实位置。
    pub fn range(&self) -> Option<(f32, f32)> {
        if self.min.is_finite() && self.max.is_finite() && self.max >= self.min {
            Some((self.min, self.max))
        } else {
            None
        }
    }
}

impl Default for MinMaxStats {
    fn default() -> Self {
        Self::new()
    }
}

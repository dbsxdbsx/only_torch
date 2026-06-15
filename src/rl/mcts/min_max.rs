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
    /// 若 min == max（所有 Q 相同或未更新），返回原始 Q
    pub fn normalize(&self, q: f32) -> f32 {
        let range = self.max - self.min;
        if range > f32::EPSILON {
            (q - self.min) / range
        } else {
            q
        }
    }
}

impl Default for MinMaxStats {
    fn default() -> Self {
        Self::new()
    }
}

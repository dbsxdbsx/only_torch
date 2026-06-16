//! MyZero 消融特征开关
//!
//! 每个开关对应一个 EZ 增量组件，用于 A/B 消融实验。
//! 全关 = canonical MuZero（S0 base）；逐个开启 = 消融序列 S1–S5。

/// 消融特征开关集合
///
/// 全部 `false` / `0.0` 等价于 canonical MuZero（S0 base）。
/// 消融序列中逐个开启，验证每个组件的增量贡献。
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSet {
    /// S1：自监督 consistency loss（SimSiam stop-grad）
    pub consistency: bool,
    /// S2：value prefix（LSTM 累计 reward 前缀，hidden 穿 MCTS 树）
    pub value_prefix: bool,
    /// S3：target network（EMA/hard 同步，配合 reanalyze）
    pub target_net: bool,
    /// S4：SVE 权重（0.0 = 关；> 0 = search value blend 进 n-step target）
    pub sve_weight: f32,
    /// S5：使用 Gumbel 搜索替代 PUCT（连续/混合动作必需）
    pub gumbel: bool,
    /// 轻量 Gumbel-A：用 completedQ 改进策略替代 visit-count 作为策略训练目标
    /// （Danihelka 2022 Eq.10-12；少模拟更稳）。
    pub completed_q_target: bool,
}

impl Default for FeatureSet {
    fn default() -> Self {
        Self {
            consistency: false,
            value_prefix: false,
            target_net: false,
            sve_weight: 0.0,
            gumbel: false,
            completed_q_target: false,
        }
    }
}

impl FeatureSet {
    /// 全关（= canonical MuZero，S0 base）
    pub fn base() -> Self {
        Self::default()
    }

    /// 是否启用了 SVE
    pub fn sve_enabled(&self) -> bool {
        self.sve_weight > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_all_off() {
        let f = FeatureSet::default();
        assert!(!f.consistency);
        assert!(!f.value_prefix);
        assert!(!f.target_net);
        assert!(!f.sve_enabled());
        assert!(!f.gumbel);
        assert!(!f.completed_q_target);
    }
}

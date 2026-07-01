//! MyZero 组件开关（消融）
//!
//! 每个开关对应一个增量组件，用于 A/B 消融实验。
//! 全关 = canonical MuZero（base）；逐个开启 = 消融序列。

/// 消融组件开关集合
///
/// 全部 `false` / `0.0` 等价于 canonical MuZero（base）。
/// 消融过程中逐个开启，验证每个组件的增量贡献。
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Components {
    /// 自监督 consistency loss（SimSiam stop-grad）
    pub consistency: bool,
    /// 自监督 reconstruction loss（latent → obs MSE；Scholz et al. 2021，见 examples/my_zero/README 文献对照表）
    pub reconstruction: bool,
    /// value prefix（LSTM 累计 reward 前缀，hidden 穿 MCTS 树）
    pub value_prefix: bool,
    /// 训练前对 sample 的 unroll 窗口重跑 MCTS 刷新标签（MuZero Reanalyze）
    /// reanalyze：position 级 MCTS 重搜 + buffer 写回（`Components.reanalyze`）。
    /// CartPole recipe 默认关；见 `.issue/items/my_zero_reanalyze_cartpole_regression.md`。
    pub reanalyze: bool,
    /// target network（EMA/hard 同步，配合 reanalyze）
    pub target_net: bool,
    /// SVE 权重（0.0 = 关；> 0 = search value blend 进 n-step target）
    pub sve_weight: f32,
    /// 使用 Gumbel 搜索替代 PUCT（连续/混合动作必需）
    pub gumbel: bool,
    /// 用 completedQ 改进策略替代 visit-count 作为策略训练目标
    /// （Danihelka 2022 Eq.10-12；少模拟更稳）。
    pub completed_q_target: bool,
    /// completedQ 的 `σ(q)=(c_visit+max_b N(b))·c_scale·q` 中的 `c_visit`（默认 50.0）
    pub cq_c_visit: f32,
    /// completedQ 的 `c_scale`（默认 1.0；论文棋类口径）。
    /// tree-level Q 归一化下向量环境也适用 1.0；旧局部 over-children min-max 才需 per-env 调小。
    pub cq_c_scale: f32,
    /// Sampled MuZero：展开时采 K 个候选 + PUCT 用 π̂_β（Hubert et al. 2021）
    /// K 由 [`sampled_params`](super::sampled_params) 按 N、sims 公式自动解析，非本字段配置。
    pub sampled: bool,
}

impl Default for Components {
    fn default() -> Self {
        Self {
            consistency: false,
            reconstruction: false,
            value_prefix: false,
            reanalyze: false,
            target_net: false,
            sve_weight: 0.0,
            gumbel: false,
            completed_q_target: false,
            cq_c_visit: 50.0,
            cq_c_scale: 1.0,
            sampled: false,
        }
    }
}

impl Components {
    /// 全关（= canonical MuZero，base）
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
        let c = Components::default();
        assert!(!c.consistency);
        assert!(!c.reconstruction);
        assert!(!c.value_prefix);
        assert!(!c.reanalyze);
        assert!(!c.target_net);
        assert!(!c.sve_enabled());
        assert!(!c.gumbel);
        assert!(!c.completed_q_target);
        assert!(!c.sampled);
    }

    #[test]
    fn cq_defaults_match_reference_qtransform() {
        let c = Components::default();
        assert!((c.cq_c_visit - 50.0).abs() < 1e-6);
        assert!((c.cq_c_scale - 1.0).abs() < 1e-6);
    }
}

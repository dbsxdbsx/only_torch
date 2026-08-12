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
    /// consistency 系数（仅 `consistency=true` 时生效；默认 [`loss::CONSISTENCY_LOSS_COEF`](super::loss::CONSISTENCY_LOSS_COEF)）
    pub consistency_coef: f32,
    /// 自监督 reconstruction loss（latent → obs MSE；Scholz et al. 2021，见 examples/my_zero/README 文献对照表）
    pub reconstruction: bool,
    /// reconstruction 系数（仅 `reconstruction=true` 时生效；默认 [`loss::RECONSTRUCTION_LOSS_COEF`](super::loss::RECONSTRUCTION_LOSS_COEF)）
    pub reconstruction_coef: f32,
    /// continuation 头 MSE 系数（恒生效，基础终止语义监督；默认 [`loss::CONTINUATION_LOSS_COEF`](super::loss::CONTINUATION_LOSS_COEF)）
    pub continuation_coef: f32,
    /// value/reward 训练目标编码：`false` = two-hot（canonical MuZero 附录 F），
    /// `true` = HL-Gauss 高斯软标签（Farebrother et al. 2024 / Simulus RaC；σ 见
    /// [`HL_GAUSS_SIGMA`](super::value_encoding::HL_GAUSS_SIGMA)）。解码端两者相同（期望 → h⁻¹）。
    /// **CartPole 消融负结果**（中位 9.8k→27.6k，2026-07-02，账本在案）→ 默认保持 two-hot；
    /// 图像域复测（大 value 噪声为其 native 场景）。
    pub hl_gauss: bool,
    /// obs symlog 无量纲化（DreamerV3 口径，`sign(x)·ln(1+|x|)`；模型 obs 入口单点，
    /// repr 输入 + recon 目标同源变换，buffer / env I/O 恒存 raw）。训练与推理必须
    /// 同口径（随 config 持久化）。见 [`obs_transform`](super::obs_transform)。
    pub obs_symlog: bool,
    /// value prefix（LSTM 累计 reward 前缀，hidden 穿 MCTS 树）
    pub value_prefix: bool,
    /// 训练前对 sample 的 unroll 窗口重跑 MCTS 刷新标签（MuZero Reanalyze）
    /// reanalyze：position 级 MCTS 重搜 + buffer 写回（`Components.reanalyze`）。
    /// CartPole recipe 默认关；见 `.issue/items/my_zero_reanalyze_cartpole_regression.md`。
    pub reanalyze: bool,
    /// ROSMO 式一步 target 刷新（reanalyze 复活阶梯一，arXiv:2210.05980）：
    /// 采样时现算 policy/value target（一步 look-ahead + 现算 bootstrap）+ 优势过滤
    /// 行为正则，**不写回 buffer**。与 `reanalyze`（全树重搜 + 写回）互斥。
    /// 见 [`rosmo`](super::rosmo)。
    pub rosmo: bool,
    /// ROSMO 行为正则系数 α（仅 `rosmo=true` 时生效；论文 Atari 口径 0.2）。
    pub rosmo_alpha: f32,
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
    /// Recurrent posterior（GRU 状态估计器）：`h(obs)` 升级为 `posterior(obs, prev_hidden, prev_action)`，
    /// 使 agent 能利用观测历史推断部分可观测环境的隐藏状态。完全可观测环境默认关，
    /// 退化为无记忆 `h(obs)` 直出 latent。配套 sequence replay + burn-in。
    pub recurrent_posterior: bool,
    /// Stochastic MuZero chance outcome 数量（Antonoglou et al. 2022 ICLR）。
    /// 默认 8（always-on）：确定性环境中 chance distribution 自然退化为单峰、KL→0；
    /// 随机环境中自动发现并利用随机结构。设 1 可退回纯确定性快路径（零 chance 开销）。
    pub num_chance_outcomes: usize,
}

impl Default for Components {
    fn default() -> Self {
        Self {
            consistency: false,
            consistency_coef: super::loss::CONSISTENCY_LOSS_COEF,
            reconstruction: false,
            reconstruction_coef: super::loss::RECONSTRUCTION_LOSS_COEF,
            continuation_coef: super::loss::CONTINUATION_LOSS_COEF,
            hl_gauss: false,
            obs_symlog: false,
            value_prefix: false,
            reanalyze: false,
            rosmo: false,
            rosmo_alpha: 0.2,
            target_net: false,
            sve_weight: 0.0,
            gumbel: false,
            completed_q_target: false,
            cq_c_visit: 50.0,
            cq_c_scale: 1.0,
            sampled: false,
            recurrent_posterior: false,
            num_chance_outcomes: 8,
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

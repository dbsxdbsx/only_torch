//! MCTS trait 定义：模型接口、搜索策略、预测器

use rand::RngCore;

use super::min_max::MinMaxStats;
use super::types::{
    ActionId, ActionPayload, CandidateSet, ChildStat, DecisionRecurrentOut, MctsConfig,
    RecurrentOut, RootOut,
};

/// MCTS 模型接口（root + recurrent，与 mctx 同构）
///
/// 实现者提供：
/// - `root`：从原始观测生成初始隐状态和先验
/// - `recurrent`：从父状态 + 动作推演下一步（确定性快路径 K=1）
///
/// Stochastic MuZero 扩展（K>1 时搜索循环调用）：
/// - `decision_recurrent`：decision node → afterstate + chance_prior
/// - `chance_recurrent`：afterstate + chance_outcome → next state
/// - `num_chance_outcomes`：chance outcome 数量
///
/// # State 是不透明的
///
/// `State` 为关联类型、对内核不透明，搜索树原样克隆 / 存储它。因此它可承载**任意**推演期
/// 状态，不止 latent：
/// - v0.23 MuZero：`State = Vec<f32>` learned latent（经 `Dynamics` + `DynamicsModel` 适配）
/// - Stochastic MuZero：state / afterstate 共用同一 `State` 类型，由 `NodeKind` 区分语义
///
/// # 后续 TODO
/// - 并行时条件加 `State: Send + Sync`
pub trait MctsModel {
    /// 隐状态类型（同时承载 state 和 afterstate）
    type State: Clone + 'static;

    /// 从原始观测生成根节点信息
    fn root(&self, obs: &[f32]) -> RootOut<Self::State>;

    /// 从父状态和动作推演子状态（确定性快路径，K=1 时搜索循环使用）
    fn recurrent(
        &self,
        state: &Self::State,
        action_id: ActionId,
        action: &ActionPayload,
    ) -> RecurrentOut<Self::State>;

    /// Stochastic MuZero chance outcome 数量。
    /// 返回 `1` 时搜索循环走确定性快路径（不创建 chance 节点）。
    fn num_chance_outcomes(&self) -> usize {
        1
    }

    /// Decision node → afterstate 转移（Stochastic MuZero，K>1 时调用）。
    ///
    /// 输出 afterstate latent + P(c|afterstate) + afterstate value。
    /// 默认实现 panic——仅当 `num_chance_outcomes() > 1` 时搜索循环才会调用。
    fn decision_recurrent(
        &self,
        _state: &Self::State,
        _action_id: ActionId,
        _action: &ActionPayload,
    ) -> DecisionRecurrentOut<Self::State> {
        unimplemented!(
            "decision_recurrent requires stochastic model (num_chance_outcomes > 1)"
        )
    }

    /// Afterstate + chance outcome → next state 转移（Stochastic MuZero，K>1 时调用）。
    ///
    /// 输出 next state + reward + value + candidates + terminal，同 `RecurrentOut`。
    fn chance_recurrent(
        &self,
        _afterstate: &Self::State,
        _chance_id: usize,
    ) -> RecurrentOut<Self::State> {
        unimplemented!(
            "chance_recurrent requires stochastic model (num_chance_outcomes > 1)"
        )
    }
}

/// 子节点选择规则（PUCT / MENTS / RENTS / ANT 等）。
pub trait SelectionRule {
    /// 选择要展开的子节点索引（从父节点视角计算 Q 值）
    fn select_child(
        &self,
        parent_visit: u32,
        parent_to_play: u8,
        children: &[ChildStat],
        stats: &MinMaxStats,
        cfg: &MctsConfig,
    ) -> usize;
}

/// 根节点策略（Dirichlet / Gumbel sequential halving / no-op）。
pub trait RootStrategy {
    /// 对根节点子节点注入探索噪声
    fn prepare_root(&self, children: &mut [ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore);

    /// 创建本次搜索的根调度器（搜索生命周期 hook）。
    ///
    /// 默认返回 [`PuctScheduler`]（不干预，行为与历史 PUCT 单叶循环**完全一致**）。
    /// Gumbel 等需要「分轮预算 + 逐轮淘汰」的策略覆盖此方法返回自定义 scheduler，
    /// 从而无需改 `mcts_search` 签名即可承载 sequential halving。
    fn make_root_scheduler(
        &self,
        num_root_children: usize,
        cfg: &MctsConfig,
    ) -> Box<dyn RootScheduler> {
        let _ = (num_root_children, cfg);
        Box::new(PuctScheduler)
    }
}

/// 搜索输出规则（最终推荐动作 + 学习 target）。
pub trait TargetRule {
    /// 搜索结束后推荐最终动作索引（训练时随机采样，评测时可贪心）
    fn recommend(&self, children: &[ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore) -> usize;

    /// 生成学习用策略目标（visit count → 概率分布）
    fn make_targets(&self, children: &[ChildStat], cfg: &MctsConfig) -> Vec<f32>;
}

/// 搜索策略组合：选择规则 + 根策略 + 输出规则。
///
/// 这是兼容层；后续 recipe 可以按三个小 trait 分别装配。
pub trait SearchPolicy: SelectionRule + RootStrategy + TargetRule {}

impl<T> SearchPolicy for T where T: SelectionRule + RootStrategy + TargetRule {}

/// 根部模拟预算调度器（搜索生命周期 hook）
///
/// 标准 MCTS 每次模拟都从根用 [`SearchPolicy::select_child`] 向下；但 Gumbel MuZero 的
/// sequential halving 需要按「分轮预算 + 逐轮淘汰」控制**根候选**——这无法用无状态的
/// `select_child` 表达。本 trait 给搜索循环一个**有状态**的根调度 hook：
///
/// - 默认实现 [`PuctScheduler`]：`is_active() == false` → 搜索循环零开销，走原 PUCT 路径。
/// - Gumbel 实现自己的 scheduler：按当前根统计逐轮缩小候选集。
pub trait RootScheduler {
    /// 是否启用根调度。默认 `false` → 搜索循环零开销、行为同历史 PUCT。
    fn is_active(&self) -> bool {
        false
    }

    /// 搜索循环开始前调用一次（`prepare_root` 之后、首次模拟之前）。
    ///
    /// Gumbel 在此采样 Gumbel 噪声并初始化 Sequential Halving 候选集。
    /// `root_to_play`：根节点执子方——双人零和时子节点 value 为子方视角，
    /// 打分中的 q̂ 须按 negamax 翻转回根方（单智能体恒 0，行为不变）。
    fn on_search_start(
        &mut self,
        root_children: &[ChildStat],
        network_value: f32,
        root_to_play: u8,
        cfg: &MctsConfig,
        rng: &mut dyn RngCore,
    ) {
        let _ = (root_children, network_value, root_to_play, cfg, rng);
    }

    /// 第 `sim_idx` 次模拟应**强制**从哪个根子节点起步（其下仍用 `select_child`）。
    ///
    /// `q_range` 为搜索至今的 tree-level Q 值范围（[`MinMaxStats::range`]；`None` = 尚无
    /// 有效范围），供 σ 归一化用——局部 over-children min-max 在 `|A|=2` 时退化为符号
    /// 开关（completedQ 侧同病已修，见 `SearchResult::q_range` 文档）。
    ///
    /// 返回 `None` → 该次模拟从根用 `select_child` 正常选择。仅在 `is_active()` 为真时被调用。
    fn next_root_child(
        &mut self,
        root_children: &[ChildStat],
        sim_idx: usize,
        cfg: &MctsConfig,
        q_range: Option<(f32, f32)>,
    ) -> Option<usize> {
        let _ = (root_children, sim_idx, cfg, q_range);
        None
    }

    /// 搜索结束后覆盖最终推荐动作索引（Gumbel 用最终幸存者）。
    ///
    /// `q_range` 语义同 [`next_root_child`](Self::next_root_child)。
    /// 返回 `None` → 用 [`SearchPolicy::recommend`]。
    fn final_recommendation(
        &self,
        root_children: &[ChildStat],
        q_range: Option<(f32, f32)>,
    ) -> Option<usize> {
        let _ = (root_children, q_range);
        None
    }
}

/// 默认根调度器：不干预，等价历史 PUCT 单叶循环。
#[derive(Debug, Clone, Default)]
pub struct PuctScheduler;

impl RootScheduler for PuctScheduler {}

/// 动作候选采样上下文：采样器决定候选所需的信息
pub struct ActionSampleContext<'a, S> {
    /// 当前节点隐状态
    pub state: &'a S,
    /// 节点在搜索树中的深度（根 = 0）
    pub depth: usize,
    /// 当前玩家（单智能体为 0）
    pub to_play: u8,
    /// 期望候选数量（连续 / 大动作空间的采样个数 K；离散可忽略）
    pub num_candidates: usize,
}

/// 采样产出的候选动作集合
pub struct ActionCandidates {
    /// 候选动作
    pub actions: Vec<ActionPayload>,
    /// 可选 proposal prior / 权重（与 `actions` 等长；`None` = 由上层均匀处理）
    pub priors: Option<Vec<f32>>,
}

/// 动作候选采样器（独立接缝）
///
/// 负责「给某节点生成 K 个候选动作 + proposal prior」，与 [`SearchPolicy`]（只消费
/// [`ChildStat`]）**解耦**。这样同一接缝同时服务：
/// - 离散：枚举固定 / 合法动作集（行为不变，见 [`DiscreteActionSampler`]）；
/// - 纯连续（Gumbel）/ 混合 / Sampled MuZero：从策略分布采样 K 个候选。
///
/// 由 learned-model adapter 在产出候选动作时调用。
pub trait ActionSampler<S> {
    /// 为某节点生成候选动作（+ 可选 proposal prior）。
    fn sample(&self, ctx: ActionSampleContext<'_, S>, rng: &mut dyn RngCore) -> ActionCandidates;
}

/// 候选展开策略（全量枚举 / Sampled MuZero / legal mask / future proposal）。
pub trait CandidateProvider {
    fn expand_candidates(
        &self,
        candidates: &CandidateSet,
        cfg: &MctsConfig,
        is_root: bool,
        rng: &mut dyn RngCore,
    ) -> CandidateSet;
}

/// 离散动作采样器：枚举固定离散动作集（行为等价现有「全量离散候选」）。
///
/// 默认实现；棋类的「按 `legal_mask` 过滤」留后续扩展（按状态变化的合法集）。
#[derive(Debug, Clone)]
pub struct DiscreteActionSampler {
    actions: Vec<ActionPayload>,
}

impl DiscreteActionSampler {
    /// 用显式离散动作集构造。
    pub fn new(actions: Vec<ActionPayload>) -> Self {
        Self { actions }
    }

    /// 从动作数 `n` 构造（`Discrete(0..n)`）。
    pub fn from_count(n: usize) -> Self {
        Self {
            actions: (0..n).map(ActionPayload::Discrete).collect(),
        }
    }
}

impl<S> ActionSampler<S> for DiscreteActionSampler {
    fn sample(&self, _ctx: ActionSampleContext<'_, S>, _rng: &mut dyn RngCore) -> ActionCandidates {
        ActionCandidates {
            actions: self.actions.clone(),
            priors: None,
        }
    }
}

/// 神经网络预测器接口（独立于 MCTS 树搜索，供外部使用）
pub trait Predictor {
    /// 单条观测预测：返回 (策略概率, 价值)
    fn predict(&self, obs: &[f32]) -> (Vec<f32>, f32);

    /// 批量预测（默认逐个调用）
    fn predict_batch(&self, obs_batch: &[Vec<f32>]) -> (Vec<Vec<f32>>, Vec<f32>) {
        let mut policies = Vec::with_capacity(obs_batch.len());
        let mut values = Vec::with_capacity(obs_batch.len());
        for obs in obs_batch {
            let (p, v) = self.predict(obs);
            policies.push(p);
            values.push(v);
        }
        (policies, values)
    }
}

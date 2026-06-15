//! MCTS trait 定义：模型接口、搜索策略、预测器

use rand::RngCore;

use super::min_max::MinMaxStats;
use super::types::{ActionPayload, ChildStat, MctsConfig, RecurrentOut, RootOut};

/// MCTS 模型接口（root + recurrent，与 mctx 同构）
///
/// 实现者提供：
/// - `root`：从原始观测生成初始隐状态和先验
/// - `recurrent`：从父状态 + 动作推演下一步
///
/// v0.22 实现：AlphaZero（State = PyObject 棋盘快照）
///
/// v0.22 实现：AlphaZero（State = PyObject 棋盘快照）
/// v0.23 实现：MuZero（State = Vec<f32> learned latent，通过 `Dynamics` + `DynamicsModel` 适配）
///
/// # 后续 TODO
/// - 并行时条件加 `State: Send + Sync`
/// - Stochastic MuZero 的 chance node 会改变 recurrent 语义（核心扩展级）
pub trait MctsModel {
    /// 隐状态类型
    type State: Clone + 'static;

    /// 从原始观测生成根节点信息
    fn root(&self, obs: &[f32]) -> RootOut<Self::State>;

    /// 从父状态和动作推演子状态
    fn recurrent(&self, state: &Self::State, action: &ActionPayload) -> RecurrentOut<Self::State>;
}

/// 搜索策略 hook（非整套替换，只注入关键决策点）
///
/// v0.22 实现：PuctPolicy（PUCT + Dirichlet + 温度采样）
///
/// # v0.23+ TODO
/// - GumbelPolicy：序贯减半根候选 + 改 recommend + make_targets（需改 prepare_root 和 recommend）
/// - RegPolicy：ACT policy ≠ LEARN policy → make_targets 需区分
/// - MENTS / RENTS / TENTS / ANT 选择变体（只改 select_child）
/// - Sampled MuZero：连续动作空间，需 ActionSampler 在展开时生成 K 个候选
pub trait SearchPolicy {
    /// 对根节点子节点注入探索噪声
    fn prepare_root(&self, children: &mut [ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore);

    /// 选择要展开的子节点索引（从父节点视角计算 Q 值）
    fn select_child(
        &self,
        parent_visit: u32,
        parent_to_play: u8,
        children: &[ChildStat],
        stats: &MinMaxStats,
        cfg: &MctsConfig,
    ) -> usize;

    /// 搜索结束后推荐最终动作索引（训练时随机采样，评测时可贪心）
    fn recommend(&self, children: &[ChildStat], cfg: &MctsConfig, rng: &mut dyn RngCore) -> usize;

    /// 生成学习用策略目标（visit count → 概率分布）
    fn make_targets(&self, children: &[ChildStat], cfg: &MctsConfig) -> Vec<f32>;
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

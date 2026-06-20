//! 动作适配：把 env 原生动作空间桥接到 MCTS 的离散候选 + `env.step` 动作向量。
//!
//! **事实 vs 选择**：动作"是离散还是连续、几档、范围"是 env 事实（库从 [`GymEnv`] 问出来）；
//! "连续怎么近似"是算法选择（由 [`ActionPlan`] 表达）。

use super::config::ActionPlan;
use crate::rl::GymEnv;
use crate::rl::mcts::ActionPayload;

/// 离散候选 idx → 连续力矩（线性映射到 `[lo, hi]`）。
fn idx_to_continuous(idx: usize, lo: f32, hi: f32, buckets: usize) -> f32 {
    if buckets <= 1 {
        return 0.5 * (lo + hi);
    }
    lo + (hi - lo) * (idx as f32) / ((buckets - 1) as f32)
}

#[derive(Debug, Clone)]
enum AdapterKind {
    /// 原生离散：idx → `[idx as f32]`
    Discrete,
    /// 连续 1 维离散化：idx → `[torque]`
    Discretized { lo: f32, hi: f32, buckets: usize },
}

/// 动作适配器：持有 MCTS 候选动作集 + idx → env 动作向量的映射。
///
/// 由 [`ActionAdapter::resolve`] 在拿到 [`GymEnv`] 后构造（动作维度等 env 事实在此问出）。
#[derive(Debug, Clone)]
pub struct ActionAdapter {
    candidates: Vec<ActionPayload>,
    kind: AdapterKind,
}

impl ActionAdapter {
    /// 从 env + 动作方案解析。动作空间事实从 env 读，方案决定连续如何近似。
    ///
    /// # Panics
    /// - `Auto` 用于连续 env（连续需指定 `Discretize`/未来 `Sampled`）。
    /// - `Discretize` 用于离散 env（离散无需离散化）。
    /// - 混合（Tuple）动作空间尚未支持。
    pub fn resolve(env: &GymEnv, plan: ActionPlan) -> Self {
        let ranges = env.get_all_action_valid_range();
        assert!(
            !ranges.is_empty(),
            "MyZero: env 未暴露任何动作维度（混合 Tuple 动作尚未支持）"
        );
        assert!(
            ranges.len() == 1,
            "MyZero: 多维 / 混合动作空间尚未支持（仅单维离散或单维连续）"
        );
        let range = &ranges[0];

        match plan {
            ActionPlan::Auto => {
                assert!(
                    range.is_discrete_action(),
                    "MyZero: 连续动作 env 须声明 .discretize(buckets)（默认 Auto 仅适用于离散 env）"
                );
                let n = range.get_discrete_action_selectable_num();
                Self {
                    candidates: (0..n).map(ActionPayload::Discrete).collect(),
                    kind: AdapterKind::Discrete,
                }
            }
            ActionPlan::Discretize { buckets } => {
                assert!(
                    !range.is_discrete_action(),
                    "MyZero: 离散动作 env 无需 Discretize（请用 ActionPlan::Auto）"
                );
                assert!(buckets >= 1, "MyZero: Discretize buckets 必须 ≥ 1");
                let (lo, hi) = range.get_continuous_action_low_high();
                Self {
                    candidates: (0..buckets).map(ActionPayload::Discrete).collect(),
                    kind: AdapterKind::Discretized { lo, hi, buckets },
                }
            }
        }
    }

    /// MCTS / `DynamicsModel` 用的候选动作集。
    pub fn candidates(&self) -> &[ActionPayload] {
        &self.candidates
    }

    /// 动作维度（= 候选数；模型输出层宽度）。
    pub fn action_dim(&self) -> usize {
        self.candidates.len()
    }

    /// 把 MCTS 选出的离散候选 idx 映射成 `env.step` 的动作向量。
    pub fn to_env(&self, idx: usize) -> Vec<f32> {
        match self.kind {
            AdapterKind::Discrete => vec![idx as f32],
            AdapterKind::Discretized { lo, hi, buckets } => {
                vec![idx_to_continuous(idx, lo, hi, buckets)]
            }
        }
    }

    /// 人类可读的动作空间描述（启动日志用）。
    pub fn describe(&self) -> String {
        match self.kind {
            AdapterKind::Discrete => format!("离散 {} 档", self.action_dim()),
            AdapterKind::Discretized { lo, hi, buckets } => {
                format!("连续→离散 {buckets} 档 ∈[{lo:.2},{hi:.2}]")
            }
        }
    }
}

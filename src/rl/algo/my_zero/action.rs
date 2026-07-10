//! 动作适配：把 env 原生动作空间桥接到 MCTS 的离散候选 + `env.step` 动作向量。
//!
//! **事实 vs 选择**：动作"是离散还是连续、几档、范围"是 env 事实（库从 [`GymEnv`] 问出来）；
//! "连续怎么近似"是算法选择（由 [`ActionPlan`] 表达）。

use super::config::ActionPlan;
use super::sampled_params::DEFAULT_CONTINUOUS_BUCKETS;
use super::schema::ActionSchema;
use crate::rl::GymEnv;
use crate::rl::mcts::ActionPayload;

/// 离散候选 idx → 连续控制量（等宽 bin **中点**，对齐 Sampled `MuZero` categorical）。
#[cfg_attr(not(test), allow(dead_code))]
pub(super) fn idx_to_continuous(idx: usize, lo: f32, hi: f32, buckets: usize) -> f32 {
    if buckets <= 1 {
        return 0.5 * (lo + hi);
    }
    let width = (hi - lo) / buckets as f32;
    (idx as f32 + 0.5).mul_add(width, lo)
}

/// 防止高维 Box 的 `B^D` 在 CPU / 内存上静默爆炸。
pub const MAX_ENUMERATED_JOINT_ACTIONS: usize = 2048;

/// 动作适配器：持有 MCTS 候选动作集 + idx → env 动作向量的映射。
///
/// 由 [`ActionAdapter::resolve`] 在拿到 [`GymEnv`] 后构造（动作维度等 env 事实在此问出）。
#[derive(Debug, Clone)]
pub struct ActionAdapter {
    candidates: Vec<ActionPayload>,
    schema: ActionSchema,
}

impl ActionAdapter {
    /// 从 env + 动作方案解析。动作空间事实从 env 读，方案决定连续如何近似。
    ///
    /// # Panics
    /// - `Auto` 用于离散 env；连续 env 亦可用（默认 B=7，见 [`DEFAULT_CONTINUOUS_BUCKETS`](super::sampled_params::DEFAULT_CONTINUOUS_BUCKETS)）。
    /// - `Discretize` 用于离散 env（离散无需离散化）。
    /// - 联合动作数超过 [`MAX_ENUMERATED_JOINT_ACTIONS`]。
    pub fn resolve(env: &GymEnv, plan: ActionPlan) -> Self {
        let ranges = env.get_all_action_valid_range();
        assert!(!ranges.is_empty(), "MyZero: env 未暴露任何动作维度");
        let buckets = match plan {
            ActionPlan::Auto => DEFAULT_CONTINUOUS_BUCKETS,
            ActionPlan::Discretize { buckets } => {
                assert!(buckets >= 1, "MyZero: Discretize buckets 必须 ≥ 1");
                buckets
            }
        };
        let discrete: Vec<usize> = ranges
            .iter()
            .filter(|r| r.is_discrete_action())
            .map(|r| r.get_discrete_action_selectable_num())
            .collect();
        let continuous: Vec<(f32, f32)> = ranges
            .iter()
            .filter(|r| !r.is_discrete_action())
            .map(|r| r.get_continuous_action_low_high())
            .collect();

        if continuous.is_empty() && !matches!(plan, ActionPlan::Auto) {
            panic!("MyZero: 纯离散动作 env 无需 Discretize（请用 ActionPlan::Auto）");
        }
        if !discrete.is_empty() && !continuous.is_empty() {
            assert!(
                discrete.len() == 1
                    && ranges[0].is_discrete_action()
                    && ranges[1..].iter().all(|range| !range.is_discrete_action()),
                "MyZero: 当前固定 Hybrid codec 只支持「一个离散 factor 在前 + 连续 factors 在后」的 Platform 布局"
            );
        }

        let schema = match (discrete.is_empty(), continuous.is_empty()) {
            (false, true) if discrete.len() == 1 => ActionSchema::Discrete {
                actions: discrete[0],
            },
            (false, true) => ActionSchema::MultiDiscrete { factors: discrete },
            (true, false) => ActionSchema::ContinuousBins {
                ranges: continuous,
                buckets,
            },
            (false, false) => ActionSchema::HybridBins {
                discrete,
                continuous,
                buckets,
            },
            (true, true) => unreachable!(),
        };
        Self::from_schema(schema)
    }

    /// 从已解析 schema 构造；供 contract tests / 非 Gym adapter 使用。
    pub fn from_schema(schema: ActionSchema) -> Self {
        let count = schema.action_count();
        assert!(
            count <= MAX_ENUMERATED_JOINT_ACTIONS,
            "MyZero: 联合动作数 {count} 超过当前枚举上限 {MAX_ENUMERATED_JOINT_ACTIONS}；\
             请降低连续 buckets 或等待无枚举 proposal backend"
        );
        let candidates = (0..count).map(|id| schema.payload(id)).collect();
        Self { candidates, schema }
    }

    /// 测试 / 无 env 单元测：构造 `n` 档原生离散动作适配器。
    #[cfg(test)]
    pub(crate) fn discrete_for_test(n: usize) -> Self {
        assert!(n >= 1, "discrete_for_test: n 必须 ≥ 1");
        Self::from_schema(ActionSchema::Discrete { actions: n })
    }

    /// MCTS / `DynamicsModel` 用的候选动作集。
    pub fn candidates(&self) -> &[ActionPayload] {
        &self.candidates
    }

    /// 动作维度（= 候选数；模型输出层宽度）。
    pub const fn action_dim(&self) -> usize {
        self.candidates.len()
    }

    pub const fn schema(&self) -> &ActionSchema {
        &self.schema
    }

    /// 把 MCTS 选出的稳定联合 ActionId 映射成 `env.step` 的动作向量。
    pub fn to_env(&self, idx: usize) -> Vec<f32> {
        self.schema.to_env(idx)
    }

    /// 人类可读的动作空间描述（启动日志用）。
    pub fn describe(&self) -> String {
        match &self.schema {
            ActionSchema::Discrete { actions } => format!("离散 {actions} 档"),
            ActionSchema::MultiDiscrete { factors } => {
                format!("MultiDiscrete {factors:?}（联合 {}）", self.action_dim())
            }
            ActionSchema::ContinuousBins { ranges, buckets } => format!(
                "{}维连续×{buckets}档（联合 {}）",
                ranges.len(),
                self.action_dim()
            ),
            ActionSchema::HybridBins {
                discrete,
                continuous,
                buckets,
            } => format!(
                "Hybrid 离散{discrete:?}+{}维连续×{buckets}档（联合 {}）",
                continuous.len(),
                self.action_dim()
            ),
        }
    }
}

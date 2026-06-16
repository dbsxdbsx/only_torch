//! Self-play 整局数据类型（AlphaZero / MuZero / EZ-V2 共用）

use super::BufferItem;

/// 整局 self-play 中的单步（MCTS 监督目标，非 SAC 五元组）。
///
/// AlphaZero 填 `reward=0.0, root_value=None`；
/// MuZero 填实际 reward 和 MCTS 根节点 value。
#[derive(Debug, Clone)]
pub struct SelfPlayStep {
    pub obs: Vec<f32>,
    /// 实际执行的动作（离散用 `vec![idx as f32]`，连续用原始向量）
    pub action: Vec<f32>,
    /// MCTS 输出的 π（visit count 归一化后的概率分布）
    pub policy_target: Vec<f32>,
    pub player: u8,
    /// 环境返回的即时 reward（AlphaZero 填 0.0）
    pub reward: f32,
    /// MCTS 搜索根节点的 value 估计（MuZero 需要，AlphaZero 可省略）
    pub root_value: Option<f32>,
    /// 该步是否为 MDP 真终止（杆倒 / 到达目标），区别于 truncation（步数上限截断）。
    ///
    /// 镜像 Gymnasium 的 `terminated`：n-step value target 仅在 `terminated` 时停止
    /// bootstrap；**truncation（如 CartPole 撞 200 步）仍需 bootstrap**，否则会系统性
    /// 低估满分局末端的 value（见 `compute_n_step_target`）。非终止步与 AlphaZero 填 `false`。
    pub terminated: bool,
    /// 可扩展附加字段（EZ value prefix 等，v0.24 引入）。
    ///
    /// 既有 MuZero / AlphaZero 路径填 [`SelfPlayStepExtras::default()`]。后续算法增量的字段一律
    /// 进 `extras`，避免在本结构上堆裸 `Option` 或全仓 struct-literal 改动。
    pub extras: SelfPlayStepExtras,
}

/// `SelfPlayStep` 的可扩展附加字段（v0.24 引入）。
///
/// 用一个结构化 extras 承载「会随算法增量增长」的字段，避免在 `SelfPlayStep` 上堆裸 `Option`
/// 造成「万能 Option 字段坟场」。**只放 v0.24 真用到的字段**——SVE / PER 等用到时再加，
/// 不预埋全程 `None` 的空字段（守主线 v0.20「禁空占位」红线）；builder 模式保证未来字段可加。
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SelfPlayStepExtras {
    /// EfficientZero value prefix 训练目标（K 步累计 reward 前缀）。
    ///
    /// MuZero / AlphaZero 路径填 `None`；EZ value prefix 填实际前缀目标。
    pub value_prefix_target: Option<f32>,
}

impl SelfPlayStep {
    /// 链式设置 EZ value prefix 训练目标（builder）。
    ///
    /// 未来 extras 新增字段时，继续以 `with_*` builder 方法暴露，调用点无需改 struct-literal。
    pub fn with_value_prefix_target(mut self, target: f32) -> Self {
        self.extras.value_prefix_target = Some(target);
        self
    }
}

/// 终局结果
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GameOutcome {
    /// 胜方 player id
    Win(u8),
    Draw,
    InProgress,
}

/// planning + self-play 族共用样本单元（AlphaZero / MuZero / EZ-V2）。
///
/// 存储单位是整局；训练时由 helper 再抽 position 展平为 (obs, π, z) batch。
/// v0.23 MuZero 在 `SelfPlayStep` 上扩展 reward/root_value 字段，不新建第三种 sample。
#[derive(Debug, Clone)]
pub struct SelfPlayGame {
    pub steps: Vec<SelfPlayStep>,
    pub outcome: GameOutcome,
}

impl BufferItem for SelfPlayGame {}

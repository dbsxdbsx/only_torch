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

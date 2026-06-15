//! Agent trait 定义（无规划 + 规划型）

/// 无规划型 Agent（SAC / DQN 族）
///
/// 给 obs 直接输出 action，不做树搜索。
/// v0.22 与 AlphaZero 一起引入（避免单算法时空抽象）。
pub trait Agent {
    /// 给定观察，返回动作向量
    ///
    /// 离散动作返回 `vec![idx as f32]`；
    /// 连续动作返回对应维度的 `Vec<f32>`。
    fn act(&self, obs: &[f32]) -> Vec<f32>;
}

/// 规划型 Agent（AlphaZero / MuZero 族）
///
/// 除了输出 action，还输出 MCTS 产出的目标策略分布。
/// 用于 self-play 数据收集（每步记录 action + target_policy）。
pub trait PlanningAgent {
    /// 返回 (action, target_policy)
    ///
    /// target_policy 是 MCTS 搜索后 visit count 归一化的概率向量，
    /// 用作监督训练目标。
    fn act_with_target(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>);
}

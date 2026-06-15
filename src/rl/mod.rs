//! 强化学习模块
//!
//! 提供与 Python Gymnasium 环境交互的 Rust 桥接层。
//! **仅支持 Gymnasium**（`>=1.3.0,<2.0`），不兼容老 gym 库。
//!
//! ## 模块结构
//!
//! - `env/` - 环境交互层（GymEnv、MinariDataset）
//!
//! ## 主要组件
//!
//! - [`GymEnv`] - Gymnasium 环境封装，支持离散/连续/混合动作空间
//! - [`MinariDataset`] - Minari 离线 RL 数据集封装
//!
//! ## 使用示例
//!
//! ```ignore
//! use only_torch::rl::{GymEnv, ActionType};
//! use pyo3::Python;
//!
//! Python::attach(|py| {
//!     let env = GymEnv::new(py, "CartPole-v1");
//!     let obs = env.reset(Some(42));
//!     let (next_obs, reward, terminated, truncated) = env.step(&[0.0]);
//!     // terminated: MDP 真终止 → 不 bootstrap
//!     // truncated: 外部截断（步数上限）→ 仍需 bootstrap
//! });
//! ```

mod env;

#[cfg(test)]
mod tests;

// 重新导出环境层的核心类型
pub use env::{
    ActionDim, ActionDimType, ActionRange, ActionType, Episode, GymEnv, MinariDataset, ObsDim,
    ObsType,
};

/// 单步交互数据（**已弃用**，Phase 1 将删除并替换为 `Transition`）
///
/// 用于经验回放缓冲区，存储一次 step 的完整信息。
///
/// **注意**：`done` 字段合并了 terminated 和 truncated，这会导致
/// CartPole 等 truncation 场景的 TD target 计算错误。
/// 新代码应直接使用 `GymEnv::step` 返回的 `(terminated, truncated)` 分离信号。
#[deprecated(note = "v0.20 Phase 1 将替换为 Transition（存 terminated + truncated）")]
#[derive(Debug, Clone)]
pub struct Step {
    /// 当前观察
    pub obs: Vec<f32>,
    /// 执行的动作
    pub action: Vec<f32>,
    /// 获得的奖励
    pub reward: f32,
    /// 下一个观察
    pub next_obs: Vec<f32>,
    /// 是否结束（terminated 或 truncated 的合并值——已弃用语义）
    pub done: bool,
}

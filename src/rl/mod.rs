//! 强化学习模块
//!
//! 提供与 Python Gymnasium 环境交互的 Rust 桥接层。
//!
//! ## 模块结构
//!
//! - `env/` - 环境交互层（GymEnv、MinariDataset）
//!
//! ## 主要组件
//!
//! - [`GymEnv`] - Gymnasium 环境封装，支持离散/连续/混合动作空间
//! - [`MinariDataset`] - Minari 离线 RL 数据集封装
//! - [`Step`] - 单步交互数据（obs, action, reward, `next_obs`, done）
//!
//! ## 使用示例
//!
//! ```ignore
//! use only_torch::rl::{GymEnv, ActionType, MinariDataset};
//! use pyo3::Python;
//!
//! // 在线交互
//! Python::attach(|py| {
//!     let env = GymEnv::new(py, "CartPole-v1");
//!     env.print_env_basic_info();
//!     let obs = env.reset(None);
//!     // ... 训练循环
//! });
//!
//! // 离线数据集
//! Python::attach(|py| {
//!     let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");
//!     let episodes = dataset.sample_episodes(10);
//!     // ... 离线训练
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

/// 单步交互数据
///
/// 用于经验回放缓冲区，存储一次 step 的完整信息。
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
    /// 是否结束（terminated 或 truncated）
    pub done: bool,
}

//! RL 环境交互层
//!
//! 提供与 Python RL 环境交互的 Rust 桥接。
//!
//! ## 主要组件
//!
//! - [`GymEnv`] - Gymnasium/Gym 环境封装，支持离散/连续/混合动作空间
//! - [`MinariDataset`] - Minari 离线 RL 数据集封装
//!
//! ## 使用示例
//!
//! ```ignore
//! use only_torch::rl::{GymEnv, MinariDataset};
//! use pyo3::Python;
//!
//! // 在线交互
//! Python::attach(|py| {
//!     let env = GymEnv::new(py, "CartPole-v1");
//!     let obs = env.reset(None);
//!     // ... 训练循环
//! });
//!
//! // 离线数据集
//! Python::attach(|py| {
//!     let dataset = MinariDataset::load(py, "D4RL/pointmaze/umaze-v2");
//!     let episodes = dataset.sample_episodes(10);
//! });
//! ```

mod gym_env;
mod minari;

// 重新导出核心类型
pub use gym_env::{ActionDim, ActionDimType, ActionRange, ActionType, GymEnv, ObsDim, ObsType};
pub use minari::{Episode, MinariDataset};

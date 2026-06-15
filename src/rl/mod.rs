//! 强化学习模块
//!
//! 提供与 Python Gymnasium 环境交互的 Rust 桥接层。
//! **仅支持 Gymnasium**（`>=1.3.0,<2.0`），不兼容老 gym 库。
//!
//! ## 模块结构
//!
//! - `env/` - 环境交互层（GymEnv、MinariDataset）
//! - `buffer/` - 经验回放（Transition、ReplayBuffer）
//!
//! ## 主要组件
//!
//! - [`GymEnv`] - Gymnasium 环境封装，支持离散/连续/混合动作空间
//! - [`Transition`] - 单步交互数据（terminated + truncated 分离）
//! - [`ReplayBuffer`] - 泛型经验回放缓冲区（有放回采样）
//! - [`MinariDataset`] - Minari 离线 RL 数据集封装
//!
//! ## 使用示例
//!
//! ```ignore
//! use only_torch::rl::{GymEnv, Transition, ReplayBuffer};
//! use pyo3::Python;
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! Python::attach(|py| {
//!     let env = GymEnv::new(py, "CartPole-v1");
//!     let mut buffer = ReplayBuffer::new(10_000);
//!     let mut rng = StdRng::seed_from_u64(42);
//!
//!     let obs_vec = env.reset(Some(42));
//!     let obs = env.flatten_obs(&obs_vec);
//!     let (next_obs_vec, reward, terminated, truncated) = env.step(&[0.0]);
//!     let next_obs = env.flatten_obs(&next_obs_vec);
//!
//!     buffer.push(Transition {
//!         obs, action: vec![0.0], reward, next_obs, terminated, truncated,
//!     });
//!
//!     let batch = buffer.sample(32, &mut rng);
//! });
//! ```

pub mod algo;
pub mod buffer;
mod env;

#[cfg(test)]
mod tests;

// 重新导出环境层的核心类型
pub use env::{
    ActionDim, ActionDimType, ActionRange, ActionType, Episode, GymEnv, MinariDataset, ObsDim,
    ObsType,
};

// 重新导出 buffer 层的核心类型
pub use buffer::{BufferItem, ReplayBuffer, Transition};

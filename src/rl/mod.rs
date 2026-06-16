//! 强化学习模块
//!
//! 提供与 Python Gymnasium 环境交互的 Rust 桥接层。
//! **仅支持 Gymnasium**（`>=1.3.0,<2.0`），不兼容老 gym 库。
//!
//! ## 模块结构
//!
//! - `agent` - Agent trait（无规划型 + 规划型）
//! - `env/` - 环境交互层（GymEnv、MinariDataset）
//! - `buffer/` - 经验回放（Transition、RolloutStep、SelfPlayGame、ReplayBuffer、RolloutBuffer）
//!
//! ## 主要组件
//!
//! - [`Agent`] / [`PlanningAgent`] - 无规划 / 规划型 Agent trait
//! - [`GymEnv`] - Gymnasium 环境封装，支持离散/连续/混合动作空间
//! - [`Transition`] - 单步交互数据（terminated + truncated 分离）
//! - [`RolloutStep`] - 单步 on-policy 采集数据（PPO / A2C 族）
//! - [`SelfPlayGame`] - 整局 self-play 样本（AlphaZero / MuZero / EZ-V2）
//! - [`ReplayBuffer`] - 泛型经验回放缓冲区（有放回采样）
//! - [`RolloutBuffer`] - 固定 n_steps 的 on-policy 采集缓冲区
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

pub mod agent;
pub mod algo;
pub mod buffer;
mod env;
pub mod mcts;

#[cfg(test)]
mod tests;

// 重新导出环境层的核心类型
pub use env::{
    ActionDim, ActionDimType, ActionRange, ActionType, Episode, GymEnv, MinariDataset, ObsDim,
    ObsType,
};

// 重新导出 agent trait
pub use agent::{Agent, PlanningAgent};

// 重新导出 buffer 层的核心类型
pub use buffer::{
    BufferItem, GameOutcome, ReplayBuffer, RolloutBuffer, RolloutStep, SelfPlayGame, SelfPlayStep,
    SelfPlayStepExtras, Transition,
};

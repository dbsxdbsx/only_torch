//! RL 模块测试
//!
//! 按 RL 组件分类的测试子模块：
//!
//! - `env/` - 环境相关测试（Gymnasium 环境、空间解析等）
//!
//! ## 设计说明
//!
//! 强化学习的经验回放（ReplayBuffer）由用户在应用层自行管理，
//! 库只提供环境交互层（GymEnv、MinariDataset）。
//! 参见 `examples/cartpole_sac/` 了解 SAC-Discrete 完整训练示例。

mod env;

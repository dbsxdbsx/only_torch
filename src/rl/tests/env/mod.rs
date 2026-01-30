//! 环境相关测试
//!
//! 测试 RL 环境的创建、交互、空间解析等功能。
//!
//! ## 测试文件
//!
//! - `gym_env.rs` - GymEnv 环境覆盖测试（离散、连续、图像等代表性环境）
//! - `action_space.rs` - 动作空间解析专项测试
//! - `obs_space.rs` - 观察空间解析专项测试
//! - `minari.rs` - Minari 离线 RL 数据集测试（MinariDataset 封装）

mod action_space;
mod gym_env;
mod minari;
mod obs_space;

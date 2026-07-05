//! 经验回放缓冲区与数据类型
//!
//! - [`Transition`]：单步交互（off-policy TD 族：SAC/DQN）
//! - [`RolloutStep`]：单步 on-policy 采集数据（PPO / A2C 族）
//! - [`SelfPlayGame`]：整局 self-play 样本（planning 族：AlphaZero/MuZero/EZ-V2）
//! - [`BufferItem`]：缓冲区元素约束 trait
//! - [`ReplayBuffer`]：泛型 FIFO + 有放回随机采样
//! - [`PerPriorities`]：位置级优先采样伴生结构（PER，`ReplayBuffer` 的 FIFO 镜像）
//! - [`RolloutBuffer`]：固定 `n_steps` 的 on-policy 采集缓冲区

mod obs;
mod per;
mod replay;
mod rollout;
mod rollout_buffer;
mod self_play;
mod transition;

pub use obs::StoredObs;
pub use per::{PER_EPS, PerPriorities};
pub use replay::ReplayBuffer;
pub use rollout::RolloutStep;
pub use rollout_buffer::RolloutBuffer;
pub use self_play::{GameOutcome, SelfPlayGame, SelfPlayStep, SelfPlayStepExtras};
pub use transition::{BufferItem, Transition};

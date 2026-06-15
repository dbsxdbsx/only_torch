//! 经验回放缓冲区与数据类型
//!
//! - [`Transition`]：单步交互（off-policy TD 族：SAC/DQN）
//! - [`SelfPlayGame`]：整局 self-play 样本（planning 族：AlphaZero/MuZero/EZ-V2）
//! - [`BufferItem`]：缓冲区元素约束 trait
//! - [`ReplayBuffer`]：泛型 FIFO + 有放回随机采样

mod replay;
mod self_play;
mod transition;

pub use replay::ReplayBuffer;
pub use self_play::{GameOutcome, SelfPlayGame, SelfPlayStep};
pub use transition::{BufferItem, Transition};

//! 经验回放缓冲区与数据类型
//!
//! - [`Transition`]：单步交互（off-policy TD 族：SAC/DQN）
//! - [`BufferItem`]：缓冲区元素约束 trait
//! - [`ReplayBuffer`]：泛型 FIFO + 有放回随机采样

mod replay;
mod transition;

pub use replay::ReplayBuffer;
pub use transition::{BufferItem, Transition};

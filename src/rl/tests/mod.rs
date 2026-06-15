//! RL 模块测试
//!
//! - `algo_sac.rs` - SAC helper 单元测试（V 值 / TD target / alpha / batch 转换）
//! - `buffer_replay.rs` - ReplayBuffer + Transition 单元测试
//! - `buffer_self_play.rs` - SelfPlayGame + ReplayBuffer 单元测试
//! - `mcts_search.rs` - MCTS 搜索引擎集成测试（mock model + backup 双场景）
//! - `env/` - 环境 + 桥接测试（Gymnasium、五子棋 Board 往返）

mod algo_sac;
mod buffer_replay;
mod buffer_self_play;
mod mcts_search;
mod env;

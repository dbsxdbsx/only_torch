//! RL 模块测试
//!
//! - `algo_sac.rs` - SAC helper 单元测试（V 值 / TD target / alpha / batch 转换）
//! - `algo_ppo.rs` - PPO helper 单元测试（GAE + loss + 优势标准化）
//! - `buffer_replay.rs` - ReplayBuffer + Transition 单元测试
//! - `buffer_self_play.rs` - SelfPlayGame + ReplayBuffer 单元测试
//! - `mcts_search.rs` - MCTS 搜索引擎集成测试（mock model + backup 双场景 + RNG 可复现 + 根调度 hook）
//! - `mcts_sampler.rs` - ActionSampler 接缝契约测试（离散默认采样器）
//! - `mcts_recurrent_state.rs` - State 携带 recurrent hidden 契约测试（hidden + prefix 增量 reward）
//! - `env/` - 环境 + 桥接测试（Gymnasium、五子棋 Board 往返）

mod algo_ppo;
mod algo_sac;
mod buffer_replay;
mod buffer_rollout;
mod buffer_self_play;
mod env;
mod mcts_cartpole_env;
mod mcts_dynamics;
mod mcts_recurrent_state;
mod mcts_sampler;
mod mcts_search;

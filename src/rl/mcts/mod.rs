//! MCTS（蒙特卡洛树搜索）内核
//!
//! 提供与 AlphaZero / MuZero 兼容的单线程 MCTS 搜索引擎。
//!
//! ## 模块结构
//!
//! - [`types`] - 核心数据类型（动作、配置、搜索结果）
//! - [`traits`] - 模型和策略 trait（MctsModel、SearchPolicy、Predictor）
//! - [`puct`] - PUCT 策略实现
//! - [`search`] - 主搜索函数 `mcts_search`
//!
//! ## 快速开始
//!
//! ```ignore
//! use only_torch::rl::mcts::{mcts_search, MctsConfig, MctsModel, PuctPolicy};
//!
//! let result = mcts_search(&model, &PuctPolicy::new(), &obs, &MctsConfig::default());
//! println!("推荐动作: {:?}", result.recommended);
//! println!("学习目标: {:?}", result.learn_policy);
//! ```

mod node;
pub mod puct;
pub mod search;
pub mod traits;
pub mod types;

// 重新导出核心公开 API
pub use puct::PuctPolicy;
pub use search::mcts_search;
pub use traits::{MctsModel, Predictor, SearchPolicy};
pub use types::{
    ActionPayload, ChildStat, MctsConfig, RecurrentOut, RootOut, SearchResult,
};

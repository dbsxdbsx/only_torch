//! SVE：Search-based Value Estimation —— v0.24 Phase 1 `+SVE` 实现。
//!
//! EfficientZero V2 增量：用 MCTS 搜索本身产出的（更可靠的）root value 修正 stale buffer 的
//! value 目标，缓解旧数据 value 漂移。与 reanalyze 协同（reanalyze 重搜，SVE 取搜索 value）。
//!
//! 具体 blend 口径（搜索 root value vs n-step bootstrap）与 reanalyze 路径耦合，随 Phase 1
//! 落地并配单测；此处先占位，保持模块结构稳定。

//! Value prefix 目标（LSTM 累计 reward 前缀）—— v0.24 Phase 1 `+value prefix`（忠实版）实现。
//!
//! Value prefix（EfficientZero V1 提出）把「逐步精确预测 reward」改为「预测从子根到第 k 步的
//! 累计（带折扣）reward 之和」，规避 reward 落点的 state-aliasing，使监督更稳。
//!
//! **忠实版**：LSTM hidden 穿过 MCTS 搜索树（见 `MctsModel::State` 不透明契约 + 第 5 根接缝
//! 契约测试 `rl::tests::mcts_recurrent_state`），搜索期 reward 取 prefix 增量。训练期的
//! 累计前缀目标计算 + LSTM value-prefix 头随 Phase 1 EZ 示例落地；此处先占位。

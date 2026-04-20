---
applyTo: "**/evolution/**/*.rs"
description: "Use when editing the NEAT-style evolution engine, genome builder, mutation logic, convergence logic, or evolution examples in only_torch."
---

# Evolution Instructions

- 保持主流程稳定：build → restore weights → train → capture weights → evaluate → accept/rollback → mutate。
- 变更前先看 [神经架构演化设计](../../.doc/design/neural_architecture_evolution_design.md)；性能排查再看 [优化候选清单](../../.doc/optimization_candidates.md)。
- 优先做最小改动，不要同时大改 `gene.rs`、`mutation.rs`、`builder.rs` 多层逻辑。
- 默认先跑小样本或针对性测试，再跑 `just example-evolution-mnist` 这类重任务。
- 长时间没有日志通常代表候选仍在评估，不一定是卡死。
- 稀疏 FM 图会明显慢于 fully-connected FM 的合并路径；调性能时优先确认 builder 是否走到高效路径。
- 新增基因字段、变异策略或收敛条件时，要同步检查序列化与回滚逻辑是否仍一致。

---
status: suspended
created: 2026-04-28
last_updated: 2026-04-28
---

# 空间域 Evolution 示例运行偏慢

## 背景

Segmentation P1 中新增了传统重叠形状语义分割、固定 slot 实例分割，以及一个最小 `segmentation evolution` 示例。传统 CNN 训练路径表现正常，但空间域 Evolution 示例明显更慢。

## 现象 / 影响

- `overlapping_shapes_semantic_segmentation` 传统示例在 debug + BLAS 下约 12 秒达到 Mean IoU 79.7%。
- `overlapping_fixed_slot_instance_segmentation` 传统示例在 debug + BLAS 下约 13.5 秒达到 Valid-slot IoU 47.8%。
- `evolution_overlapping_shapes_semantic_segmentation` 能正常完成并达到目标，但同等机器与 debug + BLAS 下约 263.6 秒才结束。
- 这类问题不只影响新增分割 evolution，既有空间域 Evolution 示例（例如 `evolution_mnist`）也应纳入同一专项排查。

## 已尝试

- 已确认新增传统分割 benchmark 可以正常训练、收敛并输出可视化结果。
- 已确认新增 segmentation evolution 示例底层逻辑可跑通：完成 2 代后达到 TargetReached，Mean IoU 约 69.9%。
- 已通过分割相关测试与 examples 编译检查，说明当前优先问题不是 correctness failure，而是空间域 Evolution 路径的 wall-clock 成本。

## 当前卡点

具体慢点尚未定位。当前怀疑可能与 Evolution 路径中的 minimizer / 候选训练评估流程有关：传统手写模型训练速度正常，但一进入 Evolution 的空间域搜索、候选构建、权重继承、训练评估循环后耗时明显放大。

## 暂缓原因

本阶段目标是先保证 Segmentation P1 的底层逻辑、shape 协议、指标和 smoke 示例成立；速度优化暂不阻塞本次交付。

## 下次恢复条件

- 当 P1 correctness 已稳定，并且需要决定 P5 Surrogate、P2 Attention、P3 多输出的优先级时恢复。
- 若演化示例运行时间成为 demo 或测试门槛，应优先恢复本问题。

## 下一步建议

- 用统一 profiler 或阶段计时拆分 `build -> restore_weights -> train -> capture_weights -> evaluate -> mutate`。
- 对比传统 CNN 训练循环与空间域 Evolution 的单候选单 epoch 成本。
- 优先排查 minimizer / optimizer step、Graph snapshot、权重继承、NodeLevel 空间算子执行路径是否存在重复计算或不必要重建。
- 将 `evolution_mnist` 与 `evolution_overlapping_shapes_semantic_segmentation` 放在同一个排查矩阵中，不要只针对分割示例做局部优化。

## 相关文件 / 命令 / 对话

- `examples/evolution/mnist/main.rs`
- `examples/evolution/overlapping_shapes_semantic_segmentation/main.rs`
- `src/nn/evolution/mod.rs`
- `src/nn/evolution/task.rs`
- `src/nn/evolution/mutation.rs`
- `cargo run --example evolution_overlapping_shapes_semantic_segmentation --features blas-mkl`

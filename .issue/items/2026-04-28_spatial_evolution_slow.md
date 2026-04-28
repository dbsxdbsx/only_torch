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
- 已落地第一批诊断与低风险优化：
  - `EvolutionCallback::on_evaluation_timing` 输出 `build / restore / train / capture / evaluate / cost` 分段计时。
  - `SupervisedTask::train()` 增加 `setup / shuffle / slice / set_value / zero_grad / backward / step / grad_norm` 二级计时。
  - `FixedEpochs` 训练预算下跳过无效的 `grad_norm` 计算。
  - segmentation 任务变异后不再无条件 `migrate_to_fm_level()`，避免 dense H/W 输出协议被 FM 分解路径干扰。
  - `Evolution::with_initial_burst()` 支持控制初始随机爆发次数，`evolution_mnist` 增加 `smoke / demo / search` 三档 profile。
  - `Conv2d` forward/backward 中将部分 `to_vec() -> Array2` 临时拷贝改为 `ArrayView2`，减少小矩阵 GEMM 输入分配。
  - `NodeInner::backward_propagate()` 改为借用当前节点梯度传播，避免每个反向节点先 clone 整块梯度张量。
  - `SoftmaxCrossEntropy` 前向融合 softmax 与 loss 计算，避免分类训练中重复扫描 logits、重复计算 max/exp，并去掉 labels 缓存 clone。
  - `NodeInner` 反向传播增加可训练祖先剪枝：父分支若不会通向任何 `Parameter`，不再计算该分支 VJP，避免普通数据输入分支产生无用梯度。
  - `Conv2d` forward 缓存每个样本的 im2col 矩阵，backward 计算 kernel 梯度时复用，避免同一 batch 内重复展开感受野。
  - `cargo run --example evolution_mnist --features blas-mkl -- --profile=smoke` 已跑通，优化后总耗时约 2.2 秒，演化段约 1.6 秒；二级计时显示主要剩余耗时仍集中在 `backward`。
- 已新增第二批搜索诊断与复杂度控制：
  - `EvaluationTimingSummary` 增加 `primary[min/avg/max]` 与 `cost[min/avg/max]`，默认日志通过 `eval-detail` 输出候选质量和 FLOPs 分布。
  - `Evolution::with_max_inference_cost()` 支持在训练前过滤超出复杂度上限的候选，默认关闭。
  - `evolution_mnist --profile=search` 暂时配置 `max_inference_cost=3_000_000 FLOPs`，并恢复 `initial_burst=8` 保留早期结构多样性。
  - `search` 曾在用户中断前运行到约第 23 代，best 约 88%，尚未达到 `final_refit` 触发区间；这说明当前主要卡点不是“最后精训不够”，而是搜索阶段没稳定找到接近 95% 的候选。
- 已新增第三批搜索空间与可用路径调整：
  - 新增空间分类初始 portfolio：`spatial_flat_mlp`、`minimal_spatial`、`spatial_lenet_tiny`，从多个合理结构族进入同一套训练/评估/选择流程。
  - 新增 P5-lite 候选预筛：按结构特征与 FLOPs 对候选做启发式排序，只完整训练 top-k；默认关闭，MNIST `search` 启用。
  - 初始 portfolio 评估后若已有候选达到 target，可直接返回 `TargetReached`，避免无意义进入随机变异循环。
  - 新增 `evolution_mnist --profile=quality`：使用 FlatMLP warm-start，15000 train / 1000 test，实测演化耗时约 3.7 秒、总耗时约 4.6 秒，Accuracy 95.5%；该 profile 作为 MNIST 示例推荐默认路径。
  - `search` 短跑显示已从原先 88% 平台提升到约 94.9%，但超过 1 分钟仍未稳定返回 95%，不适合作为用户默认体验路径。

## 当前卡点

初步证据显示 smoke 档主要耗时在候选 `train` 阶段，且 `train-detail` 基本集中在 `backward`；`shuffle / slice / set_value / zero_grad / step / grad_norm` 暂未表现为主要瓶颈。本轮通过复用 Conv2d im2col 缓存后，MNIST smoke 的 `backward` 累计从约 1.7-1.8 秒降到约 1.0 秒。

`search` 档的新问题是：portfolio + P5-lite 已显著改善早期候选质量，但完整随机搜索仍容易在 94%~95% 附近消耗超过 1 分钟；因此用户默认路径应优先使用 `quality` profile 的 portfolio warm-start，完整 `search` 保留为诊断/研究模式。

## 暂缓原因

用户已明确要求：MNIST Evolution 需要先做到接近传统示例的可用量级。当前 `quality` profile 已完成 30 秒内达标目标；完整 `search` 仍暂缓作为默认路径，后续再决定是否升级完整 P5 surrogate。该条目仍保持 `suspended`，因为 segmentation evolution 慢和完整 search 慢尚未闭环。

## 下次恢复条件

- `quality` profile 作为 MNIST 用户默认可用路径继续保持 30 秒内达标。
- 完整 `search` 若要继续优化，应先收集更多候选日志，再决定是否从 P5-lite 升级到学习型 surrogate。

## 下一步建议

- 用新增阶段计时继续对比 `evolution_mnist --profile=quality/demo/search`，重点看 `quality` 是否稳定达标、`search` 是否仍在 94%~95% 附近长时间平台。
- 对比传统 CNN 训练循环与空间域 Evolution 的单候选单 epoch 成本。
- 如果 `max_inference_cost=3_000_000` 过紧导致候选不足，优先小幅提高上限；如果过松导致后半程变慢，继续下调并观察 `cost[min/avg/max]`。
- 下一步优先排查反向传播内部：`Conv2d / Pool2d / Flatten / CrossEntropy` 各节点 VJP 占比、反向拓扑遍历中的梯度累加分配，以及多层/多分支空间结构下 im2col 缓存的内存占用边界。
- 将 `evolution_mnist` 与 `evolution_overlapping_shapes_semantic_segmentation` 放在同一个排查矩阵中，不要只针对分割示例做局部优化。

## 相关文件 / 命令 / 对话

- `examples/evolution/mnist/main.rs`
- `examples/evolution/overlapping_shapes_semantic_segmentation/main.rs`
- `src/nn/evolution/mod.rs`
- `src/nn/evolution/task.rs`
- `src/nn/evolution/mutation.rs`
- `cargo run --example evolution_overlapping_shapes_semantic_segmentation --features blas-mkl`
- `cargo run --example evolution_mnist --features blas-mkl -- --profile=smoke`
- `cargo run --example evolution_mnist --features blas-mkl -- --profile=quality`

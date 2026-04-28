根据 .doc/design/neural_architecture_evolution_design.md 和 .doc/design/spatial_vision_tasks_roadmap.md 中的路线图，列出当前的优先级和具体怎么做。

| 优先级 | 建议先后 | 当前状态 | 为什么排这里 | 具体怎么做 |
|---|---|---|---|---|
| P0 | 对齐文档与现状 | 已完成 | ASHA / spatial seed / MNIST 默认路径等文档已对齐实现，profile 概念已从示例移除 | 保持 `.doc/design/neural_architecture_evolution_design.md` 与空间任务路线图同步即可 |
| P1 | Segmentation 任务支持 | 基本完成 | 已新增传统 segmentation 示例、U-Net-lite 强基线、固定 slot instance 示例和最小 segmentation evolution 示例；原始 minutes 级慢路径已关闭 | 不再作为主线开发项，后续用更可信 benchmark 判断是否继续优化 dense forward/backward |
| P1.5 | **结构族 / P5-lite / timing 观测闭环** | 已完成当前最小闭环 | 证据显示当前 segmentation 慢点不在 scorer / ASHA；`train-detail` 已拆分 `backward_forward` / `backward_propagate`，主要剩余成本指向 dense forward | `evolution_mnist` 默认路径 seed=42 本轮 Accuracy 95.3%、总耗时约 45.3s；`evolution_overlapping_shapes_semantic_segmentation` 本轮 Mean IoU 63.0%、最新单次复测约 2.9s |
| P2 | 扩大 Segmentation Evolution benchmark | 已完成 | U-Net-lite 已给出更强传统对照；64x64 evolution 对齐版证明可接受耗时内能跑通；当前已纳入 encoder-decoder 初始族，并新增保持分辨率的 `InsertEncoderDecoderSkip` 结构变异 | `evolution_overlapping_shapes_unet_lite_segmentation` 在 target Mean IoU 0.60 下完成 5-seed 稳定性验证，seed 1..5 全部 `TargetReached`，Mean IoU 为 93.3% / 98.4% / 90.3% / 77.1% / 63.0%；后续若继续提升，再做训练预算、mutation 命中率或 dense Conv2d / ConvTranspose2d profiler |
| P3 | 多输出 / 多头任务演化 | 第一阶段已完成 | 已支持固定数量命名多头 supervised evolution，可解锁 actor-critic、辅助头、多任务学习的基础协议；检测/实例 matching 仍留到后续 | `Evolution::supervised(...)` 旧链式 API 保持兼容；新增 `SupervisedSpec` / `HeadSpec` 显式入口；`NetworkGenome` 记录 output head 元数据，`BuildResult` 暴露 `outputs`；训练按 head 加权聚合 loss，评估输出逐 head report；`predict_head` / `predict_heads` 支持选择性推理；新增 `evolution_multi_head_quadrant_radius` 作为公开用法示例 |
| P4 | Deformable Conv2d | 通用算子、传统基线与 evolution 小 benchmark 已完成 | 适合在 segmentation 有基准后做，否则难判断收益；本轮先走“通用算子 → 传统手写示例 → evolution 接入”路线 | 已新增 `NodeTypeDescriptor::DeformableConv2d`、raw node 前反向、`DeformableConv2d` Layer、PyTorch / torchvision 数值对照单测、`deformable_conv2d_segmentation` 传统示例；新增 `evolution_deformable_conv2d_segmentation`，通过 DeformableConv2d 初始族验证 evolution 主流程；已修复 DeformableConv2d 动态 batch 与 BinaryIoU BCE logits 评估问题。后续若证明命中率不足，再扩展 `ChangeFMEdgeType` |
| P5 | Surrogate 模型 | 暂停完整学习型版本；保留 P5-lite 基础 | 完整 learned surrogate 对当前 CPU-only toy benchmark 收益不确定，且容易过度工程；P5-lite 作为低成本启发式预筛继续保留 | 不作为当前主线推进；若后续真实 benchmark 出现评估成本瓶颈，再先做 Search Audit 数据与相关性验证，不直接内置万能 learned surrogate |
| P6 | Dynamic Conv2d | 未做 | 有价值但实现复杂，且 CPU-only 下成本可能偏高 | 新增动态 kernel 生成子图或专用 node；先限制小 kernel / 小通道；纳入 FM edge 类型；重点测试前向、反向和 FLOPs 估算 |
| P7 | ENAS 式权重共享 | 未做 | 潜在收益大，但会重塑权重生命周期和 Lamarckian 继承逻辑 | 建子图结构 hash → 权重库；定义共享、失效、迁移和快照规则；先只支持 Linear / Conv2d 同构子图 |
| P8 | 分布式 island model | 未做 | 工程量中等，但当前单机评估还没到必须分布式 | genome + fitness 序列化；多个 island 独立演化；周期性交换 elite；先本机多进程模拟 |
| P9 | DARTS 混合搜索 | 未做 | 几乎是另一套搜索范式，最后考虑 | 新建独立 search engine，不建议塞进现有 mutation / selection 主循环 |


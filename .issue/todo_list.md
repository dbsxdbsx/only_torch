
| 优先级 | 建议先后 | 当前状态 | 为什么排这里 | 具体怎么做 |
|---|---|---|---|---|
| P0 | 对齐文档与现状 | 已完成 | ASHA / spatial seed / MNIST 默认路径等文档已对齐实现，profile 概念已从示例移除 | 保持 `.doc/design/neural_architecture_evolution_design.md` 与空间任务路线图同步即可 |
| P1 | Segmentation 任务支持 | 基本完成 | 已新增传统 segmentation 示例、U-Net-lite 强基线、固定 slot instance 示例和最小 segmentation evolution 示例；原始 minutes 级慢路径已关闭 | 不再作为主线开发项，后续用更可信 benchmark 判断是否继续优化 dense forward/backward |
| P1.5 | **结构族 / P5-lite / timing 观测闭环** | 已完成当前最小闭环 | 证据显示当前 segmentation 慢点不在 scorer / ASHA；`train-detail` 已拆分 `backward_forward` / `backward_propagate`，主要剩余成本指向 dense forward | `evolution_mnist` 默认路径 seed=42 本轮 Accuracy 95.3%、总耗时约 45.3s；`evolution_overlapping_shapes_semantic_segmentation` 本轮 Mean IoU 63.0%、最新单次复测约 2.9s |
| P2 | 扩大 Segmentation Evolution benchmark | 已完成搜索空间第一步 | U-Net-lite 已给出更强传统对照；64x64 evolution 对齐版证明可接受耗时内能跑通，并已纳入 encoder-decoder + skip concat 初始族，但指标仍低于 U-Net-lite | `evolution_overlapping_shapes_unet_lite_segmentation` 最新 Mean IoU 53.3%、总耗时约 20.0s；下一步优先让 Pool / Deconv / skip concat 可通过 mutation 稳定探索，再按需要 profiler dense Conv2d / ConvTranspose2d |
| P3 | 多输出 / 多头任务演化 | 底层 IR 部分支持，evolution 未支持 | 这是任务层大改，适合在单输出 segmentation 跑通后做；可解锁 actor-critic、辅助头、多任务学习 | 把 `BuildResult.output: Var` 扩为 `outputs: Vec<Var>` 或兼容包装；genome 显式标记多个 output head；`TaskSpec` 支持每个输出的 target / loss / metric；`FitnessScore` 增加聚合策略；`predict()` 返回多输出结构 |
| P4 | Deformable Conv2d | 未做 | 适合在 segmentation 有基准后做，否则难判断收益 | 新增 `NodeTypeDescriptor::DeformableConv2d`；实现 raw node 前反向；扩展 FM edge 类型；让 `ChangeFMEdgeType` 能在 conv / deformable conv / pool / deconv 间切换；用 segmentation toy benchmark 验证 |
| P5 | Surrogate 模型 | 未做完整学习型版本；P5-lite 已有基础 | 阶段 E 中最现实的提速项，比 ENAS 权重共享风险低，但需要先证明候选特征对 fitness 有预测价值 | 先累计 Search Audit 数据：结构族、节点数、参数量、FLOPs、深度、变异类型、fitness、cost、timing；若启发式 scorer 与真实评估正相关，再训练简单回归器并只完整评估 top-k |
| P6 | Dynamic Conv2d | 未做 | 有价值但实现复杂，且 CPU-only 下成本可能偏高 | 新增动态 kernel 生成子图或专用 node；先限制小 kernel / 小通道；纳入 FM edge 类型；重点测试前向、反向和 FLOPs 估算 |
| P7 | ENAS 式权重共享 | 未做 | 潜在收益大，但会重塑权重生命周期和 Lamarckian 继承逻辑 | 建子图结构 hash → 权重库；定义共享、失效、迁移和快照规则；先只支持 Linear / Conv2d 同构子图 |
| P8 | 分布式 island model | 未做 | 工程量中等，但当前单机评估还没到必须分布式 | genome + fitness 序列化；多个 island 独立演化；周期性交换 elite；先本机多进程模拟 |
| P9 | DARTS 混合搜索 | 未做 | 几乎是另一套搜索范式，最后考虑 | 新建独立 search engine，不建议塞进现有 mutation / selection 主循环 |


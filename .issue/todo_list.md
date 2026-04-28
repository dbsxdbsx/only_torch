结论：当前 A / B / C / F 基础设施基本都做完了；P0 文档对齐已完成，P1 Segmentation 任务支持也基本完成。MNIST 搜索卡点已通过空间分类默认策略解决，并且用户侧不再暴露 `smoke / quality / audit / search` profile。下一步不直接做完整 P5 surrogate，而是把同一套结构族与 timing 观测迁移到 Segmentation evolution，确认剩余慢点到底在 scorer、mutation、ASHA 还是 dense spatial-to-spatial 训练成本。

理由很简单：P1 已经把空间任务闭环跑起来了，MNIST 默认路径已完成 5-seed 稳定性验证（全部达到 95%）；剩下最明显的问题是 Segmentation evolution 仍明显慢。现在最缺的不是新算子，而是能解释“候选从哪里来、P5-lite 留下了什么、真实评估是否支持 scorer 判断”的跨任务观测数据。

| 优先级 | 建议先后 | 当前状态 | 为什么排这里 | 具体怎么做 |
|---|---|---|---|---|
| P0 | 对齐文档与现状 | 已完成 | ASHA / spatial seed / MNIST 默认路径等文档已对齐实现，profile 概念已从示例移除 | 保持 `.doc/design/neural_architecture_evolution_design.md` 与 issue 记录同步即可 |
| P1 | Segmentation 任务支持 | 基本完成 | 已新增传统 segmentation 示例、固定 slot instance 示例和最小 segmentation evolution 示例；当前主要剩余问题是 evolution 路径慢 | 不再作为主线开发项，后续跟随 Search Audit 一起排查搜索和耗时 |
| P1.5 | **结构族 / P5-lite / timing 观测闭环** | MNIST 默认路径已完成；Segmentation 待迁移 | 先用证据决定是否升级完整 P5 surrogate，避免过早引入学习型预测器 | `evolution_mnist` 默认路径 5 个 seed 全部达到 95%；下一步把同一矩阵迁移到 segmentation evolution |
| P2 | Attention / Transformer 纳入演化 | 部分有基础 | `MultiHeadAttention` layer 已存在，但 evolution 还不能搜索它；适合扩展序列任务能力 | 先做 layer-block 级别接入，不急着做单 raw op；新增 `LayerConfig::SelfAttention` / `TransformerBlock`，在 `migration.rs` 展开为现有 MatMul / Softmax / Linear 子图；定义 Sequence 域规则、head 数变异、embed dim 约束和测试 |
| P3 | 多输出 / 多头任务演化 | 底层 IR 部分支持，evolution 未支持 | 这是任务层大改，适合在单输出 segmentation 跑通后做；可解锁 actor-critic、辅助头、多任务学习 | 把 `BuildResult.output: Var` 扩为 `outputs: Vec<Var>` 或兼容包装；genome 显式标记多个 output head；`TaskSpec` 支持每个输出的 target / loss / metric；`FitnessScore` 增加聚合策略；`predict()` 返回多输出结构 |
| P4 | Deformable Conv2d | 未做 | 适合在 segmentation 有基准后做，否则难判断收益 | 新增 `NodeTypeDescriptor::DeformableConv2d`；实现 raw node 前反向；扩展 FM edge 类型；让 `ChangeFMEdgeType` 能在 conv / deformable conv / pool / deconv 间切换；用 segmentation toy benchmark 验证 |
| P5 | Surrogate 模型 | 未做完整学习型版本；P5-lite 已有基础 | 阶段 E 中最现实的提速项，比 ENAS 权重共享风险低，但需要先证明候选特征对 fitness 有预测价值 | 先累计 Search Audit 数据：结构族、节点数、参数量、FLOPs、深度、变异类型、fitness、cost、timing；若启发式 scorer 与真实评估正相关，再训练简单回归器并只完整评估 top-k |
| P6 | Dynamic Conv2d | 未做 | 有价值但实现复杂，且 CPU-only 下成本可能偏高 | 新增动态 kernel 生成子图或专用 node；先限制小 kernel / 小通道；纳入 FM edge 类型；重点测试前向、反向和 FLOPs 估算 |
| P7 | ENAS 式权重共享 | 未做 | 潜在收益大，但会重塑权重生命周期和 Lamarckian 继承逻辑 | 建子图结构 hash → 权重库；定义共享、失效、迁移和快照规则；先只支持 Linear / Conv2d 同构子图 |
| P8 | 分布式 island model | 未做 | 工程量中等，但当前单机评估还没到必须分布式 | genome + fitness 序列化；多个 island 独立演化；周期性交换 elite；先本机多进程模拟 |
| P9 | DARTS 混合搜索 | 未做 | 几乎是另一套搜索范式，最后考虑 | 新建独立 search engine，不建议塞进现有 mutation / selection 主循环 |

我建议实际开工顺序是：**把 P1.5 从 MNIST 迁移到 Segmentation evolution → 得到稳定排查矩阵 → 再决定 P2 Attention 还是升级完整 P5 Surrogate**。当前 MNIST search 已闭环，剩余最大痛点是 segmentation evolution 的运行成本解释不足。
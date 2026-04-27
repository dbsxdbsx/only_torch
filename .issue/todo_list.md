结论：当前 A / B / C / F 基础设施基本都做完了，剩下主要是阶段 D 的“新任务 / 新算子扩展”和阶段 E 的“搜索效率优化”。我建议第一件先做 **Segmentation 任务支持**，不是先做 Deformable Conv 或 Dynamic Conv。

理由很简单：Segmentation 能直接复用已完成的 ConvTranspose2d、FM 粒度演化、FLOPs 选择、ASHA 等能力，能最快把阶段 C 的空间域基础设施变成一个可验证的新任务闭环；而 Deformable / Dynamic Conv 只是新零件，没任务闭环时收益不容易评估。

| 优先级 | 建议先后 | 当前状态 | 为什么排这里 | 具体怎么做 |
|---|---|---|---|---|
| P0 | 对齐文档与现状 | 小缺口 | 文档有轻微不一致：ASHA 写成三域通用，但代码里序列任务默认跳过；部分旧段落还描述 `Flatten → Linear` 空间种子 | 先更新 `.doc/design/neural_architecture_evolution_design.md`，把 ASHA 改成“非序列默认启用，序列默认跳过 / 可配置”，并统一 `minimal_spatial` 描述 |
| P1 | **Segmentation 任务支持** | 未做 | 最高性价比；能验证 Spatial / FM / ConvTranspose2d 的实际价值，不需要多输出 API | 新增 `Evolution::segmentation(...)` 或显式 `TaskSpec::Segmentation`；支持输出 `[batch, classes, H, W]`；新增 `LossType::Dice` / `Focal`，`TaskMetric::IoU` / `MeanIoU`；新增 spatial-to-spatial minimal genome：`Conv2d → Conv2d(1x1 输出头)`，不走 `Flatten`；补 toy mask / 小图像端到端测试 |
| P2 | Attention / Transformer 纳入演化 | 部分有基础 | `MultiHeadAttention` layer 已存在，但 evolution 还不能搜索它；适合扩展序列任务能力 | 先做 layer-block 级别接入，不急着做单 raw op；新增 `LayerConfig::SelfAttention` / `TransformerBlock`，在 `migration.rs` 展开为现有 MatMul / Softmax / Linear 子图；定义 Sequence 域规则、head 数变异、embed dim 约束和测试 |
| P3 | 多输出 / 多头任务演化 | 底层 IR 部分支持，evolution 未支持 | 这是任务层大改，适合在单输出 segmentation 跑通后做；可解锁 actor-critic、辅助头、多任务学习 | 把 `BuildResult.output: Var` 扩为 `outputs: Vec<Var>` 或兼容包装；genome 显式标记多个 output head；`TaskSpec` 支持每个输出的 target / loss / metric；`FitnessScore` 增加聚合策略；`predict()` 返回多输出结构 |
| P4 | Deformable Conv2d | 未做 | 适合在 segmentation 有基准后做，否则难判断收益 | 新增 `NodeTypeDescriptor::DeformableConv2d`；实现 raw node 前反向；扩展 FM edge 类型；让 `ChangeFMEdgeType` 能在 conv / deformable conv / pool / deconv 间切换；用 segmentation toy benchmark 验证 |
| P5 | Surrogate 模型 | 未做 | 阶段 E 中最现实的提速项，比 ENAS 权重共享风险低 | 记录 `(genome_features → fitness)`；先用手写轻量特征：节点数、参数量、FLOPs、深度、domain、变异类型；训练简单回归器或启发式 scorer；只完整评估 top-k 候选 |
| P6 | Dynamic Conv2d | 未做 | 有价值但实现复杂，且 CPU-only 下成本可能偏高 | 新增动态 kernel 生成子图或专用 node；先限制小 kernel / 小通道；纳入 FM edge 类型；重点测试前向、反向和 FLOPs 估算 |
| P7 | ENAS 式权重共享 | 未做 | 潜在收益大，但会重塑权重生命周期和 Lamarckian 继承逻辑 | 建子图结构 hash → 权重库；定义共享、失效、迁移和快照规则；先只支持 Linear / Conv2d 同构子图 |
| P8 | 分布式 island model | 未做 | 工程量中等，但当前单机评估还没到必须分布式 | genome + fitness 序列化；多个 island 独立演化；周期性交换 elite；先本机多进程模拟 |
| P9 | DARTS 混合搜索 | 未做 | 几乎是另一套搜索范式，最后考虑 | 新建独立 search engine，不建议塞进现有 mutation / selection 主循环 |

我建议实际开工顺序是：**P0 文档对齐半天内做掉 → P1 Segmentation 闭环 → 用它做一个稳定小 benchmark → 再决定 P2 Attention 还是 P5 Surrogate**。如果当前目标是让 evolution 模块更“能用”，先做 P1；如果当前最大痛点是运行太慢，再把 P5 提到 P2。
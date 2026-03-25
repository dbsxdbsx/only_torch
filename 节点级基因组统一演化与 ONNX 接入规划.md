# 节点级基因组统一演化与 ONNX 接入规划
## 问题定义
当前项目的演化系统以层级基因组为核心：`NetworkGenome` 存储 `LayerGene + LayerConfig + SkipEdge`，再由 `src/nn/evolution/builder.rs` 手工展开为计算图。与此同时，项目的保存/可视化/通用模型重建链路已经是节点级 IR：`GraphDescriptor` / `NodeTypeDescriptor` 定义在 `src/nn/descriptor.rs`，并由 `src/nn/graph/descriptor_rebuild.rs` 负责从 descriptor 重建图。这导致演化层和图层存在抽象断层，后续若要支持 ONNX 或细粒度演化，会持续放大维护成本。
## 当前状态
演化主表示位于 `src/nn/evolution/gene.rs (1-1000)`，其中 `LayerConfig` 是最小结构单元，`SkipEdge` 负责额外连接，`resolve_dimensions()`、`is_domain_valid()`、`validate_skip_edge_domains()` 等逻辑都围绕层级结构展开。构图逻辑位于 `src/nn/evolution/builder.rs (1-400)`，它按层类型手工实例化 `Linear`、`Rnn`、`Conv2d` 等模块，再单独处理 skip 聚合。持久化方面，`src/nn/evolution/model_io.rs` 将演化元数据单独保存，而通用图结构保存为 `GraphDescriptor`。另一方面，`src/nn/graph/descriptor_rebuild.rs (1-200)` 已具备从节点级 descriptor 直接恢复图的通用能力，这说明项目实际上已经拥有节点级 IR 基础设施，只是演化系统还没有与之统一。
## 总体架构方向
采用“一套内部表示，两类操作粒度”的方案：
* 内部统一表示改为节点级基因组，直接对齐 `NodeTypeDescriptor`
* 基因组内部表示统一为节点级，默认 `MutationRegistry` 仍然包含层级模板变异（如“插入 Linear”展开为 4 个节点组），层级模板内部操作的是 `NodeGene`，但对用户行为完全透明
* 高层模板变异的注册概率根据任务类型自适应：序列任务中 RNN/LSTM/GRU 模板有较高权重，图像任务中 Conv2d 模板有较高权重，纯 MLP 任务则不注册序列/空间模板
* 纯节点级细粒度变异（单独插入一个 ReLU、BatchNormOp 等原子操作）作为可选扩展，代码骨架预留但不注册进默认 registry，等有具体业务驱动时再开启
* 用户默认 API 与默认体验保持不变，底层表示的变化对用户透明
* 手写固定模型保存的 `.otm` 文件可作为演化种子直接导入，打通“手写训练 ↔ 演化 ↔ 继续训练”全链路
* ONNX 放在内部打通之后再做，不是首要目标
这条路线的核心收益是：演化、序列化、可视化、外部模型导入全部收敛到同一个 IR 层，不再维护两套本体模型。框架内部的三角形互通（手写训练模型 ↔ 演化 ↔ 继续手写训练）是比 ONNX 更优先的目标。请再尝试补充进去，再试一次。
## 分阶段实施方案
### 阶段 1：引入节点级基因组，但先不移除旧层级系统
第一阶段不要直接替换现有 `LayerGene` 体系，而是先在 `src/nn/evolution/gene.rs` 中新增节点级表示，并允许它与旧表示短期并存。新增的核心类型建议包括：
* `NodeGene`：持有 `innovation_number`、`node_type: NodeTypeDescriptor`、`parents: Vec<u64>`、`enabled`、`block_id: Option<u64>`（见下）和参数形状信息
* `block_id` 设计：把模板组信息直接内联到 `NodeGene` 上，而不是维护独立的 `TemplateGroup` 列表。模板展开的一组节点共享同一个 `block_id`，细粒度单节点变异插入的节点 `block_id = None`。这不仅便于 Grow/Shrink/Remove 以组为单位操作，也为将来实现 NEAT 交叉（crossover）打基础：交叉时以 block 为单位对齐，模板展开的节点组要么整体来自父本 A，要么整体来自父本 B，不会被拆散
* `GenomeKind` 或等价内部状态：用于区分当前基因组是否仍处于旧层级表示，还是已经迁移到节点级表示
第一阶段的目标不是立刻切换主流程，而是先把“节点级基因组的静态数据结构”和“统一静态分析层”做出来，具体包括：拓扶排序、形状推导、域校验、参数统计、创新号分配策略。关键设计决策：将现有分散在 `gene.rs`、`builder.rs`、`mutation.rs` 中的形状推导、域推导、skip 合法性、参数量估算等逻辑统一收口到一个命名结果结构（例如 `GenomeAnalysis`），让 mutation、builder、serializer 三个下游模块全部依赖这一个分析结果，而不是各自重复实现分析逻辑。这个命名结构就是整个重构的 IR 层界面定义，它不存数据，只表达对当前基因组的分析结论。
### 阶段 2：建立旧层级基因组到节点级基因组的单向迁移器
在 `src/nn/evolution/gene.rs` 或新建 `src/nn/evolution/migration.rs` 中实现 `LayerConfig/SkipEdge -> NodeGene` 的展开器。这个迁移器是整个重构的枢纽，因为它既服务于旧模型兼容，也服务于重构期间的灰度切换。建议先只支持当前演化系统已稳定支持的高层单元：
* `Linear` 展开为参数节点 + `MatMul` + 偏置加法节点
* `Activation` 直接映射为单个激活节点
* `Conv2d`、`Pool2d`、`Flatten` 映射到底层节点
* `Rnn/Lstm/Gru` 先作为“复合模板节点组”处理，不要求第一版就拆成完全原子级 recurrence 图
这个阶段非常关键的一点是：对于循环层不必一上来追求彻底原子化。如果现在项目里的 `Rnn/Lstm/Gru` 在图构建层仍然主要通过高层模块封装，那么可以先把它们当作“节点级基因组中的复合节点模板”，让整体架构先统一，再逐步细化循环结构的内部展开。这样能显著降低首轮重构风险。
### 阶段 3：把构图入口改成“基因组 -> GraphDescriptor -> Graph”
一旦节点级基因组可以稳定表达现有网络，就应重写 `src/nn/evolution/builder.rs` 的主路径，让它不再直接按 `LayerConfig` 手写构图，而是统一走：
`NetworkGenome -> GraphDescriptor -> Graph::from_descriptor()`
这里建议新增 `NetworkGenome::to_graph_descriptor()`，作为唯一合法的 genome 到图 IR 的转换入口。这样有几个直接收益：
* `builder.rs` 将从一大段手写图构造逻辑，收敛成更薄的一层包装
* 演化构图与通用模型加载共用同一条图恢复链路，减少偏差
* 后续 ONNX 导入只要产出合法 `NetworkGenome`，就天然可训练、可保存、可可视化
完成这一步后，`src/nn/evolution/builder.rs` 的复杂度会显著下降，维护重点从“怎么构图”转向“怎么生成合法节点基因组”。这是整个重构最重要的可维护性收益。
**阶段 3 的强验收项**：完成 `to_graph_descriptor()` 后，必须验证以下闭环稳定成立：手写建立的 `.otm` 模型 → 读取为 `GraphDescriptor` → 转换为 `NetworkGenome` → 执行一次 mutation → 再生成 `GraphDescriptor` → 再重建 `Graph`。这个闭环最简洁，也是内部 IR 统一是否真正成功的最直接证明。如果这个闭环不通，后续 ONNX 导入也只是把复杂度导入进来。
### 阶段 4：把层级变异改写为模板化节点变异，同时简化架构摘要
在 `src/nn/evolution/mutation.rs (1-500)` 现有框架基础上，保留 `Mutation trait` 与 `MutationRegistry`，但将具体变异的作用对象改成节点级基因组。重构方式建议是：
* 保留默认注册表语义，让默认 registry 仍然注册“层级模板变异”
* 每种旧层变异内部不再直接修改 `LayerGene`，而是调用 `LayerTemplate` 展开器，在节点级基因组中插入/删除/替换一组节点
* 新增可选的 fine-grained mutations，例如插入单个激活、插入归一化组、替换某个单一节点类型、在两个节点间插入 `Add/Concat`
这里建议显式引入 `LayerTemplateKind`，把今天的 `LayerConfig` 角色转化为“模板定义”，而不是“存储格式”。从架构上讲，这相当于把高层概念从数据模型中剥离，转移到操作层。
架构摘要策略同步调整：彻底移除现有的层级字符串摘要（如 `Linear(4) → ReLU → [Linear(1)]`）。维护从节点到层名的反向映射代价会随着节点级演化的深化持续增大，且在细粒度变异插入单个节点的场景下并不能准确表达网络结构。替换方案是演化日志仅输出节点数量、启用节点数和参数节点数，例如 `nodes=12 active=9 params=4`，以节点计数作为统一复杂度指标。这对高层模板变异和细粒度节点变异同样适用，不需要维护任何人类可读的层级名称映射。
### 阶段 5：迁移权重继承与参数快照机制
当前权重继承逻辑依赖 `layer_params: HashMap<u64, Vec<Var>>` 和 `weight_snapshots: HashMap<u64, Vec<Tensor>>`，位置在 `src/nn/evolution/builder.rs (1-400)` 与 `src/nn/evolution/gene.rs (200-500)`。节点级重构后，这套机制应下沉到 Parameter 节点粒度。建议改为：
* 以 Parameter 节点的 innovation number 作为最小权重继承单位
* 快照结构改为 `HashMap<u64, Tensor>` 或保留 `Vec<Tensor>` 但语义严格限定为参数节点集合
* 高层模板的 Lamarckian 继承不再依赖“某层有几个参数”这种隐式约定，而依赖模板组中哪些 Parameter 节点被保留、哪些形状变化
这一步做完后，Grow/Shrink/Replace 的权重继承将更精确，也更适合后续 ONNX 导入模型继续演化。
### 阶段 6：升级持久化格式，提供旧格式迁移
`src/nn/evolution/model_io.rs` 当前保存的是 v2 风格演化元数据，其中 `GenomeSerialized` 仍然基于 `layers + skip_edges`。这一层需要升级，但必须兼容旧文件。建议：
* 定义新的 genome serialized 结构，字段改为 `nodes + template_groups + training_config + meta`
* 加载时根据 metadata 或 JSON 字段判断是旧格式还是新格式
* 若是旧格式，则先按旧结构读取，再调用阶段 2 的迁移器转成新节点级基因组
这样可以保证已保存的 `.otm` 演化模型不失效，也避免用户因为内部重构丢掉历史结果。
### 阶段 7（延后）：增加 ONNX 导入模块
等前 6 个阶段完成、框架内部三角形互通验证后，再引入 ONNX。模块应放在图模块下，例如 `src/nn/graph/onnx_import.rs`，而非演化模块下——因为 ONNX 导入的第一个产物是 `GraphDescriptor`，再由 `NetworkGenome::from_graph_descriptor()` 转一步进入演化系统。并通过 Cargo feature 隔离依赖，避免影响默认编译体验。ONNX 导入流程建议拆成四层：
1. protobuf 解析层：读 `ModelProto/GraphProto`
2. 符号表层：建立 tensor name 到 gene/node 的映射
3. 算子映射层：把 ONNX `op_type` 映射到 `NodeTypeDescriptor`
4. Genome 装配层：构造合法的 `NetworkGenome`，同时填充参数快照
第一版不要追求支持所有 ONNX 算子，而是先支持与当前框架能力重叠最大的核心子集：`MatMul/Gemm`、`Add`、`Relu`、`Sigmoid`、`Tanh`、`Conv`、`Flatten`、`Concat`、`BatchNormalization`、`Softmax`。对于不支持的算子，要明确报错并返回 unsupported op 信息，而不是静默忽略。
## 测试策略
**`src/nn/tests/`（节点和图层计算测试）完全不动**。这些测试测的是节点前向/反向计算正确性，和演化模块重构无关。`tests/convergence.rs`、`tests/selection.rs`、`tests/task.rs`、`tests/evolution.rs`（集成测试）几乎不动，因为它们测的是公开 API 行为和每代演化结果的合理性，与内部基因组表示无关。需要更新的测试按阶段分布如下：
阶段 1 对应测试：新增 `NodeGene` 数据结构单元测试 + 60+ 种 `NodeTypeDescriptor` 的 shape inference 覆盖测试（逐类型验证输入形状 → 输出形状规则）。这是整个重构唯一需要大批量新增的测试，但每条很简单，且层级节点已在 `src/nn/tests/node_*.rs` 中隐式验证过计算正确性。
阶段 2 对应测试：新增迁移器测试，验证每种 `LayerConfig` 变体展开后的节点组结构正确性。
阶段 3 对应测试：更新 `tests/builder.rs`，将原有构图测试改为验证 `to_graph_descriptor()` + `Graph::from_descriptor()` 的等价性（即展开后再重建的图和原图等价）。
阶段 4 对应测试：更新 `tests/mutation.rs`，将原有变异测试改为验证模板展开后节点组的合法性；`tests/gene.rs` 根据新 `NodeGene` 结构大部分重写。
阶段 5 对应测试：更新 `tests/builder.rs` 中的权重继承测试，验证 Parameter 节点粒度的 snapshot/restore 正确性。
阶段 6 对应测试：更新 `tests/model_io.rs`，验证新格式保存/加载 + 旧格式自动迁移。
## 推荐的具体落地顺序
建议按下面顺序推进，而不是并行大拆：
1. 为节点级基因组建立最小数据结构、shape inference、合法性校验
2. 写旧层级基因组到节点级基因组的迁移器
3. 用迁移后的节点级表示生成 `GraphDescriptor`
4. 将 `builder.rs` 改为通过 `Graph::from_descriptor()` 构图
5. 把默认层级变异改为模板化节点插入
6. 迁移权重继承与 `.otm` 序列化
7. 最后做 ONNX 导入
这样安排的好处是，每一步都能单独验收，而且第 4 步完成时系统已经拿到最大收益：内部表示统一，后续 ONNX 和 fine-grained evolution 都只是“新增能力”，不再是“推翻主干”。
## 风险点与规避策略
### RNN/LSTM/GRU 的底层表示复杂度最高
循环结构是这次重构里最容易失控的部分。建议第一阶段先允许它们在节点级基因组里以“复合模板组”存在，不强制立即展开成完整时间递归图。否则你会在表示层、shape inference、mutation、weight inheritance 四个地方同时被放大复杂度。
### 形状推导会替代大量今天隐含在 builder 里的逻辑
现在很多维度规则、序列返回规则、空间域规则散落在 `gene.rs` 和 `builder.rs`。节点级化之后，必须建立统一的 shape/domain inference 层，否则 mutation、导入、保存、重建会各写一套规则，维护成本更高而不是更低。
### 不要同时维护两套长期并存的正式演化系统
短期迁移期可以并存，但目标必须明确：最终只有一套正式基因组表示。否则测试矩阵、保存格式、变异注册表、权重继承都将出现双倍维护负担，这正是这次架构升级最需要避免的结果。
## 验收标准（阶段 1-6 完成后）
完成这套改造后，应至少满足以下能力：
* 默认 `Evolution::supervised(...).run()` 的用户体验不变
* 内部演化在节点级进行，高层模板变异按任务类型自适应注册
* 可通过额外配置开启或关闭细粒度节点变异
* 训练日志不再依赖层级字符串摘要，而统一输出节点数量、启用节点数量和参数节点数量
* 现有演化 examples 继续可运行
* 旧 `.otm` 演化文件可加载（旧格式自动迁移）
* 节点级基因组可以直接生成 `GraphDescriptor` 并构图
* 手写固定模型保存的 `.otm` 文件可作为演化种子导入，并继续演化
* 演化结果可保存后继续作为手写训练的起点，打通“手写训练 ↔ 演化 ↔ 继续训练”三角形互通
## 我对这个项目的建议决策
对于你这个项目，最正确的做法不是“先把 ONNX 做出来”，也不是“先把所有层都拆成最底层原子图”，而是先完成内部表示统一。只要 `NetworkGenome` 真正和 `NodeTypeDescriptor` 对齐，ONNX、自定义模板、手工 seed 模型、细粒度变异都会自然落位。反过来，如果不先统一表示，任何一个功能都会变成一次额外的兼容层工程。
因此我建议你把这次工作定义为一次“演化内核重构”，而不是“新增 ONNX 支持”或“新增细粒度变异”。ONNX 和细粒度演化都应该被视为这次内核重构完成后的直接产物。
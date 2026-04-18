# 节点级基因组统一演化与 ONNX 接入规划

## 问题定义
当前项目的演化系统仍以层级基因组为历史核心：`NetworkGenome` 维护 `LayerGene + LayerConfig + SkipEdge`，再由 `src/nn/evolution/builder.rs` 逐层展开为图。而框架内部真正通用、可保存、可可视化、可重建的 IR 已经是节点级：`GraphDescriptor` / `NodeTypeDescriptor` 定义在 `src/nn/descriptor.rs`，并由 `src/nn/graph/descriptor_rebuild.rs` 负责重建计算图。

这导致项目在“演化内核”和“图内核”之间长期维护两套本体模型：

* 演化系统理解的是层、skip、模板、主路径
* 图系统理解的是节点、父边、张量形状、拓扑序

如果继续在这两个世界之间叠加新功能，无论是更细粒度的变异、手写模型导入、继续训练、还是未来 ONNX 接入，都会不断放大复杂度和维护成本。

## 当前状态
当前代码已经取得了非常关键的进展：

* 非序列任务默认已经迁移到 NodeLevel 运行，Flat/Spatial 主路径基本收口
* `GenomeAnalysis` 已作为节点级静态分析快照存在
* `NetworkGenome -> GraphDescriptor -> Graph::from_descriptor()` 已成为可运行主路径
* Flat/Spatial `.otm` 持久化主格式已向 NodeLevel 收敛
* 手写固定模型 `.otm` 与演化系统之间已经打通基本闭环
* LayerLevel `SkipEdge` 已不再是 Flat/Spatial 默认主路径中的核心表达

但距离“彻底完成节点级统一内核”仍有几类明显缺口：

* NodeLevel 仍缺少真正的通用连边变异，无法自然长出 ResNet/DenseNet 式绕连
* RNN/LSTM/GRU 等记忆单元尚未节点级化，序列路径仍依赖 LayerLevel
* LayerLevel 目前仍兼具“历史内核表示”和“用户入口层”两种角色，边界不够清晰
* ONNX 仍然只能排在内核统一完成之后，否则只是把复杂度继续导入系统

## 终态设计目标
本次规划的最终目标不是“保留两套表示并尽量兼容”，而是明确收口为下面的架构：

### 一、内核只保留 NodeLevel
演化系统内部唯一正式基因组表示应是 NodeLevel，直接对齐 `NodeTypeDescriptor`。所有默认演化、持久化、构图、日志、可视化、导入导出都围绕 NodeLevel 展开。

### 二、LayerLevel 只允许作为用户入口 DSL 或模板规范存在
未来可以保留“层”的概念，但它不再是内核表示，而只承担以下角色：

* 用户手动写 seed 模型时的高层模板语言
* 传统层式网络导入时的前端适配语法
* `LayerTemplateSpec` / `ModelSeedSpec` / `HighLevelTemplate` 这类用户侧构造器

也就是说，LayerLevel 可以作为“输入语言”保留，但不应该继续作为“内部世界模型”保留。

### 三、SkipEdge 彻底降解为普通节点图连接
最终系统里不再单独维护 `SkipEdge` 概念。任何所谓 skip、绕连、残差、dense 连接，本质上都只是 NodeLevel DAG 中的普通前向连接，必要时配合：

* 聚合节点：`Add` / `Concat` / `Maximum` / `Stack+Mean`
* 投影节点：Flat 域 `Linear`、Spatial 域 `1x1 Conv2d`

### 四、记忆单元也必须进入 NodeLevel 本体
如果序列路径长期保留在 LayerLevel，那么系统就永远无法真正完成统一内核。因此 RNN/LSTM/GRU 的节点级化不是可选优化，而是删除 LayerLevel 内核地位的前提条件。

### 五、ONNX 是内核统一后的自然产物，而不是先导目标
只有在 NodeLevel 真正成为统一内核后，ONNX 导入才是一个清晰、低耦合的增量功能。否则它只会迫使系统再维护第三套兼容语义。

## 总体架构分层
为了避免后续继续混淆“用户入口”和“内核本体”，建议明确采用四层结构：

### 1. 用户入口层
面向用户的构造方式，可以是：

* 手写层式模型
* `LayerTemplateSpec`
* 手写 `.otm` 模型
* 未来 ONNX 文件

这一层允许继续使用“层”这个概念，因为它是用户熟悉的输入语言。

### 2. 降级/翻译层
负责把用户入口统一转成 NodeLevel 内核可理解的形式，典型入口包括：

* `LayerLevel -> NodeLevel` 迁移器
* `GraphDescriptor -> NetworkGenome::from_graph_descriptor()`
* 未来 `ONNX -> GraphDescriptor -> NetworkGenome`

### 3. NodeLevel 演化内核层
这是系统的唯一正式本体层，负责：

* 拓扑表达
* 形状/域分析
* 默认模板变异
* 细粒度节点变异
* 通用连边变异
* 权重继承与快照
* 持久化主格式

### 4. 图执行层
统一走：

`NetworkGenome -> GraphDescriptor -> Graph::from_descriptor()`

这层只负责按蓝图施工，不再承担高层架构推导职责。

## 分阶段实施方案

### 阶段 1：建立 NodeLevel 数据结构与统一分析层
这一阶段的目标是建立 NodeLevel 的静态数据模型与统一分析接口，但暂不删除旧 LayerLevel 系统。

核心工作：

* 引入 `NodeGene`
* 引入 `GenomeKind` 或等价状态，允许迁移期双表示并存
* 把 `block_id` 直接内联到节点上，用于模板组操作与未来 crossover 对齐
* 建立统一的 `GenomeAnalysis`

`GenomeAnalysis` 需要统一承担：

* 拓扑排序
* 环检测
* 形状推导
* 域推导
* 参数统计
* 输出可达性检查
* 连接合法性检查

硬约束：

* `GenomeAnalysis` 必须是不可变快照
* 外部语义必须是 `genome.analyze() -> GenomeAnalysis`
* 任何 mutation、migration、builder 修正之后都必须重新分析，不允许共享可变 analysis 状态

### 阶段 2：建立 LayerLevel / 手写图到 NodeLevel 的统一降级器
这一阶段的目标是让所有历史入口都能单向降级到 NodeLevel。

需要支持的来源：

* `LayerConfig/SkipEdge -> NodeGene`
* `GraphDescriptor -> NetworkGenome::from_graph_descriptor()`
* 手写固定模型 `.otm` -> `GraphDescriptor` -> `NetworkGenome`

高层模板初版支持：

* `Linear`
* `Activation`
* `Conv2d`
* `Pool2d`
* `Flatten`
* `Dropout`
* `Rnn/Lstm/Gru` 暂作为复合模板节点组存在，不要求第一版即完全原子化

这一阶段的关键目标不是删旧代码，而是确保所有旧入口都能安全降级到 NodeLevel。

### 阶段 3：统一构图主路径到 GraphDescriptor
一旦 NodeLevel 可以稳定表达现有网络，`builder.rs` 主路径就应收口为：

`NetworkGenome -> GraphDescriptor -> Graph::from_descriptor()`

需要新增并稳定以下能力：

* `NetworkGenome::to_graph_descriptor()` 成为唯一合法的 genome 到图 IR 转换入口
* `Graph::from_descriptor()` 不再额外承担高层行为推导
* 今天 builder 中所有隐含决策都要下沉为 descriptor 可显式表达的信息

必须显式固化的行为包括但不限于：

* 序列模块的 `return_sequences`
* Pool2d 在 kernel 超出空间尺寸时的退化策略
* 动态 batch 维规则
* 输入/输出节点命名与参数节点映射方式

强验收项：

* 手写 `.otm` 模型 -> `GraphDescriptor` -> `NetworkGenome` -> mutation -> `GraphDescriptor` -> `Graph` 闭环稳定成立

### 阶段 4：把默认层级变异重写为 NodeLevel 模板变异
这一阶段的目标是让默认演化从“修改 LayerGene”转向“修改 NodeGene 模板组”，但保留用户默认体验不变。

核心原则：

* 默认 registry 仍然对用户表现为“插入 Linear / Conv / Activation / Pool”这类高层动作
* 但内核实现改成对 NodeLevel 的模板组插入、删除、替换、扩缩

建议引入：

* `LayerTemplateKind`
* `TemplateExpander`
* `TemplateMutationContext`

这一层实际上把今天 `LayerConfig` 的角色从“存储格式”转成“操作模板”。

日志与复杂度摘要同步收口：

* 不再维护层级字符串摘要
* 统一输出 `nodes=... active=... params=...`

### 阶段 5：把权重继承与参数快照收口到 Parameter 节点粒度
NodeLevel 重构后，Lamarckian 继承与快照应下沉到 Parameter 节点粒度，而不是继续依赖“某层有几个参数”这种隐式约定。

建议收口为：

* `HashMap<u64, Tensor>`，键为 Parameter 节点 innovation number
* Grow/Shrink/Replace 的继承逻辑以 Parameter 节点是否保留、是否 shape compatible 为判断依据
* 模板组只作为批量操作边界，不再作为权重继承的真实单位

这一阶段完成后，Flat/Spatial 演化、手写模型 seed、未来 ONNX seed 都能共用一套参数快照语义。

### 阶段 6：完成 Flat/Spatial 持久化主格式的 NodeLevel 收口
这一步的目标是把 Flat/Spatial 两类演化模型的保存/加载语义彻底固定到 NodeLevel 上。

明确原则：

* Flat/Spatial 的主格式只有 NodeLevel
* 旧 LayerLevel 演化文件不再作为长期正式兼容目标
* 开发期允许放弃历史旧文件自动兼容，以减少维护负担
* 序列模式暂时保留现状，等记忆单元节点级化后再统一

`model_io.rs` 的定位应改成：

* NodeLevel Flat/Spatial genome 的单一、稳定、可验证的保存/加载实现
* 而不是 LayerLevel/NodeLevel 双格式兼容枢纽

### 阶段 7：为 NodeLevel 增加通用跨层连接变异，彻底吸收 SkipEdge 语义
这一阶段的目标不是继续维护 `SkipEdge`，而是让 NodeLevel 真正具备“图演化”能力。

当前 NodeLevel 已经具备：

* 节点级父边表达
* DAG 构图能力
* 聚合节点类型
* 形状/域分析基础

但当前仍缺少：

* 真正的节点级连边变异
* 连边后的自动聚合逻辑
* 形状不兼容时的自动投影逻辑

建议新增：

* `AddConnection`
* `RemoveConnection`

并配套新增规则：

* 新增连接必须满足拓扑序，不能成环
* 如果目标已有主输入，则自动插入聚合节点
* 如果形状不兼容，则自动插入投影节点
* Flat 域优先用 `Linear` 投影
* Spatial 域优先用 `1x1 Conv2d` 投影
* 所有连接合法性统一由 `GenomeAnalysis` 校验

完成这一阶段之后：

* `SkipEdge` 不再是独立概念
* ResNet、DenseNet 以及其他任意合法前向 DAG 连接都由 NodeLevel 普通图连边生成
* LayerLevel SkipEdge 进入明确弃用态

### 阶段 8：完成记忆单元的 NodeLevel 化，收口序列路径
这一阶段是彻底删除 LayerLevel 内核地位的决定性阶段。

必须完成的工作：

* 为 RNN/LSTM/GRU 定义 NodeLevel 表达
* 明确 State/Input/Hidden/Cell 等内部节点的边界
* 给 recurrence 图建立统一的 shape / domain / temporal legality 规则
* 支持序列路径的 `to_graph_descriptor()` 与 `Graph::from_descriptor()` 闭环
* 让序列模式默认也运行在 NodeLevel，而不是继续依赖 LayerLevel

这里不要求第一版就把循环展开成“时间步完全展开图”，但至少要做到：

* 记忆单元在内核语义上属于 NodeLevel，而不再属于 LayerLevel 复合层

这一步完成前，LayerLevel 还不能从内核中彻底删除。

### 阶段 9：把 LayerLevel 从内核表示降级为用户入口 DSL
这一阶段的目标不是“完全不允许层”，而是把层从内核表示中清除出去。

最终保留方式应类似：

* `LayerTemplateSpec`
* `ModelSeedSpec`
* `HighLevelTemplate`
* 用户侧辅助 builder

但以下内容应从正式内核中移除：

* `LayerGene` 作为正式基因组本体的地位
* `SkipEdge` 作为正式连接本体的地位
* LayerLevel builder 主路径
* LayerLevel 持久化主路径
* LayerLevel 默认 mutation 主路径
**阶段 9 附加任务：NodeLevel 构图的可视化 Cluster 保障**

问题背景：
* `build_layer_level()` 通过 RAII `NodeGroupContext` 自动为节点打 `NodeGroupTag`（Layer / Recurrent cluster）
* `build_from_nodes()` 经 `to_graph_descriptor()` → `rebuild_into()` 逐节点重建，无上下文，cluster 标签丢失
* `NodeDescriptor` 中无 `block_id` 字段，信息未被传递给重建图

解决方案：
* 在 `build_from_nodes()` 构图完成后，调用 `backfill_node_group_tags(genome, &rebuild.node_map)`
* 利用 `node_main_path()` 返回的 `NodeBlock::block_id`，对同块所有节点（含 `Parameter`）补填 `NodeGroupTag`
* `NodeBlockKind` → `GroupStyle` 映射：`Linear/Conv2d/Pool2d/Flatten/Dropout` → `Layer`；`Rnn/Lstm/Gru` → `Recurrent`；`SkipAgg/Unknown` → 跳过（不打 cluster 标签）

效果：
* 演化变异（InsertLayerMutation 等）插入的层有 `block_id` → 展示 Cluster ✅
* 从 LayerLevel 迁移来的初始基因组节点有 `block_id` → 展示 Cluster ✅
* `build_layer_level()` 遗留路径本来就有 `NodeGroupContext` → 展示 Cluster ✅
* 散节点（单独激活等，`block_id = None`）→ 不展示 cluster，符合预期 ✅
这一阶段完成后，LayerLevel 只剩“输入语言”意义，不再是“内部世界模型”。

### 阶段 10（延后）：增加 ONNX 导入模块
等前 1-9 阶段完成、内核真正统一后，再做 ONNX。模块应放在图模块下，例如 `src/nn/graph/onnx_import.rs`，而不是演化模块下。

推荐流程拆成四层：

1. protobuf 解析层：读取 `ModelProto/GraphProto`
2. 符号表层：建立 tensor name 到 node/gene 的映射
3. 算子映射层：把 ONNX `op_type` 映射到 `NodeTypeDescriptor`
4. Genome 装配层：构造合法的 `NetworkGenome` 并填充参数快照

第一版只支持与框架现有能力重叠最大的核心子集：

* `MatMul/Gemm`
* `Add`
* `Relu`
* `Sigmoid`
* `Tanh`
* `Conv`
* `Flatten`
* `Concat`
* `BatchNormalization`
* `Softmax`

对不支持算子必须明确报错，不允许静默忽略。

## 测试策略
`src/nn/tests/` 中关于节点前向/反向正确性的测试原则上不动，因为它们测试的是算子本身，不是演化内核重构。

需要跟随阶段推进的测试重点如下：

### 阶段 1
* `NodeGene` 数据结构测试
* 全量 `NodeTypeDescriptor` 的 shape inference 测试
* `GenomeAnalysis` 不可变快照测试

### 阶段 2
* `LayerConfig -> NodeGene` 展开测试
* `GraphDescriptor -> NetworkGenome` 导入测试
* 旧模型与新 seed 导入一致性测试

### 阶段 3
* `to_graph_descriptor()` 与 `Graph::from_descriptor()` 等价性测试
* `return_sequences` 一致性测试
* Pool2d 退化策略一致性测试
* 手写 `.otm` -> 演化 -> 再构图闭环测试

### 阶段 4
* NodeLevel 模板变异测试
* 架构摘要由层级字符串切换为节点计数后的日志测试
* 旧 mutation 测试按模板组语义重写

### 阶段 5
* Parameter 节点粒度 snapshot/restore 测试
* Grow/Shrink/Replace 后部分继承测试
* Flat / Spatial 模板的权重继承一致性测试

### 阶段 6
* Flat/Spatial NodeLevel `.otm` 保存/加载往返一致性
* 手写模型 `.otm` 导入为演化种子测试
* Flat/Spatial 新格式拒绝旧 LayerLevel 主格式的测试
* 序列路径暂不统一时不被误伤的回归测试

### 阶段 7
* `AddConnection/RemoveConnection` 的 DAG 合法性测试
* 聚合节点自动插入测试
* 投影节点自动插入测试
* ResNet 式单绕连 NodeLevel 构图/训练/保存测试
* DenseNet 式多源连接 NodeLevel 构图/训练/保存测试
* 原有 LayerLevel SkipEdge 测试逐步收缩为迁移兼容测试，并最终删除

### 阶段 8
* RNN/LSTM/GRU 的 NodeLevel 表达合法性测试
* 序列模式 `to_graph_descriptor()` / rebuild 闭环测试
* 序列路径默认 NodeLevel 化后的训练/保存/加载测试

### 阶段 9
* 用户 DSL / 模板入口降级到 NodeLevel 的测试
* LayerLevel 内核路径删除后的兼容入口测试
* 不再允许直接依赖 LayerLevel 正式持久化/构图主路径的回归测试
* `build_from_nodes()` 后 Linear 块参数节点含 `GroupStyle::Layer` NodeGroupTag 的测试
* `build_from_nodes()` 后 RNN/LSTM/GRU 块参数节点含 `GroupStyle::Recurrent` NodeGroupTag 的测试

### 阶段 10
* ONNX 最小模型导入测试
* ONNX -> `GraphDescriptor` -> `NetworkGenome` -> build 闭环测试
* unsupported op 明确报错测试

## 推荐的具体落地顺序
建议按下面顺序推进，而不是并行大拆：

1. 建立 NodeLevel 最小数据结构与统一分析层
2. 打通所有旧入口到 NodeLevel 的单向降级器
3. 把构图主路径收口到 `GraphDescriptor`
4. 把默认层级变异改成 NodeLevel 模板变异
5. 把权重继承下沉到 Parameter 节点粒度
6. 先完成 Flat/Spatial 持久化主格式收口
7. 再为 NodeLevel 增加通用连边变异，彻底吸收 SkipEdge
8. 然后完成记忆单元节点级化，让序列路径也迁移到 NodeLevel
9. 最后把 LayerLevel 从内核降级成用户入口 DSL
10. 等内核完全统一后，再做 ONNX

这样安排的好处是：

* 每一步都可以单独验收
* 前 6 步先稳住 Flat/Spatial 主路径
* 第 7 步解决真正的“图演化”能力
* 第 8 步解决最后一个阻碍 LayerLevel 删除的核心缺口
* 第 9 步才真正完成内核收口
* 第 10 步的 ONNX 是顺势接入，而不是再次重构主干

## 风险点与规避策略

### 一、记忆单元节点级化是最大技术难点
RNN/LSTM/GRU 的节点级表达、状态语义与形状推导会同时触发表达层、分析层、构图层和变异层复杂度上涨。规避策略是：

* 先让它们成为 NodeLevel 复合模板组
* 再逐步细化 recurrence 内部结构
* 不要在同一轮里同时追求“完全原子图”和“默认可演化”

### 二、连边变异很容易破坏图合法性
NodeLevel 增加通用连边能力后，最容易失控的是：

* 环路
* 形状不兼容
* 域错配
* 输出不可达

规避策略是：

* 所有连边都必须通过 `GenomeAnalysis`
* 把投影/聚合自动化，避免调用方自行补图
* 只允许前向连接，不允许隐式回边

### 三、不要把 LayerLevel 的“用户入口价值”和“内核表示价值”混为一谈
层式写法对用户依然有价值，但这不意味着 LayerLevel 还应继续存在于内核中。规避策略是：

* 尽早在文档和代码中明确：LayerLevel 只保留为用户入口 DSL
* 不再把它用于默认持久化、默认 mutation、默认构图主路径

### 四、ONNX 不能过早接入
如果在序列路径仍未节点级化、LayerLevel 仍未降级前接入 ONNX，系统会被迫维护更多入口兼容。规避策略是：

* 明确把 ONNX 后置到内核统一之后
* 让 ONNX 导入只需要关心 `GraphDescriptor` 和 `NodeTypeDescriptor`

## 验收标准（阶段 1-9 完成后）
在不考虑 ONNX 的前提下，演化内核统一完成时，应至少满足：

* 默认 `Evolution::supervised(...).run()` 用户体验不变
* 内部正式表示只有 NodeLevel
* Flat / Spatial / Sequential 三类任务都运行在 NodeLevel 内核上
* LayerLevel 不再作为默认构图、默认持久化、默认 mutation 的正式内核表示
* LayerLevel 最多只保留为用户入口 DSL / 模板规范 / seed 适配器
* NodeLevel 可以表达并生成任意合法前向跨层连接
* `SkipEdge` 不再是独立概念
* ResNet、DenseNet 及其他 DAG 式绕连拓扑都可由 NodeLevel 自然表达和演化
* 节点级基因组可以直接生成 `GraphDescriptor` 并稳定构图
* Flat/Spatial/Sequential 演化结果都可以保存、加载、继续演化
* 手写固定模型 `.otm` 文件可作为演化种子导入，并继续演化
* 演化结果可保存后继续作为手写训练的起点，打通“手写训练 ↔ 演化 ↔ 继续训练”三角形互通

## 最终验收标准（阶段 10 完成后）
在完成 ONNX 后，额外应满足：

* ONNX 可导入为 `GraphDescriptor`
* ONNX 导入模型可转为 `NetworkGenome`
* ONNX 导入模型可训练、可保存、可继续演化
* unsupported op 会明确报错，不会静默失败

## 我对这个项目的建议决策
对于这个项目，最正确的做法不是“优先把 ONNX 做出来”，也不是“继续长期保留 LayerLevel 作为双轨正式系统”，而是先完成演化内核统一。

真正正确的战略定义应该是：

* 这是一场 **演化内核重构**
* ONNX、细粒度变异、手写模型 seed、用户层式 DSL 都只是这次重构完成后的自然产物

只要 `NetworkGenome` 真正与 `NodeTypeDescriptor` 对齐，只要记忆单元和跨层连接都完成 NodeLevel 化，系统就会得到一个真正完备、灵活、可扩展的演化内核。到了那个时候，LayerLevel 才可以从内核中退出，而不再成为后续每一项新功能的兼容负担。
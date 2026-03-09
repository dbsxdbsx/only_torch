# 更新日志

## [0.14.0] - 2026-03-09

### 新增

- **feat(evolution): 神经架构演化模块 MVP（核心特色功能）**
  - 完整的 Genome-centric 层级演化系统，用户只需提供数据和目标——零模型代码
  - `gene.rs`: 基因数据结构（`NetworkGenome`、`LayerGene`、`SkipEdge`、`TrainingConfig` 等）
  - `mutation.rs`: `Mutation` trait + `MutationRegistry`，内置 12 种变异操作
    - 结构变异：`InsertLayer`、`RemoveLayer`、`AddSkipEdge`、`RemoveSkipEdge`
    - 参数变异：`GrowHiddenSize`、`ShrinkHiddenSize`、`ChangeActivation`
    - 训练超参数变异：`MutateLearningRate`、`MutateOptimizer`、`MutateBatchSize`、`MutateLossFunction`
    - 聚合变异：`ChangeAggregateStrategy`
  - `builder.rs`: Genome → Graph 自动转换 + Lamarckian 权重继承（跨代权重复用）
  - `convergence.rs`: `ConvergenceDetector` 训练收敛检测（loss plateau + gradient norm 双判据）
  - `task.rs`: `EvolutionTask` trait + `SupervisedTask` 监督学习实现，支持 full-batch / mini-batch
  - `callback.rs`: 回调接口（`EvolutionCallback` + `DefaultCallback`），支持自定义日志/停止策略
  - `Evolution` 主控结构体：Builder 模式 API，`run()` 驱动完整演化主循环
  - `EvolutionResult`：`predict()` 推理 + `visualize()` 计算图可视化
  - `SkipEdge` DAG 拓扑演化：支持 `Add`/`Concat`/`Mean`/`Max` 四种聚合策略
  - `NetworkGenome` Display：主路径摘要 + skip edge 注解 + 重名层自动消歧
  - 停滞探测机制：连续 N 代 primary 未提升后强制结构变异
  - 完整的单元测试和集成测试覆盖

- **feat(evolution): 新增 2 个演化示例**
  - `evolution_xor`: XOR 零模型代码演化，从 `Input(2) → [Linear(1)]` 自动发现解决方案
  - `evolution_iris`: Iris 鸢尾花演化，150 样本自动 mini-batch + CrossEntropy 推断

- **feat(examples): 新增中国象棋棋子 CNN 分类器示例**
  - 15 类分类（空位 + 红方 7 子 + 黑方 7 子），28x28 合成 patch
  - Conv(3→16) → Pool → Conv(16→32) → Pool → FC(1568→128) → FC(128→15)
  - 运行时数据增强（ColorJitter）、early stopping、per-class 准确率报告

- **feat(nn): 批量新增 18 项基础节点（节点总数 41 → 53）**
  - 已在 0.13.0 CHANGELOG 中列出（该批提交实际落入本版本）

- **feat(nn): 将 ReLU 从 LeakyReLU 中独立为一等节点**

- **feat: 可选 BLAS 加速（Intel MKL / OpenBLAS）**
  - 通过 `--features blas-mkl` 或 `--features blas-openblas` 启用
  - justfile 自动检测本地 BLAS 后端（MKL > OpenBLAS > 纯 Rust）

### 性能优化

- **perf(conv2d): im2col + GEMM 替换嵌套循环卷积，训练速度提升 2.6-4.4x**
- perf(conv2d): 反向传播 im2col 批量化，N 次小 GEMM 合并为 1 次大 GEMM
- perf(conv2d): 前向传播 padded_input 缓存改用 move 消除 clone
- perf(nn): 反向传播全局优化——in-place 梯度累加 + ReLU 融合 + MaxPool 预分配
- perf(nn): `GradResult` 零拷贝梯度传递 + benchmark 基础设施
- perf(optimizer): `set_value_owned` 零拷贝参数更新 + Adam 临时分配优化

### 修复

- fix: 消除编译警告 + 补充 `GradResult::Negated` 路径单元测试
- fix: 补齐 roadmap 遗漏项（Tensor 测试 + 独立节点 + Var API）

### 重构

- refactor: 计算图表示中 LeakyReLU 替换为 ReLU
- refactor(evolution): 移除所有 Phase N 工程阶段注释
- refactor(evolution): `EvolutionError` + 延迟实例化，`supervised()` 恢复无错构造
- refactor(evolution): 隐藏 `Graph`，`EvolutionResult` 仅暴露 `predict()` / `visualize()` API
- refactor(examples): 更新中国象棋模型架构和数据增强

### 文档

- docs: 归档已完成的规划文档，整合至 architecture_roadmap
- docs: 更新性能优化文档，反映 Phase 1-5 完成状态
- docs: 更新文档反映 roadmap 完成状态
- docs: 新增 oneDNN CPU 内核优化参考
- docs: 数据共享可视化已通过 source_id 实现，更新未来方向

### 其他

- feat: Phase 1-5 feature expansion（CNN / data augmentation / Transformer / API convenience methods / Repeat node / Chunk / Norm variants / error refinement / utility activation methods）

## [0.13.0] - 2026-02-14

### 破坏性变更

- **feat: 全面分离 Stack 与 Concat 为独立操作**
  - `Stack` 和 `Concat` 不再合并为同一节点，各自拥有独立的语义和实现
  - 新增 `Var::cat` 便捷方法（对应 PyTorch 的 `torch.cat`）

- **refactor(nn): 将 Detach 从 Identity 标志位拆分为独立节点类型**
  - `Detach` 不再是 `Identity` 的特殊标志，而是完整独立的计算图节点

- **refactor(vis): 统一节点分组机制，删除旧 LayerGroup/RecurrentLayerMeta 体系**
  - 新的节点分组上下文机制取代旧式 `LayerGroup` / `RecurrentLayerMeta`

### 新增

- **feat(graph): 实现通用 CSE（公共子表达式消除）节点去重机制**

- **feat(nn): 新增概率分布模块**
  - `Categorical`：离散分类分布（支持 log_prob / entropy / sample）
  - `Normal`：正态分布
  - `TanhNormal`：Tanh 压缩正态分布（SAC 连续动作策略核心）

- **feat(nn): 新增计算图节点**
  - `Exp`：指数函数
  - `Clip`：值域裁剪
  - `Sqrt`：平方根
  - `Negate`：取负（补全基础算术运算对称性）

- **feat(nn): 批量新增基础节点（18 项，节点总数 41 → 53）**
  - 7 个现代激活函数节点：`GELU`、`Swish/SiLU`、`ELU`、`SELU`、`Mish`、`HardSwish`、`HardSigmoid`
  - 形状操作节点：`Narrow`（沿轴连续切片）、`Permute`（维度重排 / 转置）
  - 条件/筛选节点：`Where`（掩码选择）、`TopK`（取前 K 大值）、`Sort`（沿轴排序）
  - 3 个 Var 便捷方法（无独立 NodeType）：`squeeze`、`unsqueeze`、`split`
  - 统一 Tensor → Node → Var 三层架构，每层均有独立测试
  - 11 个 Python 对照脚本（PyTorch 前向值 + Jacobian 验证）

- **refactor(nn): 补齐 3 个已有节点的 Tensor 层方法**
  - `LeakyReLU`、`SoftPlus`、`Step` 的前向计算下沉到 Tensor 方法，统一三层调用路径
  - 附带将 `Concat` 内的 `slice_along_axis` 重构为 `Tensor::narrow`

- **feat(vis): Graph 快照可视化 + 多 Loss 路径边着色**
  - 支持在任意时刻对计算图进行快照可视化
  - 多 Loss 场景下自动为不同 Loss 路径着色

- **feat(vis): 节点分组上下文机制 + 分布 cluster 可视化**
  - 基于上下文的灵活分组，支持概率分布模块的 cluster 展示

- **feat(vis): Tensor source_id 追踪 + 同源数据节点链式虚线标注**
  - 追踪数据来源，同源输入以虚线可视化关联

- **feat(rl): 新增 SAC 示例**
  - SAC-Continuous Pendulum 示例
  - Moving-v0 Hybrid SAC 示例（方式 B — 独立连续分支）

### 修复

- fix(vis): 修复 `.dot` 输出中同源数据虚线边顺序不确定的问题
- fix(vis): 修复 RNN/LSTM/GRU 场景 Input 节点未归入模型 scope 的 bug
- fix(docs): 移除公开文档中的本地私有路径

### 重构

- refactor: 大文件按功能域拆分，降低单文件复杂度
- refactor(examples): 8 个示例改用 snapshot 可视化 + GAN 多 Loss 着色 + detach 节点命名
- refactor(examples): 4 个示例从逐样本训练改为 full-batch 模式
- refactor(test): 将内联单元测试迁移到独立 tests/ 目录

### 测试

- test(tensor): 补充 source_id corner case 单元测试

### 文档

- 新增 Input 节点语义与数据共享可视化设计文档
- 新增 RL 路线图，整理 RL 相关文档过时内容
- 新增 SAC 数学基础分析文档

### 其他

- chore: Minari 联网测试加 `#[ignore]`，justfile 细化测试命令
- chore: rustfmt 格式化 + lint 清理

## [0.12.0] - 2026-02-12

### 破坏性变更

- **refactor(nn): 动态图架构迁移（方案 C）**
  - `Var` 持有 `Rc<NodeInner>`，节点生命周期由引用计数自动管理
  - 移除 `ModelState`、`Criterion` — 不再需要闭包式缓存机制
  - 移除 `GraphInner::new_*_node()` / `forward(NodeId)` / `get_node_value(NodeId)` 等旧 API
  - 新 API：`Graph` + `Var` 算子重载 + `Module` trait + `Optimizer`

- **refactor(nn): 移除旧式循环机制**
  - 删除 `connect_recurrent` / `step` / `backward_through_time` 等旧 API
  - 删除 `StepSnapshot` / `recurrent_edges` / `prev_values` 等旧字段
  - 展开式 RNN/LSTM/GRU 设计完全取代旧式显式时间步方案

- **refactor(nn): 移除 `backward_ex()` 和 `retain_graph` 参数**
  - 动态图架构下节点自动管理生命周期，`retain_graph` 不再需要
  - 统一使用 `backward()` 即可支持多 loss 梯度累积、多次反向传播

### 新增

- **feat(nn): PyTorch 风格动态图 API**
  - `graph.input()` / `graph.parameter()` 创建变量
  - `&a + &b`、`a.matmul(&b)` 等算子重载
  - `var.forward()` / `var.backward()` 自动前向/反向传播
  - `var.mse_loss()` / `var.cross_entropy_loss()` 等损失函数方法链

- **feat(rl): 强化学习基础设施**
  - `GymEnv`：与 Python Gymnasium 环境交互
  - `Minari`：离线 RL 数据集加载
  - CartPole SAC-Discrete 示例（Twin Q、自动温度调节、目标网络软更新）

- **feat(nn): RNN/LSTM/GRU 展开式设计**
  - 一次性处理整个序列，标准 `backward()` 自动完成 BPTT
  - 支持动态 batch_size 和变长序列

### 测试

- **test: 全量测试迁移完成**
  - 1579 个单元测试全部通过（0 failed, 0 ignored）
  - 12 个 Batch 的节点测试从旧 API 迁移到新 API
  - 16 个示例全部迁移到新 API 并验证通过（含 cartpole_sac RL 示例）

### 文档

- 更新 README：移除 `ModelState` 引用，更新为新 API 描述
- 更新动态图设计文档状态为"已完成"

## [0.11.0] - 2026-01-29

### 新增

- **feat(tensor): 实现统一的 Stack 操作**
  - 覆盖 PyTorch 的 `stack` 和 `cat` 功能
  - 支持 Tensor 层和节点层操作

- **feat(nn): 多输入/多输出 API**
  - `forward2`/`forward3` 多输入前向传播
  - `ModelState` 支持多输出及 `retain_graph` 反向传播
  - 新增 `dual_input_add`、`siamese_similarity`、`dual_output_classify`、`multi_io_fusion` 示例

- **feat(nn): 新增损失函数**
  - `MAE`（Mean Absolute Error）损失节点
  - `BCE Loss` 二元交叉熵损失（支持多标签分类）
  - `Huber Loss`（Smooth L1 Loss）

- **feat(metrics): 评估指标模块**
  - 分类指标：Accuracy、Precision、Recall、F1Score 等
  - 回归指标：MSE、MAE、R² 等
  - 统一 API，用户无需导入 `Metric` trait

- **feat(nn): Dropout 正则化节点**
  - 支持训练/推理模式自动切换

- **feat(tensor): Abs 绝对值算子**
  - Tensor 层和节点层完整支持

### 重构

- **refactor: 统一浮点类型为 f32**
  - 移除 `f64` 过度设计，简化代码

- **refactor: 统一损失节点命名**
  - `MSELoss` → `MSE`，与其他损失节点命名风格一致

### 文档

- docs: README 添加多输入/多输出示例说明

### 其他

- fix: rust lint 修复

## [0.10.2] - 2026-01-28

### 重构

- **refactor(graph): 模块化重构 graph.rs 为 graph/ 目录结构**
  - 拆分为 `core.rs`、`forward.rs`、`backward.rs`、`visualization.rs` 等子模块
  - 提升代码可维护性，为 NEAT 演化架构做准备

- **refactor(cnn): 统一 CNN 层为 Batch-First 4D 格式**
  - Conv2d/MaxPool2d/AvgPool2d 输入输出格式统一为 `[N, C, H, W]`

- **refactor: 统一术语，明确 Batch-First 设计原则**
  - 文档和代码注释统一使用 Batch-First 术语

- **refactor: 代码质量提升**
  - 统一错误信息格式，避免"节点"前缀重复
  - 改进参数文件格式错误的提示信息
  - 清理代码注释中的版本/阶段历史痕迹
  - 清理冗余代码并新增通用下载模块 (`src/utils/download.rs`)

### 文档

- **docs: 新增 NEAT 神经架构演化设计文档**
  - 新增 `.doc/design/neural_architecture_evolution_design.md`
  - 整合循环边变异机制设计
  - 为后续 NEAT/强化学习功能做架构准备

## [0.10.1] - 2026-01-25

### 新增

- **feat(rnn): 添加 RNN 展开缓存机制**
  - 支持动态 batch，避免重复展开相同序列长度的计算图

### 修复

- **fix(rnn): 修复 RNN/LSTM/GRU 缓存 key 问题**
  - 缓存 key 仅用 seq_len 导致变 batch 失效，现已修正

### 重构

- **refactor(vis): 统一可视化 API**
  - 默认启用层分组显示
  - 可视化边线从 ortho 改为 polyline
  - 优化循环层时间步标签及 ZerosLike 节点样式

### 文档

- 新增计算图可视化指南 (`.doc/design/visualization_guide.md`)

### 其他

- test: 更新 forward 行为测试以反映新设计
- chore: rust lint format

## [0.10.0] - 2026-01-25

### 重构

- **refactor(nn): 统一 Input 节点类型架构**
  - 将 `Input` 和 `GradientRouter` 统一为 `InputVariant` 枚举
  - 三种变体：`Data`（通用输入）、`Target`（Loss 目标值）、`Smart`（模型入口，原 GradientRouter）
  - 详见 [设计文档](.doc/design/input_node_unification_design.md)

- **refactor(nn): 可视化样式区分不同输入类型**
  - `Data`：浅蓝色，标签 `Input`
  - `Target`：浅橙色，标签 `Target`
  - `Smart`：浅绿色，标签 `Input`

### 新增

- **feat(examples): 所有示例添加计算图可视化**
  - 新增 `.dot` 和 `.png` 文件：xor、iris、sine_regression、california_housing、mnist、parity_rnn_fixed_len、parity_rnn_var_len、parity_lstm_var_len、parity_gru_var_len
  - 更新 mnist_gan 可视化

### 文档

- 新增 Input 节点统一设计文档
- README 可视化示例改用 examples 目录图片

## [0.9.0] - 2026-01-22

### 新增

- **feat(nn): DynamicShape 动态形状系统**
  - 新增 `DynamicShape` 类型，支持动态维度（类似 Keras 的 `None`）
  - 所有节点实现 `dynamic_expected_shape()` 和 `supports_dynamic_batch()`
  - `NodeDescriptor` 存储 `dynamic_shape` 用于可视化和序列化
  - 可视化中动态维度显示为 `?`（如 `[?, 128]`）

- **feat(nn): GradientRouter 节点和函数式 detach 机制**
  - 新增 `GradientRouter` 节点，支持动态梯度路由
  - 实现 `DetachedVar` 轻量 detach 包装
  - 支持 GAN 训练的 `fake.detach()` 模式

- **feat(nn): ModelState 智能缓存 + Criterion 损失封装**
  - `ModelState` 按特征形状缓存计算图，忽略 batch 维度
  - `MseLoss` / `CrossEntropyLoss` PyTorch 风格封装
  - `ForwardInput` trait 统一输入类型

- **feat(nn): PyTorch 风格 RNN/LSTM/GRU API**
  - `Rnn`/`Lstm`/`Gru` struct + `forward()` 模式
  - 支持变长序列（`BucketedDataLoader`）
  - `ZerosLike` 节点动态生成初始隐藏状态

- **feat(data): PyTorch 风格 DataLoader**
  - `DataLoader` 统一批处理接口
  - `BucketedDataLoader` 变长序列分桶

- **feat(tensor): argmax/argmin 方法**
  - 分类任务预测必需

### 示例

- 新增 10 个完整示例：
  - `xor`: 基础 MLP
  - `sine_regression`: 回归任务
  - `iris`: 多分类
  - `mnist`: 图像分类（MLP + CNN）
  - `mnist_gan`: GAN 训练 + detach
  - `california_housing`: 房价回归
  - `parity_rnn_fixed_len`: RNN 定长
  - `parity_rnn_var_len`: RNN 变长 + 智能缓存
  - `parity_lstm_var_len`: LSTM 变长
  - `parity_gru_var_len`: GRU 变长

### 修复

- fix(layer): RNN/LSTM/GRU 层 h0/c0 不再缓存，每次 forward 动态创建
  - 解决 `BucketedDataLoader` 变长批次的形状不兼容问题

### 重构

- refactor(nn): `check_shape_consistency` 使用 `DynamicShape.is_compatible_with_tensor()`
- refactor(seed): Graph seed 自动传播到 Layer

### 测试

- 单元测试从 822 增加到 1017
- 所有节点新增 DynamicShape 单元测试
- 新增 `node_softmax.rs`、`node_zeros_like.rs` 测试文件

## [0.8.0] - 2026-01-20

### ⚠️ 破坏性变更 (Breaking Changes)

- **refactor(layer)!: 统一所有 Layer 为 PyTorch 风格 API**
  - `Linear`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `Rnn`, `Lstm`, `Gru` 统一为 struct + `forward()` 模式
  - 旧函数式 API 已删除
  - 详见 [架构 V2 设计](.doc/design/architecture_v2_design.md)

- **refactor(nn): 移除 `ScalarMultiply` 和 `ChannelBiasAdd` 节点**
  - 功能由通用 `Add`/`Subtract`/`Multiply` + 广播替代
  - `Conv2d` bias 形状从 `[1, C]` 改为 `[1, C, 1, 1]`

- **refactor(optimizer): 统一优化器 API**
  - V1 API 已删除，V2 成为默认实现
  - Optimizer 内部持有图引用，`zero_grad()`/`step()` 不再需要 `&mut Graph` 参数

### 新增

- **feat(tensor): 实现完整 NumPy 风格广播机制**
  - Tensor 层：8 个运算符（`+`/`-`/`*`/`/` 及其 `Assign` 版本）支持广播
  - Node 层：`Add`/`Subtract`/`Multiply`/`Divide` 支持广播
  - 工具函数：`broadcast_shape()`, `sum_to_shape()`
  - 新增 `Subtract` 节点
  - 详见 [广播机制设计](.doc/design/broadcast_mechanism_design.md)

- **feat(nn): 实现 Module trait 和 PyTorch 风格 API**
  - `Module` trait：`parameters()` 返回 `Vec<Var>`
  - `Var` 支持算子重载（`&a + &b`）和链式调用（`x.relu().sigmoid()`）
  - `Graph` 句柄：`Rc<RefCell<GraphInner>>` 允许 `Var` 持有图引用

### 重构

- refactor(layer): 简化 Layer 层，使用原生广播替代 `ones @ bias` 模式
- refactor(test): 改进 RNN/LSTM/GRU reset 测试的健壮性

### 文档

- docs: 更新架构 V2 设计文档，添加广播机制设计决策
- docs: 新增广播机制设计文档

### 测试

- 单元测试从 ~800 增加到 822+
- 新增 V2 集成测试：`test_mnist_linear_v2.rs`, `test_mnist_batch_v2.rs`

## [0.7.0] - 2026-01-08

### ⚠️ 破坏性变更 (Breaking Changes)

- **refactor(autodiff): 自动微分 API 统一 (Jacobian → VJP)**
  - 删除 Jacobian 模式，统一使用 VJP (Vector-Jacobian Product)
  - API 重命名：
    - `forward_node()` → `forward()`
    - `backward_nodes()` / `backward_batch()` → `backward()`
    - `clear_jacobi()` / `clear_grad()` → `zero_grad()`
    - `one_step()` / `one_step_batch()` / `update()` → `step()`
  - 删除：所有节点的 `jacobi` 字段、`calc_jacobi_to_a_parent()` 方法
  - `backward()` 返回 `f32` (loss 值)，简化训练循环
  - 详见 [自动微分统一设计](.doc/design/autodiff_unification_design.md)

## [0.6.0] - 2026-01-01

### 新增

- feat(layer): **Phase 3 完成** - RNN/LSTM/GRU Layer API
  - `rnn()`: Vanilla RNN 层 (h_t = tanh(x@W_ih + h_{t-1}@W_hh + b))
  - `lstm()`: LSTM 层 (4 门: 输入门、遗忘门、候选细胞、输出门)
  - `gru()`: GRU 层 (2 门: 重置门、更新门)
  - 所有层支持 BPTT 训练与层分组可视化
  - 集成测试验收：RNN 95.3%、LSTM 93.8%、GRU 90.6% 准确率
- feat: 实现 State 节点与 BPTT 循环机制
  - 支持时序状态记忆
  - `graph.step()` / `backward_through_time()` API
- feat: 添加 Sign 节点（Tensor 层 + NN 节点层）
  - 输出 {-1, 0, 1}，与 PyTorch 行为一致
- feat: 添加 Conv2d bias 支持与层分组可视化功能
  - 新增 ChannelBiasAdd 节点用于 bias 广播
  - 新增 `LayerGroup` 和 `save_visualization()` 实现层分组可视化

### 性能优化

- perf: 优化赋值算子 (+=/-=/*=/÷=) 并减少不必要的 clone
  - jacobi 累加、优化器梯度计算等处避免临时张量分配

### 重构

- refactor: 重组 Python 测试目录结构 (`tests/python/layer_reference/`)
- refactor(test): 增强 `assert_err!` 宏，支持多种简洁语法
  - 新增 `Variant(literal)`、`ShapeMismatch(exp, got, msg)` 等语法
  - 重构所有测试文件，消除冗长的 if guard 形式

### 测试

- test: 补充各层 PyTorch 数值对照及覆盖测试
  - 层测试总数从 128 增加到 143
  - 新增 AvgPool2d/MaxPool2d/Linear/Conv2d 的 forward/backward PyTorch 对照
  - 新增 RNN/LSTM/GRU batch_backward、chain_batch_training 等测试

### 文档

- docs: 新增五层架构设计文档 (`architecture_v2_design.md`)
- docs: 添加记忆机制设计文档及 NEAT/EXAMM 论文笔记
- docs: 更新梯度流控制设计文档
- docs: 修复 README 笔误 (waht→what, ndoes→nodes, fis→fix)

### 其他

- chore: 删除 README 中已完成的正确性验证 section（所有项已被现有测试覆盖）

## [0.5.0] - 2025-12-27

### 新增

- feat: 实现计算图序列化与可视化功能
  - `GraphDescriptor` 统一 IR 设计
  - `save_model()` / `load_model()` 模型保存加载（JSON + bin）
  - `to_dot()` / `save_visualization()` Graphviz 可视化
  - `summary()` / `summary_markdown()` Keras 风格摘要输出
- feat: 实现完整的梯度流控制机制
  - `no_grad_scope()` 无梯度作用域
  - `detach_node()` / `attach_node()` 梯度截断
  - `backward_nodes_ex(..., retain_graph)` 多次反向传播
- feat: 优化器 `with_params()` 方法，支持指定参数列表优化（用于 GAN/迁移学习）
- feat(Input): Input 节点拒绝设置雅可比矩阵

### 文档

- docs: 添加 Graph 序列化与可视化设计文档
- docs: 添加梯度流控制设计文档 (no_grad/detach/retain_graph)
- docs: README 添加计算图可视化展示
- docs: 精简 README TODO 列表

### 重构

- refactor: 将 Python 测试脚本移至 `tests/python/` 目录
- refactor: summary 标题改为中文「模型摘要」

### 其他

- chore: 添加 MNIST GAN 示例
- chore: 修正 GitHub 语言检测，忽略 issues 目录

## [0.4.0] - 2025-12-22

### 新增

- feat(layer): 实现 Linear 层（Batch-First 设计）
- feat: 实现 Conv2d 节点（2D 卷积）
- feat: 实现 MaxPool2d 节点（2D 最大池化）
- feat: 实现 AvgPool2d 节点（2D 平均池化）
- feat: 添加 CNN Layer 便捷函数 (conv2d, max_pool2d, avg_pool2d) 及 MNIST CNN 集成测试
- feat: 添加 Softplus 激活函数节点
- feat(nn): 实现 MSELoss 损失节点
- feat: California Housing 房价回归数据集与集成测试

### 性能优化

- perf: 使用 Rayon 并行化 CNN 层 (conv2d, max_pool2d, avg_pool2d)
- perf: 添加 dev profile 优化配置以加速 debug 模式下的计算密集测试
- perf: 为 SoftmaxCrossEntropy 添加 Rayon 并行优化

### 文档

- docs: 更新 CNN 节点状态为已完成

## [0.3.0] - 2025-12-21

### 新增

- feat: 实现 ScalarMultiply 和 Multiply 节点，修复 batch 训练梯度链
- feat: 添加带种子的随机函数以确保集成测试可重复性
- feat: 实现 Tanh 节点和 XOR 集成测试 (MVP M2+M3 完成)
- feat: M4 - 验证 Graph 动态扩展能力（NEAT 友好性）
- feat: M4b - Graph 级别种子 API
- feat: 实现 Sigmoid 激活节点 + jacobi_diag() 重构
- feat: 实现 SoftmaxCrossEntropyLoss 融合节点
- feat: 实现 data 模块（DataLoader + MNIST 数据集）
- feat: 实现 Batch Forward/Backward 机制
- feat: MNIST batch 测试添加 bias 支持
- feat: 实现 LeakyReLU/ReLU 激活函数节点
- feat: 为 Tensor 实现 AbsDiffEq trait，统一测试中的浮点比较
- feat: 实现 Reshape 节点
- feat: 实现 Flatten 节点

### 重构

- refactor: 统一集成测试命名规范
- refactor: 重构 tensor_slice 宏解决临时值生命周期问题

### 文档

- docs: 添加 API 分层与种子管理设计文档
- docs: 更新文档反映阶段二核心完成

### 其他

- chore: 统一术语规范，API 参数 axis 改为 dim

## [0.2.0] - 2025-12-20

### 新增

- feat: 实现优化器架构 (SGD/Adam) 及相关测试

### 重构

- refactor(optimizer): 模块化测试并封装内部实现细节

### 文档

- 架构设计重构：`.doc/high_level_architecture_design.md` 全面重写
- Hybrid 执行引擎设计：借鉴 MXNet hybrid 思想，设计 Eager/Graph 双模式执行方案
- 五层架构设计：用户 API 层、演化 API 层、执行引擎层、中间表示层、底层计算层
- OTMF 模型格式设计：OnlyTorch Model Format 规范，支持演化信息和跨语言部署
- NEAT 演化 API 设计：完整的演化模型接口、基因表示和演化引擎
- PyTorch 风格 API 设计：Module trait、functional 模块、优化器系统
- 整理全部文档

### 其他

- chore: update .gitignore
- chore: 将 MatrixSlow Python 参考项目纳入版本控制
- chore: 应用 clippy 和 rustfmt 自动修复

## [0.1.0] - 2025-07-23

### 文档

- 搁置底层计算图重构计划，当前重心为完善上层 API。

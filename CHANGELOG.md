# 更新日志

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
  - 新增 `LayerGroup` 和 `save_visualization_grouped()` 实现层分组可视化

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

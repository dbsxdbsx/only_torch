# 更新日志

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

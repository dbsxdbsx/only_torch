# 更新日志

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

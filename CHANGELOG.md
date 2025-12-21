# 更新日志

## [0.3.0] - 2025-12-21

### 新增

- **Sigmoid 激活节点**：实现 `σ(x) = 1/(1+e^(-x))` 前向传播和 `σ(x)*(1-σ(x))` 反向传播
- **SoftmaxCrossEntropyLoss 融合节点**：数值稳定的 log-sum-exp 实现，简化梯度 `p - y`
- **DataLoader 模块**：完整的数据加载基础设施
  - `MnistDataset`：支持自动下载、gzip 解压、IDX 解析、本地缓存
  - `transforms`：`normalize_pixels`、`one_hot`、`flatten_images` 数据转换
  - `DataError`：统一的数据加载错误类型
- **Tensor 新方法**：`sigmoid()`、`exp()`、`ln()`、`max_value()`、`min_value()`、`jacobi_diag()`

### 改进

- **jacobi_diag() 重构**：统一 Tanh/Sigmoid 等元素级激活节点的 Jacobian 计算逻辑
- **Graph 反向传播**：正确处理 `SoftmaxCrossEntropyLoss` 的 `assistant_parent`（标签节点）

### 测试

- MNIST MLP MVP 集成测试：验证 DataLoader + 网络 + 训练循环的基本逻辑
- SoftmaxCrossEntropyLoss 单元测试：覆盖前向/反向、10 类分类、与 Linear 层集成等场景
- Sigmoid 单元测试：覆盖标量/向量/矩阵输入的前向/反向传播
- MNIST 数据加载测试、transforms 单元测试

### 依赖

- 新增 `flate2`（gzip）、`dirs`（跨平台目录）、`ureq`（HTTP 下载）

## [0.2.0] - 2025-08-01

### 文档

- **架构设计重构**：`.doc/high_level_architecture_design.md`全面重写，从221行扩展到775行
- **Hybrid执行引擎设计**：借鉴MXNet hybrid思想，设计Eager/Graph双模式执行方案
- **五层架构设计**：设计用户API层、演化API层、执行引擎层、中间表示层、底层计算层
- **OTMF模型格式设计**：设计OnlyTorch Model Format规范，支持演化信息和跨语言部署
- **NEAT演化API设计**：设计完整的演化模型接口、基因表示和演化引擎
- **PyTorch风格API设计**：设计Module trait、functional模块、优化器系统
- **实现路径规划**：制定5阶段开发计划，包含优先级和时间估算
- **MXNet hybrid研究**：深度分析MXNet hybrid特性对项目的启发价值
- **设计理念升级**：从简单的PyTorch风格API升级为支持演化的Hybrid框架设计
- **向后兼容设计**：确保现有底层Graph/Node API在新架构中完全可用

## [0.1.0] - 2025-07-23

### 文档

- 搁置底层计算图重构计划，当前重心为完善上层API。

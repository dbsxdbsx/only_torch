# 更新日志

## [0.2.0] - 2025-08-01

### 新增

- **Hybrid执行引擎设计**：借鉴MXNet hybrid思想，支持Eager/Graph双模式执行
- **五层架构体系**：用户API层、演化API层、执行引擎层、中间表示层、底层计算层
- **OTMF模型格式**：OnlyTorch Model Format，支持演化信息和跨语言部署
- **NEAT演化API**：完整的演化模型接口、基因表示和演化引擎设计
- **PyTorch风格API**：Module trait、functional模块、优化器系统设计
- **详细实现路径**：5阶段开发计划，包含优先级和时间估算

### 改进

- **架构文档重构**：`.doc/high_level_architecture_design.md`全面重写，从221行扩展到775行
- **设计理念升级**：从简单的PyTorch风格API升级为支持演化的Hybrid框架
- **向后兼容保证**：确保现有底层Graph/Node API完全可用

### 文档

- 新增MXNet hybrid特性深度分析和对比研究
- 完善演化优先设计原则和竞争优势分析
- 提供完整的使用示例和API设计规范

## [0.1.0] - 2025-07-23

### 文档

- 搁置底层计算图重构计划，当前重心为完善上层API。

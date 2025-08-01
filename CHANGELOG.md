# 更新日志

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

# AGENTS.md

尽量使用中文沟通、注释、报错与提交信息；术语可保留英文原文。

## 项目定位

only_torch 是一个纯 Rust 的 PyTorch 风格玩具框架，当前重点是：
- 动态计算图 + autograd
- CPU only、f32 only
- NEAT 风格神经架构演化
- 可选 BLAS 加速
- 通过 `pyo3` 对接 Python Gymnasium

处理需求时请优先保持：CPU 友好、接口直观、跨平台、易扩展。

## 日常命令

本项目统一使用 `just`：

```bash
just build                 # 调试构建
just check                 # 仅检查编译
just test                  # 常规测试
just test-filter <pattern> # 定位单个测试/模块
just test-serial           # 单线程调试（pyo3 / 竞态问题优先用它）
just lint                  # clippy
just fmt                   # rustfmt
just bench-conv2d          # 卷积基准
just example-xor           # 最小传统示例
just example-evolution-mnist # 演化版 MNIST 示例
just example-cartpole-sac  # RL 示例，需 Python + gymnasium
```

除非用户明确要求，否则不要默认运行耗时 bench、RL 示例或 `test-all`。

## 架构心智模型

- `src/tensor/`：纯张量层；只管数据，不感知计算图。
- `src/nn/nodes/`：原子计算节点；新增 op 通常先改这里。
- `src/nn/var/`：面向用户的 `Var` API；运算符重载和链式调用在这里。
- `src/nn/graph/`：图执行、`train/eval`、`no_grad`、可视化、序列化。
- `src/nn/layer/`：`Linear`、`Conv2d`、`Rnn` 等高层模块。
- `src/nn/evolution/`：基因、变异、builder、收敛与演化主流程。
- `src/data/`、`src/metrics/`、`src/rl/`、`src/vision/`：数据、指标、强化学习和图像支持。

## 改动前先看这些文档

按场景查阅，不要重复发明设计：

- 广播 / shape 问题：[广播机制设计](.doc/design/broadcast_mechanism_design.md)
- 梯度流、`detach()`、`no_grad()`：[梯度流控制设计](.doc/design/gradient_flow_control_design.md)
- 梯度清零与累积：[梯度清零与累积设计](.doc/design/gradient_clear_and_accumulation_design.md)
- Node 与 Layer 的边界：[节点与层边界设计](.doc/design/node_vs_layer_design.md)
- 演化系统：[神经架构演化设计](.doc/design/neural_architecture_evolution_design.md)
- DataLoader / 变长序列：[数据加载设计](.doc/design/data_loader_design.md)
- 开发环境 / rust-analyzer：[开发环境配置](.doc/dev_environment_setup.md)
- 术语规范：[术语约定](.doc/terminology_convention.md)

## 高频约定与坑点

- **必须手动** `optimizer.zero_grad()`；不要假设框架会自动清梯度。
- 本项目强调**显式 broadcast**；不要按 PyTorch 的隐式广播习惯直接写。
- `Module` trait 只统一 `parameters()`；`forward()` **不是**通用 trait 方法。
- `graph.no_grad(|| { ... })` 是图级上下文；`var.detach()` 是局部截断梯度。
- 新增 op 时，一般要同时改 `raw_node` 实现与 `Var` 便捷接口。
- RL / pyo3 相关测试容易有导入竞态；优先 `serial_test` 或 `just test-serial`。
- 演化阶段长时间无日志，通常表示候选仍在评估，不一定是卡死。
- 稀疏 FM 图会比 fully-connected FM 合并路径慢很多；排查性能时优先确认 builder 路径。

## 常见改动流程

### 新增或修改算子
1. 在 `src/nn/nodes/raw_node/` 实现前向与反向逻辑。
2. 在 `src/nn/var/` 暴露用户侧 API。
3. 补充单元测试；复杂数值最好同时补 Python 参考。

### 新增或修改 Layer
1. 参考已有 `Linear`、`Conv2d`、`Rnn/Lstm/Gru` 的写法。
2. 不要强行把 `forward()` 抽成统一 trait 签名。
3. 确保 `parameters()` 返回完整可训练参数。

### 修改 Evolution
1. 保持主流程稳定：build → restore weights → train → capture → evaluate → accept/rollback → mutate。
2. 优先跑针对性测试或小示例，不要直接全量 MNIST。
3. 若改到性能路径，顺手查看 [`.doc/optimization_candidates.md`](.doc/optimization_candidates.md)。

## 测试约定

- 源码旁测试：各模块内部 `mod tests`。
- 集成测试：`tests/` 下每个 `.rs` 文件尽量只放一个 `#[test]`。
- 数值验证：复杂算子优先参考 `tests/*.py` 中的 PyTorch 对照实现。
- 普通验证优先用 debug 构建；仅在性能或发布场景才切 release。

## 参考目录

- `examples/traditional/`：手写网络的标准用法。
- `examples/evolution/`：演化 API 的标准用法。
- `MatrixSlow/`：仓库内 Python 参考实现，适合遇到架构问题时对照思路。

## 输出约束

- 不要在 public-facing 文本里暴露本机绝对路径或私有资料。
- 若已有设计文档，优先链接，不要整段复制进新文档。
- 修改说明尽量简洁、可执行、可验证。

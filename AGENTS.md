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

## 当前版本与焦点

| 项 | 内容 |
|----|------|
| **版本** | `0.20.0`（2026-06-15；本地可能超前 `origin/master`，以 `git log` / `CHANGELOG.md` 为准） |
| **刚闭环** | **v0.20.0 RL 环境/buffer/smoke**：Gymnasium-only `GymEnv`（legacy 清零）、`Transition`+`ReplayBuffer<T>` 入库、Platform-v0 替代 Moving-v0、smoke 门禁 |
| **当前主线** | **强化学习** v0.21.0：SAC helper 入库 + 示例瘦身 + LunarLander-v3；见 [RL 路线图](.doc/design/rl_roadmap.md) |
| **刻意暂缓** | 演化 **阶段 D**（`CellAttention` ONNX、`Attention` Net2Net 函数保持、Conv2d Attention、3D batched MatMul）——与 RL 零耦合，见 [记忆机制设计 — Phase D](.doc/design/memory_mechanism_design.md#-后续-phase-d刻意未做) |
| **路线展望** | RL 主线 v0.20–v0.24：**环境一律 `python/gym_env/<游戏>/` + `GymEnv` 桥接**（无 Rust 棋盘）。**架构跑通**：SAC/MuZero/PPO 用 **`CartPole-v0` ≥195**；**终极调优**：v0.24 **[EfficientZero V2](https://arxiv.org/abs/2403.00564)（EZ-V2，第二代，唯一）**（全 `-v1` 环境）。详见 [RL 主线实施计划](../c:/Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md) |

**进度符号**（设计文档统一口径）：✅ Phase 验收范围内已完成 · ⏳ 已识别、留后续 Phase · 🔲 可选增强 · 📦 已归档历史路径。

**接手 RL 时建议顺序**：读 `rl_roadmap.md` → 配环境 [`.doc/rl_python_env_setup.md`](.doc/rl_python_env_setup.md)（**仅 Gymnasium**）→ `just test-filter buffer_replay`（15 测试确认 buffer）→ `just smoke-cartpole-sac`（管线验证）→ 推进 v0.21（helper 入库 + 示例瘦身）。

**接手 Attention 阶段 D 时**：先读 [记忆机制设计 — 实现状态速览](.doc/design/memory_mechanism_design.md#-实现状态速览)（含 105 个相关单元测试与 IT-* 示例表），勿假设「打勾 = ONNX 也做完」。

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
just clean                 # cargo clean，彻底清理构建产物
just clean-cache           # 清大体积编译缓存，保留 benchmark 结果
just bench-smoke           # 约 30 秒快速性能回归
just bench-save <name>     # 保存 Criterion baseline（重构前）
just bench-compare <name>  # 与 baseline 对比（重构后）
just bench-macro           # hyperfine 跑 release example 宏基准
just bench-conv2d          # 卷积基准
just example-xor           # 最小传统示例
just example-evolution-mnist   # 演化版 MNIST 示例
just examples-memory-unit      # parity RNN/LSTM/GRU/Transformer + 演化序列示例聚合
just example-cartpole-sac      # RL 示例，需 Python + gymnasium
```

除非用户明确要求，否则不要默认运行耗时 bench、RL 示例或 `test-all`。

## 架构心智模型

- `src/tensor/`：纯张量层；只管数据，不感知计算图。
- `src/nn/nodes/`：原子计算节点；新增 op 通常先改这里。
- `src/nn/var/`：面向用户的 `Var` API；运算符重载和链式调用在这里。
- `src/nn/graph/`：图执行、`train/inference`、Mode 契约、可视化、序列化。
- `src/nn/layer/`：`Linear`、`Conv2d`、`Rnn` / `Lstm` / `Gru`、`MultiHeadAttention`、`TransformerEncoder` 等高层模块。
- `src/nn/evolution/`：基因、变异、builder、收敛与演化主流程。
- `src/data/`、`src/metrics/`、`src/rl/`：数据、指标、强化学习；`src/data/` 同时承载 `Transform`（image-only）与 `SampleTransform`（image + label 同步）两套变换契约。
- `src/vision/`：图像支持，按职能划分为 `io / color / geom / filter / draw / mask / preprocess / viz / detection / cv`。`detection/` 闭环收口 `BBox / NMS / mAP-friendly 类型 / Backbone 契约 / loss 组合 / label 同步变换 / YOLO 标签解析`，并通过 `adapter::yolo::v5` 适配第三方 YOLOv5 ONNX 输出解码；`mask/` 是像素级 mask 处理（argmax / 多类→前景 / mask→ASCII）；`viz/` 是展示画布工具（`Palette` 调色板、像素放大、alpha 混合、5x3 像素字体 `TinyFont`）；`cv/` 收纳 OpenCV 风格的传统 CV 算法（PyTorch / JAX 不收录的部分）。

## 改动前先看这些文档

按场景查阅，不要重复发明设计：

- 广播 / shape 问题：[广播机制设计](.doc/design/broadcast_mechanism_design.md)
- 训练 / 推理模式与 `detach()`：[Mode 设计](.doc/design/mode_design.md)
- 梯度清零与累积：[梯度清零与累积设计](.doc/design/gradient_clear_and_accumulation_design.md)
- Node 与 Layer 的边界：[节点与层边界设计](.doc/design/node_vs_layer_design.md)
- 演化系统：[神经架构演化设计](.doc/design/neural_architecture_evolution_design.md)
- 记忆 / RNN / Attention（含 Phase 进度与留坑表）：[记忆机制设计](.doc/design/memory_mechanism_design.md)
- 强化学习（当前主线）：[RL 路线图](.doc/design/rl_roadmap.md)、[Python 环境配置](.doc/rl_python_env_setup.md)
- 空间视觉任务路线：[空间视觉任务路线图](.doc/design/spatial_vision_tasks_roadmap.md)
- DataLoader / 变长序列：[数据加载设计](.doc/design/data_loader_design.md)
- 开发环境 / rust-analyzer：[开发环境配置](.doc/dev_environment_setup.md)
- 术语规范：[术语约定](.doc/terminology_convention.md)

## 高频约定与坑点

- **必须手动** `optimizer.zero_grad()`；不要假设框架会自动清梯度。
- 本项目强调**显式 broadcast**；不要按 PyTorch 的隐式广播习惯直接写。
- `Module` trait 只统一 `parameters()`；`forward()` **不是**通用 trait 方法。
- `graph.train()` / `graph.inference()` 是图执行的二态切换：`Inference` 同时关闭层训练分支、跳过 backward 缓存，并让 `backward()` 直接报错；临时切换用 `graph.inference_scope(|g| { ... })`；`var.detach()` 是节点级局部梯度截断（与 mode 正交）。
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

### 修改 RL
1. 改前读 [RL 路线图 §7](.doc/design/rl_roadmap.md#7-v0200-实施计划) 与 [`.github/instructions/rl.instructions.md`](.github/instructions/rl.instructions.md)。
2. **Phase 0 优先**：`GymEnv` 仅 `gymnasium.make`，禁止 `import gym`；**Phase 0b**：混合动作用 **`Platform-v0`**（`hybrid-platform`），不用 gym-hybrid / Moving。
3. v0.20.0 入库 `Transition` + `ReplayBuffer`（删 `Step`）；`SacAgent` 仍只在示例。
4. 验证：`just test-filter rl`；训练 `just example-cartpole-sac`；发版前 `SMOKE=1` / smoke just 目标（见路线图 §7.4）。

### 修改 Evolution
1. 保持主流程稳定：build → restore weights → train → capture → evaluate → accept/rollback → mutate。
2. 优先跑针对性测试或小示例，不要直接全量 MNIST。
3. 若改到性能路径，顺手查看 [`.doc/optimization_candidates.md`](.doc/optimization_candidates.md)。

## 测试约定

- **主流：模块下 `tests/` 子目录**，例如 `src/nn/tests/`、`src/tensor/tests/`、`src/vision/tests/`、`src/data/tests/`、`src/metrics/tests/`、`src/nn/evolution/tests/`。每个测试文件聚焦一个主题（命名如 `node_<op>.rs` / `layer_<name>.rs` / `transform_<name>.rs`），由对应模块的 `tests/mod.rs` 集中索引。这是事实上的 default。
- **源码旁内嵌 `mod tests`**：仅用于极少数"小且紧贴单一 struct" 的场景（≤ 2 个 test、≤ 30 行、不依赖跨模块 fixture），用 `super::*` 直通私有 API 即可。规模一旦超出就归位到风格 A。
- **顶层 `tests/`**：仅用于跨 crate / 跨模块、需要从 public API 走端到端的集成测试，每个 `.rs` 文件尽量只放一个 `#[test]`。
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

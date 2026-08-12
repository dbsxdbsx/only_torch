# Documentation Index

`.doc/` 存放项目长期知识：架构、配置、流程、排障与术语。面向协作者与自动化工具；不是任务清单，也不是未闭环问题现场（后者见 [`.issue/`](../.issue/README.md)）。

## 目录契约

```text
.doc/
├── README.md                    # 本文件：索引与约定
├── architecture_roadmap.md      # 仓库级架构路线图
├── rl_python_env_setup.md       # RL Python / Gymnasium 环境配置
├── dev_environment_setup.md     # 开发环境 / rust-analyzer
├── terminology_convention.md    # 术语约定
├── design/                      # 专题设计文档（英文 snake_case）
├── performance/                 # 性能策略、候选、流程、战报
├── paper/                       # 论文阅读笔记
├── reference/                   # 外部参考摘要
├── assets/                      # 文档配图（若有）
└── _archive/                    # 已替代 / 过期文档（默认不进主索引）
```

约定：

- 文件名英文 `snake_case`；一文一题；中文正文，术语可保留英文。
- 图片放 `assets/`，文内用相对路径。
- `_archive/` 文首须写归档原因、替代文档与删除条件；默认不进本索引。
- 新增有效 `.doc` 文档后，应能从本文件或根目录 `AGENTS.md` 的「改动前先看这些文档」链到。

## 知识索引（现行）

### 总览与环境

| 文档 | 用途 |
|------|------|
| [architecture_roadmap.md](architecture_roadmap.md) | 仓库级架构与模块地图 |
| [rl_python_env_setup.md](rl_python_env_setup.md) | RL 的 Python / Gymnasium / Platform / MinAtar 配置 |
| [dev_environment_setup.md](dev_environment_setup.md) | 本机开发与 rust-analyzer |
| [terminology_convention.md](terminology_convention.md) | 术语 |

### 设计（`design/`）

| 文档 | 用途 |
|------|------|
| [rl_myzero_status.md](design/rl_myzero_status.md) | **RL / MyZero 唯一战略权威**（状态、组件裁决、处方表、backlog） |
| [broadcast_mechanism_design.md](design/broadcast_mechanism_design.md) | 显式广播 |
| [mode_design.md](design/mode_design.md) | Train / Inference 与 detach |
| [gradient_clear_and_accumulation_design.md](design/gradient_clear_and_accumulation_design.md) | 梯度清零与累积 |
| [node_vs_layer_design.md](design/node_vs_layer_design.md) | Node 与 Layer 边界 |
| [neural_architecture_evolution_design.md](design/neural_architecture_evolution_design.md) | NEAT 风格演化 |
| [memory_mechanism_design.md](design/memory_mechanism_design.md) | RNN / Attention / Phase 进度 |
| [spatial_vision_tasks_roadmap.md](design/spatial_vision_tasks_roadmap.md) | 空间视觉任务路线 |
| [threading_model.md](design/threading_model.md) | Rayon / 线程与分配 |
| [data_loader_design.md](design/data_loader_design.md) | DataLoader / 变长序列 |
| [batch_mechanism_design.md](design/batch_mechanism_design.md) | Batch 机制 |
| [distributions_design.md](design/distributions_design.md) | 概率分布 |
| [optimizer_architecture_design.md](design/optimizer_architecture_design.md) | 优化器 |
| [graph_serialization_design.md](design/graph_serialization_design.md) | 序列化 |
| [input_node_semantics_design.md](design/input_node_semantics_design.md) | Input 语义 |
| [api_layering_and_seed_design.md](design/api_layering_and_seed_design.md) | API 分层与 seed |
| [onnx_import_strategy.md](design/onnx_import_strategy.md) | ONNX 导入 |
| [visualization_guide.md](design/visualization_guide.md) | 计算图可视化 |

### 性能（`performance/`）

| 文档 | 用途 |
|------|------|
| [optimization_strategy.md](performance/optimization_strategy.md) | 全局策略与约束 |
| [optimization_candidates.md](performance/optimization_candidates.md) | 候选与否决 |
| [benchmark_workflow.md](performance/benchmark_workflow.md) | 基准流程与台账 |
| [optimization_log.md](performance/optimization_log.md) | 已实施战报 |

### 强化学习文档分工（勿再建第三份路线图）

| 角色 | 路径 |
|------|------|
| 战略 / 裁决 / backlog | [design/rl_myzero_status.md](design/rl_myzero_status.md) |
| Python 环境 | [rl_python_env_setup.md](rl_python_env_setup.md) |
| 示例入口（薄） | [examples/my_zero/README.md](../examples/my_zero/README.md) |
| 环境数字账本 | `examples/my_zero/<env>/README.md` |
| 未闭环现场 | [`.issue/items/`](../.issue/items/) |

编辑器本地的 `*.plan.md` 等仅会话草稿，**不**作仓库权威。

## 与其他目录的边界

| 目录 | 职责 |
|------|------|
| `README.md`（根） | 给人看的简介与示例表 |
| `AGENTS.md` | AI / 协作者 onboarding：命令、焦点、改前必读 |
| `.doc/` | 可复用长期知识（本文） |
| `.issue/` | 暂缓未闭环问题现场 |
| `CHANGELOG.md` | 版本历史 |

---
status: suspended
created: 2026-06-15
updated: 2026-06-15
---

# EZ-V2（v0.24）收口后：验收口径 + 后续研究方向 backlog

> **用途**：v0.24 开工前/收口后的架构锚点——明确「EZ-V2 做完要验什么」、以及「做完还缺什么、以后往哪扩」。
> **不是** v0.24 发版阻塞项；是 v0.25+ 规划与接缝设计的索引。
> **权威计划**：[RL 主线实施计划](../../../c:/Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md) · [rl_roadmap.md §5.8](../../.doc/design/rl_roadmap.md)

---

## 一、v0.24 EZ-V2 必须验证什么（发版门禁）

### 1.1 两层验收，不要混用

| 层级 | 算法 | 环境 ID | 门禁 | 状态 |
|------|------|---------|------|------|
| **架构跑通** | SAC / MuZero / PPO | **`CartPole-v0`** | greedy(temp=0) eval 20 局均值 **≥195** | ✅ v0.23.1 已收口 |
| **终极调优** | **EfficientZero V2（唯一）** | **一律 `-v1` / 新版 ID** | **各任务 EZ 专属指标**（见下表） | 🔲 v0.24 目标 |

**CartPole 在 EZ 层用 `CartPole-v1`（500 步上限），不再用 195 门槛**——与 v0/v1 分层、与 SAC/MuZero/PPO 的 v0 门禁 deliberately 区分。

### 1.2 EZ-V2 六格矩阵：验什么、验到什么程度

| # | 模式 | 示例路径 | 环境 | 验收侧重 | 达成形式 |
|---|------|----------|------|----------|----------|
| 1 | 完美信息博弈 | `examples/efficientzero/gomoku/` | `Gomoku-*-v0` | **同训练量胜率 ≥ v0.22 AlphaZero** | 固定局数/步数 budget 下对弈胜率（非 CartPole 式 reward 门槛） |
| 2 | 向量离散 | `examples/efficientzero/cartpole/` | **`CartPole-v1`** | **EZ 任务指标**（样本效率或 return 阈值，发版前在示例内写死并文档化） | 仍 **plugged self-play + MCTS**；greedy eval 或固定 episode return（**非** v0 的 195） |
| 3 | 图像离散 | `examples/efficientzero/atari/` | `ALE/*-v5` | **pixel pipeline + 训练闭环** | CNN 表征 + 完整 train loop；CPU 下分数**不作 SOTA 对标**，以「能学、loss 有限、无 panic」+ 合理 return 趋势为主 |
| 4 | 连续高维 | `examples/efficientzero/ant/` | `Ant-v5` 等 | **训练闭环 + Gumbel 搜索跑通** | plugged self-play；MuJoCo return 达 EZ 示例内写死的 best-effort 阈值 |
| 5 | 混合 Tuple | `examples/efficientzero/platform/` | `Platform-v0` | **离散+连续复合动作闭环** | Gumbel + hybrid action 编码；pipeline 跑通 + 示例内指标 |
| 6 | 离线（**可选**） | `examples/efficientzero/minari_pointmaze/` 或 `examples/offline/minari_pointmaze/` | **Minari** 数据集 | **pipeline smoke only** | 数据加载 + 一步/短训；**不进 EZ 性能门禁** |

### 1.3 训练范式：plugged 为主，offline 为辅

| 问题 | 定案 |
|------|------|
| EZ-V2 是否全部改成 offline？ | **否**。主路径仍是 **plugged self-play**（环境交互产局 → buffer → reanalyze 调强 → 训练），与 MuZero Unplugged 的「reanalyze 重搜」一致，但**不是**纯 Minari 离线 RL。 |
| Minari 是什么？ | [Minari](https://minari.farama.org/)：Farama 生态的**离线 RL 数据集**标准（类似 D4RL）。我们 v0.24 仅作 **可选 pipeline 验证**（能否 load 数据、跑一步训练），**不作**发版主验收。 |
| 与 SAC/MuZero/PPO 验收形式是否一样？ | **部分同构**：向量/图像 env 仍有 **greedy eval / episode return**；棋类用 **胜率**；Atari 不作 Gym 195 式单一数字。共性：**示例内写死判定函数 + release 手跑 + CHANGELOG 记录**。 |
| 发版命令（plan §验收） | `efficientzero_gomoku` / `cartpole` / `atari` / `ant`；`dqn_atari` 等辅助示例仅 pipeline |

### 1.4 v0.24 算法增量（须在 EZ 示例中可开关/可观测）

- value prefix
- reanalyze 调强（默认开或按 env 配置；降级见 plan）
- SVE
- Gumbel 连续搜索
- target network（EZ 增强）
- 图像 CNN obs pipeline（Atari）

**降级路径**：完整 reanalyze 实现超 ~1 **开发周** → v0.24 发 value prefix + SVE，完整 reanalyze 推 v0.25（plan 已定）。

---

## 二、EZ-V2 做完之后还缺什么（相对「MuZero 研究全谱系」）

> EZ-V2 是本项目 **v0.20–v0.24 RL 主线的工程收口**，不是 MuZero 所有变体的超集。下表 = 「地基打好后，扩展时知道缺哪块」。

### 2.1 已在 v0.22–v0.24 地基 / EZ 覆盖的

| 能力 | 落地位置 |
|------|----------|
| 离散 MCTS + learned dynamics | v0.23.1 MuZero canonical |
| categorical + latent norm + absorbing | v0.23.1 |
| reanalyze 机制 | `reanalyze_game`（v0.24 调强） |
| 离散/连续/混合 action **类型** | `ActionPayload`；SAC 已验 continuous/hybrid |
| 五子棋 self-play | v0.22 AlphaZero → v0.24 EZ 对标 |
| Gumbel 连续搜索 | v0.24 EZ 计划 |
| value prefix + SVE | v0.24 EZ 计划 |

### 2.2 EZ-V2 做完仍缺、按优先级排序的扩展方向

| 优先级 | 方向 | 与 EZ 关系 | 用到的接缝 | 参考 |
|--------|------|------------|------------|------|
| P1 | **完整 reanalyze 集成**（若 v0.24 降级） | EZ 核心，非新算法 | buffer + 训练循环 | MuZero Unplugged |
| P1 | **MCTS 并行**（rayon / virtual loss） | CPU 性能，非算法 | `predict_batch`、树内核 | mctx / 经典 MCTS 并行 |
| P2 | **Stochastic MuZero**（chance node） | **EZ 不包含**；真·结构随机环境 | **动树核心**（decision/chance 节点） | Antonoglou et al. 2022 |
| P2 | **纯 offline SOTA**（Minari 主路径） | EZ reanalyze 是半离线；纯 offline 是另一档 | buffer 来源 + 无 env step | MuZero Unplugged / CQL 类 |
| P3 | **Sampled MuZero** | 与 Gumbel 目标重叠，机制不同 | `ActionSampler` + 节点动作集 | Hubert et al. 2021 |
| P3 | **MENTS / RegPolicy / ANT** | 搜索变体，非 EZ 必需 | `SearchPolicy` 插件 | §5.10 rl_roadmap |
| P3 | **PER** | 样本优先级 | buffer 包装层 | 长期 backlog |
| P4 | **不完全信息**（扑克等） | **另一算法家族** | IS-MCTS / CFR，非 Zero 补丁 | 不在当前路线 |
| P4 | **多智能体 N>2** | self-play 契约不成立 | PettingZoo 级框架 | v0.25+ backlog |
| P4 | **19×19 围棋 / 象棋** | 算力问题 | 同 AZ/EZ，scale up | 明确不做于 v0.24 |
| P4 | **演化 NEAT + RL** | 与 RL 零耦合 | `src/nn/evolution/` | v0.25+ |
| P4 | **GPU / 分布式** | 与 CPU-only 约束冲突 | — | 明确不做 |

### 2.3 架构上 v0.24 收口后应「可插」而不必「全做」

| 接缝 | v0.22 已定 | 后续加实现不改签名 |
|------|------------|-------------------|
| `MctsModel` | root + recurrent | Stochastic → chance 语义（**例外：动 recurrent 契约**） |
| `SearchPolicy` | PUCT | Gumbel（v0.24）/ MENTS / Sampled |
| `ReplayBuffer<SelfPlayGame>` | 整局存储 | value prefix / reanalyze 元数据扩展 `SelfPlayStep` |
| `MuZeroConfig` | 按环境 sim 等 | 新 env 只加配置，不改库签名 |
| obs pipeline | 向量 | Atari CNN wrapper（v0.24） |

**原则**：新研究方向先问「能否只加 `SearchPolicy` / 示例 / `MuZeroConfig`」；只有 Stochastic / 不完全信息才动树或换算法族。

---

## 三、是否还要回头看 MuZero 周边版本？

| 来源 | v0.24 后是否还需单独做 | 说明 |
|------|------------------------|------|
| MuZero 原文 Reanalyze | 已入库；v0.24 调强 | 不必另起炉灶 |
| MuZero Unplugged | 部分覆盖 | EZ + reanalyze ≈ 半离线；纯 offline 主路径 = P2 backlog |
| Sampled MuZero | 可选 | Gumbel 已覆盖连续搜索主路径；Sampled 作备选研究 |
| Stochastic MuZero | **需要时单独立项** | EZ 不替代 |
| Gumbel MuZero | **并入 EZ-V2** | v0.24 必做 |
| EfficientZero V1 | **不单独追** | V2 已含 V1 思想 + SVE |
| LightZero / mctx | **参照实现** | 加变体时查接缝对照，不必 port 全库 |

---

## 四、建议的 v0.25+ 里程碑（草案，非承诺）

| 版本 | 焦点 | 触发条件 |
|------|------|----------|
| v0.25 | 完整 reanalyze（若 v0.24 降级）+ MCTS 并行 smoke | v0.24 发版后 |
| v0.25+ | Stochastic MuZero spike（单一 env，如 2048） | 有明确随机 env 需求 |
| v0.25+ | Minari 离线主路径（非 smoke） | 有 offline 产品需求 |
| v0.25+ | 演化 + RL 联合 | NEAT 阶段 D 或 RL 稳定后 |

---

## 五、恢复 / 更新本 issue 的条件

- v0.24 发版后：把 §一 验收表逐项标 ✅/❌，降级项写入 CHANGELOG。
- 新增研究方向立项前：在 §二 增行，标注接缝与参考论文。
- 某方向开始实现：可拆独立 `.issue/items/` 或 `.doc/design/` 设计节，本文件保持索引角色。

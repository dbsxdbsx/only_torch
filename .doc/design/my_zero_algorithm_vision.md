# MyZero 算法纲领与路线决策

> **定位**：记录 MyZero 及 RL 母算法选型的**战略层**结论——我们讨论过什么、为什么可能做、为什么大概率不做。  
> **不是**实施 checklist（见 [RL 路线图 §8](./rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿)）；**不是**消融实测表（见 [MyZero 示例总览](../../examples/my_zero/README.md)）。  
> **创建日期**：2026-06-20 · **状态**：讨论沉淀，条目可随新证据修订

---

## 1. 文档分工（读哪份）

| 文档 | 回答什么 |
|------|----------|
| **本文** | 算法哲学、文献谱系、做/不做决策、环境选型 |
| [rl_roadmap.md §8](./rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿) | v0.25 阶段、代码组织、消融顺序 |
| [examples/my_zero/README.md](../../examples/my_zero/README.md) | 组件×环境实测矩阵、命令、门禁 |
| [rl.instructions.md](../../.github/instructions/rl.instructions.md) | 改 RL 代码时的 agent 约束 |

---

## 2. 核心原则（已定）

> **用语**：本节是 MyZero 的**战略定锚**（选什么算法、怎么算成功、优化什么目标）。  
> **算法孰优孰劣**另见 §2.2 **首要评价指标**——有取舍时以该指标为准（产品文档里常称 North Star /「北极星指标」，本项目统一用「首要评价指标」）。

### 2.1 战略定锚

1. **母算法**：`src/rl/algo/my_zero/` —— 项目**唯一**的 `*Zero` 实现；learned-model MCTS + 消融驱动叠组件。
2. **成功判据**：各环境 **greedy(temp=0) eval** 达门槛（见 [示例 README](../../examples/my_zero/README.md)）；训练期随机性只是探索手段。
3. **优化目标（implicit）**：**期望回报最大化** / minimax，**不是** SAC 式 \(R + \alpha H(\pi)\) 最大熵目标。

> **可对外表述**：Zero 系默认认为——在标准 MDP 回报问题下，收敛后的**执行策略应是确定性的**（visit/value argmax）；POMDP 上应在 **belief 上**做 greedy，而非默认「最优 = 最大熵分布」。MaxEnt RL 回答的是**另一个 well-posed 问题**（meta-POMDP / 任务不确定下的鲁棒性），见 Eysenbach & Levine 2021。

### 2.2 首要评价指标

**env-steps-to-solved**：达到同一性能门槛所需的真实环境交互（env-steps）**越少越好**。

- **wall-clock**：研究迭代速度的约束，**不是**评价算法优劣的第一标尺。
- 三条推论（模型侧开销可压、红利前提是模型够准等）：见 [MyZero README — 评判口径](../../examples/my_zero/README.md#评判口径所有环境通用)。

---

## 3. 双轨架构（已定）

不存在单一「万金油」RL 底座；采用 **双轨**：

| 轨道 | 角色 | 典型环境 |
|------|------|----------|
| **MyZero** | 旗舰、样本效率、规划+学模型 | CartPole → Pendulum → Platform → Gomoku → … |
| **SAC / PPO** | model-free 基线、对照、开箱即用 | CartPole / Pendulum / Platform（Hybrid SAC 已通） |

**为什么保留 SAC**：离散/连续/混合动作已验证；Platform 等 MyZero 尚未覆盖；跨算法比 env-step 需要对照。

**为什么不把 SAC 升格为母算法**：棋类/self-play 需重造 MCTS 与 visit 蒸馏；与 MyZero 栈正交。

---

## 4. 文献谱系（讨论备忘）

两条线**几乎不交叉**，勿混为一谈：

```text
DeepMind Zero 系（回报最大化 + 部署 greedy）
  MuZero → EfficientZero → Gumbel MuZero → Stochastic MuZero → BetaZero(POMDP belief)

MaxEnt-MCTS 系（搜索内 Boltzmann backup，非完整 MuZero 栈）
  MENTS → TENTS/RENTS → ANTS → BTS/DENTS(NeurIPS 2023，修正 MENTS 目标错位)
```

| 工作 | 与 MyZero 关系 | 备注 |
|------|----------------|------|
| **MuZero** | 代码与组件已吸收进 `my_zero/` | Schrittwieser et al. 2020 · arXiv:1911.08265 |
| **EfficientZero** | consistency / value_prefix / reanalyze / SVE 等 | Ye et al. 2021 · arXiv:2111.00176 |
| **Scholz et al. 2021（reconstruction）** | CartPole ✅ reconstruction loss | *Improving Model-Based RL with Internal State Representations through Self-Supervision* · arXiv:2102.05599 |
| **Gumbel MuZero** | **计划**：Gumbel-root / completedQ | Danihelka et al. 2022 · arXiv:2111.00301；少 sims 的 policy improvement；**非 MaxEnt** |
| **Stochastic MuZero** | 远期：随机转移 | 学 \(p(s'\|s,a)\)，不要求策略随机 |
| **BetaZero** | 远期：POMDP / belief MCTS | **非 MaxEnt**；长 horizon 部分可观测 |
| **BTS/DENTS** | 若改 `SearchPolicy` backup 时**参考** | 比 MENTS/ANTS 理论更干净；未接 MuZero 全家桶 |
| **ANTS / MENTS** | **不**作为 MuZero 替代品全盘复刻 | Atari planning + 预训练 Q 习惯；与 DM 棋类栈不同范式 |
| **SAC / MaxEnt RL** | 卫星基线 | meta-POMDP 理论价值；**无 MCTS** |
| **Klein 2021 A0C 硕士论文** | 局部参考 | 连续 AZ + 训练 loss 固定 α 熵正则 + tanh policy；**搜索阶段无 entropy**；非 SAC⊕AZ 有机统一 |

**ANTS 是不是「AZ + MaxEnt 最好结合」？** —— 在「MaxEnt-MCTS + 在线闭环 + Atari」窄口径里是里程碑；**不是** 2026 全问题终局。同线后继看 **DENTS**；官方 Zero 演进看 **Gumbel / Stochastic / BetaZero**。

---

## 5. 决策表：可能做 / 暂缓 / 不做

图例：`🔲 可能做` · `⏸ 暂缓/远期` · `❌ 大概率不做` · `✅ 已在主线`

### 5.1 组件与方向

| 项 | 裁决 | 理由 |
|----|------|------|
| consistency | ✅ CartPole 已验收 | EfficientZero（Ye et al. 2021）+ SimSiam（Chen & He 2020） |
| reconstruction | ✅ CartPole 已验收 | Scholz et al. 2021 *Improving Model-Based RL with Internal State Representations through Self-Supervision*（arXiv:2102.05599）；seed=42 ~11.7k env-steps |
| value_prefix / target_net / SVE | ✅ 已在库，消融驱动 | EZ 谱系；CartPole value_prefix ❌ |
| completedQ / Gumbel-root | 🔲 Pendulum 起重点 | DM 官方 policy improvement 线 |
| Sampled MuZero（高维连续候选） | ⏸ 离散化够用后再上 | CartPole/Pendulum 规模暂不需要 |
| BTS/DENTS 式 backup | 🔲 仅当动 `SearchPolicy` | 避免 MENTS 式「max-entropy 最优 ≠ 回报最优」 |
| ANTS 自适应温度闭环 | ⏸ 借鉴思想，不搬整套 | 与 MuZero 训练循环结构不同 |
| Stochastic dynamics 头 | ⏸ Platform / 随机 env 需要时 | Stochastic MuZero 路线 |
| Belief-space MCTS | ⏸ 明确 POMDP 需求时 | BetaZero 路线 |
| 训练侧 SAC 式 soft Bellman | ❌ 不并入 MyZero trainer | 换目标函数；保留 SAC 示例即可 |
| MENTS/ANTS 替换 PUCT 为默认 | ❌ | 问题设定不同；非 EZ/MuZero 主栈 |
| SAC 扩展棋类 self-play | ❌ | 需从零造搜索+蒸馏 |
| DQN 入库 | ❌ | v0.21 已定；SAC 离散已覆盖 |
| 单一算法覆盖全动作×全状态×POMDP | ❌ | 不现实；双轨 + 插件 |

### 5.2 概念澄清（避免再讨论时跑偏）

| 说法 | 对错 |
|------|------|
| hidden task / meta-POMDP（Eysenbach） | **≠** 仅 sparse terminal reward |
| 随机转移环境 | 最优策略**仍常可确定性**（同一 state 固定 action） |
| POMDP 最优策略 | **未必** raw observation 上 greedy；information-gathering 问题里 **π\* 可本身随机** |
| Zero 系「证明不要 MaxEnt」 | **无**此类定理；是**问题设定 + FOMDP 经典结果 + 工程传统** |
| Klein A0C = AlphaZero + SAC 最佳统一 | **否**；熵正则只在训练 loss，MCTS 无 MaxEnt backup |

---

## 6. 环境 × 默认算法 × 插件位（规划）

> 默认算法 = **首选验证/MyZero 推进**；基线 = 应对照。Platform MyZero 子目录待建。

| 环境 | 观测 | 动作 | MyZero 状态 | 默认对照 | 远期插件位 |
|------|------|------|-------------|----------|------------|
| CartPole-v1 | 向量 | 离散 | ✅ 回归哨兵 ~**11.7k** steps（cons+recon） | SAC ~82k | completedQ ✅；Gumbel 边际 |
| Pendulum-v1 | 向量 | 连续 | ⏳ 失败区间，先诊断可学习性 | SAC | Gumbel-root、连续候选 |
| Platform-v0 | 向量 | 混合 Tuple | — 未实现 | Hybrid SAC ✅ | `ActionAdapter` Tuple、混合 MCTS |
| Gomoku | 离散棋盘 | 离散 |  backlog | — | self-play、legal_mask |
| 随机转移 / 长 horizon | — | — | ⏸ | — | stochastic dynamics |
| 部分可观测 | — | — | ⏸ | — | belief（BetaZero 类） |

**Platform 阻塞点（已知）**：MyZero `ActionAdapter` 仅 Auto/Discretize，Tuple 混合未支持——扩 Platform 前先扩动作适配层，而非换 SAC 母算法。

---

## 7. 推荐阅读顺序（接手 RL 战略时）

1. 本文 §2–§5（5 分钟）
2. [MyZero README 矩阵](../../examples/my_zero/README.md) + CartPole/Pendulum 子 README
3. [rl_roadmap.md §8](./rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿) 实施顺序
4. 若动搜索：**Gumbel MuZero (2022)** → 若考虑熵 backup：**DENTS (2023)**，**不是**先读 ANTS
5. 若动 POMDP：**BetaZero (2024)** + 教科书 POMDP（Tiger 等反例）
6. 若理解为何要随机策略：**Eysenbach & Levine 2021**（meta-POMDP，与 MCTS 正交）

---

## 8. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-06-20 | 初版：沉淀 SAC vs MyZero、MaxEnt 谱系、ANTS/BTS/Gumbel/BetaZero、POMDP/greedy 口径、Klein 论文评估、双轨决策 |
| 2026-06-20 | §2 用语：「北极星」拆为 **核心原则** + **首要评价指标**（env-steps-to-solved） |
| 2026-06-21 | §5.1 / §6：CartPole reconstruction 验收 ~11.7k env-steps（consistency + reconstruction） |

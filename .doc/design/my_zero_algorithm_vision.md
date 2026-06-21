# MyZero 算法纲领与路线决策

> **定位**：记录 MyZero 及 RL 母算法选型的**战略层**结论——我们讨论过什么、为什么可能做、为什么大概率不做。  
> **不是**实施 checklist（见 [RL 路线图 §8](./rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿)）；**不是**消融实测表（见 [MyZero 示例总览](../../examples/my_zero/README.md) — 组件×环境矩阵）。**组件→论文**对照见本文 §4.1。  
> **创建日期**：2026-06-20 · **状态**：讨论沉淀，条目可随新证据修订

---

## 1. 文档分工（读哪份）

| 文档 | 回答什么 |
|------|----------|
| **本文** | 算法哲学、文献谱系、做/不做决策、环境选型；**组件→论文**见 §4.1 |
| [rl_roadmap.md §8](./rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿) | v0.25 阶段、代码组织、消融顺序、改动纪律 |
| [examples/my_zero/README.md](../../examples/my_zero/README.md) | 组件×环境实测矩阵、命令、门禁 |
| [rl.instructions.md](../../.github/instructions/rl.instructions.md) | 改 RL 代码时的 agent 约束（含搬运≠改进） |

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
  MuZero → EfficientZero → Gumbel MuZero → Stochastic MuZero
  （文献叙事常并列 BetaZero(POMDP belief)——**另一问题类、本项目 ❌ 不学**，见 §5.3）

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
| **BetaZero** | **❌ 不学**（2026-06-21 定稿） | Moss et al. 2024 · [2306.00249](https://arxiv.org/abs/2306.00249)；belief 规划、**已知** \(T,O\)——与黑盒 MyZero 设定冲突，见 §5.3 |
| **BTS/DENTS** | 若改 `SearchPolicy` backup 时**参考** | 比 MENTS/ANTS 理论更干净；未接 MuZero 全家桶 |
| **ANTS / MENTS** | **不**作为 MuZero 替代品全盘复刻 | Atari planning + 预训练 Q 习惯；与 DM 棋类栈不同范式 |
| **SAC / MaxEnt RL** | 卫星基线 | meta-POMDP 理论价值；**无 MCTS** |
| **Klein 2021 A0C 硕士论文** | 局部参考 | 连续 AZ + 训练 loss 固定 α 熵正则 + tanh policy；**搜索阶段无 entropy**；非 SAC⊕AZ 有机统一 |

**ANTS 是不是「AZ + MaxEnt 最好结合」？** —— 在「MaxEnt-MCTS + 在线闭环 + Atari」窄口径里是里程碑；**不是** 2026 全问题终局。同线后继看 **DENTS**；DeepMind Zero 演进我们跟 **Gumbel / Stochastic**；**BetaZero 不纳入**（§5.3）。

### 4.1 组件文献对照（单一事实源）

> 组件×环境**实测状态**见 [MyZero 示例 README — 内部组件进展](../../examples/my_zero/README.md#内部组件进展团队--promote-时改-recipers)。本节只回答「每个库内组件对应哪篇论文」。

| 组件 | 论文（英文原标题 · 年份 · arXiv） | 备注 |
|------|-----------------------------------|------|
| **base**（MuZero 三网络 + latent MCTS） | Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* · 2020 · [1911.08265](https://arxiv.org/abs/1911.08265) | categorical value/reward、K 步 unroll、visit 蒸馏 |
| **consistency** | Ye et al., *Mastering Atari Games with Limited Data*（EfficientZero）· 2021 · [2111.00176](https://arxiv.org/abs/2111.00176) | EZ 自监督一致性 |
| | Chen & He, *Exploring Simple Siamese Representation Learning with SimSiam* · 2020 · [2011.10566](https://arxiv.org/abs/2011.10566) | 一致性 **loss 实现**参照（非 EZ 原文独有） |
| **reconstruction** | Scholz et al., *Improving Model-Based Reinforcement Learning with Internal State Representations through Self-Supervision* · 2021 · [2102.05599](https://arxiv.org/abs/2102.05599) | 独立于 EZ 谱系 |
| **reanalyze** | Ye et al., *Mastering Atari Games with Limited Data*（EfficientZero）· 2021 · [2111.00176](https://arxiv.org/abs/2111.00176) | position 级 MCTS 重搜 + buffer 写回；MuZero 系亦有同源机制 |
| **value_prefix** | 同上（EfficientZero）· [2111.00176](https://arxiv.org/abs/2111.00176) | LSTM reward 前缀 + hidden 穿树 |
| **target_net** | 同上（EfficientZero）· [2111.00176](https://arxiv.org/abs/2111.00176) | hard / EMA 同步 target network |
| **SVE** | 同上（EfficientZero）· [2111.00176](https://arxiv.org/abs/2111.00176) | search value 修正 stale target；现实现为固定权重，自适应 mixed target 见 §5.1 |
| **completedQ** | Danihelka et al., *Policy Improvement by Planning with Gumbel*（Gumbel MuZero）· 2022 · [2111.00301](https://arxiv.org/abs/2111.00301) | 训练侧策略 target（Eq.10–12） |
| **Gumbel-root** | 同上（Gumbel MuZero）· [2111.00301](https://arxiv.org/abs/2111.00301) | 搜索侧序贯减半（`GumbelPolicy` + `MyZeroSearchPolicy` 已接训练循环） |
| **连续采样候选**（Sampled MuZero） | Hubert et al., *Learning and Planning in Complex Action Spaces with Deep Neural Networks* · 2021 · [2104.06303](https://arxiv.org/abs/2104.06303) | 大/连续动作空间：每节点采 K 候选 + PUCT 用 π̂_β；**B/K 选型**见 [issue](../../.issue/items/my_zero_action_space_sampled_policy.md) |

---

## 5. 决策表：可能做 / 暂缓 / 不做

图例：`🔲 可能做` · `⏸ 暂缓/远期` · `❌ 大概率不做` · `✅ 已在主线`

### 5.1 组件与方向

| 项 | 裁决 | 理由 |
|----|------|------|
| consistency | ✅ CartPole 已验收 | EfficientZero（Ye et al. 2021）+ SimSiam（Chen & He 2020） |
| reconstruction | ✅ CartPole 已验收 | Scholz et al. 2021 *Improving Model-Based RL with Internal State Representations through Self-Supervision*（arXiv:2102.05599）；seed=42 ~12.2k env-steps @ sims=20 |
| value_prefix / target_net | ✅ 已在库，消融驱动 | EZ 谱系；CartPole value_prefix ❌ |
| SVE | ✅ 已在库，消融驱动 | 现固定权重 blend stale target；CartPole 训练循环待接 |
| SVE 自适应 mixed target | 🔲 Phase 2 改进候选 | EZ 论文为自适应 mixed target，非固定权重；须单独消融 |
| consistency 正规化（SimSiam target encoder + EMA） | 🔲 Phase 2 改进候选 | 现简化实现 CartPole 有效；见 [Pendulum 诊断 issue](../../.issue/items/pendulum_failure_diagnosis.md) §六 |
| value head 上游 target（Pendulum 坍缩） | 🔲 Phase 2 改进候选 | head 容量已证伪；根因在上游 n-step / 搜索，见同上 issue |
| completedQ / Gumbel-root | ❌ CartPole 实测失败；⏸ 复测留 `\|A\| > n` 环境 | 见 [issue](../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md)；库内已实现 |
| Sampled MuZero（高维连续候选） | ✅ CartPole recipe 已开 + release 压测（2026-06-22）；⏸ factorized B=7 待 Pendulum | cons+recon+Sampled：greedy **491.6** @ ep300 · **15,193** env-steps（N=2 K_eff=2 退化全枚举，仍过 475）；**B/K/N 公式**见 [issue](../../.issue/items/my_zero_action_space_sampled_policy.md) |
| BTS/DENTS 式 backup | 🔲 仅当动 `SearchPolicy` | 避免 MENTS 式「max-entropy 最优 ≠ 回报最优」 |
| ANTS 自适应温度闭环 | ⏸ 借鉴思想，不搬整套 | 与 MuZero 训练循环结构不同 |
| Stochastic dynamics 头 | ⏸ Platform / 随机 env 需要时 | Stochastic MuZero 路线 |
| Belief-space MCTS（BetaZero） | **❌ 不学** | 2026-06-21 定稿；需已知 \(T,O\)，与 Gymnasium 黑盒目标冲突，见 §5.3 |
| POMDP / 部分可观测（MyZero 侧） | 🔲 默认 history / 帧堆叠 | 表征层扩展；**非** belief MCTS |
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
| BetaZero = MyZero 的 POMDP 下一站 | **否**；已知 \(T,O\) 的 belief 规划，与学 latent 模型正交 |
| 万金油 MyZero 应内置 BetaZero | **否**；黑盒 env 只有 `step(obs,a)`，见 §5.3 |

### 5.3 BetaZero 裁决（2026-06-21 定稿）

**结论**：论文已读（本地 `BetaZero_Belief-State_POMDP_2306.00249.pdf`），**明确不学习、不实现、不并入 MyZero**。

| 维度 | BetaZero（Moss et al. 2024） | 本项目 MyZero |
|------|------------------------------|---------------|
| 问题设定 | POMDP **规划**；\(T,O,R\) **已知** | 黑盒交互；通常只知 obs / action 空间 |
| 环境模型 | **不学**转移/观测；粒子滤波 + belief MCTS | **学** value-equivalent latent dynamics |
| 规划对象 | belief \(b\) + \(\phi(b)\) 摘要 | latent state（+ 可选 history） |
| 典型场景 | 地质勘探、CCS 等**自带仿真器** | Gymnasium `make` / `step` |

**为何不学**：

1. **与母算法哲学相反**：MuZero/MyZero 的价值是「无规则、从交互学模型」；BetaZero 假设模型已写在仿真器里。
2. **与接口不匹配**：Gymnasium 不给 \(T(s'|s,a)\)、\(O(o|a,s')\)；硬接 belief MCTS 不现实。
3. **与 Stochastic MuZero 正交**：后者管**完全可观测 + 结构随机**；BetaZero 管**部分可观测 + 已知模型**——不是「再叠一层就 universal」。

**POMDP / 部分可观测时 MyZero 怎么办**（不引入 BetaZero）：

- **默认**：表征吃 **history / 帧堆叠**（与 Atari 帧堆叠同理；[RL 路线图 §2.2.1b](./rl_roadmap.md)）。
- **继续强化 latent**：consistency + reconstruction（已验收）帮表征保留观测信息。
- **真·结构随机且可观测**：远期 **Stochastic MuZero**（chance node），不是 belief MCTS。
- **不完全信息博弈**（扑克等）：仍 **❌ 不在路线**（information-set MCTS / CFR，另一算法族）。

**可单独借鉴、不打包 BetaZero**：Q-weighted visit policy、prioritized action widening 等训练/搜索技巧——若未来有证据再单项 ablation，**不**引入 belief 树或已知 \(T,O\) 依赖。

---

## 6. 环境 × 默认算法 × 插件位（规划）

> 默认算法 = **首选验证/MyZero 推进**；基线 = 应对照。Platform MyZero 子目录待建。

| 环境 | 观测 | 动作 | MyZero 状态 | 默认对照 | 远期插件位 |
|------|------|------|-------------|----------|------------|
| CartPole-v1 | 向量 | 离散 | ✅ 回归哨兵 ~**12.2k** steps（cons+recon · PUCT · sims=20） | SAC ~82k | completedQ / Gumbel ❌ CartPole · [issue](../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md) |
| Pendulum-v1 | 向量 | 连续 | ⏳ 失败区间，先诊断可学习性 | SAC | Gumbel-root、连续候选 |
| Platform-v0 | 向量 | 混合 Tuple | — 未实现 | Hybrid SAC ✅ | `ActionAdapter` Tuple、混合 MCTS |
| Gomoku | 离散棋盘 | 离散 |  backlog | — | self-play、legal_mask |
| 随机转移 / 长 horizon | — | — | ⏸ | — | stochastic dynamics |
| 部分可观测 | — | — | 🔲 | — | history / 帧堆叠（**非** BetaZero） |

**Platform 阻塞点（已知）**：MyZero `ActionAdapter` 仅 Auto/Discretize，Tuple 混合未支持——扩 Platform 前先扩动作适配层，而非换 SAC 母算法。

---

## 7. 推荐阅读顺序（接手 RL 战略时）

1. 本文 §2–§5（5 分钟）
2. [MyZero README 矩阵](../../examples/my_zero/README.md) + CartPole/Pendulum 子 README
3. [rl_roadmap.md §8](./rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿) 实施顺序
4. 若动搜索：**Gumbel MuZero (2022)** → 若考虑熵 backup：**DENTS (2023)**，**不是**先读 ANTS
5. 若理解 POMDP 与 MyZero 边界：本文 **§5.3**（BetaZero **不学**）+ 教科书 POMDP（Tiger 等反例）
6. 若理解为何要随机策略：**Eysenbach & Levine 2021**（meta-POMDP，与 MCTS 正交）

---

## 8. 修订记录

| 日期 | 变更 |
|------|------|
| 2026-06-20 | 初版：沉淀 SAC vs MyZero、MaxEnt 谱系、ANTS/BTS/Gumbel/BetaZero、POMDP/greedy 口径、Klein 论文评估、双轨决策 |
| 2026-06-20 | §2 用语：「北极星」拆为 **核心原则** + **首要评价指标**（env-steps-to-solved） |
| 2026-06-21 | §5.1 / §6：CartPole reconstruction 验收 ~11.7k env-steps（consistency + reconstruction） |
| 2026-06-21 | §5.1：拆分 SVE / 补 Phase 2 改进候选（SVE 自适应、consistency 正规化、value 上游 target） |
| 2026-06-21 | §4.1：组件文献对照迁入本文（单一事实源）；示例 README 改链入 |
| 2026-06-21 | §5.3：BetaZero 论文已读，**❌ 不学**（已知 \(T,O\) belief 规划 vs 黑盒 MyZero）；POMDP 默认 history |

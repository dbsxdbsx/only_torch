# MyZero 组件：问题与可测性

> **定位**：只回答两件事——每个旋钮在解决什么、环境侧能不能测到（以及何时能测）。  
> **不是**战略 / 裁决权威。开不开、promote 与否只认 [状态总览](rl_myzero_status.md) §3。  
> **不是**第三份路线图。不抄组件×环境效果矩阵，不写剂量。  
> 代码对照：[`route.rs`](../../src/rl/algo/my_zero/route.rs) 目前只实现阶段 ① 的一部分；生产路径仍是 `env_id` recipe。

写本文是为了先把「结构 → 锁栈」的思路钉死，再决定要不要把结构路由接到 `MyZero::new`。

---

## 0. 一无所知时怎么学（只谈逻辑，不谈某环境测过正负）

假设对新环境零先验。要学的是「怎么配置学习器」，不是先猜它叫什么名字。

学习器自己有一个事实：**value/reward 进的是有限个原子**（现在 41 个、变换域 `[-20,20]`）。未知单位的实数进有限码本，逻辑上必须一开始就有一层**无墙、可逆、小处近线性**的压缩。这是算法的性质，不是环境的性质，所以**不必跑、不必等证伪**。

代码已是 always-on `symlog`（[`value_transform`](../../src/rl/algo/my_zero/value_transform.rs)），再进 `[-20,20]`。

| 进网的实数 | 逻辑默认（开训前锁死） | 不要做的 |
|------------|------------------------|----------|
| 加总后的 return / value | 永远 `symlog`，再进 `[-20,20]` | 线性预缩放；`tanh`；中途改尺度 |
| 连续观测（向量 / 图像 / 棋盘 / aux） | 永远 `symlog`（模型入口；buffer 仍存 raw）。观测**不进** 41 原子 | 读 `Box` 比维间倍数；设 10× / 100× 门槛 |
| token ID | 不变换（查表前必须是整数） | 对 ID 做 `symlog` |

buffer 只存 raw。观测也不读 Box：维间「差 10 倍」会沿链叠成 100 倍，门槛本身站不住。线性 `reward_scale` 已删除，量纲交给 always-on `symlog`。

还需要「学」的只剩：

1. **① 接口事实**（不是跑分）：空间还是向量、离散还是连续、joint `N` 与已有 `sims` 算完是否 `K_eff < N`。Gym space + 搜索预算里就有。  
2. **`!` 有网之后**：标签刷新有没有比旧标签更接近回报。没网就没有这个数。

本节不引用任何环境消融结果。

---

## 0.1 两档（测的是适应症，不是效果）

| 阶段 | 何时 | 能看见什么 | 不能拿它决定什么 |
|------|------|------------|------------------|
| **① init / schema** | `gymnasium.make` 之后、建网之前 | 观测类型与维数、动作类型、联合动作数 \|A\| | 该组件训完有没有用 |
| **`!` 训练中裁判（先不做）** | 必须先有网才可能定 | 新旧 target 谁更接近回报等 | 开局探测；本阶段路由把它当成 `—` |

本表只问：有没有固定谓词，从而给出开/关默认。训完涨不涨分一律不在本表——那是 3-seed，对所有组件都成立，不必再标一档。  
母算法不换：结构路由只决定 MyZero 里哪些损失 / 子网开着，不改去 PPO / SAC。  
改图结构的组件（Stochastic 子网、decoder、posterior）必须在开训前锁死。

图例：

| 标记 | 意思 |
|------|------|
| `①` | **正式用神经网络训练之前就能定。** 看 Gym space（0 局），还没有 MyZero 网。 |
| `!` | 有网之后才可能用数字裁判决定开（reanalyze / ROSMO 等）。**逻辑口号清楚，自动开关未写。** 现阶段路由当作 `—`，环境测试先不管。 |
| `—` | 测不到：没有环境谓词，路由上不靠探测去开 |

量纲见 §0：value 与连续观测都是 always-on `symlog`，不是路由。中途热切仍然不合适。

`!` 的过程才是训练本身，尺子是当前网算出来的标签 / KL / 预测误差。

**`!` 以后怎么判 reanalyze / ROSMO（先不做，可用 ≠ 涨分）**

两者都在做同一件事：buffer 里的 policy/value 标签过期了，用**当前网**再算一版来训练。reanalyze 是全树重搜并写回；ROSMO 是只向前看一步、算完即弃。

判定不看「是不是图像 / 奖励稀不稀」，看三个数谁更接近：

| 符号 | 哪来的 | 要不要网 |
|------|--------|----------|
| `z` | 这条轨迹后面真正拿到的回报（环境事实） | 不要 |
| `ν_old` | 当初 self-play 写进 buffer 的 MCTS 标签 | 当时的网（已过期） |
| `ν_new` | 现在重算的标签（全树 = reanalyze，一步 = ROSMO） | **要，当前网** |

规则（状态总览已写；**机制代码有，比较 `|ν_new−z|` 并翻开关的代码没有**）：

- `|ν_new − z|` 稳定小于 `|ν_old − z|` → 刷新在帮标签，机制**可用**（仍须 3-seed 才谈涨分）
- `|ν_new − z|` 更大 → 弱网在自己改自己，**不可用**；reanalyze 还会把坏标签写回 buffer
- 早期网很弱时默认当不可用（「弱模型阶段禁止」）

这就是必须先有模型：没有当前网，就没有 `ν_new`，三个数比不成。CartPole 上全树 reanalyze 学崩、ROSMO 无增益，都是这条尺子（加 3-seed）裁过的，不是结构路由裁的。

---

## 1. `Components` 消融开关

| 组件 | 解决什么问题 | 可测性 | 不同环境怎么测（或为什么测不到） |
|------|--------------|--------|----------------------------------|
| **consistency** | latent 对齐下一帧编码，给 dynamics 额外自监督（稀疏任务信号时世界模型不够用） | ① | 谓词：空间观测 → 开；否则关。**图像 / 棋盘**开，**向量**关。CartPole 生产额外开了，是 overlay，不是这条谓词。 |
| **reconstruction** | decoder 对齐真实 obs，逼 latent 留住任务相关比特 | ① | 谓词：`Flat` 且维数 ∈ (0, 16] → 开；否则关。高维向量 / 图像 / 棋盘 / Tokens 关。 |
| **Sampled** | 一次 search 每个节点只展开 K 个 joint，而不是整张菜单 | ① | 谓词：`K_eff < N`（K 公式见 [`sampled_params`](../../src/rl/algo/my_zero/sampled_params.rs)）。读已有 `sims`，不负责选定 sims。连续 / 混合先算 joint `N`，同一句话，不是类型必开。CartPole N=2 → 关；Pendulum N=7 sims=20 → 开；9×9 / 15×15 → 开。生产 CartPole 额外开了 Sampled，是 overlay。 |
| **Stochastic K** | 同一 `(s,a)` 对应多种下一状态；chance 变量吸收随机性 | — | **任何环境**：几局随机几乎撞不到同一 `(s,a)` 两次；商业游戏也没有 snapshot。故 **K=8 always-on**，确定性退化为单峰。自适应 K 仍是悬案，不要用随机轨迹去关。StochasticCartPole 的随机性是适配层事实。 |
| **Gumbel-root** | 少 sim 时根节点用 Gumbel 替代 PUCT，决策更稳 | ① | 谓词看得到（连续动作或 `sims ≪ \|A\|`），默认关：与 Sampled 暂不兼容，连续动作走 Sampled。 |
| **completedQ** | 少 sim 时用 completed Q 当策略目标，替代 visit-count | ① | 谓词看得到（`sims / \|A\|`），默认关：本仓库未证实该适应症。 |
| **obs symlog** | 把进网的连续观测压到可训尺度 | — | 与 value 同类：算法入口性质，不靠环境探测。连续观测 always-on；**token ID 除外**。不读 `Box`，不设维间倍数门槛。**禁止中途热开**。 |
| **HL-Gauss** | value/reward 宽 support、高噪声时，用高斯软标签代替 two-hot | — | 随机策略回报看不出像样策略下的 value 噪声。CartPole 已劣化；图像未复测。路由关。 |
| **recurrent posterior** | 单帧观测不 Markov，用历史推断隐状态 | — | Gym shape 看不出缺了哪一维。缺速度的 CartPole 只是 `Flat(3)`。须适配层声明（或 `obs_mask`）。POMDP-lite 已负裁；默认关。 |
| **reanalyze** | replay 标签陈旧时，全树重搜并写回 | `!` | 现阶段当 `—`。CartPole 已学崩。先不做自动开。 |
| **ROSMO** | 同上，但只做一步 look-ahead、不写回 | `!` | 现阶段当 `—`。CartPole 无增益、棋盘有害。先不做自动开。 |
| **target_net** | n-step 尾值用滞后 V，减轻自举抖动 | — | 不是环境结构，看不见 bootstrap 噪不噪。MinAtar 已负。路由关。 |
| **SVE** | 把搜索价值混进 n-step target | — | 接线未完成。无结构谓词。不要用环境类型猜。 |
| **value_prefix** | 用 LSTM 累加 reward 前缀，改善长程信用 | — | 适应症像「长地平线 + 稠密逐步奖励」，但实现有断链（训 LSTM、搜普通 reward head）。未修前路由不得开。地平线长度不够当谓词。 |

系数（`consistency_coef` / `reconstruction_coef` / `rosmo_alpha` / `cq_*`）是剂量，不是「开不开」；不在本表探测。

---

## 2. 不在 `Components` 里的过程旋钮

这些不是环境结构，阶段 ① 不该用它们「自动选学习方法」。

| 旋钮 | 解决什么问题 | 可测性 | 说明 |
|------|--------------|--------|------|
| **KL 自适应 lr** | 用探针 KL 自动升降学习率，少手调 lr | `!` | 现阶段当 `—`。CartPole 灾难级。 |
| **PER** | 按 \|ν−z\| 重排消费，补覆盖 | `!` | 现阶段当 `—`。棋盘负。 |
| **主动数据** | 按 world-model 误差去要新数据 | `!` | 现阶段当 `—`。3A0 负裁。 |
| **（已删）reward_scale** | 曾线性预缩放逐步 reward | — | 已删除。量纲由 always-on `symlog` + `[-20,20]` 吸收。旧 `.otm` 里的字段加载时忽略。 |

`sims` / `td_steps` / `lr` / `trains_per_episode` 是训练预算与超参，不是「环境测出来的组件」。Sampled 只**读取**已有 `sims` 来算 `K_eff`，不按探测去改 sims。MinAtar 上改 sims / td 已有负诊断，不按探测自动改。

---

## 3. 按环境原型：schema 看见什么

| 原型 | ① init 已够 | 仍然看不见 |
|------|-------------|------------|
| **CartPole 式** `Flat(4)` + 离散 2 | 低维 → recon 默认；N=2 → `K_eff=N`，Sampled 关；非空间 → consistency 非结构必开；连续观测 → obs_symlog | 确定性、要不要 overlay 开 consistency/Sampled |
| **Pendulum 式** 连续动作 | N=B=7、sims=20 → `K_eff=5<7`，Sampled 开；低维 → recon 默认；连续观测 → obs_symlog | n-step 方差会不会塌（诊断项，不是组件探测） |
| **Platform 混合** | joint N 通常 `K_eff<N` → Sampled；Flat(10) → recon | — |
| **MinAtar / Atari 式** 网格 + 小离散 | 空间 → consistency 默认；非低维向量 → recon 关；N≤5 → Sampled 关，N≥6（常见 sims）→ 开 | 该不该 HL-Gauss / target_net / 加 sims |
| **五子棋式** 大棋盘 | 空间 → consistency 默认；N=81/225 → `K_eff<N`，Sampled 结构默认 | 生产 recipe=base（Sampled 关）是 overlay，不是探测失败 |
| **StochasticCartPole** | 与 CartPole 同 schema | 随机性靠声明；K 保持 always-on |
| **POMDP-lite** | 若无人声明，只看见更短的 Flat | 必须声明；默认不开 posterior |
| **Tokens** | 非空间、非低维 Flat → 两套自监督都关 | 序列任务要不要别的记忆机制（超出本表） |

---

## 4. 和现有代码的关系

- [`route_components`](../../src/rl/algo/my_zero/route.rs) = 阶段 ①。单测 [`tests/route.rs`](../../src/rl/algo/my_zero/tests/route.rs)。`value_support_noisy` / `partially_observable` 仍须声明。
- value/reward 编码 = always-on `symlog`。单测 [`tests/value_transform.rs`](../../src/rl/algo/my_zero/tests/value_transform.rs) / [`tests/value_encoding.rs`](../../src/rl/algo/my_zero/tests/value_encoding.rs)。
- 连续观测 `symlog` = always-on（[`ObservationSchema::uses_obs_symlog`](../../src/rl/algo/my_zero/schema.rs)）；token 关。runner 物化 / 训练时按 schema 注入，不读 Box。其余 ① 未接到 `MyZero::new`；`!` 不实现。

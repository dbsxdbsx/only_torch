# MyZero 组件：问题与可测性

> **定位**：只回答两件事——每个旋钮在解决什么、环境侧能不能测到（以及何时能测）。  
> **不是**战略 / 裁决权威。开不开、promote 与否只认 [状态总览](rl_myzero_status.md) §3。  
> **不是**第三份路线图。不抄组件×环境效果矩阵，不写剂量。  
> 代码对照：[`route.rs`](../../src/rl/algo/my_zero/route.rs) 目前只实现阶段 ① 的一部分；生产路径仍是 `env_id` recipe。

写本文是为了先把「结构 → 短探 → 锁栈」的思路钉死，再决定要不要把探测接到 `MyZero::new`。

---

## 0. 三阶段（测的是适应症，不是效果）

| 阶段 | 何时 | 能看见什么 | 不能拿它决定什么 |
|------|------|------------|------------------|
| **① init / schema** | `gymnasium.make` 之后、建网之前 | 观测类型与维数、动作类型、联合动作数 \|A\| | 该组件训完有没有用 |
| **② 短探** | 随机策略跑几局，**只对原始轨迹做算术** | 各维 min/max、\|r\| 量级、奖励是否几乎全 0 | 需要网络才能算的量（policy、value、预测误差） |
| **`!` 训练中裁判（先不做）** | 必须先有网才可能定 | 新旧 target 谁更接近回报等 | 开局探测；本阶段路由把它当成 `—` |

本表只问：有没有固定谓词，从而给出开/关默认。训完涨不涨分一律不在本表——那是 3-seed，对所有组件都成立，不必再标一档。  
母算法不换：探完只决定 MyZero 里哪些损失 / 子网开着，不改去 PPO / SAC。  
改图结构的组件（Stochastic 子网、decoder、posterior）必须在开训前锁死。

图例：

| 标记 | 意思 |
|------|------|
| `①` `②` | **正式用神经网络训练之前就能定。** ① 看 Gym space（0 局）；② 最多用环境自己的随机动作（`action_space.sample()`）跑几局做算术。都还没有 MyZero 网。当前只做这两档。 |
| `!` | 有网之后才可能用数字裁判决定开（reanalyze / ROSMO 等）。**逻辑口号清楚，自动开关未写。** 现阶段路由当作 `—`，环境测试先不管。 |
| `—` | 测不到：没有环境谓词，路由上不靠探测去开 |

② 现在实际只有 **obs symlog** 和 **`reward_scale`**（后者是适配，不是 `Components`）。  
奥卡姆：默认关，**不必为它们单独开探局**。训练自己会往 buffer 里写 raw 观测/奖励，隔一段时间用同一套算术看决定有没有被证伪即可。

**中途热切不合适。** 这两个改的是输入变换 / 奖励量纲，不是多加一项 loss。前面按 raw 尺度训过的权重，后面突然 symlog 或改 `reward_scale`，语义对不上（`obs_transform` 已写：训推必须同口径）。  
证伪了：停这趟或记下，**下一趟从一开始就开**；不要在同一趟里后半段打开。这不是 `!`：尺子仍是环境算术，不看网的标签。

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

这就是必须先有模型：没有当前网，就没有 `ν_new`，三个数比不成。CartPole 上全树 reanalyze 学崩、ROSMO 无增益，都是这条尺子（加 3-seed）裁过的，不是探测裁的。

---

## 1. `Components` 消融开关

| 组件 | 解决什么问题 | 可测性 | 不同环境怎么测（或为什么测不到） |
|------|--------------|--------|----------------------------------|
| **consistency** | latent 对齐下一帧编码，给 dynamics 额外自监督（稀疏任务信号时世界模型不够用） | ① | 谓词：空间观测 → 开；否则关。**图像 / 棋盘**开，**向量**关。CartPole 生产额外开了，是 overlay，不是这条谓词。 |
| **reconstruction** | decoder 对齐真实 obs，逼 latent 留住任务相关比特 | ① | 谓词：`Flat` 且维数 ∈ (0, 16] → 开；否则关。高维向量 / 图像 / 棋盘 / Tokens 关。 |
| **Sampled** | 动作 catalog 太大或连续时，只展开 K 个候选，避免无界 / 爆炸分支 | ① | **连续 / 混合**（Pendulum、Platform）：init 必须开。**大离散**（五子棋 15×15=225）：init 比联合动作与阈值 128。**小离散**（CartPole N=2、MinAtar 数个动作）：init 可知「只是簿记」。生产 CartPole 额外开了 Sampled，是 overlay，不是结构必然。 |
| **Stochastic K** | 同一 `(s,a)` 对应多种下一状态；chance 变量吸收随机性 | — | **任何环境**：几局随机几乎撞不到同一 `(s,a)` 两次；商业游戏也没有 snapshot。故 **K=8 always-on**，确定性退化为单峰。自适应 K 仍是悬案，不要用短探关。StochasticCartPole 的随机性是适配层事实，不是探测出来的。 |
| **Gumbel-root** | 少 sim 时根节点用 Gumbel 替代 PUCT，决策更稳 | ① | 谓词看得到（连续动作或 `sims ≪ \|A\|`），默认关：与 Sampled 暂不兼容，连续动作走 Sampled。 |
| **completedQ** | 少 sim 时用 completed Q 当策略目标，替代 visit-count | ① | 谓词看得到（`sims / \|A\|`），默认关：本仓库未证实该适应症。 |
| **obs symlog** | 连续特征跨数量级时压到可训尺度 | ①+② | 默认关。Flat 才可能开；图像/棋盘 ① 即可排除。证伪用 buffer 里 raw 尺度，**禁止中途热开**（须重开一趟）。 |
| **HL-Gauss** | value/reward 宽 support、高噪声时，用高斯软标签代替 two-hot | — | 短探看的是随机策略回报，不是像样策略下的 value 噪声。CartPole 已劣化；图像未复测。路由关。 |
| **recurrent posterior** | 单帧观测不 Markov，用历史推断隐状态 | — | Gym shape 看不出缺了哪一维。缺速度的 CartPole 只是 `Flat(3)`。须适配层声明（或 `obs_mask`）。POMDP-lite 已负裁；默认关。短探不能当 POMDP 检测器。 |
| **reanalyze** | replay 标签陈旧时，全树重搜并写回 | `!` | 现阶段当 `—`。CartPole 已学崩。先不做自动开。 |
| **ROSMO** | 同上，但只做一步 look-ahead、不写回 | `!` | 现阶段当 `—`。CartPole 无增益、棋盘有害。先不做自动开。 |
| **target_net** | n-step 尾值用滞后 V，减轻自举抖动 | — | 不是环境结构。短探看不见 bootstrap 噪不噪。MinAtar 已负。路由关。 |
| **SVE** | 把搜索价值混进 n-step target | — | 接线未完成。无结构谓词。不要用环境类型猜。 |
| **value_prefix** | 用 LSTM 累加 reward 前缀，改善长程信用 | — | 适应症像「长地平线 + 稠密逐步奖励」，但实现有断链（训 LSTM、搜普通 reward head）。未修前路由不得开。局长可粗探，不够当谓词。 |

系数（`consistency_coef` / `reconstruction_coef` / `rosmo_alpha` / `cq_*`）是剂量，不是「开不开」；不在本表探测。

---

## 2. 不在 `Components` 里的过程旋钮

这些不是环境结构，阶段 ① ② 都不该用它们「自动选学习方法」。

| 旋钮 | 解决什么问题 | 可测性 | 说明 |
|------|--------------|--------|------|
| **KL 自适应 lr** | 用探针 KL 自动升降学习率，少手调 lr | `!` | 现阶段当 `—`。CartPole 灾难级。 |
| **PER** | 按 \|ν−z\| 重排消费，补覆盖 | `!` | 现阶段当 `—`。棋盘负。 |
| **主动数据** | 按 world-model 误差去要新数据 | `!` | 现阶段当 `—`。3A0 负裁。 |
| **reward_scale** | 线性预缩放，让**累计** value 落在 support 中央 | ② | 默认 1。`h(x)` always-on，本身不封顶；贴边来自 `clamp` 到已写死的 `[-20,20]`。证伪则**整次训练重来、空 buffer**、从第一步就乘 0.1。禁止同一 buffer 混两种 scale。 |

`sims` / `td_steps` / `lr` / `trains_per_episode` 是训练预算与超参，不是「环境测出来的组件」。MinAtar 上改 sims / td 已有负诊断，不按探测自动改。

---

## 3. 按环境原型：init 看见什么、短探还能补什么

| 原型 | ① init 已够 | ② 短探还可能补 | 仍然看不见 |
|------|-------------|-----------------|------------|
| **CartPole 式** `Flat(4)` + 离散 2 | 低维 → recon 默认；N=2 → Sampled 非必须；非空间 → consistency 非结构必开 | 特征尺度小 → symlog 关；reward +1 稠密 → 不必 HL-Gauss | 确定性、要不要 overlay 开 consistency/Sampled |
| **Pendulum 式** 连续动作 | 连续 → Sampled 必须；低维 → recon 默认 | \|r\| 大 → `reward_scale`；特征尺度 → 是否 symlog | n-step 方差会不会塌（诊断项，不是组件探测） |
| **Platform 混合** | 混合 → Sampled；Flat(10) → recon | 同上（尺度 / 奖励） | — |
| **MinAtar / Atari 式** 网格 + 小离散 | 空间 → consistency 默认；非低维向量 → recon 关；N 小 → Sampled 关 | 像素范围（多已在 space 里）→ 不开 symlog | 该不该 HL-Gauss / target_net / 加 sims |
| **五子棋式** 大棋盘 | 空间 → consistency 默认；225≥128 → Sampled 结构默认 | 0/1 棋盘 → 不开 symlog | 生产 recipe=base（Sampled 关）是 overlay，不是探测失败 |
| **StochasticCartPole** | 与 CartPole 同 schema | 短探几乎判不出随机 | 随机性靠声明；K 保持 always-on |
| **POMDP-lite** | 若无人声明，只看见更短的 Flat | 轨迹看不出缺速度 | 必须声明；默认不开 posterior |
| **Tokens** | 非空间、非低维 Flat → 两套自监督都关 | — | 序列任务要不要别的记忆机制（超出本表） |

---

## 4. 和现有代码的关系

- [`route_components`](../../src/rl/algo/my_zero/route.rs) = 阶段 ①。单测 [`tests/route.rs`](../../src/rl/algo/my_zero/tests/route.rs)。
- [`probe_trajectories`](../../src/rl/algo/my_zero/probe.rs) = 阶段 ②（人造轨迹算术）。单测 [`tests/probe.rs`](../../src/rl/algo/my_zero/tests/probe.rs)。
- 两档都未接到 `MyZero::new`；`!` 不实现。接到生产是改行为：一次一项 + 3-seed，并处理 CartPole overlay。

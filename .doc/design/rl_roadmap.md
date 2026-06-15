# 强化学习路线图

> 本文档记录 RL 模块的当前状态、设计决策、已知差距与未来方向，
> 供后续开发者快速了解全貌并确定下一步工作。
>
> **创建日期**：2026-02-14
> **当前状态**：环境层 + 概率分布 + SAC 三变体示例已完成；**v0.18.0 后起为项目当前主线**（v0.20.0 目标：Gymnasium-only 环境 + `Transition`/`ReplayBuffer` 入库）
> **接手入口**：项目 [AGENTS.md](../../AGENTS.md#当前版本与焦点) · 环境 [rl_python_env_setup.md](../rl_python_env_setup.md) · 实施计划见 [§7](#7-v0200-实施计划)

---

## 目录

1. [当前状态](#1-当前状态)
2. [架构与设计决策](#2-架构与设计决策)
3. [已知差距与技术债](#3-已知差距与技术债)
4. [SAC 技术笔记：统一 Actor Loss 公式](#4-sac-技术笔记统一-actor-loss-公式)
5. [未来方向](#5-未来方向)
6. [参考](#6-参考)
7. [v0.20.0 实施计划](#7-v0200-实施计划)

---

## 1. 当前状态

### 1.1 已完成组件

| 组件 | 位置 | 说明 |
|------|------|------|
| **GymEnv** | `src/rl/env/gym_env.rs` | 支持离散 / 连续 / 混合动作空间、图像观察、自定义环境 |
| **MinariDataset** | `src/rl/env/minari.rs` | 离线 RL 数据集封装，提供 episode 采样 |
| **Step**（待删） | `src/rl/mod.rs` | 死代码；v0.20.0 由 **`Transition`** 取代，无 deprecated 过渡 |
| **Categorical** | `src/nn/distributions/` | 离散分类分布（probs / log_probs / entropy / sample） |
| **Normal** | `src/nn/distributions/` | 正态分布，支持重参数化采样（rsample） |
| **TanhNormal** | `src/nn/distributions/` | Squashed Gaussian，带 Jacobian 修正的 log_prob |
| **SAC-Discrete** | `examples/sac/cartpole/` | **CartPole-v0**（架构跑通；发版验收 **reward ≥ 195**） |
| **SAC-Continuous** | `examples/sac/pendulum/` | Pendulum-v1，连续动作，TanhNormal 策略 |
| **Hybrid SAC** | `examples/sac/platform/`（由 `moving/` 迁移） | **`Platform-v0`**（[`hybrid-platform`](https://pypi.org/project/hybrid-platform/)，Gymnasium 生态，**不用** gym-hybrid） |

### 1.2 目录结构

```
src/rl/
├── mod.rs              # 导出 env + buffer + algo + mcts + agent
├── agent.rs            # Agent / PlanningAgent 双 trait（v0.22 新增）
├── env/
│   ├── mod.rs
│   ├── gym_env.rs      # GymEnv：gymnasium.make + 规划桥接（legal_mask/snapshot/restore）
│   └── minari.rs       # MinariDataset（离线数据，独立 pip 包）
├── buffer/
│   ├── transition.rs   # Transition（off-policy 五元组）
│   ├── replay.rs       # ReplayBuffer<T: BufferItem>
│   └── self_play.rs    # SelfPlayStep / SelfPlayGame / GameOutcome（v0.22 新增）
├── algo/sac/           # SAC 函数式 helper（v0.21 新增）
├── mcts/               # MCTS 搜索引擎（v0.22 新增）
│   ├── mod.rs
│   ├── types.rs        # ActionPayload, RootOut, RecurrentOut, ChildStat, SearchResult, MctsConfig
│   ├── traits.rs       # MctsModel, SearchPolicy, Predictor
│   ├── node.rs         # Node, Edge, Tree（arena 模式）
│   ├── search.rs       # mcts_search 主循环
│   └── puct.rs         # PuctPolicy（Dirichlet 噪声 + UCB 选择 + 温度采样）
└── tests/
    ├── env/            # 环境测试（#[serial]）
    ├── buffer_replay.rs
    ├── buffer_self_play.rs  # SelfPlayGame 入库测试（v0.22 新增）
    └── algo_sac.rs     # SAC helper 测试（v0.21 新增）

python/gym_env/                  # 自定义 Gymnasium 环境包（v0.22 新增）
├── pyproject.toml               # pip install -e python/gym_env
├── __init__.py                  # gymnasium.register 集中入口
└── gomoku/                      # 五子棋
    ├── board.py                 # Board：纯规则 + clone/restore/legal_mask（增量 check_winner）
    ├── env.py                   # GomokuSelfPlayEnv（无对手）+ GomokuEnv（带 naive 对手）
    └── opponents.py             # naive 对手策略（random/naive0-3）

examples/sac/
├── README.md
├── cartpole/           # 主线 + smoke
├── pendulum/
├── platform/           # Hybrid SAC，Platform-v0
└── lunarlander/        # LunarLander-v3 离散 SAC（v0.21 新增）
```

---

## 2. 架构与设计决策

### 2.1 库只提供环境层，Buffer / Agent 由用户管理

**结论**：经评估，ReplayBuffer 与监督学习的 DataLoader 抽象层次不同，不应强行复用。

| 组件 | DataLoader | ReplayBuffer |
|------|-----------|--------------|
| 数据来源 | 静态数据集 | 动态增长的经验池 |
| 遍历方式 | 每 epoch 看到完整数据 | 随机采样固定数量 |
| 生命周期 | 训练前确定 | 训练中持续增长 |

详见 [DataLoader 设计文档](./data_loader_design.md) Phase 2 段落。

### 2.2 在线环境：仅 Gymnasium，不回退 OpenAI Gym

**结论（v0.20.0）**：`GymEnv` **只**调用 `gymnasium.make`；删除 `use_legacy_gym`、`py.import("gym")` 及双套 reset/step 分支。

**老 gym / 其他库环境**：不在 Rust 层兼容；改在 **Python 侧用 [`shimmy`](https://shimmy.farama.org/) 适配成标准 Gymnasium 环境**再经 `GymEnv` 接入（Rust 不 `import gym`；**注意 numpy ≥ 2.0 下未维护的老 gym 常因 `np.bool8` 无法加载，shimmy 也救不回**）。`GymEnv` 加载失败时给中文友好报错并指引 shimmy（**指路**，非自动识别老 gym）。详见 [环境搭建 — 老 gym 兼容](../rl_python_env_setup.md)。

| 概念 | 说明 |
|------|------|
| **Gymnasium** | OpenAI Gym 的官方继任者（Farama），API 与维护线以它为准 |
| **与「老 gym」** | 不是「真包含」：标准环境在 Gymnasium 里；**不采用** gym-hybrid（硬依赖老 `gym`）；混合动作用 **`hybrid-platform` / Platform-v0** |
| **扩展环境** | 通过 **extras** 或独立包安装，例如 `gymnasium[mujoco]`、`gymnasium[atari]`、自定义 `gymnasium.register` |
| **离线数据** | **不在** `gymnasium` 核心里；用 **Minari**（`pip install minari`），库侧已有 `MinariDataset` |

详见 [RL Python 环境搭建 — 生态分层](../rl_python_env_setup.md#生态分层gymnasium--扩展包--minari)。

#### 2.2.1 环境能力矩阵（Gymnasium-only 目标）

| 观察 \\ 动作 | 离散 | 连续 | 混合（Tuple 离散+连续） |
|-------------|------|------|-------------------------|
| **向量** | **CartPole-v0**（SAC/MuZero/PPO 架构验收 ≥195）、LunarLander-v3（离散） | Pendulum-v1、MountainCarContinuous-v0 | **`Platform-v0`**（`hybrid-platform`，见 §2.2.2） |
| **图像** | ALE/Breakout-v5 等（`gymnasium[atari]`） | 少见；可用 MuJoCo 等 | 无标准内置；需自定义 |
| **离线** | — | — | Minari 数据集（`MinariDataset`，非 `env.step`） |

**仅装 `pip install gymnasium` 即可覆盖**：离散 + 连续向量（主线 CartPole / Pendulum）。
**需 extras**：Box2D、MuJoCo、Atari 图像。
**混合动作**：`pip install hybrid-platform`（**不**装 gym-hybrid / 老 `gym`）。

#### 2.2.1b 覆盖边界与明确非目标（2026-06-15 定稿）

> 上表只列 obs × action 两个轴。下表补充**确定性**和**信息结构**两个轴，防止「全 ✅ 矩阵」误导。

| 游戏/环境类型 | 覆盖 | 靠什么机制 | 版本 | 说明 |
|---|:---:|---|---|---|
| 完全信息·零和·双人棋类 | ✅ | `MctsModel` + negamax + `legal_mask` | AZ v0.22 → EZ v0.24 | 围棋 19×19 / 象棋是**算力**问题，非架构问题 |
| 单智能体（Atari / MuJoCo） | ✅ | learned latent + γ backup + CNN 表征 | MZ v0.23 → EZ v0.24 | |
| 离散 / 连续 / 混合 action | ✅ | legal_mask / 采样器 + Gumbel + 动作-keyed 节点 | 全程 | 混合 action 在 MCTS 线证据最薄（SAC 侧已稳） |
| POMDP / 部分可观测 | ✅ | 表征层吃 history / 帧堆叠，MCTS 核心不改 | MZ/EZ | 与 Atari 帧堆叠同理 |
| 轻度随机（Atari sticky action 等） | ✅ | learned 模型吸收成近似确定性 latent | MZ/EZ | MuZero 原文即如此 |
| **真·结构随机（骰子 / 2048 / 随机生成）** | ⚠️ | 需 Stochastic MuZero chance node（afterstate + chance 两类节点交替） | **未排期 backlog** | **会动树核心**（节点类型 / backup / transition），非填缝；见 §5.10 |
| **不完全信息（扑克 / 斗地主 / 麻将）** | ❌ | 需 information-set MCTS / CFR / DMC，属**另一算法家族** | 不在路线 | 完全信息 MCTS 无法处理隐藏手牌 |
| **多智能体 N>2 / general-sum / 合作博弈** | ❌ | self-play 只 1v1；negamax 只对二人零和成立 | 不在路线 | |

**总结**：新类型 = 加实现、不改 `mcts_search` 签名（新环境进 Python、新算法换 `SearchPolicy`、新模型换 `MctsModel`）。**唯一会动签名的是 chance node**——已诚实标为核心扩展，不藏。

#### 2.2.2 混合动作：`Platform-v0`（已定案，取代 gym-hybrid）

| 项 | 内容 |
|----|------|
| **环境 ID** | `Platform-v0` |
| **Python 包** | [`hybrid-platform`](https://pypi.org/project/hybrid-platform/)（Gymnasium API fork，原 [gym-platform](https://github.com/cycraig/gym-platform) / Masson et al. 2016） |
| **依赖** | `gymnasium`、`numpy`、`pygame` — **无** `gym` |
| **用法** | `import gymnasium as gym` + `import gym_platform` → `gym.make("Platform-v0")` |
| **任务** | 横版跳台：run / hop / leap + 连续参数；躲敌人、过沟、到终点（**不是** Moving 的俯视角进绿圈） |
| **弃用** | gym-hybrid、`Moving-v0`、`Sliding-v0`；Rust 不再 `import gym` / `gym_hybrid` |
| **示例** | `moving_sac` → **`platform_sac`**，`moving/` → **`platform/`**（Phase 0b） |
| **Rust** | `GymEnv` 在 `Platform-v0` 前 `import gym_platform`；观察为 Tuple(Box, Discrete)，`GymEnv` 按子空间暴露分块结构，**展平为单 `Vec<f32>` 由上层 flatten helper 按 space 原生顺序处理（不赌顺序）** |

**论文与示意图**：[arXiv:1509.01644](https://arxiv.org/abs/1509.01644) Figure 4/6；[gym-platform 截图](https://github.com/cycraig/gym-platform/blob/master/img/platform_domain.png)。

### 2.3 Transition 使用 Vec\<f32\>，不使用 Tensor

**理由**：
- **轻量**：Buffer 存储大量历史数据，`Vec<f32>` 内存紧凑
- **去耦合**：存储层不依赖计算图；训练时批量转为 Tensor 即可
- **灵活**：离散动作存 `[action_index as f32]`，连续动作存 `[a1, a2, ...]`，混合动作存 `[discrete, cont1, cont2, ...]`

### 2.4 对 rustRL 的态度：参考设计，不直接迁移

[rustRL](https://github.com/dbsxdbsx/rustRL) 是基于 tch-rs 的 SAC / SAC-Hybrid 实现。其核心设计思路（Buffer 结构、Policy trait、哑值统一公式）有很高参考价值，但代码与 tch-rs 深度耦合（`tch::Tensor`、`tch::nn::VarStore` 等），直接迁移代价大于重写。

当前三个 SAC 示例已参考 rustRL 的架构重新实现，使用 only_torch 的原生类型。

### 2.5 MCTS 抽象边界：内核吃 model、不吃 env；选择规则可插拔（v0.22 定形）

> **决策（2026-06-07 体检）**：库内 MCTS 是 Zero 全家族（AlphaZero / MuZero / EfficientZero V2 及其改良）的共享骨架，**最贵的返工点是搜索签名**。因此 v0.22 day-1 就把「会改一片」的接缝留好，只实现最小集（PUCT + 确定性 + 单线程）。变体清单见 §5.10。

| 接缝 | v0.22 怎么定形 | 变体压力来自 |
|------|----------------|--------------|
| **模型抽象（root+recurrent）** | `mcts_search` 吃 `MctsModel`（root + recurrent 两段 = DeepMind [mctx](https://github.com/google-deepmind/mctx)），**不吃 `&GymEnv`**；`recurrent` 一次返回 next_state/reward/terminal/prior/value/候选动作 | MuZero/EZ 在 learned latent 滚动、无 env 可 snapshot；`Dynamics`(v0.23) 作为实现复用同一 search |
| **搜索策略（整套流程）** | 可插拔单位 `SearchPolicy`（选择+展开+推荐+根部处理），只实现 `PuctPolicy` | Gumbel（序贯减半改根部+recommendation）、RegPolicy(ACT/SEARCH)、MENTS/ANT |
| **SearchResult 暴露原始统计** | 每个根孩子 `{action, N, Q, prior}` + 推荐动作 + target_policy | RegPolicy 的 ACT/LEARN、Sampled 校正、reanalyze/SVE 在搜索外算 |
| **候选动作来源** | 离散=`legal_mask`；连续/混合=采样器，**禁**从 `prior.len()` 默认全量 | Sampled MuZero、连续/混合动作 |
| **节点动作集** | 孩子 `Vec<(Action, Child)>` / map，禁定长数组 | 同上 |
| **backup 视角统一** | value/reward 视角 + player-to-move + 单智能体 γ + 双智能体 negamax（self-play `-1`）+ terminal backup | 五子棋(2p) 与 CartPole MuZero(1p) 共用 |
| **batch 叶子评估（接口形状）** | `predict_batch` + `recurrent` 可 batch（CPU 摊销 pyo3/网络）；并行推迟 | EZ-V2 / 所有 learned model |

参照：mctx（泛型 `search` + 可插拔 `muzero_policy`/`gumbel_muzero_policy`）、[LightZero](https://github.com/opendilab/LightZero)（tree / policy / buffer 分层，按变体层叠）。**Rust 单份树**即 LightZero 的 ctree，省掉其 Python/C++ 双份维护。签名细节见 [RL 主线实施计划 v0.22 §C](../../c:/Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md)。

**v0.22 必需 vs 之后的数值件**：AlphaZero 自对弈本身要 **Dirichlet 根噪声 + 温度采样**（v0.22 必做）；**scalar↔categorical value 支持变换、树内 min-max 归一化** 属 MuZero(v0.23+)——**接口预留、实现推迟**，且 `SearchPolicy` 须能访问 value normalizer。

---

## 3. 已知差距与技术债

按影响程度排序：

### 3.1 🟡 `Step` 死代码 → **`Transition`（已定案）**

`src/rl/mod.rs` 的 `Step` 未被示例使用；三示例各自 `Experience`。v0.20.0：

- **删除** `Step`（无 `deprecated`、无 type alias）
- **新增** `pub struct Transition`（字段：`obs, action: Vec<f32>, reward, next_obs, terminated, truncated`——**镜像 Gymnasium，勿合并成单一 `done`**；TD target 用 `1 - terminated` bootstrap，CartPole-v0 撞 200 步是 `truncated` 仍需 bootstrap）
- 三示例（cartpole / pendulum；moving 待环境恢复）改用 `rl::{ReplayBuffer, Transition}`

| 示例 | action 编码 |
|------|------------|
| CartPole | `vec![idx as f32]`，读取时集中 `action[0] as usize` |
| Pendulum | 连续向量 |
| Platform（Hybrid SAC） | 混合展平：离散 run/hop/leap + 对应连续参数（与 Moving 编码方式可能不同，迁移时对齐） |

### 3.1b 🔴 `GymEnv` 仍含 legacy gym 回退（Phase 0 首要清理）

`gym_env.rs` 在 `gymnasium.make` 失败时会 `import gym`。Phase 0 删除；Phase 0b 以 **Platform-v0** 恢复混合测例与 Hybrid SAC 示例，不再使用 Moving。

### 3.2 🟡 三个示例之间存在大量代码重复

以下代码在 cartpole / pendulum / moving 三个示例中**逐字复制**：

| 重复代码 | 大约行数 × 3 |
|----------|-------------|
| `ReplayBuffer`（VecDeque + push + sample） | ~35 行 |
| `SacConfig` 结构 | ~20 行 |
| 训练循环骨架（采样→Critic→Actor→Alpha→软更新） | ~80 行 |

**处理选项**：

- **A. 提取 `examples/sac/common.rs`**：共享 Buffer、Config 等通用代码
- **B. 保持独立**：每个示例自包含，方便独立阅读

### 3.3 🟢 Actor Loss 风格不统一

三个示例的 Actor Loss 写法不一致（详见 [§4](#4-sac-技术笔记统一-actor-loss-公式)）。如果未来要做统一 Agent 框架，Pendulum 的写法需要调整。

### 3.4 🟢 Agent 接口命名不统一

| 示例 | 推理方法名 | 返回类型 |
|------|-----------|---------|
| CartPole | `sample_action()` | `(usize, Tensor)` |
| Pendulum | `sample_action()` | `(Tensor, Tensor)` |
| Moving | `select_action()` | `(usize, Vec<f32>)` |

方法名和签名不一致，不利于未来抽象 Policy trait。

---

## 4. SAC 技术笔记：统一 Actor Loss 公式

> 本节记录 SAC Actor Loss 的"哑值统一"技巧，来源于 rustRL 的设计分析。

### 4.1 核心思想：用"哑值"统一三种 Action 类型

标准 SAC Actor Loss 公式：

$$
L = -E_π[Q(s,a)] - α·H(π)
$$

等价地展开为以 `log_prob` 为核心的形式：

$$
L = mean( Σ_d π(d) × (α_d·log π(d) + α_c·log π_c(d) - Q(d)) )
$$

通过对不同动作类型**填充哑值**，三种情况可以走**同一段代码**：

| 类型 | `prob_d` | `log_prob_d` | `log_prob_c` |
|------|----------|-------------|-------------|
| **Discrete** | 真实 softmax 概率 | $log(prob + ε)$ | **zeros**（哑值） |
| **Continuous** | **ones**（哑值） | **≈ 0**（哑值） | 真实 TanhNormal log_prob |
| **Hybrid** | 真实概率 | 真实 log_prob | 真实 log_prob |

各类型退化时哑项自然消零：
- **Discrete**：$prob_d × (Q + (-α_d × log_prob_d) + 0)$ — 连续部分为 0
- **Continuous**：$1.0 × (Q + 0 + (-α_c × log_prob_c))$ — 离散概率为 1，log_prob_d ≈ 0
- **Hybrid**：完整公式

### 4.2 当前项目的实现情况

| 示例 | 公式风格 | 是否统一 |
|------|---------|---------|
| **Moving (Hybrid)** | $probs * (log_probs * α_d + log_prob_c * α_c - Q)$ | ✅ 标准哑值统一公式 |
| **CartPole (Discrete)** | $probs * (log_probs * α - Q)$ | ⚠️ 形式接近，但无连续哑值项 |
| **Pendulum (Continuous)** | $log_prob_sum * α - Q$ | ❌ 完全不同的风格 |

Moving 的 Brake 动作（纯离散，无连续参数）用 `zero_lp_var` 作为哑值，验证了这一技巧。但 CartPole 和 Pendulum 各自走了简化写法。

### 4.3 为什么不用 `entropy()` 构建 Loss？

虽然数学上 $Σ π·(Q - α·log π) ≡ E_π[Q] + α·H(π)$，但以 `entropy()` 拆分构建 loss 有两个问题：

**问题一：拆开后反而更复杂。** 统一公式变成三个独立项的拼接，每个项的 shape 和语义不同，失去了一行公式的简洁性。而且 `entropy_d` 需要取反，`log_prob_c` 不需要——符号处理不一致，容易出错。

**问题二：通用 Hybrid 场景下分解不成立。** 上面的分解依赖一个前提——$log π_c$ 不依赖离散动作 `d`。但在更一般的 Hybrid 设计中（每个离散动作对应不同的连续分布参数），$log π_c(a_c|d)$ 随 `d` 变化：

$$
Σ_d π_d × [Q(d) - α_d·log π_d(d) - α_c·log π_c(a_c | d)]
$$

此时 $π_d$ 和 $log π_c(·|d)$ 在求和内部耦合，**不可能**分解成 `entropy_d` + `entropy_c` 的形式。

**结论**：`entropy()` 的定位是**监控工具**（训练时打日志看分布是否坍缩），不应是 loss 的构建块。Actor Loss 应该始终用 `probs()` + `log_probs()` 组合构建。

---

## 5. 未来方向

> 均为"可做但不紧迫"的方向，按自然推进顺序排列。

### 5.1 整理现有 SAC 示例（**v0.21 主线**）

**工作量**：小–中

- 完成 `Transition` + buffer 入库（§3.1，v0.20 已落）
- 统一三个示例的 Actor Loss 风格为哑值统一写法（§4.2）
- 统一 Agent 接口命名（`sample_action`）
- 算法 helper 入库 `src/rl/algo/sac/`：critic_update / actor_update / alpha_update / soft_update（**函数式**，无状态）
- **否决** `examples/sac/common.rs`（examples 树共享文件是反模式）
- 三 SAC 示例瘦身到 ≤ 150 行 main.rs；新 LunarLander SAC ≤ 80 行
- 目录重组：`examples/traditional/sac/` → `examples/sac/`（为后续 AZ/MZ 顶层目录腾位置）

### 5.2 Beta 分布

**工作量**：中

作为 TanhNormal 的备选连续策略后端。Beta 分布天然有界（[0, 1]，可缩放），无需 Jacobian 修正，在高维连续动作 + 严格边界控制的场景有优势。

详见 [Beta 分布备选方案分析](../../examples/sac/beta_distribution_note.md)。

**依赖**：需要实现 `src/nn/distributions/beta.rs`。

### 5.3 优先级经验回放（PER）

**工作量**：中

当前三个示例均使用均匀随机采样。PER（Prioritized Experience Replay）按 TD-error 优先采样高价值经验，配合重要性采样权重修正偏差。

**注意**：按当前设计决策，PER 应作为示例或独立 crate，不纳入库核心。

### 5.4 更多 RL 算法示例

| 算法 | 类型 | 适用场景 | 落地版本 |
|------|------|---------|----------|
| **PPO** | On-policy | 通用性最强；on-policy buffer 与 SAC 不同 | **v0.23**（与 `Rollout`/`RolloutBuffer` 一起落） |
| **DQN** | Off-policy, 离散 | 教学示例；功能上是 SAC-Discrete 的真子集 | 长期 backlog（v0.21 已**决定不做**） |
| **TD3** | Off-policy, 连续 | SAC 的确定性策略对标；与 SAC 高度重叠 | 长期 backlog |

PPO 需要 Clamp 节点（ratio clipping），当前已实现（详见 [节点类型规划](./future_node_types.md)）。

**关于 v0.21 不加 DQN 的决定**（2026-05-27）：SAC 已覆盖离散 / 连续 / 混合三种动作空间，DQN 仅做纯离散，在「`ReplayBuffer<Transition>` / helper 抽象边界」上没有新增压力，仅有教学价值。v0.21 主线聚焦「helper 入库 + 示例瘦身 + LunarLander-v3 离散 SAC 验证复用」，DQN 推至 backlog。

### 5.5 `Agent` / `PlanningAgent` 双 trait（**v0.22 主线**）

> v0.21 之前**故意推迟**，避免单算法时空抽象；v0.22 与 AlphaZero 一同引入。

```rust
pub trait Agent {
    fn act(&self, obs: &[f32]) -> Vec<f32>;
}

pub trait PlanningAgent {
    /// MCTS 之类先模拟再决策的算法用这个 trait
    fn act_with_target(&self, obs: &[f32]) -> (Vec<f32>, Vec<f32>);  // (action, target_distribution)
}
```

| Trait | 实现者 | 何时引入 |
|-------|--------|----------|
| `Agent::act` | SAC / DQN / PPO / TD3 | v0.22 |
| `PlanningAgent::act_with_target` | AlphaZero / MuZero / EfficientZero V2 | v0.22 |

### 5.6 AlphaZero / 五子棋（**v0.22 主线**）

> **环境宪法（2026-05-27）**：环境 **100% Python**，目录 **`python/gym_env/<游戏>/`**（**扁平包**：`python/gym_env/` 即 `import gym_env`，无 `gym_env/gym_env/`）；Rust 只桥接 `GymEnv`。**MCTS 在 `src/rl/mcts/`**；**`board.py` 只写规则**，**不在 Env 里写 MCTS**。

| 层 | 位置 | 说明 |
|----|------|------|
| **Board** | `python/gym_env/gomoku/board.py` | `legal_mask`、`clone`/`restore`、终局（**非** MCTS） |
| **Gym Env** | `python/gym_env/gomoku/env.py` | 薄包装：`reset`/`step`/`obs`，委托 `Board` |
| **MCTS** | `src/rl/mcts/` | `mcts_search<M: MctsModel, P: SearchPolicy>(model, policy, cfg) -> SearchResult`（吃 model+策略不吃 env，见 §2.5）；PUCT、树、negamax backup |
| **规划桥接** | `src/rl/env/gym_env.rs` | 转调 `unwrapped.board` 的 snapshot/restore 等 |
| `Predictor` | `src/rl/mcts/predictor.rs` | `predict_batch` |
| 数据结构 | `SelfPlayGame`（`src/rl/buffer/self_play.rs`） | `impl BufferItem` |
| 示例 | `examples/alphazero/gomoku/` | `Gomoku-selfplay-v0` 训练；`Gomoku-naive*-v0` 评测 |

**安装**：`pip install -e python/gym_env`。v0.22 默认 **9×9** 验收；性能瓶颈：缩棋盘/sims → MCTS 批量调 Python（仍无 Rust 棋盘）。

### 5.6.1 算法验收分层（2026-05-27 定稿）

| 层级 | 算法 | CartPole 环境 | 门禁 | 说明 |
|------|------|---------------|------|------|
| **架构跑通** | SAC、MuZero、PPO | **`CartPole-v0`**（满分 200，solved = **195**） | 发版 / 示例：**单局 reward ≥ 195**（或 100 局均值 ≥ 195） | v0.20 **smoke** 不断言 reward，只验管线 |
| **终极调优** | **[EfficientZero V2](https://arxiv.org/abs/2403.00564)（EZ-V2，第二代，唯一）** | **`-v1` / 新版 ID** | 各任务专属指标 | 离散+连续+视觉+低维；混合 Tuple 为工程扩展 |

端到端表见 [RL 主线实施计划](../../c:/Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md)。

### 5.7 MuZero + PPO（**v0.23 主线**）

> Reviewer 终审（2026-05-27）将原「v0.23 MuZero + EfficientZero + 环境矩阵」拆为 v0.23 / v0.24。v0.23 聚焦「`Rollout`/`RolloutBuffer` + `Dynamics` 落地 + 第一个 model-based 闭环」。

引入 **learned dynamics** 与 **on-policy buffer**（与 SAC 共用 **CartPole-v0 ≥ 195** 标准，见 §5.6.1）：

| 组件 | 位置 | 说明 |
|------|------|------|
| `Rollout` + `RolloutBuffer` | `src/rl/buffer/rollout.rs` / `rollout_buffer.rs` | on-policy 段；**不** `impl BufferItem`（独立容器，见主计划 buffer 三分法）；含 PPO 用的 `log_probs / values` 字段 |
| `Dynamics` trait | `src/rl/mcts/dynamics.rs` | `initial_state` / `recurrent`；**`impl MctsModel`** 复用 v0.22 `mcts_search` 不改签名（见 §2.5） |
| MuZero 示例 | `examples/muzero/cartpole/` | **`CartPole-v0`**；简化版（**无 reanalyze**）；**reward ≥ 195** |
| PPO 示例 | `examples/ppo/cartpole/` | **`CartPole-v0`**；`Rollout`/`RolloutBuffer` + `src/rl/algo/ppo/`；**reward ≥ 195** |

### 5.8 EfficientZero V2 终极调优 + 多模式矩阵（**v0.24 主线**）

> **算法定稿**：v0.24 **只以 [EfficientZero V2（EZ-V2）](https://arxiv.org/abs/2403.00564)**（ICML 2024 Spotlight，第二代）为终极调优算法；**不以** EfficientZero V1（NeurIPS 2021）为发版目标。参考：[EfficientZeroV2](https://github.com/Shengjiewang-Jason/EfficientZeroV2)。
>
> v0.24 起性能调优只落在 EZ-V2；环境 ID **一律 `-v1` / 新版**（与架构层 `CartPole-v0` 区分）。EZ-V2 在 V1 的 value prefix / reanalyze 之上增加 **Gumbel 连续动作搜索**、**SVE** 等。

| 模式 | 示例 | 环境 ID | 验收 |
|------|------|---------|------|
| 完美信息博弈 | `examples/efficientzero/gomoku/` | `Gomoku-*-v0` | 同训练量胜率 ≥ AlphaZero |
| 向量离散 | `examples/efficientzero/cartpole/` | **`CartPole-v1`** | EZ 任务指标（非 195） |
| 图像离散 | `examples/efficientzero/atari/` | `ALE/*-v5` | 训练闭环 |
| 连续高维 | `examples/efficientzero/ant/` | `Ant-v5` 等 | 训练闭环 |
| 混合动作 | `examples/efficientzero/platform/` | `Platform-v0` | Tuple 离散+连续 |
| 离线（可选） | `examples/efficientzero/minari_pointmaze/` | Minari | 离线评估 |

- **降级路径**：完整 reanalyze 超 1 周 → value prefix + SVE only，完整 reanalyze 推 v0.25
- **辅助 pipeline**（`examples/dqn/atari/` 等）：仅跑通，**不进** EZ 性能门禁

### 5.9 长期 backlog（≥ v0.25）

- Beta 分布（§5.2 旧 → 长期）
- 优先级经验回放 PER（示例或独立 crate，不入库核心）
- DQN / TD3（教学价值，非架构必需；SAC 已覆盖）
- MCTS rayon 并行 / virtual loss（v0.22 单线程版稳定后视性能再决定）
- 演化（NEAT）+ RL 联合搜索
- 多智能体 / 分布式 self-play（与 CPU only 约束冲突，需重新评估）

### 5.10 MCTS / Zero 变体 backlog（接缝已留，做不做按需）

> v0.22 内核已留好 4 根接缝（见 §2.5）；下列变体都是「加实现」而非「改签名」，故归 backlog——**可能做、也可能不做**，按兴趣/需要推进。每条标注用到哪根接缝 + 公开参考。

| 变体 | 用到的接缝 | 公开参考 |
|------|-----------|---------|
| **Gumbel MuZero**（小 sims 仍保证策略提升） | 选择规则 | Danihelka et al. 2022 (ICLR, OpenReview `bERaNdoegnO`)；mctx `gumbel_muzero_policy` |
| **MENTS / RENTS / TENTS**（最大熵树搜索） | 选择 + soft backup | Xiao et al. 2019 (NeurIPS)；Dam et al. 2021 |
| **ANT（自适应熵树搜索）** | 选择 + backup | Adaptive Entropy Tree Search |
| **MCTS as Regularized Policy Opt**（不依赖搜索次数） | 选择 / 目标 | Grill et al. 2020 (ICML) |
| **Sampled MuZero**（连续/大动作空间采样 K 个） | 节点动作集 | Hubert et al. 2021 (arXiv:2104.06303) |
| **Stochastic MuZero**（随机环境 chance node / afterstate） | **核心树扩展**（decision/chance 两类节点交替，改 backup/transition 形状）——非「加实现不改核心」 | Antonoglou et al. 2022 (ICLR, OpenReview `X6D9bAHhBQ1`) |
| **连续 / 复合动作 AlphaZero·MuZero** | 节点动作集 + 转移 | A0C (Moerland et al. 2018)；continuous/composite-action MuZero |
| **MCTS 并行**（virtual loss / 批量叶子评估） | 内核（B3 已留 `predict_batch`） | 经典 root / leaf / tree 并行 |
| **完整 reanalyze**（off-policy 重算目标） | buffer 元数据 | MuZero Reanalyze / MuZero Unplugged |

> 这些**不进 v0.20–v0.24 发版门禁**；EfficientZero V2（v0.24 唯一终极调优）已内含 value prefix / SVE / Gumbel 连续搜索，其余变体属研究性扩展。统一参照库：[LightZero](https://github.com/opendilab/LightZero)（AlphaZero/MuZero/Sampled/Stochastic/Gumbel/EfficientZero 全谱系）、[mctx](https://github.com/google-deepmind/mctx)。

---

## 6. 参考

### 项目内文档

- [RL Python 环境搭建指南](../rl_python_env_setup.md) — Gymnasium-only、扩展包、Minari、五子棋
- [SAC 示例总览](../../examples/sac/README.md) — Entropy、Alpha、Target Entropy 核心概念
- [SAC 数学基础分析](../../examples/sac/sac_mathematical_foundations.md) — Hybrid 6 种模式、收敛性、KL 散度
- [Beta 分布备选方案](../../examples/sac/beta_distribution_note.md) — 有界连续策略分析
- [概率分布模块设计](./distributions_design.md) — Categorical / Normal / TanhNormal API 设计
- [DataLoader 设计](./data_loader_design.md) — Phase 2 段落记录了 Buffer 解耦决策
- [待扩展节点类型](./future_node_types.md) — PPO 等算法可能需要的节点

### 外部参考

- [rustRL](https://github.com/dbsxdbsx/rustRL) — 基于 tch-rs 的 SAC / SAC-Hybrid 实现（本项目的设计参考来源）
- Haarnoja et al. 2018 — SAC v1 / v2
- Christodoulou 2019 — SAC-Discrete
- Delalleau et al. 2019 — Hybrid SAC
- Chou et al. 2017 — Beta Policy

---

## 7. v0.20.0 实施计划

> 2026-05-27 定稿：环境 **最先**、**Gymnasium-only**；`Transition` 取代 `Step`（直接删除）；CartPole **smoke** 为同一示例的短跑模式。
>
> **「发版」口径（2026-06-07）**：本项目 RL 主线各版本的「发版 / 已发」指 **bump 版本号 + 更新 `CHANGELOG.md`**，**不**执行 `cargo publish`。
>
> **v0.20 之后的版本路线**：§5.1 v0.21 示例瘦身 / §5.5 v0.22 双 trait / §5.6 v0.22 AlphaZero / **§5.7 v0.23 MuZero + PPO** / **§5.8 v0.24 EfficientZero V2 + 多模式矩阵**；端到端表参见 [RL 主线实施计划](../../c:/Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md)。
>
> **Reviewer 终审（2026-05-27）裁决要点**（已并入本路线图）：v0.20 Conditional Go；原 v0.23 拆分为 v0.23 + v0.24；MCTS v0.22 单线程；**终极算法定为 EfficientZero V2（第二代）**；EZ-V2 reanalyze 可降级为 value prefix + SVE only。
>
> **Creator 环境宪法（2026-05-27）**：`python/gym_env/<游戏>/`；MCTS 在库、Board 在 Python；详见 [RL 主线实施计划 v0.22](../../c:/Users/Administrator/.cursor/plans/rl_主线实施计划_5966956a.plan.md)。

### 7.1 里程碑一句话

CartPole SAC 在 Windows + 仅 Gymnasium 下可复现；`GymEnv` 无 legacy gym；Hybrid SAC 跑 **Platform-v0**；`ReplayBuffer` + `Transition` 入库；`cartpole_sac` smoke；文档与示例路径一致。

### 7.2 Phase 顺序（环境优先）

| Phase | 焦点 | 工时 | 验收 |
|-------|------|------|------|
| **0** | **Gymnasium-only `GymEnv`** + 文档/测试对齐 | S–M | **legacy 清零三条 grep**（`rg 'import\("gym"\)' src/rl` + `rg use_legacy_gym src/rl` + `rg gym_hybrid src/rl` 均为零）；离散/连续 Gymnasium 测例绿 |
| **0b** | **Platform-v0**（hybrid-platform）+ 示例/测例迁移 | S–M | `gymnasium.make('Platform-v0')`；`platform_sac`；`test_07` 更新 |
| **1** | `buffer/` + `Transition`（存 `terminated`+`truncated`，非 `done`）；删 `Step`；`GymEnv::step` 透出两信号；三示例接 buffer | M | cartpole / pendulum / platform 用库 buffer |
| **2** | 文档路径、`just examples-rl`、`py-gym-platform` | S | 无 gym-hybrid 推荐路径；`rg examples/sac` 清零 |
| **3** | `cartpole_sac` smoke（`SMOKE=1` / `just example-cartpole-sac-smoke`） | S | 数分钟内跑通整条训练链；**不以 reward 收敛为通过条件** |
| **4** | CHANGELOG / AGENTS / v0.20.0 | S | 发版前手跑 smoke + `just test-filter rl` |

### 7.3 Phase 0 代码要点（最先做）

1. `gym_env.rs`：仅 `gymnasium.make`；删除 `use_legacy_gym`、gym 分支、双 API reset/step。
2. 错误信息：环境未注册时提示安装对应 **gymnasium extra**（`gymnasium>=1.3.0,<2.0`）或注册自定义环境，**不**提示安装 `gym`。
3. `src/rl/tests/env/`：删除 Moving/Sliding 测例；**Phase 0b** 增加 Platform-v0 混合测例。
4. `.doc/rl_python_env_setup.md`：批次 6 改为 **hybrid-platform**；移除 gym-hybrid 安装说明。
5. `try_import_env_module`：`Platform-v0` → `import gym_platform`（删 `gym_hybrid`）。
6. 验证：`just py-gym-basic` → `just test-serial`（或 `just test-filter rl`）。

**Phase 0 legacy 大清理范围（2026-06-15 全局扫描定稿）**：

> 三条 grep（`import("gym")` / `use_legacy_gym` / `gym_hybrid`）覆盖的代码是**同一套耦合 legacy 机制**（gym 回退 + 双 API 分支 + gym-hybrid 导入），要么整体存在要么整体不存在，无"选择性保留"场景。

| 文件 | 删什么 | 改什么 |
|------|--------|--------|
| `src/rl/env/gym_env.rs` | `use_legacy_gym` 字段 + 所有 `if self.use_legacy_gym` 分支（reset/step/get_module_name）；`try_make_env` gym 回退分支；`extract_gym_reset_obs` 辅助函数；`try_import_gym_env_module` 的 `gym_hybrid` 分支；doc 注释里 Moving/gym 回退描述（4 处） | `try_make_env` → gymnasium 失败直接 panic + 中文指引；`step` 返回类型透出 `(terminated, truncated)` 不再合并 `done`；`try_import_gym_env_module` 改为 `Platform-v0 → import gym_platform`（Phase 0b） |
| `src/rl/tests/env/gym_env.rs` | 4 个 `Moving-v0`/`Sliding-v0` 测例（含 `get_module_name() == "gym"` 断言） | Phase 0b 换成 Platform-v0 测例 |
| `tests/python/gym/test_07_hybrid.py` | 整文件的 `import gym` + `import gym_hybrid` + Moving/Sliding | Phase 0b 重写为 `import gymnasium` + `import gym_platform` + `Platform-v0` |
| `examples/traditional/sac/moving/` | — | Phase 0b 整目录迁移为 `platform/`（`Moving-v0` → `Platform-v0`），Cargo.toml `moving_sac` → `platform_sac` |
| `.md` 文档里的 `gym-hybrid` | **不删**（历史说明"弃用 gym-hybrid"，不是代码依赖） | — |

### 7.4 Smoke 含义

- **不是**新二进制；是 **`cartpole_sac` 同一 example** 在 `SMOKE=1` 下的短参数（约 3 episode、小 buffer）。
- **仅 CartPole**（主线、依赖最少）。
- **不进**默认 CI；**进入** v0.20.0 发布前手动清单。

### 7.5 明确不做

- OpenAI Gym / `pip install gym` 回退路径
- `Policy` trait、库内 `SacAgent`
- `examples/traditional/sac/common.rs`（本版）
- 默认 CI 跑 500 episode 或 `just examples-traditional` 作 RL 门禁

### 7.6 Backlog（非 v0.20.0）

- 无（混合环境已定为 Platform-v0）
- Minari **离线训练**示例（`MinariDataset` 已有，无训练 example）
- PER、DQN/PPO/TD3、Beta 分布

### 7.7 2026-06-07 体检补充决策（buffer / 环境 API 形状）

> Reviewer 体检 + 架构师裁决，5 项 API 形状决策在开工前定案，与 RL 主线实施计划 plan 同步。**凡与前文 §3.1 / §7.2 / §7.3 的字段或 Phase 描述冲突，以本节为准。**

1. **终止语义（镜像 Gymnasium）**：`Transition` 字段由 `done` 改为 **`terminated` + `truncated` 两个 bool**（取代 §3.1 的 `done`）。TD target 用 $r + γ·(1 - terminated)·V(next)$；`CartPole-v0` 撞 200 步是 `truncated`、**仍需 bootstrap**，合并成 `done` 会算错 loss（[Gymnasium 官方](https://gymnasium.farama.org/main/tutorials/gymnasium_basics/handling_time_limits/)）。`GymEnv::step` 须透出两个信号（Phase 0 一并改）；便捷 `is_episode_end() = terminated || truncated`。
2. **`BufferItem` 砍 `Send`**：改为 `Clone + 'static`。CPU-only 单线程无跨线程需求；`T` 须为纯 owned 数据（不持 `PyObject` / 借用）；真要并行（v0.23+）再加。
3. **`sample` 语义边界**：`ReplayBuffer::sample` = 按**存储单位**随机有放回抽样，**非训练采样器**。`Transition` 存储单位 == 训练单位；`SelfPlayGame`（v0.22）整局存、position 取，需两级采样（helper 或独立 `GameBuffer`），不由本 `sample` 承诺覆盖。
4. **采样实现红线**：`sample` 直接 `rng.gen_range(0..len)` 有放回，**禁止** `(0..len).collect()` 建全长索引；返 owned `Vec<T>`，不返 `&T` / 索引借用。
5. **`GymEnv` 错误策略 + seed**：维持 **panic 为主**（程序员错误继续 `panic!` / `expect`），**不**全面 `Result` 化（教学玩具避免过度设计）；可复现靠显式注入 RNG（`reset(Some(seed))` / `sample(.., rng)` / `StdRng::seed_from_u64`）。

**Phase 修订**（取代 §7.2 / §7.3 对应描述）：Phase 0 增「`step` 透出 `terminated` / `truncated` + 单测一例 truncation」；Phase 1 示例改 **切片迁移**（CartPole → Pendulum → Platform，逐个跑绿，不一次性三连改）。

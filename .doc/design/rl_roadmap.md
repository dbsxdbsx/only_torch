# 强化学习路线图

> 本文档记录 RL 模块的当前状态、设计决策、已知差距与未来方向，
> 供后续开发者快速了解全貌并确定下一步工作。
>
> **创建日期**：2026-02-14
> **当前状态**：环境层 + 概率分布 + SAC 三变体示例已完成；架构层 / 通用组件暂不推进

---

## 目录

1. [当前状态](#1-当前状态)
2. [架构与设计决策](#2-架构与设计决策)
3. [已知差距与技术债](#3-已知差距与技术债)
4. [SAC 技术笔记：统一 Actor Loss 公式](#4-sac-技术笔记统一-actor-loss-公式)
5. [未来方向](#5-未来方向)
6. [参考](#6-参考)

---

## 1. 当前状态

### 1.1 已完成组件

| 组件 | 位置 | 说明 |
|------|------|------|
| **GymEnv** | `src/rl/env/gym_env.rs` | 支持离散 / 连续 / 混合动作空间、图像观察、自定义环境 |
| **MinariDataset** | `src/rl/env/minari.rs` | 离线 RL 数据集封装，提供 episode 采样 |
| **Step** | `src/rl/mod.rs` | 单步交互数据结构（`Vec<f32>` 字段） |
| **Categorical** | `src/nn/distributions/` | 离散分类分布（probs / log_probs / entropy / sample） |
| **Normal** | `src/nn/distributions/` | 正态分布，支持重参数化采样（rsample） |
| **TanhNormal** | `src/nn/distributions/` | Squashed Gaussian，带 Jacobian 修正的 log_prob |
| **SAC-Discrete** | `examples/sac/cartpole/` | CartPole-v0，离散动作，Categorical 策略 |
| **SAC-Continuous** | `examples/sac/pendulum/` | Pendulum-v1，连续动作，TanhNormal 策略 |
| **Hybrid SAC** | `examples/sac/moving/` | Moving-v0，混合动作，双温度（α_d + α_c） |

### 1.2 目录结构

```
src/rl/
├── mod.rs              # 模块入口，导出核心类型，定义 Step 结构
├── env/
│   ├── mod.rs
│   ├── gym_env.rs      # GymEnv 实现（~1000 行）
│   └── minari.rs       # MinariDataset 实现（~340 行）
└── tests/
    └── env/            # 环境测试（serial_test 串行执行）

examples/sac/
├── README.md                        # SAC 算法说明（Entropy / Alpha / Target Entropy）
├── sac_mathematical_foundations.md   # 数学基础（Hybrid 6 种模式、KL 散度等）
├── beta_distribution_note.md        # Beta 分布备选方案分析
├── cartpole/                        # SAC-Discrete 完整示例
├── pendulum/                        # SAC-Continuous 完整示例
└── moving/                          # Hybrid SAC 完整示例
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

### 2.2 Step 使用 Vec\<f32\>，不使用 Tensor

**理由**：
- **轻量**：Buffer 存储大量历史数据，`Vec<f32>` 内存紧凑
- **去耦合**：存储层不依赖计算图；训练时批量转为 Tensor 即可
- **灵活**：离散动作存 `[action_index as f32]`，连续动作存 `[a1, a2, ...]`，混合动作存 `[discrete, cont1, cont2, ...]`

### 2.3 对 rustRL 的态度：参考设计，不直接迁移

[rustRL](https://github.com/dbsxdbsx/rustRL) 是基于 tch-rs 的 SAC / SAC-Hybrid 实现。其核心设计思路（Buffer 结构、Policy trait、哑值统一公式）有很高参考价值，但代码与 tch-rs 深度耦合（`tch::Tensor`、`tch::nn::VarStore` 等），直接迁移代价大于重写。

当前三个 SAC 示例已参考 rustRL 的架构重新实现，使用 only_torch 的原生类型。

---

## 3. 已知差距与技术债

按影响程度排序：

### 3.1 🟡 `Step` 类型是死代码

`src/rl/mod.rs` 中定义了 `Step` 结构，但三个 SAC 示例**全都没有使用它**——各自定义了独立的 `Experience` 类型：

| 示例 | 自定义类型 | action 字段 |
|------|-----------|------------|
| CartPole | `Experience { action: usize, ... }` | 离散索引 |
| Pendulum | `Experience { action: Vec<f32>, ... }` | 连续值 |
| Moving | `Experience { action: Vec<f32>, ... }` | 混合展平 `[d, a1, a2]` |

**处理选项**：

- **A. 统一使用 Step**：让 `Step.action` 为 `Vec<f32>`（已如此定义），三个示例改为使用 `rl::Step`。离散动作存为 `vec![action as f32]`
- **B. 移除 Step**：既然设计决策是 Buffer 由用户管理，库层面也不需要定义 Step
- **C. 保持现状**：Step 作为"推荐但不强制"的参考结构

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

```
L = -E_π[Q(s,a)] - α·H(π)
```

等价地展开为以 `log_prob` 为核心的形式：

```
L = mean( Σ_d π(d) × (α_d·log π(d) + α_c·log π_c(d) - Q(d)) )
```

通过对不同动作类型**填充哑值**，三种情况可以走**同一段代码**：

| 类型 | `prob_d` | `log_prob_d` | `log_prob_c` |
|------|----------|-------------|-------------|
| **Discrete** | 真实 softmax 概率 | `log(prob + ε)` | **zeros**（哑值） |
| **Continuous** | **ones**（哑值） | **≈ 0**（哑值） | 真实 TanhNormal log_prob |
| **Hybrid** | 真实概率 | 真实 log_prob | 真实 log_prob |

各类型退化时哑项自然消零：
- **Discrete**：`prob_d × (Q + (-α_d × log_prob_d) + 0)` — 连续部分为 0
- **Continuous**：`1.0 × (Q + 0 + (-α_c × log_prob_c))` — 离散概率为 1，log_prob_d ≈ 0
- **Hybrid**：完整公式

### 4.2 当前项目的实现情况

| 示例 | 公式风格 | 是否统一 |
|------|---------|---------|
| **Moving (Hybrid)** | `probs * (log_probs * α_d + log_prob_c * α_c - Q)` | ✅ 标准哑值统一公式 |
| **CartPole (Discrete)** | `probs * (log_probs * α - Q)` | ⚠️ 形式接近，但无连续哑值项 |
| **Pendulum (Continuous)** | `log_prob_sum * α - Q` | ❌ 完全不同的风格 |

Moving 的 Brake 动作（纯离散，无连续参数）用 `zero_lp_var` 作为哑值，验证了这一技巧。但 CartPole 和 Pendulum 各自走了简化写法。

### 4.3 为什么不用 `entropy()` 构建 Loss？

虽然数学上 `Σ π·(Q - α·log π) ≡ E_π[Q] + α·H(π)`，但以 `entropy()` 拆分构建 loss 有两个问题：

**问题一：拆开后反而更复杂。** 统一公式变成三个独立项的拼接，每个项的 shape 和语义不同，失去了一行公式的简洁性。而且 `entropy_d` 需要取反，`log_prob_c` 不需要——符号处理不一致，容易出错。

**问题二：通用 Hybrid 场景下分解不成立。** 上面的分解依赖一个前提——`log π_c` 不依赖离散动作 `d`。但在更一般的 Hybrid 设计中（每个离散动作对应不同的连续分布参数），`log π_c(a_c|d)` 随 `d` 变化：

```
Σ_d π_d × [Q(d) - α_d·log π_d(d) - α_c·log π_c(a_c | d)]
```

此时 `π_d` 和 `log π_c(·|d)` 在求和内部耦合，**不可能**分解成 `entropy_d` + `entropy_c` 的形式。

**结论**：`entropy()` 的定位是**监控工具**（训练时打日志看分布是否坍缩），不应是 loss 的构建块。Actor Loss 应该始终用 `probs()` + `log_probs()` 组合构建。

---

## 5. 未来方向

> 均为"可做但不紧迫"的方向，按自然推进顺序排列。

### 5.1 整理现有 SAC 示例

**工作量**：小

- 处理 `Step` 死代码（§3.1 的选项 A / B / C）
- 统一三个示例的 Actor Loss 风格为哑值统一写法（§4.2）
- 统一 Agent 接口命名（`select_action`）
- 可选：提取 `examples/sac/common.rs` 减少重复

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

**工作量**：大

| 算法 | 类型 | 适用场景 |
|------|------|---------|
| **DQN / Double DQN** | Off-policy, 离散 | 入门级，验证基础设施 |
| **PPO** | On-policy | 通用性最强的算法之一 |
| **TD3** | Off-policy, 连续 | SAC 的确定性策略对标 |

PPO 需要 Clamp 节点（ratio clipping），当前已实现（详见 [节点类型规划](./future_node_types.md)）。

### 5.5 统一 Agent / Policy 框架

**工作量**：大

如果算法示例增多，可能需要抽象统一的 Policy trait：

```rust
pub trait Policy {
    /// 根据观察选择动作（explore=true 时带探索噪声）
    fn select_action(&self, obs: &[f32], explore: bool) -> Vec<f32>;
    /// 从一个 batch 更新策略，返回训练统计
    fn update(&mut self, batch: &[Step]) -> PolicyStats;
}
```

但目前只有 SAC 一族算法，抽象为时尚早。建议在有 2+ 种不同算法后再考虑。

---

## 6. 参考

### 项目内文档

- [RL Python 环境搭建指南](../rl_python_env_setup.md) — Windows 下 Gymnasium / MuJoCo / Minari / gym-hybrid / 五子棋环境的安装与验证
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

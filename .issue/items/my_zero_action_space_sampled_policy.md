---
status: suspended
created: 2026-06-21
updated: 2026-06-21
---

# MyZero · 动作空间 / Sampled MuZero 决策备忘（B、K、连续、复合）

> **用途**：接 Pendulum / Platform MyZero 前必读；避免把「论文 explicit」与「工程推断」混谈，避免 B/K 概念绑错。
> **状态**：`suspended` —— 非阻塞 bug；CartPole + Pendulum 1D 搜索层已接；Pendulum **未过 −200 门禁**（见 [Pendulum 诊断 §十](./pendulum_failure_diagnosis.md#十consreconsampled-压测2026-06-21)）。
> **论文本地副本**：`AI论文/Muzero复合action.pdf`（= Hubert et al. ICML 2021 · Sampled MuZero · arXiv:2104.06303）
> **关联**：[MyZero 纲领 §5](../../.doc/design/my_zero_algorithm_vision.md) · [RL 路线图 §2.5 / §5.10](../../.doc/design/rl_roadmap.md) · [Pendulum 诊断](./pendulum_failure_diagnosis.md) · [post_ez_v2 backlog §Sampled](./post_ez_v2_research_backlog.md)

---

## 一、论文写死了什么 vs 我们推断什么

| 内容 | Sampled MuZero 论文 | 工程推断（未在论文 Platform 实验） |
|------|---------------------|-----------------------------------|
| 采 K 个候选 + PUCT 用 π̂_β | ✅ §4–5、Appendix D | — |
| 纯离散大 \|A\|（Go、Atari） | ✅ 有实验 | — |
| 连续：factorized categorical **B=7** / Gaussian | ✅ Appendix A（主路径 categorical） | — |
| **Composite / hybrid**（离散维 + 连续维） | ❌ 无专章、无公式、无 Platform 数据 | factorized π + 采 K 个 **joint** hybrid 向量 |
| Platform-v0 门禁 / 对标 SAC hybrid | ❌ | 需自测 |

**结论**：复合 action **在机制上可套** Sampled（不枚举 \|A\|），但 **hybrid 具体 head 与采样伪代码不是论文原文**——实现前以本 issue + Tang & Agrawal factorized 为准，勿声称「论文已写清 Platform」。

---

## 二、符号与统一超参（团队定稿 2026-06-22）

| 符号 | 含义 |
|------|------|
| **N** | **joint 候选总数**（算法层「纯离散 \|A\|」）；见 §2.1 |
| **B** | 连续 **每一维** categorical 档数（连续 / hybrid 的连续维） |
| **K** | 每个 MCTS 节点 expand 时 **子节点条数**（**K 个完整 joint 动作**） |

### 2.1 统一 K 公式（纯离散 / 连续 / 复合共用）

```text
K = min( max(5, N / 2),  floor(sims × 2 / 3) )
```

- **下限 5**：小空间时 `max(5, N/2)` 常落到 5；\|A\| 或 N 更小时由实现 **`min(K, N)`** 自动收成 N（不必写进 recipe）。
- **`sims × 2/3`**：大 N 时压搜索宽度，避免 K 远大于一次 search 能分 visit 的量级。
- **实现**：`sampled.rs` 已有 `k.min(n)`；配置层只记上式即可。

**1D 连续特例**：N = B，上式等价于把纯离散式里的 \|A\| **换成 B**——**仅 1D 成立**。

**多维 / 复合**：N 为 **笛卡尔积大小**（或 mask 后合法 joint 数），**不能**只把 \|A\| 换成「某一维的 B」；仍用同一公式，大 N 时主要靠 `sims×2/3` 截断。

### 2.2 N 怎么算（按动作空间类型）

| 类型 | N（joint 候选总数） | B | 示例（sims=20） |
|------|---------------------|---|-----------------|
| 纯离散 | N = \|A\| | — | CartPole \|A\|=2 → K 算 5，实际 **2** |
| 1D 连续 | N = **B** | **7**（默认，对齐 Sampled MuZero Appendix；可 A/B **10**） | B=7 → K=**5** |
| D 维连续 factorized | N = **B^D** | 每维 **7** | B=7,D=2 → N=49 → K=**13** |
| 复合 hybrid | N = **\|A_d\| × ∏ B_i**（连续维 factorized） | 连续维每维 **7** | \|A_d\|=3, 两连续维 B=7 → N=147 → K=**13** |
| 多离散维 | N = **∏ \|A_i\|** | — | 5×4=20 → K=**10** |

有 **legal action mask** 时：N 用 **合法 joint 数** N_legal，公式不变。

### 2.3 B 与 bin 代表值（连续维）

- **B 默认 7**（论文 Appendix）；Pendulum 首版跟论文；需要更细离散化再试 B=10。
- **每档固定一个连续值**（执行 / MCTS 均 deterministic，**不在 bin 内 uniform 随机**）。
- **对齐 Sampled MuZero**：`[low, high]` **等宽 B 段**，每档取 **区间中点**（bin center）。
- **现状**：`action.rs` bin 中点 + `recipe.rs` 对 Pendulum 注入 B=7；`Auto` 连续 env 亦默认 B=7。

### 2.4 算法层心智模型（MCTS 不拆语义）

1. 设计期定 B、离散维 \|A_d\| → **N 固定**。
2. 每个 joint = **一条边 / 一个完整 action**（复合亦然）；expand 从 N 里按 π **无放回采 K 个**。
3. Policy 用 **factorized** 建网（π∝∏π_d）；MCTS 不解析子 action 语义。
4. 单次 search 内节点 expand 的 K 条边 **固定**；下一步 search 可重采。

**K 与 B 不要绑成 K ≤ B**（1D 常 K≤B；多维时 K 可 **>** 单维 B）。

**K 与 sims**：K 来自上式；sims 是 simulation 次数，在 K 条边上分 visit，不是 K×sims 条边各走一次。

### FAQ：1D 连续里 B=7 时还要 K 吗？能只调一个吗？

**不能混成一个数**——B 管 **策略/执行精度**，K 管 **搜索每节点看几岔**（不同层）：

| | **B** | **K** |
|--|-------|-------|
| 管什么 | 网络每维几档；env 实际能输出几种力矩 | MCTS 每个节点 expand 几条边 |
| 1D B=7, K=5 | π 仍对 **7 档** 分配概率 | 每次 search 只把 **5 档** 放进树（按 π 加权采） |
| 1D 可设 K=B=7 | 7 档精度 | 搜索枚举全部 7 档（小维可行） |

**K < B 时 B 仍有意义**：未进树的 2 档仍存在于 policy head 与 visit 蒸馏目标里；下一步 search 或 π 变后可能进入 K。若 **B=3**，即使 K=3 也只能有 3 档精度，无法更细。

**多维（D>1）必须两个都定**：B=7/D 维 → 组合空间 7^D；K=5 表示只采 **5 个完整 D 维向量** 进树，**不能**用 K 替代 B。

**1D 简化**：可设 `K = min(K_default, B)`；仅 Pendulum 时可只调 B、令 K=B，但论文多维实验 **K 可 > B（按维）**（如 K=20, B=7）。

**采样是否固定**：单次 search 内每个节点 expand 的 K 条边固定；**π 随训练变 → 下一步 search 重新采 K 个**（β=π，非 uniform）。

---

## 三、按动作空间类型选型（recipe 速查）

| 类型 | B | N | K（sims=20 示例） | 策略表示 |
|------|---|---|-------------------|----------|
| 纯离散（CartPole） | — | \|A\| | \|A\|=2 → **2** | discrete head |
| 纯离散（\|A\|=20） | — | 20 | **10** | discrete + Sampled |
| 1D 连续（Pendulum） | **7** | B | **5** | factorized categorical |
| D 维连续 | **7**/维 | B^D | 见 §2.2（N 大时 ≈**13**） | factorized |
| 复合（Platform） | 连续维 **7** | \|A_d\|×∏B | 见 §2.2 | hybrid + factorized joint |

**不推荐**：为省 N 把 hybrid **手工展平**成 giant one-hot \|A\|（难训）；用 factorized + joint 采 K 即可。

**大棋类 / Atari 超大 \|A\|**：仍可用统一公式，但 sims 常 **≫20**；若 K 仍偏大，**recipe 单独 override K**（勿假设全局 sims=20）。

**与 SAC hybrid 差异**：Zero 系 = factorized π + **MCTS 从 N 个 joint 采 K 个** + visit 蒸馏。

### FAQ：factorized 与「切太粗」是两类问题

- **B 太粗（离散化误差）**：加大 B / bin 中点更细 → 某 bin 能盖住 optimal。
- **factorized π=∏π_d**：网络不能表达任意 joint 形状；MCTS 可部分纠偏，强配对 joint 可能要 mask 或 joint head（见前文 2×2 反对角例）。**与「每条边是否完整 action」无关。**

---

## 四、当前代码状态（2026-06-22）

| 项 | 状态 |
|----|------|
| `src/rl/mcts/sampled.rs` | ✅ K 候选 + π̂_β + 根 Dirichlet 后采样 |
| `MctsConfig::sampled_k` + `Components::sampled` | ✅ runner 按 §2.1 公式从 N、sims 解析 K |
| CartPole recipe `sampled` | ✅ 开；N=2 → K_eff=2（退化全枚举） |
| `sampled_params.rs` | ✅ B/N/K 解析 + 单测 |
| `scatter_policy_target`（K→full action_dim） | ✅ 2026-06-21 | Pendulum B=7 K=5 训练必需 |
| factorized categorical **policy head** | ⏸ 1D 可跳过 | D=1 flat softmax ≡ factorized；多维/Platform 待做 |
| Pendulum cons+recon+Sampled | ❌ greedy **−942** @ 120k steps | 门禁 −200；见 [诊断 §十](./pendulum_failure_diagnosis.md#十consreconsampled-压测2026-06-21) |
| 按 state 动态采 K（连续/hybrid） | 🔲 | `DynamicsModel` 仍构造期固定离散表 |
| Platform MyZero 示例 | 🔲 仅 SAC 有 hybrid 参考 |

### 4.1 CartPole release 压测（seed=42 · sims=20 · 2026-06-22）

| 指标 | +Sampled（当前 recipe） | 无 Sampled 基线（2026-06-21） |
|------|-------------------------|-------------------------------|
| greedy 达标 | **491.6** @ ep300 | **500.0** @ ep250 |
| total_env_steps | **15,193** | **12,186** |
| wall-clock | **109s** | **~80s** |
| 门槛 475 | ✅ | ✅ |
| eval×10 / run×1 | 491.6 / **500.0** | — |

**解读**：K_eff=2 时搜索宽度与全枚举 MuZero 等价；指标差 ~25% env-steps 主要来自 Sampled 根路径（Dirichlet 时机）与 RL 方差，非「只搜 1 个动作」类机制错误。SMOKE=1 管线已通过。

---

## 五、暂缓原因

- CartPole 门禁与 cons+recon 消融优先；Sampled 在 \|A\|=2 上为退化路径，**验证价值有限**。
- Pendulum 尚有 [value 上游 / 离散 MCTS 能否学会](./pendulum_failure_diagnosis.md) 未闭环；不宜并行开 hybrid。
- 复合 action **无论文门禁数字**，需 Pendulum 连续/factorized 探路后再定 Platform recipe。

---

## 六、下次恢复条件（接 Pendulum / Platform 前）

1. 读本文 + Sampled MuZero §4–5、Appendix A/D。
2. Recipe 按 §2.1 算 K（CartPole N=2→2；Pendulum N=B=7→5）；实现 `min(K,N)` 已存在。
3. **Pendulum**：factorized B=7、bin 中点、K 按公式；greedy 门禁后再试 B=10 / Gaussian。
4. **Platform**：在 Pendulum 路径验证 Sampled 后，接 hybrid head + `Platform-v0` smoke；对标 SAC hybrid 示例，**不引用论文 hybrid 表格**（不存在）。
5. 大棋类 / Atari 大 \|A\|：单独调 K（50 级），勿用全局 K=5。

---

## 七、相关文件

- 实现：`src/rl/mcts/sampled.rs` · `search.rs` · `src/rl/algo/my_zero/sampled_params.rs` · `component.rs` · `recipe.rs` · `runner.rs`
- 接缝：`src/rl/mcts/traits.rs`（`ActionSampler`）· `action.rs`（`Discretize { buckets }` 与 B 的关系）
- 文献：Hubert et al. 2021；连续 factorized 实现参照 Tang & Agrawal 2020（Sampled MuZero Appendix 引用）

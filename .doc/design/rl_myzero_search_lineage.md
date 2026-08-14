# MyZero 搜索派系与温度

> **定位**：文献取舍与温度契约。说明为什么走官方 Zero 搜索线、不走最大熵树搜，以及现行 \(\tau\) 怎么用。  
> **不是**战略 / 组件裁决权威。Gumbel / Sampled / completedQ **开不开、promote 与否**只认 [状态总览](rl_myzero_status.md) §3。  
> **不是**第三份路线图。不抄效果矩阵，不用某环境消融正负当路由理由。  
> 落盘日期：2026-08-14。论文判定同步记入 [reading_log](../paper/reading_log.md)。

---

## 0. 一句话

搜索与出手走 **PUCT / Sampled / Gumbel** 这一派（政策先验 + 政策改进）；**不**把树内 backup / 选边改成最大熵（MENTS / RENTS / TENTS / ANTS），也不把玻尔兹曼选边（BTS / DENTS）换进主算法。

温度只做评测开关：非评测 \(\tau=1\)，评测 / `run` \(\tau=0\)。无课程表、无自适应树内温度。

---

## 1. 两条线

```text
官方 Zero 线（走这条）
  AlphaZero / MuZero PUCT
    → Grill 2020（MCTS ≈ 正则化政策优化）
    → Sampled MuZero 2021（大 |A| 抽菜单）
    → Gumbel 2022（少 sim 时保证政策改进）
    → Inverse-RPO 等后续（未精读、不纳入）

最大熵树搜线（不走这条）
  MENTS 2019（Shannon + 软 Bellman + E2W）
    → RENTS / TENTS 2021（KL / Tsallis；JAIR 2024 收成 α-divergence）
    → ANTS 2021/22（用目标熵反解树内 τ）
    → BTS / DENTS 2023（玻尔兹曼只用于选边，推荐回到硬 Q）
```

分叉点只有一句：**规划目标还是不是「回报最大化」**。官方线始终优化回报，用 \(\pi\) 先验和（可选的）政策改进来分配少模拟。最大熵线把熵写进 backup 或目标，温度会改「正确答案是谁」。

玻尔兹曼分布 \(\pi(a)\propto\exp(Q(a)/\alpha)\) 在最大熵线里是选边工具；Gumbel-Max 加在 **政策先验的 logit** 上，用来无放回抽样菜单。两者都跟 \(\exp(\cdot)\) 有亲戚，问的不是同一个问题。

---

## 2. 为什么走官方线

Zero 系的资产是 **\(\pi\) 先验 + 用搜索改进 \(\pi\)**，不是「无先验的 UCT 再加一层熵」。

| 官方机制 | 解决什么 | 和最大熵的差别 |
|----------|----------|----------------|
| PUCT | 有先验时怎么在树上选边 | 探索项吃 \(\pi\)，不是 \(\exp(Q/\alpha)\) |
| Sampled | \(\|A\|\) 大时先抽菜单再搜 | 菜单来自 \(\pi\)，不是软最优 |
| Gumbel-Top-k + Sequential Halving | \(n\ll\|A\|\) 时根上访不完所有边 | 保证 \(\mathbb{E}[q(\text{出手})]\ge\sum_a\pi(a)q(a)\)，不是换目标 |
| completedQ | 少 visit 时政策目标不能是整数比 | 未访问边用 `vmix` 补全，仍是对 \(\pi\) 的改进 |
| 二元 \(\tau\) | 训练要探索、评测要贪心 | 只拧「搜完怎么对环境出手」，不拧树内 backup |

少模拟的 correctness 官方已经用政策改进讲清楚了（Gumbel 第 1–3 页的反例：只在先验最大的 \(n\) 个里比 Q，期望可以比原 \(\pi\) 更差）。MyZero 要接的是这一套，不是再发明树内温控。

Gumbel-root / completedQ **现在 recipe 默认关**（状态总览 §3），那是 promote 裁决，不是「官方线否决」。代码与融合契约仍按官方三件套分层：Sampled = 候选层，Gumbel = 根层，completedQ = 目标层。

---

## 3. 为什么不走最大熵线

不是「某环境没涨分」，是机制对不上 Zero 学习环。

1. **软目标可以推荐错动作。** MENTS 的最优是软 Bellman；Painter 用改过的 D-chain 证明：\(\alpha=1\) 时软 Q 与真 Q 选边相反，多搜也纠正不过来。ANTS 仍在这条目标上，只是用目标平均熵反解 \(\tau\)。
2. **有强 \(\pi\) 之后，树内再加熵没有额外好处。** Painter 在围棋上用 KataGo 的 \(\pi\)/\(V\)：DENTS 与 BTS 打平；MENTS / TENTS 打不过 PUCT。熵 bonus 的价值在无网、稀疏奖励的规划器里，不是 Zero 学习环。
3. **BTS 也不能当「可拆零件」。** 理论 BTS = 玻尔兹曼选边 + 硬 max-Bellman。围棋实装改成平均回报，并强制 \(\alpha(N)\to 0\)，再混一次 \(\pi\)。限时 5 秒下 trials 约 5375 vs PUCT 1054，胜率优势先来自吞吐，不是选边公式。与 PUCT / Gumbel 抢同一层；`acting_simulations=1` 时收集不搜树，整套空转。
4. **接 learned \(g\) 更贵。** ANTS 原文主要是真模型、一次展开全部后继、训 Q。叶子若用一步 \(g+f\) 拼软 Q，值 / 政策目标都要改成软 Bellman，和 visit / completedQ 对不齐。
5. **这条线没有长成活学派。** ANTS（arXiv:2102.06808）引用很少；后续能接到棋盘 \(\pi\) 先验的，基本只有 Painter 2023，而它自己在否定软目标。

Sampled MuZero **没有**和 MENTS 家族对打。DeepMind 把正则化收进 PUCT / Gumbel，用 Sampled 解决动作空间。

---

## 4. 现行温度契约

代码里的 `MctsConfig.temperature` **不是** target-net EMA 的 `tau`。

| 场景 | \(\tau\) | 行为 |
|------|---------|------|
| self-play、reanalyze、政策目标 | `1.0`（`SELF_PLAY_TEMPERATURE`） | 原样按 \(\pi\) 或 visit 抽；目标不 `visit^{1/\tau}` 拧尖 |
| greedy eval、`run` | `0.0`（`EVAL_TEMPERATURE`） | argmax；Gumbel 噪声在 \(\tau<10^{-6}\) 时关闭 |

常量在 [`runner.rs`](../../src/rl/algo/my_zero/runner.rs)。已删除 `temp_hold_episodes` / `temp_decay_episodes`。

**不要做的：**

- 按训练步 / 回合的 \(1\to 0.5\to 0.25\) 日历（Atari MuZero 附录有，不是通解）
- 地板 0.25、回撤加温、用 \(2\sigma\) 之类硬凑的控制器
- ANTS 式「目标熵反解树内 \(\tau\)」
- \(T>1\) 进常规档
- 把同一颗 `temperature` 同时拧抽样、visit 目标、Gumbel 开关到中间值（连续 0.25 会 1+1<2）

官方也不是「\(\tau\) 只降不升」一条：棋盘 AlphaZero 是每局前约 30 手 \(\tau=1\)、之后贪心，**下一局又从 1 开始**。那是领域加料，不是万金油默认。万金油只看是不是评测。

与 Sampled / Gumbel 共处：Sampled 的 \(\beta=\pi\)（不调温再抽 K）；Gumbel 训练噪声开、评测关；政策目标用 completedQ 时不要再拧 visit。Gumbel 与 Sampled 在代码里仍互斥。

---

## 5. 读过的论文

判定口径与 [reading_log](../paper/reading_log.md) 相同：采纳 / 证据归档 / 否决 / 跳过。PDF 本机留存不入库。

### 5.1 官方线（走）

| 论文 | 出处 | 判定 | 对本项目 |
|------|------|------|----------|
| Silver et al. *A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play* | *Science* 2018 | 采纳（母算法） | PUCT、自博弈、visit 目标 |
| Schrittwieser et al. *Mastering Atari, Go, chess and shogi by planning with a learned model* | *Nature* 2020 | 采纳（母算法） | learned \(g\)、K 步 unroll |
| Grill et al. *Monte-Carlo tree search as regularized policy optimization* | ICML 2020，[arXiv:2007.12509](https://arxiv.org/abs/2007.12509) | 证据归档 | PUCT ≈ 正则化政策优化；Gumbel / completedQ 的理论前身 |
| Hubert et al. *Learning and planning in complex action spaces*（Sampled MuZero） | ICML 2021，[arXiv:2104.06303](https://arxiv.org/abs/2104.06303) | 采纳 | 大 \(\|A\|\) 抽 K；`K_eff<N` 才开 |
| Danihelka et al. *Policy improvement by planning with Gumbel* | ICLR 2022 | 采纳（组件在库，recipe 默认关） | 少 sim 政策改进；根上 Top-k + Sequential Halving；completedQ |

Inverse-RPO / 方差感知先验 UCT（arXiv:2512.21648，2025-12）是 Grill 线后续，**未精读、不纳入**。

### 5.2 最大熵树搜线（不走）

| 论文 | 出处 | 判定 | 一句话 |
|------|------|------|--------|
| Xiao et al. *Maximum Entropy Monte-Carlo Planning*（MENTS） | NeurIPS 2019 | 否决（路径） | 软 Bellman + E2W；温度是正确性参数 |
| Dam et al. *Convex Regularization in Monte-Carlo Tree Search*（RENTS / TENTS） | ICML 2021 | 否决（路径） | KL / Tsallis 换正则；仍是树内 backup，不是出手 \(\tau\) |
| Dam et al. *A Unified Perspective on Value Backup and Exploration in MCTS* | *JAIR* 2024，[arXiv:2202.07071](https://arxiv.org/abs/2202.07071) | 否决（路径） | 上篇期刊版；\(\alpha\)-divergence 统一；作者写明不和 MuZero 比 |
| Kozakowski et al. *Planning and Learning using Adaptive Entropy Tree Search*（ANTS） | arXiv:2102.06808，IJCNN 2022 | 否决（路径） | 目标熵反解树内 \(\tau\)；学习环里第一次能跑 MaxEnt MCTS；引用少 |
| Painter et al. *Monte Carlo Tree Search with Boltzmann Exploration*（BTS / DENTS） | NeurIPS 2023，[arXiv:2404.07732](https://arxiv.org/abs/2404.07732) | 否决（纳入）· 证据归档 | 钉死「软最优 ≠ 回报最优」；玻尔兹曼只配用在选边；围棋实装退回平均回报 + \(\alpha(N)\to 0\)，无运行时零件可拆 |

Dam 后来的 CATS / PATS（ICML 2024 workshop，分布价值 + Thompson Sampling）已离开最大熵，**跳过**。

---

## 6. 明确不做

- 树内软 Bellman / E3W / 目标熵反解 \(\tau\)
- 玻尔兹曼选边替换 PUCT 或 Gumbel
- Alias 表加速（瓶颈是网络前向与模型展开，不是对 \(\|A\|\) 做 argmax）
- 温度课程表、回热控制器、\(T>1\) 常规档
- 用「Painter 围棋 57–43」或任何单环境消融，回过来推翻本节的机制取舍

以后若有人再提「MaxEnt MCTS / 自适应温度 / 玻尔兹曼选边」，先读本文 §3，不要从零论证。

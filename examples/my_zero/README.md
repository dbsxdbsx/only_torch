# MyZero — 统一 Model-Based RL 算法

> only_torch 的终极强化学习算法：一个持续进化的 learned-model MCTS 实现，
> 以消融实验驱动逐步叠加组件，最终覆盖全动作空间与全环境类型。

## 设计理念

- **一个算法，持续迭代**：不再为每篇论文建独立实现（MuZero / EfficientZero），而是维护一个不断进化的 MyZero
- **奥卡姆剃刀**：每叠一个组件必须用消融证明其价值，保证不回归
- **从简到繁**：CartPole → Pendulum → Platform → 更多环境
- **判别环境原则**：组件的去留必须在「能分辨出它价值」的环境上验证；在分辨不出的环境（如 value_prefix 之于 CartPole）上得到的结论，**不当作全局裁决**

## 组件 × 环境 效果矩阵

> **单一事实源（dashboard）**：一行一个组件、一列一个环境，格子是该组件在该环境的**实测裁决**。
> 满屏 ⏳ 是好事——它就是「待办地图」。**表头的环境名是链接**，点一下即进入该环境子文档（配置 / 实测明细）。

图例：`✅ 实测有效(留下)` · `❌ 实测有害(该环境删)` · `➖ 中性(测过无显著增益亦无害)` · `⏳ 待测` · `⏸ 此规模不适用` · `— 未开始`

> **➖ 不等于 ❌**：中性表示该配置下组件未带来可测增益，**不强制进该环境默认栈**，开关保留；跨环境汇总后再定全局默认（CartPole ✅ + 他环境 ➖ → 仍可全局默认开）。

| 组件 | 开关 | [CartPole-v1](cartpole/README.md) | [Pendulum-v1](pendulum/README.md) | Platform-v0 |
|------|------|:---:|:---:|:---:|
| consistency¹ | `EZ_CONS=1` | ✅ | ➖ ᶜ | — |
| value_prefix² | `EZ_VP=1` | ❌ ᵃ | ⏳ | — |
| target_net² | `EZ_TARGET=1` | ⏳ | ⏳ | — |
| SVE² | `EZ_SVE=0.5` | ⏳ | ⏳ | — |
| completedQ³ | `CQ=1` | ✅ | ➖ ᶜ | — |
| Gumbel-root⁴ | （待实现） | ⏳ ᵇ | ⏳ | — |

- ᵃ value_prefix 在 CartPole（reward 恒 +1）退化成「步数计数器」→ 有害；但它是 EfficientZero 在 **Atari / 稀疏奖励 / 长 horizon** 的关键组件，**CartPole 删 ≠ 组件坏**，留待判别环境重测（故 Pendulum 标 ⏳ 而非沿用 ❌）。详见 [CartPole 详情](cartpole/README.md)。
- ᵇ Gumbel-root 的判别环境是 **大动作空间 / 连续动作**（Pendulum 起）；CartPole（2 动作）上 completedQ 已打满，预计仅边际收益。
- ᶜ Pendulum @ sims=50/16（seed=42）：均未达 −200；cons+CQ 相对 cons-only **无增益（➖）**，含低 sims A/B（−1287 vs −1440）。详见 [Pendulum 详情](pendulum/README.md)。

**脚注（源论文）**

1. `consistency` ← SimSiam（Chen & He 2021）/ EfficientZero
2. `value_prefix` / `target_net` / `SVE` ← EfficientZero（Ye et al. 2021）
3. `completedQ` ← Grill et al. 2020《MCTS as Regularized Policy Optimization》+ Gumbel MuZero（Danihelka et al. 2022）
4. `Gumbel-root` ← Gumbel MuZero（Danihelka et al. 2022）
5. 连续采样候选 ← Sampled MuZero（Hubert et al. 2021）
6. Dirichlet 根噪声 ← MuZero（Schrittwieser et al. 2020）

> **搜索内部·已读未取**：Sequential Halving⁴（仅当 `sims < 动作数`，CartPole(2)/Pendulum(9) 用不上）、非根确定性选择⁴（论文 Fig.7 增益很小）、连续采样候选⁵（离散化已覆盖当前格，高维连续再上）、Dirichlet 根噪声⁶（MuZero 原配，计划由 Gumbel-root 取代）。
>
> 注：早期 commit 用 `S0/S1/S2` 指代 `base/+consistency/+value_prefix`，现统一以开关名为准。

## 评判口径（所有环境通用）

| 维度 | 操作定义 | 说明 |
|------|---------|------|
| **好（good）** | greedy(temp=0) eval 10 局固定 seed 均值达门槛 | **唯一成功判据**；`avg_R`（自对弈分）带探索噪声，永远偏低，**不作判据** |
| **稳（stable）** | 多 seed（≥3）都达标，取中位数 | 排除单 seed spike |
| **快（fast）** | env-steps-to-solved 为主，wall-clock 为辅 | 同算法消融看 wall-clock；跨算法看 env-step（样本效率） |

## 各环境

> 环境名即链接，点击进入对应子文档（与上方矩阵表头一致）。

| 环境 | 动作类型 | 门禁 | 状态 |
|------|---------|------|------|
| [**CartPole-v1**](cartpole/README.md) | 离散（2） | greedy eval ≥ 475 | ✅ 又好又稳，回归哨兵 |
| [**Pendulum-v1**](pendulum/README.md) | 纯连续（1） | return ≥ -200 | sims=50/16 ➖ 未达标；completedQ 无增益 |
| **Platform-v0** | 混合 Tuple | return 趋势上升 | 待实现（子目录待建） |

## 快速开始

```bash
# 最快达标路径（CartPole · +consistency +completedQ · sims=16 · ~40 秒）
EZ_CONS=1 CQ=1 SIMS=16 cargo run --example my_zero_cartpole --release

# 管线自检（~30 秒）
SMOKE=1 cargo run --example my_zero_cartpole
```

> 各环境完整命令、超参与实测，见上表「详情」链接。

## 算法核心

MyZero 采用 MuZero 的三网络架构：

- **Representation** h：obs → latent（min-max 归一化）
- **Dynamics** g：(latent, action) → (next_latent, reward)
- **Prediction** f：latent → (policy, value)

训练期 K 步 unroll + categorical value/reward 表示 + absorbing state 终止处理。
搜索期 MCTS 在 learned latent 空间推演（不碰真环境）。

## 与旧代码的关系

- `examples/muzero/`：MuZero canonical 参考实现，保留为消融对照基线
- `examples/efficientzero/`：EZ-V2 多模式示例，保留为组件参考
- `src/rl/algo/my_zero/`：MyZero 库模块（配置 + 消融开关）
- `src/rl/algo/muzero/`：MuZero helper（support / value_transform / n_step 等），MyZero 直接复用

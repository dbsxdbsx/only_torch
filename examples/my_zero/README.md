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

> **失败区间 ≠ 中性**：若某环境整体还没学会（成绩远未达门禁），该环境的格子记 `⏳`（待 clean A/B），**不能**因为「加了组件没变好」就打 `➖`——那只是无信号，不是中性裁决。

| 组件 | 开关 | [CartPole-v1](cartpole/README.md) | [Pendulum-v1](pendulum/README.md) | Platform-v0 |
|------|------|:---:|:---:|:---:|
| consistency¹ | `CONSISTENCY=1` | ✅ | ⏳ ᶜ | — |
| value_prefix² | `VALUE_PREFIX=1` | ❌ ᵃ | ⏳ | — |
| target_net² | `TARGET_NET=1` | ⏳ | ⏳ | — |
| SVE² | `SVE=0.5` | ⏳ | ⏳ | — |
| completedQ³ | `CQ=1` | ✅ | ⏳ ᶜ | — |
| Gumbel-root⁴ | （待实现） | ⏳ ᵇ | ⏳ | — |

- ᵃ value_prefix 在 CartPole（reward 恒 +1）退化成「步数计数器」→ 有害；但它是 EfficientZero 在 **Atari / 稀疏奖励 / 长 horizon** 的关键组件，**CartPole 删 ≠ 组件坏**，留待判别环境重测（故 Pendulum 标 ⏳ 而非沿用 ❌）。详见 [CartPole 详情](cartpole/README.md)。
- ᵇ Gumbel-root 的判别环境是 **大动作空间 / 连续动作**（Pendulum 起）；CartPole（2 动作）上 completedQ 已打满，预计仅边际收益。
- ᶜ Pendulum 目前整体在**失败区间**（greedy eval 全部 −900 ~ −1700，远未达 −200），属「还没学会」而非「组件中性」。此前 cons / cons+CQ 的实测无判别力，**已撤销原 ➖ 裁决**改记 ⏳，待可学习性诊断通过后再做 clean A/B。详见 [Pendulum 详情](pendulum/README.md)。

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

> **首要评价指标**：算法「更好」≜ **达到同一性能门槛所需的真实环境交互（env-steps）更少**。有取舍时以此为准（产品文档里常称 North Star /「北极星指标」，本项目统一用「首要评价指标」）。理由：真实交互往往昂贵 / 危险 / 慢，model-based 的全部价值就是把试f错从真实世界搬进脑内模型。由此三条推论：
>
> 1. **env-steps 第一**：即使 wall-clock 增加，只要 env-steps 下降、且增量开销落在**模型侧**（搜索 / 训练），仍算有意义的更新——模型侧开销可经算法 / 并行 / 硬件优化压缩。
> 2. **wall-clock 是约束、不是目标**：它是「研究迭代速度」的代理。本项目 **CPU-only**，模型侧并非无限可压，故口径里的「为辅」不可丢。
> 3. **红利的前提是「模型够准」**：sample-efficiency 来自「在准的模型里脑内多想几步」。模型不准时，脑内规划是在错地图找路，反而比 model-free **更费**真实交互。故任一环境上，先确认 learned model 立住，再谈组件增益。

## 各环境

> 环境名即链接，点击进入对应子文档（与上方矩阵表头一致）。

| 环境 | 动作类型 | 门禁 | 状态 |
|------|---------|------|------|
| [**CartPole-v1**](cartpole/README.md) | 离散（2） | greedy eval ≥ 475 | ✅ 又好又稳，回归哨兵 |
| [**Pendulum-v1**](pendulum/README.md) | 纯连续（1） | return ≥ -200 | 诊断中：当前全在 −900~−1700（失败区间），先查可学习性 |
| **Platform-v0** | 混合 Tuple | return 趋势上升 | 待实现（子目录待建） |

## 快速开始

```bash
# 最快达标路径（CartPole · +consistency +completedQ · sims=16 · ~40 秒）
CONSISTENCY=1 CQ=1 SIMS=16 cargo run --example my_zero_cartpole --release

# 管线自检（~30 秒）
SMOKE=1 cargo run --example my_zero_cartpole
just smoke-my-zero-cartpole   # 同上（just 封装）
just smoke-my-zero-pendulum
```

> 各环境完整命令、超参与实测，见上表「详情」链接。

## 训练与推理生命周期

| 时机 | 内存权重 | 说明 |
|------|---------|------|
| 训练中 self-play / 梯度更新 | **latest** | 每步都在变 |
| `.train()` 返回的实例 | **latest** | 训末权重，不自动换回 best |
| 训后直接 `.eval()` / `.run()` | **latest** | 与返回实例一致 |
| 磁盘 `best.otm` | **best** | periodic greedy eval 创新高时写入 |
| 显式 `.load_model(path)` | **best**（或任意已存 `.otm`） | path 不含 `.otm` 后缀 |

`TrainReport`：`final_greedy` = latest 权重上的 greedy；`best_greedy` / `model_path` = 训练期历史 best 与落盘路径。

```rust
let mz = MyZero::new("CartPole-v1").solved(475.0).max_episodes(2000).train()?;
mz.eval(10)?; // latest

MyZero::new("CartPole-v1")
    .load_model(mz.train_report().unwrap().model_path.unwrap())?
    .run(Some(10))?; // best
```

## 算法核心

MyZero 采用 MuZero 的三网络架构：

- **Representation** h：obs → latent（min-max 归一化）
- **Dynamics** g：(latent, action) → (next_latent, reward)
- **Prediction** f：latent → (policy, value)

训练期 K 步 unroll + categorical value/reward 表示 + absorbing state 终止处理。
搜索期 MCTS 在 learned latent 空间推演（不碰真环境）。

## 代码组织

- `src/rl/algo/my_zero/`：**自包含**的 MyZero 库，也是项目**唯一**的 `*Zero` 实现——配置（5 层）+ 网络（`network.rs`）+ 训练循环（`runner.rs`）+ 全部算法组件（value_encoding / value_transform / n_step / reanalyze / loss / consistency / value_prefix / target_net / sve）。
- `examples/my_zero/*/main.rs`：声明 `env_id` + 训练契约 + `.train()?`；训后同一实例 `.eval(n)?` / `.run(Some(1))?` 用 **latest**；要看 **best** 用 `.load_model(path)?`（path 不含 `.otm` 后缀，见 `TrainReport.model_path`）。

> 组件的论文出处（MuZero / EfficientZero / SimSiam / Gumbel 等）见上方矩阵脚注——那是**学术溯源**，与代码依赖无关。

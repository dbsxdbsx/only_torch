# MyZero — 统一 Model-Based RL 算法

> only_torch 的终极强化学习算法：learned-model MCTS，按环境内置已验收组件组合，持续迭代。  
> 战略层（做/不做、文献谱系）：[MyZero 算法纲领](../../.doc/design/my_zero_algorithm_vision.md) · 实施顺序：[RL 路线图 §8](../../.doc/design/rl_roadmap.md#8-v025-myzero-统一算法2026-06-16-方向定稿)

## 设计理念

- **一个算法，持续迭代**：维护不断进化的 MyZero，不再为每篇论文建独立实现
- **用户零组件概念**：`MyZero::new(env_id)` 自动套用库内按环境配置的组件组合；团队 promote 组件时只改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)
- **从简到繁**：CartPole → Pendulum → Platform → 更多环境

## 环境 × 状态

| 环境 | 动作类型 | 门禁 | 状态 |
|------|---------|------|------|
| [**CartPole-v1**](cartpole/README.md) | 离散（2） | greedy eval ≥ 475 | ✅ 回归哨兵（continuation 二值门修复 · 默认 td=5：**~13.1k** env-steps；soft `γ·c` 旧实现曾退化到 30.2k，见 cartpole/README） |
| [**Pendulum-v1**](pendulum/README.md) | 纯连续（1） | return ≥ -200 | 诊断中（当前 recipe 复用 CartPole 栈作诊断，不代表组件已裁决） |
| **Platform-v0** | 混合 Tuple | return 趋势上升 | 待实现 |

## 内部组件进展（团队 · promote 时改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)）

> **单一事实源 / 待办地图**：一行一个组件、一列一个环境。满屏 ⏳ 即下一步可测项。
> 图例：`✅ 已验收` · `❌ 实测有害` · `⏳ 待测` · `⏸ 此环境不适用` · `— 未实现`
> 用户 API **不暴露**组件开关；`recipe.rs` 记录当前内置栈。矩阵表示“该组件在该环境是否已完成效果裁决”，不等同于“当前诊断栈是否临时启用”。

| 组件         | [CartPole-v1](cartpole/README.md) | [Pendulum-v1](pendulum/README.md) | Platform-v0 | 备注 |
| ------------ | :-------------------------------: | :-------------------------------: | :---------: | --------------------------- |
| consistency  |                ✅                 |                 ⏳                 |      —      | SimSiam 一致性 loss |
| reconstruction |              ✅                 |                 ⏳                 |      —      | CartPole ~12.2k steps @ sims=20 |
| reanalyze    | ⏸ 已接写回，CartPole 未开启（[issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md)） | ⏳ | — | position 级 MCTS 重搜 |
| value_prefix |                ❌                 |                 ⏳                 |      —      | CartPole 有害（≠ 全局坏） |
| target_net   |                 ⏳                 |                 ⏳                 |      —      | 已入库，训练循环待接 |
| SVE          |                 ⏳                 |                 ⏳                 |      —      | 已入库，训练循环待接；🔲 改进：固定权重 → 自适应 mixed target |
| completedQ   | ❌ 2×2 无收益（visit ~12k vs CQ ~30–34k steps） |                 ⏳                 |      —      | CartPole 不 promote · [issue](../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md) |
| Gumbel-root  | ❌ sims=10/20 未收敛（greedy 峰值 154/123） |                 ⏳                 |      —      | 库内已实现；CartPole 暂缓 · 同上 issue |
| 连续采样候选 |              ✅ 接入不回归           |                 ⏳                 |      —      | CartPole N=2 退化全枚举；Pendulum B=7 首次真实子采样诊断 |

> 论文全称与 arXiv：[算法纲领 §4.1 — 组件文献对照](../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)

**CartPole-v1 当前内置**：base + **consistency + reconstruction + Sampled**；reanalyze 写回已实现，CartPole 暂不开启（[issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md)）。

**Pendulum-v1 当前内置**：复用 **consistency + reconstruction + Sampled** 做诊断栈（B=7 · sims=20 · γ=0.997 · `reward_scale(0.1)`），但仍在失败区间；不要据此给这些组件下 ✅/❌ 裁决。

## 快速开始

```bash
cargo run --example my_zero_cartpole --release
```

```rust
use only_torch::rl::algo::my_zero::MyZero;

let mz = MyZero::new("CartPole-v1")
    .solved(475.0)
    .max_episodes(2000)
    .save_model_when_eval("models/my_zero/CartPole-v1/seed_42/best")
    .train()?;

mz.load_model_if_exists("models/my_zero/CartPole-v1/seed_42/best")?.eval(10)?;
```

## 训练与推理生命周期

| 时机 | 权重 | 说明 |
|------|------|------|
| `.train()` 返回 / 训后直接 eval | **latest** | 训末权重 |
| periodic greedy eval 创新高 | 写入 `{path}.otm` | 须 `.save_model_when_eval(path)` |
| `.load_model_if_exists(path)` 后 eval | **best** | 部署 / 演示用 |

`TrainReport`：`final_greedy` = latest；`best_greedy` / `model_path` = 训练期 periodic eval 最优（有落盘时）。

## 评判口径

| 维度 | 操作定义 |
|------|---------|
| **好** | greedy(temp=0) eval 达门槛（唯一成功判据） |
| **快** | env-steps-to-solved 为主，wall-clock 为辅 |

## 算法核心

MuZero 三网络（Representation / Dynamics / Prediction）+ latent MCTS + K 步 unroll + categorical value/reward。Replay / n-step 显式区分 `terminated / truncated / continuation`：真终止 `continuation=0`，普通步与 time-limit truncation `continuation=1`。Dynamics 同时学习 continuation；MCTS imagined edge 的 discount 用 **binary gate `gamma * (1 - done)`**（`done` 由 continuation 头阈值化），与 n-step value target 的二值 continuation 口径一致——**不**用 soft `gamma * predicted_continuation` 对每条健康边连续衰减（确定性/无终止环境下软折扣只注方差，见 cartpole/README 结论）。该基础语义不列入上方消融组件矩阵。母论文见 [算法纲领 §4.1 — base](../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)。

## 代码组织

- `src/rl/algo/my_zero/`：自包含 MyZero 库（**唯一** `*Zero` 实现）
- `recipe.rs`：按 `env_id` 内置组件开关（内部维护，文件名保留 recipe）
- `examples/my_zero/*/main.rs`：薄示例（env + 训练契约 + opt-in 落盘）

# MyZero — 统一 Model-Based RL 算法

> only_torch 的终极强化学习算法：learned-model MCTS，按环境内置已验收组件组合，持续迭代。  
> 战略层（做/不做、文献谱系、战略目标）：[MyZero 算法纲领](../../.doc/design/my_zero_algorithm_vision.md) · 当前状态与方向：[RL 路线图](../../.doc/design/rl_roadmap.md) · **实测数字唯一账本**：[CartPole 基准账本](cartpole/README.md)

## 设计理念

- **一个算法，持续迭代**：维护不断进化的 MyZero，不再为每篇论文建独立实现
- **用户零组件概念**：`MyZero::new(env_id)` 自动套用库内按环境配置的组件组合；团队 promote 组件时只改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)
- **优先轴（2026-07 起）**：磨观测空间（图像/CNN）与 self-play（[纲领 §2.3](../../.doc/design/my_zero_algorithm_vision.md#23-战略目标与优先轴2026-07-01-定稿)）；连续/混合动作降级

## 环境 × 状态

| 环境 | 动作类型 | 门禁 | 状态 |
|------|---------|------|------|
| [**CartPole-v1**](cartpole/README.md) | 离散（2） | greedy eval ≥ 475 | ✅ 回归哨兵（官方口径 3-seed 中位 env-steps，数字见[账本](cartpole/README.md)） |
| [**Pendulum-v1**](pendulum/README.md) | 纯连续（1） | return ≥ -200 | 诊断中·已降级（当前 recipe 复用 CartPole 栈作诊断，不代表组件已裁决；[issue](../../.issue/items/pendulum_failure_diagnosis.md)） |
| 图像离散（Atari-100k 类） | 离散 | 任务指标 | v0.26 P0（[路线图 §5](../../.doc/design/rl_roadmap.md#5-v026-方向2026-07-01-战略转向定稿)） |
| Gomoku（→ 象棋） | 离散棋盘 | 胜率 | v0.26 P1（self-play 踏脚石；环境已备 `python/gym_env/gomoku/`） |
| Platform-v0 | 混合 Tuple | return 趋势上升 | 已降级，待具体需求 |

## 内部组件进展（团队 · promote 时改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)）

> **裁决矩阵**：一行一个组件、一列一个环境；表示「该组件在该环境是否已完成效果裁决」，不等同于「当前诊断栈是否临时启用」。
> 图例：`✅ 已验收` · `❌ 实测有害/无增益` · `⏳ 待测` · `⏸ 此环境不适用` · `— 未实现`
> 用户 API **不暴露**组件开关；实测数字一律见[基准账本](cartpole/README.md)，本表只记裁决结论。

| 组件         | [CartPole-v1](cartpole/README.md) | [Pendulum-v1](pendulum/README.md) | 图像/Gomoku | 备注 |
| ------------ | :-------------------------------: | :-------------------------------: | :---------: | --------------------------- |
| consistency  |                ✅                 |                 ⏳                 |      ⏳      | SimSiam 一致性 loss |
| reconstruction |              ✅※                |                 ⏳                 |      ⏳      | ※ autograd 修复后 3-seed 中位不再显示增益（方差大、未回滚），v0.26 P0 重标定 loss 系数后复裁，见[账本结论](cartpole/README.md#结论v025-收口) |
| Sampled（连续采样候选） |       ✅ 接入不回归        |                 ⏳                 |      ⏸      | CartPole K=N 退化全枚举、与无 Sampled 逐步等价；Pendulum B=7 才是真实子采样 |
| reanalyze    | ❌ 当前实现有害（[issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md)） | ⏳ | **v0.26 战略组件** | 「实时轻 acting + 离线重 reanalyze」解耦，图像线优先验证 |
| value_prefix |                ❌                 |                 ⏳                 |      ⏳      | CartPole 有害（≠ 全局坏） |
| target_net   |                 ⏳                 |                 ⏳                 |      ⏳      | 已入库，训练循环待接 |
| SVE          |                 ⏳                 |                 ⏳                 |      ⏳      | 已入库，训练循环待接；🔲 改进：固定权重 → 自适应 mixed target |
| completedQ   | ❌ 系统性慢于 visit（[issue](../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md)） |                 ⏳                 |      ⏳      | 复测留 `\|A\| > sims` 场景 |
| Gumbel-root  | ❌ 未收敛（同上 issue） |                 ⏳                 | ⏳ 少 sim acting 候选 | 库内已实现；低延迟 acting 场景 v0.26 复测 |

> 论文全称与 arXiv：[算法纲领 §4.1 — 组件文献对照](../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)

**CartPole-v1 当前内置**：base + **consistency + reconstruction + Sampled**（PUCT · sims=20 · td=5 · continuation 二值门）。

**Pendulum-v1 当前内置**：复用 CartPole 栈做**诊断**（B=7 · sims=20 · `reward_scale(0.1)`），仍在失败区间；不要据此给组件下 ✅/❌ 裁决。

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
| **快** | env-steps-to-solved 为主（官方口径 **3-seed 中位 + 达标率**），wall-clock 为辅 |

CartPole 是 sanity/regression **哨兵**（叠组件不崩的证明），不是组件价值证明台；不拿 wall-clock 判生死（[纲领 §2.2/§2.3](../../.doc/design/my_zero_algorithm_vision.md#22-首要评价指标)）。

## 算法核心

MuZero 三网络（Representation / Dynamics / Prediction）+ latent MCTS + K 步 unroll + categorical value/reward。Replay / n-step 显式区分 `terminated / truncated / continuation`：真终止 `continuation=0`，普通步与 time-limit truncation `continuation=1`。Dynamics 同时学习 continuation；MCTS imagined edge 的 discount 用 **binary gate `gamma * (1 - done)`**（`done` 由 continuation 头阈值化），与 n-step value target 的二值 continuation 口径一致——**不**用 soft `gamma * predicted_continuation` 对每条健康边连续衰减（确定性/无终止环境下软折扣只注方差，见账本结论）。该基础语义不列入上方消融组件矩阵。母论文见 [算法纲领 §4.1 — base](../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)。

## 代码组织

- `src/rl/algo/my_zero/`：自包含 MyZero 库（**唯一** `*Zero` 实现）
- `recipe.rs`：按 `env_id` 内置组件开关（内部维护，文件名保留 recipe）
- `tests/baseline_matrix_bench.rs`：发版基线增量链（base → +cons → +recon → promoted，`--ignored` 手动）
- `examples/my_zero/*/main.rs`：薄示例（env + 训练契约 + opt-in 落盘）

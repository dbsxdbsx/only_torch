# MyZero 强化学习状态总览

> **定位**：only_torch RL 模块的**唯一战略 / 裁决 / 待办权威源**。  
> 本文合并自原 8 份文档（2026-07-23），并在 2026-08-12 将示例总览中的组件矩阵与处方表并入，消除双事实源。  
> **文档分工**见下方 §0；环境配置另见 [rl_python_env_setup.md](../rl_python_env_setup.md)；实测数字按环境分账：  
> [CartPole](../../examples/my_zero/cartpole/README.md) / [Gomoku](../../examples/my_zero/gomoku/README.md) / [MinAtar](../../examples/my_zero/minatar/README.md) / [Pong](../../examples/my_zero/pong/README.md)。

---

## 0. 文档权威（DevOps 契约）

| 角色 | 路径 | 写什么 |
|------|------|--------|
| **战略 / 组件裁决 / backlog** | 本文 | 算法定位、环境状态、组件矩阵、处方表、验收协议、未完成事项 |
| **组件可测性（非裁决）** | [rl_myzero_component_detectability.md](rl_myzero_component_detectability.md) | 每个旋钮解决什么、环境侧能否测到；当前只做 ① schema，`!`（训练中裁判）当 `—`；value 与连续观测量纲是 always-on `symlog`，不是路由；**不**写 promote |
| **搜索派系与温度（非裁决）** | [rl_myzero_search_lineage.md](rl_myzero_search_lineage.md) | 为何走 PUCT / Sampled / Gumbel、不走最大熵树搜；现行 \(\tau\) 契约；文献表 |
| **Python / Gymnasium 配置** | [rl_python_env_setup.md](../rl_python_env_setup.md) | 解释器、依赖、Platform / MinAtar / ALE 安装 |
| **示例入口（薄）** | [examples/my_zero/README.md](../../examples/my_zero/README.md) | 理念摘要、环境索引、快速开始；**不**维护组件裁决副本 |
| **环境数字账本** | `examples/my_zero/<env>/README.md` | 仅该环境的实测数字与口径列 |
| **未闭环现场** | `.issue/items/` | 负结果、诊断、一级风险；链回本文，不另立路线图 |
| **会话草稿（非权威）** | 编辑器本地 `*.plan.md` 等 | 可作草稿；结论落盘只认本文 / 账本 / issue |

已删除并并入本文的历史文件（勿再链接）：`rl_roadmap.md`、`rl_closure_plan.md`、`rl_phase1_image_plan.md`、`rl_phase1_report.md`、`my_zero_algorithm_vision.md`、`my_zero_world_model_foundation.md`、`my_zero_simulus_ablation_plan.md`、`_archive/rl_roadmap_v020_v024.md`。

**当前主线状态**：crate **v0.26.0**（2026-08-12）。RL 未恢复为开发主线。MinAtar 三刀便宜诊断均负：acting-sims=8（best 1.5）、独立 target_net（best 2.3）、td_steps 10→5（best 2.6 vs 对照 4.1）。FNC / COMBO 整包不进主线；ROSMO 默认关。下次接手顺序：本文 → [环境配置](../rl_python_env_setup.md) → `just test-filter rl` → `just smoke-rl`。

---

## 1. MyZero 是什么

`src/rl/algo/my_zero/` — 项目**唯一**的 MuZero 系实现。核心思路：

- **Learned world model + MCTS**：纯 learned dynamics 做树搜索（不依赖真规则 / env snapshot）
- **消融驱动叠组件**：一次一项、3-seed 统计、数字进账本
- **万金油目标**：`MyZero::new(env_id)` 一个入口覆盖向量 / 棋盘 / 图像 / 离散 / 连续 / 混合
- **用户零组件概念**：按环境内置 recipe；团队 promote 时只改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)

**战略目标**：中国象棋（离散 self-play）+ 商业图像游戏（图像 obs + 实时 + 样本贵）。

**评价指标**：env-steps-to-solved（达同一门槛所需的环境交互越少越好）；wall-clock 是约束不是评价。

**算法核心（非消融组件）**：Representation / Dynamics / Prediction + latent MCTS + K 步 unroll + categorical value/reward。Replay / n-step 显式区分 `terminated / truncated / continuation`；MCTS imagined edge discount 用 binary gate `gamma * (1 - done)`。

---

## 2. 架构概览

| 层 | 位置 | 说明 |
|----|------|------|
| 算法主体 | `src/rl/algo/my_zero/` | 模型 + 训练循环 + MCTS + recipe |
| 环境 | `src/rl/env/` | `GymEnv`（Gymnasium-only）+ `MinariDataset` |
| 搜索 | `src/rl/mcts/` | PUCT / Gumbel / Sampled；吃 model 不吃 env |
| buffer | `src/rl/buffer/` | `SelfPlayGame` / `ReplayBuffer` / `RolloutBuffer` |
| model-free | `src/rl/algo/{sac,ppo}/` | 纯 helper；Agent 在 `examples/` |

**通用契约（v0.25 落地）**：`ObservationSchema`（Flat/Image/Board/ImageDense/Tokens）· `ActionSchema+Codec`（Discrete/MultiDiscrete/Continuous/Hybrid）· `LatentState` · `WorldModel`。

---

## 3. 已验收环境与组件

### 3.1 环境状态

| 环境 | 门槛 | 状态 | recipe |
|------|------|------|--------|
| **CartPole-v1** | greedy ≥ 475 | ✅ 哨兵冻结（旧账本中位 ~9.8k，3/3；**2026-08-14 改走结构路由，数字待重跑**） | 结构路由：reconstruction + obs `symlog` + Stochastic K=8 |
| **Gomoku** (vs random) | 胜率 ≥ 95% | ✅ 已闭环（recipe=base；历史里程碑见[棋盘账本](../../examples/my_zero/gomoku/README.md)） | base（全组件关） |
| **StochasticCartPole** | K=8 3/3 | ✅ Stochastic MuZero 验收 | always-on K=8 |
| **MinAtar/Breakout** | greedy ≥ 8 | ⚠️ 正式 3-seed 0/3（中位 best **3.9**）；acting-sims=8 负（best 1.5）；独立 target_net 负（best 2.3）；td_steps=5 负（best 2.6，不 promote） | consistency ON |
| **ALE/Pong-v5** | greedy ≥ −18 | ❌ 预算不足（0/3 平直） | consistency ON |
| **Pendulum-v1** | — | ⏳ 诊断态（n-step target 方差坍缩） | 同 CartPole（诊断，不代表裁决） |
| **Platform-v0** | — | ✅ 管线 smoke 通过，未做基准 | Sampled |
| Schema toys | loss 有限 | ✅ 接口纵切 | — |
| CartPole POMDP-lite | greedy ≥ 400 | ❌ posterior ON 0/3 | 默认关 |

### 3.2 组件裁决（浓缩）

| 组件 | CartPole | Gomoku | 图像 | 全局 |
|------|----------|--------|------|------|
| consistency (coef 2) | ✅ promote | 中性 / 弱阳未过线 | 未定 | ON（native=图像） |
| reconstruction (coef 16) | ✅ promote | 发现集未过未见 seed | 未定 | ON |
| Sampled MuZero | ✅ | — | — | ON |
| Stochastic (K=8) | ✅ | 无系统回归 | smoke | always-on |
| completedQ | ❌ | ❌ 中性 | — | OFF |
| Gumbel-root | ❌ | 中性 | — | OFF |
| reanalyze (旧全树版) | ❌ | ❌ | 战略待图像 | OFF |
| ROSMO reanalyze | 中性 / 无增益 | ❌ | 待图像基线 | OFF |
| HL-Gauss | ❌ | — | 待测 native | OFF |
| obs symlog | 逻辑 always-on | — | always-on | always-on（token 除外） |
| PER | — | ❌ | — | OFF |
| value_prefix | ❌（有断链 bug） | — | — | OFF |
| target_net / SVE | 入库待测 | — | — | OFF |
| recurrent posterior | ❌ | — | — | OFF |
| KL 自适应 lr | ❌ 灾难级 | ✅ 无害 | — | OFF |
| 主动数据 proxy | — | ❌ | — | OFF |

promote 时改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)；论文全称见 [reading_log](../paper/reading_log.md)。

### 3.3 组件 × 环境裁决矩阵（详细）

> 一行一个组件、一列一个环境；表示「是否已完成效果裁决」，不等同于「当前诊断栈是否临时启用」。  
> 图例：`✅` 已验收 · `❌` 有害/无增益 · `⏳` 待测 · `⏸` 不适用 · `—` 未实现 · `⚠️` 未过线弱阳  
> 实测数字一律见各环境账本，本表只记裁决结论。

| 组件 | CartPole-v1 | Pendulum-v1 | 图像 | Gomoku | 备注 |
|------|:-----------:|:-----------:|:----:|:------:|------|
| consistency | ✅ | ⏳ | ⏳ | ⚠️ 带内弱阳不 promote | SimSiam；棋盘 CNN 底座弱阳 <0.30 线 |
| reconstruction | ✅※ | ⏳ | ⏳ | ❌ 未见 seed 未复现 | ※ CartPole coef=16；Gomoku coef=1 仅发现集 |
| Sampled | ✅ 接入不回归 | ⏳ | ⏸ | ⏸ | 小离散 K=N 退化全枚举 |
| HL-Gauss | ❌ 显著劣化 | ⏸ | ⏳ native | ⏸ | 开关留库，图像域复测 |
| obs symlog | 逻辑 always-on | 逻辑 always-on | 逻辑 always-on | ⏸ token | 不读 Box；token ID 除外 |
| reanalyze | ❌ [issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) | ⏳ | 战略组件 | ⏸ | acting/reanalyze 解耦待做 |
| ROSMO | ❌ 无增益 | ⏳ | ⏳ | ❌ 有害 | recipe 默认关 |
| PER | — | — | — | ❌ | buffer 伴生件，默认关 |
| value_prefix | ❌※断链 | ⏳ | ⏳ | ⏸ | 训 LSTM / 搜普通 reward head |
| target_net | ⏳ | ⏳ | ❌ 独立接线负（best 2.3 vs 对照 4.1） | ⏳ | 默认关；只换 n-step 尾值不够 |
| SVE | ⏳ | ⏳ | ⏳ | ⏳ | 同上 |
| completedQ | ❌ | ⏳ | ⏳ | ❌ 中性 | 全域关 |
| Gumbel-root | ❌ | ⏳ | ⏳ | ❌ 中性 | 少 sim acting 降档候选 |
| recurrent posterior | ❌ [issue](../../.issue/items/my_zero_pomdp_lite_posterior_negative.md) | ⏳ | ⏳ | ⏸ | 代码保留默认关 |
| Stochastic K=8 | ✅ | ❌ 诊断失败区间 | 💨 smoke | ✅ 无系统回归 | always-on；确定性可退化为单峰 |
| KL 自适应 lr | ❌ 灾难级 | ⏸ | ⏸ | ✅ 无害 | 默认关 |

### 3.4 处方表（跨域路由规则）

> 矩阵记**结果**；本表记**规则**（机制 / 适应症 / 禁忌 / 剂量）。新环境先查表路由，再一次预注册 A/B，禁止网格搜索。剂量判据：宽平台 = 真机制；刀锋最优点 = 过拟合签名。  
> 「适应症能不能从环境测到、何时测」见 [组件可测性](rl_myzero_component_detectability.md)，不在本表展开。

| 组件 | 机制 | 适应症 | 禁忌症 | 剂量 |
|------|------|--------|--------|------|
| consistency | latent 对齐下一帧编码（SimSiam stop-grad） | 稀疏任务信号 × dynamics 瓶颈；图像 native | 棋盘两裁未过线；可冻结真值时优先 recon | coef=2.0 |
| reconstruction | decoder 对齐真实 obs | 稀疏信号 × obs 比特任务相关 | 高维无关比特；跨 seed 不迁移 | CartPole 16；Gomoku 不 promote |
| PER | `p=\|ν−z\|^α` 重排消费 | 覆盖不足 × 冻结事实 target | 与 recon 均匀监督冲突；弱网自评刷新 | α=0.6，默认关 |
| HL-Gauss | value/reward 高斯软标签 | 图像高噪声宽 support | 窄 support 低噪声（CartPole 劣化） | σ 见 `value_encoding` |
| obs symlog | 连续观测无量纲化 | 所有进网的连续观测 | token ID（查表前不能压） | 无参数 |
| reanalyze | 全树重搜刷新 target | 陈旧数据 × 高 replay × 离线算力 | CartPole 有害；棋盘终局事实非 native | 待图像域 |
| ROSMO | 一步 look-ahead 刷新 | 同上且自评质量足够 | 弱模型自指；新鲜低 replay | α=0.2 + prior top-16 |
| completedQ / Gumbel-root | 少 sim 根决策 | 理论 sims≪\|A\|（实测未证实） | sims≫\|A\|（CartPole 劣化） | 全域关 |
| KL 自适应 lr | 探针 KL 调 lr | 大 batch × 探针同源 | 小 batch × 大 buffer（CartPole 发散） | kl_targ=0.02，默认关 |
| Sampled | 候选采样 + 先验校正 | `K_eff < N`（读已有 sims） | `K_eff = N` 时全展开；recipe 可 overlay 开 | `sampled_params` |

未列组件（value_prefix / target_net / SVE）：接线未完成或断链在修，不写推测处方。课程 / 真规则树为诊断脚手架，不进本表。

**当前内置 recipe 摘要**：

- CartPole：结构路由（recon + obs `symlog` + K=8；consistency / Sampled 谓词为关。PUCT · acting_sims=1 · think/eval sims=20 · td=5）
- Pendulum：诊断 overlay 仍是 cons + recon + Sampled（B=7 · acting_sims=1 · think/eval sims=20）——勿据此裁决组件，也不跟 CartPole 路由变薄
- Gomoku：base 全关（Flat MLP · negamax MC · sims=100）
- MinAtar：consistency ON（见 [minatar 账本](../../examples/my_zero/minatar/README.md)）

---

## 4. 验收协议（三层）

| 层 | 命令 | 判什么 |
|----|------|--------|
| 单元/集成测试 | `just test-filter rl` | 正确性 |
| 管线 smoke | `just smoke-rl` | 管线通、loss 有限 |
| 统计基线 | `SEEDS=3 cargo run --example my_zero_cartpole --release` | 3-seed 中位 env-steps + 达标率 |

**改动纪律**：搬运（行为零变化）可批量；改进（改行为）一次一项 + 3-seed 消融。  
「变慢 ≠ 失败」；新实测即新基线；仅不收敛 / 不达标记 issue。

---

## 5. 未完成事项（收口时的 backlog）

### 5.1 高优先级（下次接手 RL 时）

| 项 | 说明 | 关联 |
|----|------|------|
| MinAtar 图像效率下一刀 | 三刀便宜诊断均负（sims=8 / target_net / td_steps=5）。默认仍 sims=25、td=10、无独立 target。勿砍 `trains_per_episode`；若继续只允许等步数加训，否则暂停 RL | [minatar 账本](../../examples/my_zero/minatar/README.md) |
| target_net + SVE 终审 | 独立 target_net 已负；SVE 仍未接。禁止把 SVE 当补丁绑回去 | — |
| value_prefix 断链修复 | 训练用 LSTM head，搜索用普通 reward head → 静默错误 | — |
| Pendulum 终审 | n-step target 方差坍缩；可能需 1-step + target_net | [pendulum issue](../../.issue/items/pendulum_failure_diagnosis.md) |

### 5.2 中优先级

| 项 | 说明 |
|----|------|
| 图像线预算标定 | Pong CPU 需 ~120k updates (batch256)，当前只跑了 9.6k (batch16) |
| reanalyze acting/reanalyze 解耦 | 实时轻 acting + 离线重 reanalyze |
| loss 优先回放 | Simulus B1，随 reanalyze 一起做 |
| HL-Gauss 图像域复测 | CartPole 负但图像是 native 场景 |
| 中国象棋 | Gomoku 踏脚石已立；规则引擎 / Gym 环境待推 |

### 5.3 明确不做

- BetaZero（belief 规划）· CFR 族（不完全信息）· 多智能体 N>2 · SAC 升格母算法
- DreamerV3 派（棋类非 SOTA）· Simulus tokenizer/RetNet · **FNC / COMBO 整包**（问题设定与 MyZero 不同，禁止当主线组件）
- 演化 + RL 联合搜索：仅长期 backlog，无实施计划；触发条件 = 需要演化 MyZero backbone
- **ROSMO 当救命药 / 绿灯后默认开**：棋盘有害、CartPole 无增益；无裁判探针（轨迹回报 + 少量新鲜搜索）不开；弱模型阶段禁止

---

## 6. 一级风险

**CPU-only × 图像 CNN × MCTS × 实时**：spike 实测裁决 GO（CNN 只进 representation 1 次/步，不被 sims 放大；sims=50 仍 < 4ms）。真实瓶颈 = 训练吞吐 × 消融轮次。详见 [.issue/items/cpu_only_mcts_image_realtime_risk.md](../../.issue/items/cpu_only_mcts_image_realtime_risk.md)。

免规划世界模型（Simulus 等）是该风险的实证退路；可拆组件（HL-Gauss、loss 优先回放）挂 §5，不另立消融计划文档。

---

## 7. 关键设计决策备忘

- **母算法 EZ-V2 / MCTS 系**（非 DreamerV3）：ICML 2024 证明 Zero 系跨 66 任务 50 胜 DreamerV3
- **生产搜索始终 learned dynamics**：true-rules 仅 `cfg(test)` diagnostic
- **CartPole 只是哨兵**：不用它判组件价值，更不拿 wall-clock 判生死
- **优先轴**：观测空间（CNN/图像）+ self-play；Pendulum/Platform 降级为诊断/smoke
- **recon_coef=16 是话语权旋钮**：非单位换算，跨量纲需重标
- **value/reward always-on `symlog`**：未知量纲进 41 原子前必须无墙压缩；与连续观测入口公式相同，但观测不进 41 原子。线性 `reward_scale` 已删除
- **Stochastic K=8 always-on**：确定性环境自动退化为单峰（自适应 K 为会话悬案，未落盘改契约前仍以此为准）
- **Sampled MuZero 候选层正交**：K_eff=|A| 时与 PUCT 等价，仅余 ~26% wall-clock 簿记开销
- **三件套正交分层**：Sampled=候选层 / Gumbel=根层 / completedQ=目标层（融合契约；全域默认关后两项）
- **温度是评测开关**：非评测恒 `τ=1`（原样 π / visit，政策目标不拧尖）；评测 / `run` 恒 `τ=0`（Gumbel 噪声关）。无 hold/decay 课程表，与观测/动作类型无关。文献取舍与「为何不走最大熵树搜」见 [搜索派系与温度](rl_myzero_search_lineage.md)
- **sims 是 wall-clock 预算不是「越大越好」**：MinAtar 默认仍 25。2026-08-13 单 seed 1h：`SIMS=8` **不买吞吐**（27,980 steps / 1h，约 7.8 step/s，低于对照 sims=25 的 ~9.1），greedy 锁 1.2–1.5（对照 seed 42 best 4.1）。无与游戏无关的路由规则则不 promote
- **标签刷新要有裁判**：库存 MCTS vs 一步 look-ahead，用轨迹回报 / 少量新鲜搜索比较，不用环境类型（图像/稠密奖励）猜；ROSMO 默认关
- **target_net 可独立接线，但 MinAtar 上不够**：只换 n-step 尾值（滞后 V 替换 buffer `root_value`），hard 400。2026-08-13 单 seed：best **2.3** vs 对照 seed 42 的 4.1，等步数更低，不 promote。每步 EMA 已否决
- **td_steps 默认仍 10**：2026-08-13 单 seed 1h `TD_STEPS=5`：best **2.6** / 32,134 steps，等步数低于对照 seed 42（~32k 时 4.1）。缩短 n 不加税也不买吞吐（~8.9 vs ~9.1 step/s），不是 2–4 分天花板的原因，不 promote

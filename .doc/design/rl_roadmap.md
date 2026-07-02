# 强化学习路线图（当前态）

> **定位**：RL 模块的**当前状态 + 验收协议 + 下一版方向**。战略层（为什么选这条路）见 [MyZero 算法纲领](./my_zero_algorithm_vision.md)；实测数字**唯一账本**是 [examples/my_zero/cartpole/README.md](../../examples/my_zero/cartpole/README.md)。
> v0.20–v0.24「每版一个算法」时代的设计决策与实施计划已整体归档：[rl_roadmap_v020_v024.md](./archive/rl_roadmap_v020_v024.md)（MCTS 接缝设计、能力边界矩阵、SAC 技术笔记等长期有效内容也在归档 §2.5 / §2.2.1b / §4，按需链回，不重复维护）。
>
> **创建日期**：2026-02-14 · **本版重写**：2026-07-02（v0.25 收口）

---

## 1. 文档分工（先读这张表）

| 文档 | 回答什么 |
|------|----------|
| **本文** | 当前模块状态、验收协议、v0.25 结果、v0.26 方向 |
| [my_zero_algorithm_vision.md](./my_zero_algorithm_vision.md) | 算法哲学、文献谱系、做/不做决策、战略目标 |
| [examples/my_zero/README.md](../../examples/my_zero/README.md) | 组件×环境实测矩阵、命令、门禁 |
| [examples/my_zero/cartpole/README.md](../../examples/my_zero/cartpole/README.md) | **基准账本**（所有 benchmark 数字的唯一事实源） |
| [rl_python_env_setup.md](../rl_python_env_setup.md) | Python / Gymnasium 环境搭建 |
| [rl.instructions.md](../../.github/instructions/rl.instructions.md) | 改 RL 代码时的 agent 约束 |
| [archive/rl_roadmap_v020_v024.md](./archive/rl_roadmap_v020_v024.md) | 历史：v0.20–v0.24 设计决策与实施计划 |

---

## 2. 当前状态（v0.25 收口，2026-07-02）

### 2.1 双轨架构

| 轨道 | 位置 | 状态 |
|------|------|------|
| **MyZero**（主线，项目唯一 `*Zero` 实现） | `src/rl/algo/my_zero/`（自包含：模型 + 训练循环 + MCTS 接入 + recipe） | CartPole ✅ 哨兵；Pendulum ⏳ 诊断中 |
| **SAC / PPO**（model-free 基线） | 纯函数 helper 入库 `src/rl/algo/{sac,ppo}/`；Agent 与训练循环留在 `examples/{sac,ppo}/` | SAC 四环境 + PPO CartPole 示例可跑 |

支撑层：`src/rl/env/`（`GymEnv` Gymnasium-only + `MinariDataset`）、`src/rl/buffer/`（`Transition` / `ReplayBuffer` / `RolloutBuffer` / `SelfPlayGame`）、`src/rl/mcts/`（搜索内核：PUCT / Gumbel / Sampled，吃 model 不吃 env——接缝设计见[归档 §2.5](./archive/rl_roadmap_v020_v024.md#25-mcts-抽象边界内核吃-model不吃-env选择规则可插拔v022-定形)）。

### 2.2 MyZero 组件裁决现状

组件由 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs) 按 env 注入，用户 API 不暴露开关。

| 状态 | 组件 |
|------|------|
| ✅ CartPole promoted | consistency + reconstruction + Sampled（PUCT · sims=20 · td=5 · continuation 二值门） |
| ❌ CartPole 负结果（代码保留、recipe 关） | completedQ / Gumbel-root（[issue](../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md)）、reanalyze 写回（[issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md)）、value_prefix |
| ⏳ 已入库待接/待测 | target_net、SVE |

组件×环境全矩阵见 [MyZero 总览](../../examples/my_zero/README.md#内部组件进展团队--promote-时改-recipers)；实测数字一律以[基准账本](../../examples/my_zero/cartpole/README.md)为准。

---

## 3. 验收协议（三层，发版固定关卡）

| 层 | 命令 | 判什么 | 何时跑 |
|----|------|--------|--------|
| **单元/集成测试** | `just test`（RL 子集 `just test-filter rl`，~229 个 `#[test]`，其中 33 个依赖 Python） | 正确性 | 每次改动 |
| **管线 smoke** | `just smoke-rl`（聚合 7 目标：MyZero cartpole/pendulum + PPO cartpole + SAC cartpole/pendulum/platform/lunarlander） | 管线通、loss 有限；**不验收敛** | 每次发版 |
| **统计基线** | MyZero：`SEEDS=3 cargo run --example my_zero_cartpole --release`；增量链 4 档：`my_zero::tests::baseline_matrix_bench`（`--ignored` 手动）；PPO/SAC：`SEED=42/43/44` 各跑一次 | **3-seed 中位 env-steps-to-solved + 达标率**（官方哨兵口径） | 发版前 / 改算法行为后 |

### 3.1 基线判据原则（2026-07-02 定稿）

- **变慢 ≠ 失败**：重测数字（哪怕比历史口径慢）直接接受为新基线并回填账本；历史行仅作方向性参考。
- 只有「跑完仍无法收敛 / 无法达标（CartPole greedy < 475）」才记 known-fail `.issue`，且不阻塞发版。
- 账本每行必须带**口径列**（profile / BLAS / seeds / 日期）；跨口径对比 wall-clock 无效，env-steps 有效。
- CartPole 是 **sanity/regression 哨兵**（叠组件不崩的证明），**不是**组件价值的证明台，更不拿 wall-clock 判生死——规划红利要到更难的环境验证（见 §5）。

### 3.2 改动纪律（搬运 ≠ 改进，沿用）

- **搬运**（挪代码、改 import、折 config）：行为零变化 → 可批量；CartPole 哨兵过即无回归。
- **改进**（改行为）：**一次一项**，单独 A/B 消融（3-seed 统计口径），单独过哨兵；不得与搬运混提交。

---

## 4. v0.25 结果小结（MyZero 统一）

1. **MyZero 成为唯一 `*Zero` 实现**：算法主体 + 全部组件（value_encoding / n_step / consistency / reconstruction / value_prefix / target_net / SVE / reanalyze / completedQ / Gumbel / Sampled）统一进库 `src/rl/algo/my_zero/`；旧 `muzero/` + `efficientzero/` 模块与示例整体删除。
2. **用户 API 定形**：`MyZero::new(env_id)` 链式 builder + `train/eval/run` 生命周期 + `.otm` 统一持久化；示例瘦身至 ~40 行。
3. **框架级 autograd 修复改变基线语义**：MSE/MAE/BCE/Huber 反向此前忽略 `upstream_grad`（作中间 loss 时丢链式缩放），修复后辅助 loss 回到正确量级——**所有历史 benchmark 数字失效**，哨兵改 3-seed 统计口径并全量重测（数字见[基准账本](../../examples/my_zero/cartpole/README.md)）。同批：非连续张量守卫、MCTS recurrent 单趟前向提速 ~3.4×、训练 batch-native（batch=8 快 2.2×）。
4. **负结果沉淀**：completedQ / Gumbel-root / reanalyze 在 CartPole 实测无增益或有害 → recipe 关、issue 记录（不否定其在 `|A| > sims`、低延迟 acting、图像等场景的价值，v0.26+ 复测）。
5. **Pendulum 转入可学习性诊断**：value 头容量已证伪为瓶颈，根因收敛到上游 target/搜索（[issue](../../.issue/items/pendulum_failure_diagnosis.md)）；**不作为发版门禁**。

---

## 5. v0.26 方向（2026-07-01 战略转向定稿）

> 依据：真实目标是**中国象棋**（离散完美信息 self-play）与**商业图像游戏**（图像 obs + 实时 + 样本贵）——两者都不在「动作空间广度」轴上。路线从「磨动作空间（Pendulum 连续 / Platform 混合）」**转向「磨观测空间（图像/CNN）+ self-play」**。完整论证见[算法纲领 §2.3](./my_zero_algorithm_vision.md#23-战略目标与优先轴2026-07-01-定稿)。

| 优先级 | 方向 | 说明 |
|--------|------|------|
| **P0** | loss 系数重标定消融 | 旧基线部分依赖 autograd bug 放大辅助 loss；修复后显式调大 cons/recon 系数有望收回样本效率——v0.26 第一个消融 |
| **P0** | CNN 图像表征 + 图像离散基准（Atari-100k 类） | 商业游戏直接代理；复用已验收 consistency + reconstruction（自监督正是图像+少样本的命门组件） |
| **P1** | Gomoku self-play → 象棋踏脚石 | `SelfPlayGame` / negamax backup / legal_mask 地基已在库；环境 `python/gym_env/gomoku/` 已备 |
| **P1** | reanalyze 复活 + acting/reanalyze 解耦 | 「实时轻 acting（少 sim / policy 先验）+ 离线重 reanalyze（榨样本）」是商业游戏路线的战略组件；CartPole 负结果不构成否定 |
| **P2** | Gumbel 少 sim acting 复测 | 在 `|A| > sims` 或低延迟场景重估（CartPole `sims ≫ |A|` 不构成否定） |
| **降级** | Pendulum / Platform | 不在两大目标关键路径；Pendulum 保留诊断态，Platform 待具体需求再排 |

**一级风险**（显式管理）：CPU-only × 图像 CNN × MCTS × 实时的结构性冲突——见 [.issue/items/cpu_only_mcts_image_realtime_risk.md](../../.issue/items/cpu_only_mcts_image_realtime_risk.md)。

## 6. 长期 backlog（可能做，不承诺）

Stochastic MuZero（chance node，动树核心）· MENTS/DENTS 熵 backup · MCTS 并行（virtual loss / 批量叶子）· PER · Beta 分布 · DQN/TD3（教学）· 演化+RL 联合搜索。详细谱系与接缝对照见[归档 §5.10](./archive/rl_roadmap_v020_v024.md#510-mcts--zero-变体-backlog接缝已留做不做按需)与[算法纲领 §5](./my_zero_algorithm_vision.md#5-决策表可能做--暂缓--不做)。

**明确不做**：BetaZero（belief 规划，[纲领 §5.3](./my_zero_algorithm_vision.md#53-betazero-裁决2026-06-21-定稿)）· 不完全信息博弈（CFR 族）· 多智能体 N>2 · SAC 升格母算法。

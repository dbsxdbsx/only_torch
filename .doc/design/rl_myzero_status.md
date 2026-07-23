# MyZero 强化学习状态总览

> **定位**：only_torch RL 模块的全部现状——算法定位、已验收结果、未完成事项。
> 本文合并自原 8 份文档（2026-07-23 收口整理）。
> 环境配置另见 [rl_python_env_setup.md](../rl_python_env_setup.md)；实测数字按环境分账：
> [CartPole](../../examples/my_zero/cartpole/README.md) / [Gomoku](../../examples/my_zero/gomoku/README.md) / [MinAtar](../../examples/my_zero/minatar/README.md)。

---

## 1. MyZero 是什么

`src/rl/algo/my_zero/` — 项目**唯一**的 MuZero 系实现。核心思路：

- **Learned world model + MCTS**：纯 learned dynamics 做树搜索（不依赖真规则/env snapshot）
- **消融驱动叠组件**：一次一项、3-seed 统计、数字进账本
- **万金油目标**：`MyZero::new(env_id)` 一个入口覆盖向量/棋盘/图像/离散/连续/混合

**战略目标**：中国象棋（离散 self-play）+ 商业图像游戏（图像 obs + 实时 + 样本贵）。

**评价指标**：env-steps-to-solved（达同一门槛所需的环境交互越少越好）；wall-clock 是约束不是评价。

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

### 3.1 环境支柱

| 环境 | 门槛 | 状态 | recipe |
|------|------|------|--------|
| **CartPole-v1** | greedy ≥ 475 | ✅ 哨兵冻结（中位 ~9.8k env-steps，3/3） | consistency + reconstruction(coef16) + Sampled |
| **Gomoku** (vs random) | 胜率 ≥ 95% | ✅ M0–M4 闭环 | base（全组件关） |
| **StochasticCartPole** | K=8 3/3 | ✅ Stochastic MuZero 验收 | always-on K=8 |
| **MinAtar/Breakout** | greedy ≥ 8 | ⚠️ 管线能学（peak 4.1）未达标 | consistency ON |
| **ALE/Pong-v5** | greedy ≥ −18 | ❌ 预算不足（0/3 平直） | consistency ON |
| **Pendulum-v1** | — | ⏳ 诊断态（n-step target 方差坍缩） | 同 CartPole |
| **Platform-v0** | — | ✅ 管线 smoke 通过，未做基准 | Sampled |

### 3.2 组件裁决

| 组件 | CartPole | Gomoku | 图像 | 全局 |
|------|----------|--------|------|------|
| consistency (coef 2) | ✅ promote | 中性 | 未定 | ON（native=图像） |
| reconstruction (coef 16) | ✅ promote | 中性 | 未定 | ON |
| Sampled MuZero | ✅ | — | — | ON |
| Stochastic (K=8) | ✅ | — | — | always-on |
| completedQ | ❌ | ❌ | — | OFF |
| Gumbel-root | ❌ | 中性 | — | OFF |
| reanalyze (旧全树版) | ❌ | ❌ | — | OFF |
| ROSMO reanalyze | 中性 | ❌ | — | OFF |
| HL-Gauss | ❌ CartPole | — | 待测 | OFF |
| value_prefix | ❌（有断链 bug） | — | — | OFF |
| target_net / SVE | 入库待测 | — | — | OFF |
| recurrent posterior | ❌（容量不足） | — | — | OFF |
| 主动数据 proxy | — | ❌ | — | OFF |

---

## 4. 验收协议（三层）

| 层 | 命令 | 判什么 |
|----|------|--------|
| 单元/集成测试 | `just test-filter rl` | 正确性 |
| 管线 smoke | `just smoke-rl` | 管线通、loss 有限 |
| 统计基线 | `SEEDS=3 cargo run --example my_zero_cartpole --release` | 3-seed 中位 env-steps + 达标率 |

**改动纪律**：搬运（行为零变化）可批量；改进（改行为）一次一项 + 3-seed 消融。

---

## 5. 未完成事项（收口时的 backlog）

### 5.1 高优先级（下次接手 RL 时）

| 项 | 说明 | 关联 issue |
|----|------|-----------|
| MinAtar pilot 调参 | peak 4.1 / 门槛 8；需增大预算或调 lr/sims | `examples/my_zero/minatar/README.md` |
| target_net + SVE 终审 | 入库但零实测裁决（缺裁决场） | — |
| value_prefix 断链修复 | 训练用 LSTM head，搜索用普通 reward head → 静默错误 | — |
| Pendulum 终审 | n-step target 方差坍缩；可能需 1-step + target_net | `.issue/items/pendulum_failure_diagnosis.md` |

### 5.2 中优先级

| 项 | 说明 |
|----|------|
| 图像线预算标定 | Pong CPU 需 ~120k updates (batch256)，当前只跑了 9.6k (batch16) |
| reanalyze acting/reanalyze 解耦 | 实时轻 acting + 离线重 reanalyze 架构 |
| loss 优先回放 | Simulus B1，随 reanalyze 一起做 |
| HL-Gauss 图像域复测 | CartPole 负但图像是 native 场景 |
| 中国象棋 | Gomoku 踏脚石已立；15×15→象棋待推 |

### 5.3 明确不做

- BetaZero（belief 规划）· CFR 族（不完全信息）· 多智能体 N>2 · SAC 升格母算法
- DreamerV3 派（棋类非 SOTA）· Simulus tokenizer/RetNet

---

## 6. 一级风险

**CPU-only × 图像 CNN × MCTS × 实时**：spike 实测裁决 GO（CNN 只进 representation 1 次/步，不被 sims 放大；sims=50 仍 < 4ms）。真实瓶颈 = 训练吞吐 × 消融轮次。详见 `.issue/items/cpu_only_mcts_image_realtime_risk.md`。

---

## 7. 关键设计决策备忘

- **母算法 EZ-V2 / MCTS 系**（非 DreamerV3）：ICML 2024 证明 Zero 系跨 66 任务 50 胜 DreamerV3
- **生产搜索始终 learned dynamics**：true-rules 仅 `cfg(test)` diagnostic
- **CartPole 只是哨兵**：不用它判组件价值，更不拿 wall-clock 判生死
- **recon_coef=16 是话语权旋钮**：非单位换算，跨量纲需重标
- **Stochastic K=8 always-on**：确定性环境自动退化为单峰
- **Sampled MuZero 候选层正交**：K_eff=|A| 时与 PUCT 等价，仅余 ~26% wall-clock 簿记开销

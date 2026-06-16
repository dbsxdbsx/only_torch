# EfficientZero V2（v0.24）示例矩阵

> 全程 **learned-model**（含五子棋博弈，不用 perfect-model），严格遵照 EfficientZero V2。
> 算法 helper 在 [`src/rl/algo/efficientzero/`](../../src/rl/algo/efficientzero/)，搜索内核复用
> [`src/rl/mcts/`](../../src/rl/mcts/)。

## EZ 相对 MuZero 的增量（本版引入）

- **自监督 consistency**（SimSiam stop-grad）
- **value prefix（忠实版）**：LSTM hidden 穿过 MCTS 树、搜索期 reward 取前缀增量
- **target network**（EZ 稳定性增强）
- **reanalyze 调强**（batch-time，用 target net）+ **SVE**
- **Gumbel 搜索**（连续 / 混合 / 大动作空间）

## 示例矩阵与验收口径

| 示例 | 环境 | 动作 | 验收口径 | 状态 |
|------|------|------|----------|------|
| `cartpole/` | `CartPole-v1` | 离散 | **达标**：greedy eval ≥ 阈值（建议 ≥450；非 v0 的 195） | Phase 1 |
| `pendulum/` | `Pendulum-v1` | 纯连续 | **达标**：greedy eval 平均 return ≥ 阈值（建议 ≥ -200，对比 SAC 基线） | Phase 2a |
| `platform/` | `Platform-v0` | 混合 Tuple | **闭环**：return 趋势上升（不强制达标分数） | Phase 2b |
| `gomoku/` | `Gomoku-*-v0` | 离散博弈 | **best-effort**：vs `naive3` 胜率 ≥ 阈值（CPU 学不动可降棋盘/sims/对手级别） | Phase 3 |
| `atari/` | `ALE/*-v5` | 图像离散 | **pipeline smoke**：能学、loss 有限、无 panic（CPU 不追分数） | Phase 4 |
| `ant/` | `Ant-v5` | 高维连续 | **pipeline smoke**：高维连续管线跑通（不追分数） | Phase 4 |
| `minari/` | Minari 数据集 | 离线 | **pipeline smoke**：load + 一步训练（不进性能门禁） | Phase 4 |

## 两套判据，分开跑、不混用

- **达标门禁**：固定 eval seeds / 训练 max episode / eval 局数 / `num_simulations`，greedy(temp=0) eval 均值过阈值。
- **SMOKE**：所有示例支持 `SMOKE=1` 短跑（约数局 + 1 次训练），只验「管线跑通、loss 有限、无 panic」，**不**断言分数。默认 CI 不跑训练。

## 可复现性

- 搜索随机性经注入的 seeded RNG 控制（`mcts_search(.., rng)`），eval 用 `env.reset(Some(seed))`；
  greedy eval（temp=0 + exploration_fraction=0）输出本就确定。

## 平台

- `cartpole` / `pendulum` / `platform` / `gomoku`：跨平台。
- `atari` / `ant` / `minari`：**Linux 优先验证，Windows best-effort**（MuJoCo/Atari 在 Windows 历史脆弱）。

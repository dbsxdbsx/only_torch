# MyZero · Gomoku 9×9（棋盘账本）

> [← 返回 MyZero 总览](../README.md)
>
> **本文件是棋盘（self-play）支柱 benchmark 数字的唯一账本（owner）**：issue / roadmap / AGENTS 一律链到这里。每行实测带口径（profile / BLAS / seeds / 日期）。
>
> 无独立 example：入口为库内手动档 bench（`src/rl/algo/my_zero/tests/gomoku_m*_bench.rs`）
> 与 `just smoke-my-zero-gomoku` 冒烟关卡；环境 `python/gym_env/gomoku/`。

双人零和 · 9×9 · 5 连胜 · |A|=81 · **万金油铁律**：树内全程 learned dynamics，真环境仅 self-play/eval 走子、根节点 legal_mask、终局判定三处。

## 运行

```bash
# 冒烟（3 局全路径：训练 / 快照 / gating / naive 梯队）
just smoke-my-zero-gomoku

# M2 预注册正裁（3 seeds，约 5 分钟/seed）
cargo test --release --features blas-mkl gomoku_m2_3seed -- --ignored --nocapture --test-threads=1

# M3 消融臂（一次一臂）
cargo test --release --features blas-mkl gomoku_m3_<arm> -- --ignored --nocapture --test-threads=1
# arm ∈ {gumbel, base_s16, gumbel_s16, cons, recon, cnn, budget, replay32, lr3e3, augment, combo}
```

## 当前 recipe（M4 定型，2026-07-05）

**base 组件全关** + Flat MLP + negamax MC target + sims=100 + D4 增广**关**
（[`recipe.rs::board_stack`](../../../src/rl/algo/my_zero/recipe.rs)）。
裁决依据 = 下方 M3 九臂消融：无单臂过 promote 线；唯一弱阳性 D4 增广留复核入口
（[naive0 战术墙 issue](../../../.issue/items/gomoku_naive0_tactical_wall.md) §四）。

## M2 预注册正裁（2026-07-04 · 当前官方基线）

**口径**：release + MKL · 400 局满预算 · sims=100 · 半程快照(ep200) · 终局 40 局双闸门（随机开局 2 步 + 黑白镜像成对）· naive 梯队 20 局/档 · seeds 42/43/44 · 日志 `.bench/gomoku_m2_3seed_rerun_20260704.log`。

**双硬门槛**：vs random ≥ 0.95 且 vs 半程快照 gating ≥ 0.55，逐 seed 判。

| seed | vs random | vs 快照 | naive0 | env-steps | 达标 |
|------|-----------|---------|--------|-----------|------|
| 42 | 1.000 | 0.975 | 0.15 | 17,996 | ✅ |
| 43 | 0.950 | 0.925 | 0.00 | 19,129 | ✅ |
| 44 | 1.000 | 0.950 | 0.10 | 18,290 | ✅ |

**裁决：3/3 达标（中位 vs random 1.000 / vs 快照 0.950）——棋盘支柱立柱。**

## M3 组件消融总账（2026-07-04/05 · 均 3-seed 同 M2 载体 · 中位口径）

主指标 = naive0 胜率（base 在 random 已饱和，naive 档是唯一有上升空间的尺子）；vs random / vs 快照为回归护栏。日志 `.bench/gomoku_m3_*_2026070{4,5}.log`。

| 臂 | vs random | vs 快照 | naive0 | 裁决 |
|---|---|---|---|---|
| base（对照） | 1.000 | 0.950 | 0.00–0.15 | — |
| Gumbel+completedQ (s100) | 1.000 | 0.875 | 0.05–0.10 | 中性（无 CartPole 式灾难；n>\|A\| regime） |
| base / Gumbel (s16) | 0.925 / 0.975 | 0.925 / 0.825 | ≈0 | 中性（\|A\|≫sims native 复裁；负结果 issue 关闭条件达成） |
| consistency (coef 2) | 0.950 | 0.975 | 0.00–0.10 | 中性（MLP base 初裁） |
| reconstruction (coef 16) | 0.925 | 0.550 | 0.00–0.10 | 中性偏害（coef 为 CartPole 域标定，未重标） |
| CNN 表征（stride-1 塔） | 0.950 | 0.750 | 0.00–0.10 | 中性——表征假设被削弱 |
| 预算 ×5（2000 局） | 0.925 | 0.500 | 0.10–0.15 | 中性；后半程平台化，证伪「纯交互不足」 |
| replay ×8 | 0.875 | 0.825 | 0.00–0.05 | 中性偏害（小 buffer 过磨） |
| lr 3e-3 | 1.000 | 0.725 | 0.00–0.05 | 中性 |
| **D4 增广** | 0.975 | 0.975 | **0.00–0.25（中位 0.15）** | **弱阳性**（唯一正信号，未达 promote 线） |
| 组合（增广+RR32+lr3e3） | 1.000 | 0.675 | 0.05–0.25 | 不超增广单臂，「配方合力」未兑现 |

## naive0 战术墙裁决臂（M4 后 · [issue §四](../../../.issue/items/gomoku_naive0_tactical_wall.md) 前台主线）

| 臂 | vs random | vs 快照 | naive0 | 裁决 |
|---|---|---|---|---|
| ① sims 100→400（2026-07-05） | 0.950 | 0.900 | **0.05/0.15/0.05（中位 0.05）** | **平——搜索预算嫌疑排除**（预注册线 ≥0.3 未达），根因收敛「规则学习税」假设 → 进 ② 树内真规则诊断臂。日志 `.bench/gomoku_naive0_sims400_20260705.log` |
| ② 树内真规则（2026-07-05，**5-seed** 42–46；首跑后按哨兵复裁先例扩 seed，协议修订记 bench 注释） | 0.950 | 0.450 | **0.10/0.00/0.45/0.20/0.20（中位 0.20，max 0.45 + naive1 2/5 非零）** | **方向性正信号、未坐实**（未达 ≥0.3 部分坐实线，但非 ≈0.1 排除形态：中位翻倍、上限 3×）——判读 = 真规则只解锁 acting 侧，训练 value/policy 信号仍是瓶颈；战略分叉不触发，③ replay32×ROSMO 臂顺位。gating 走低为「双方同用真规则树抹平前后期差距」的测量形态。开关 `true_rules_tree`（默认关，learned 路径逐 bit 不变）。日志 `.bench/gomoku_naive0_true_rules{,_ext}_20260705.log` |
| ③ replay×8 × ROSMO 刷新（2026-07-05） | **0.425/0.625/0.475** | 0.475 | **全 0** | **有害（护栏崩塌）**——vs random 从 replay32 臂的 0.875 崩到 ~0.5（≈随机水平）。判读 = 弱模型下一步 look-ahead 的 adv 是噪声，`prior×exp(adv)` 现算 target 形成自指反馈（网络向自己的噪声改进分布学习、失去 MCTS 搜索改进信号）；**「policy target 过期」嫌疑同时被反证**（存量 MCTS target 比现刷 target 好得多）。ROSMO 棋盘格 ❌。开关 `rosmo_refresh`（默认关）。日志 `.bench/gomoku_naive0_rr32_rosmo_20260705.log` |

### 结论（M3/M4 收口）

1. **recipe = base 定型**：九臂无一显著抬升主指标，「变慢 ≠ 失败」但「不动 = 不 promote」。
2. **naive0 战术墙为结构性遗留**：头号假设 = MuZero 规则学习税（参照 AlphaZero 实现树内真规则、零规则学习负担）；后续裁决入口（sims 400 臂 / 树内真规则诊断臂 / 增广大预算复核）见 [issue](../../../.issue/items/gomoku_naive0_tactical_wall.md)，不阻塞支柱。
3. **Gumbel / completedQ 负结果 issue 终局归档**：|A|≫sims（s16）复裁中性、无灾难，CartPole 灾难确系 n≫|A| regime + 前置双 bug（greedy 去噪 + q_range）所致；棋盘域两组件按中性处置（不 promote、代码保留）。

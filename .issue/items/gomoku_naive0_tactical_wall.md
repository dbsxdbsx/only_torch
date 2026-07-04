---
status: active
created: 2026-07-05
updated: 2026-07-05
owners: []
reviewers: []
---

# Gomoku · naive0 战术墙：9 臂消融全平/弱阳，疑似 MuZero 规则学习税

> **状态**：active——M3 消融批次收口档案。棋盘支柱本身已立（M2 双门槛 3/3），
> 本 issue 记录的是"更上一层"的战术能力瓶颈，不阻塞 M4 工程收口。
> **关联**：收口规划 §3（Gomoku 主线）· `src/rl/algo/my_zero/board.rs` ·
> bench 载体 `tests/gomoku_m3_bench.rs` · 日志 `.bench/gomoku_m3_*_2026070{4,5}.log`

## 一、现象

M2 base（9×9 · Flat MLP · 组件全关 · sims=100 · 400 局）吊打 random（≥0.95）、
被 naive0 吊打（胜率 ≈0.00–0.15）。naive0 仅有"一步胜必走、对手一步胜必挡"
两条规则——瓶颈是最基础的战术视野。

## 二、M3 消融总账（2026-07-04/05，均 3-seed 同载体，中位口径）

| 臂 | vs random | vs 快照 | naive0 | 裁决 |
|---|---|---|---|---|
| base | 1.000 | 0.950 | 0.00–0.15 | 对照 |
| Gumbel+completedQ (s100) | 1.000 | 0.875 | 0.05–0.10 | 中性（无 CartPole 式灾难，前置双修复生效） |
| base/Gumbel (s16) | 0.925/0.975 | 0.925/0.825 | ≈0 | 中性；少 sim acting 可用降档 |
| consistency (2.0) | 0.950 | 0.975 | 0.00–0.10 | 中性（MLP base 初裁） |
| reconstruction (16.0) | 0.925 | 0.550 | 0.00–0.10 | 中性偏害（coef 为 CartPole 域标定，未重标） |
| CNN 表征（stride-1 塔） | 0.950 | 0.750 | 0.00–0.10 | 中性——表征假设被削弱 |
| 预算 ×5（2000 局） | 0.925 | 0.500 | 0.10–0.15 | 中性；后半程 gating≈0.5 = 平台化，证伪"纯交互不足" |
| replay ×8 | 0.875 | 0.825 | 0.00–0.05 | 中性偏害（小 buffer 过磨） |
| lr 3e-3 | 1.000 | 0.725 | 0.00–0.05 | 中性 |
| **D4 增广** | 0.975 | 0.975 | **0.00–0.25（中位 0.15）** | **弱阳性**（唯一正信号，未达 promote 线） |
| 组合（增广+RR32+lr3e3） | 1.000 | 0.675 | 0.05–0.25 | 不超增广单臂，"配方合力"未兑现 |

## 三、根因假设（按当前置信排序）

1. **MuZero 规则学习税**（头号）：参照实现 junxiaosong/AlphaZero_Gomoku（8×8/5
   连 2000–3000 局出好模型）是 **AlphaZero**——树内真规则，网络零规则学习负担。
   我们树内全程 learned dynamics，"必挡"要求 dynamics 先学会"不挡即负"的想象，
   1.8 万 env-steps 远不够。MuZero 论文棋类训练量亦远大于 AlphaZero 同级。
2. **搜索预算差距**：参照 400 playouts vs 我们 100 sims（未消融，便宜可测）。
3. 温度/探索调度细节差异（参照 KL 自适应 lr + 温度常 1.0）。

## 四、后续裁决入口（不阻塞 M4，择机执行）

- [ ] **sims 100→400 臂**（一行改动；搜索质量嫌疑的直接裁决）
- [ ] **树内真规则诊断臂**（AlphaZero 式：树内调 board snapshot/restore 真推演——
      基建已有 `GymEnv::snapshot/restore`；若真规则版快速翻越 naive0，假设 1 坐实，
      并为「acting 期真规则 / 训练期 dynamics」的混合架构提供依据）
- [ ] 增广弱阳性复核：更大预算（2000 局）× 增广，若显著则 promote 进棋盘 recipe
- [ ] recon 棋盘 coef 重标（{1,4} pilot）——现 16 为 CartPole 域值，偏害嫌疑

## 五、复现

```bash
cargo test --release --features blas-mkl gomoku_m3_<arm> -- --ignored --nocapture --test-threads=1
# arm ∈ {gumbel, base_s16, gumbel_s16, cons, recon, cnn, budget, replay32, lr3e3, augment, combo}
```

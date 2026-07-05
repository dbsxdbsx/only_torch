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

## 三、根因假设（2026-07-05 ①②③ 臂裁决后修订）

1. **训练信号瓶颈**（②臂后上位；③臂后子嫌疑收窄）：真规则树解锁 acting 上限但复现
   不稳（§四-②）→「必挡」的知识主要缺在 **value/policy 训练信号**里。子嫌疑修订：
   ~~过期 policy target~~ **已反证**（③臂：现刷 target 反而崩盘）；剩余 =
   **negamax MC 高方差 value**（每局仅终局 ±1 信号，无中间 credit assignment）与
   **探索覆盖不足**（战术关键局面在 self-play 分布中稀有）。
2. **MuZero 规则学习税**（方向性支持、未坐实）：真规则 vs learned dynamics 的差
   = naive0 中位 0.10 → 0.20 + 上限 0.15 → 0.45——有贡献但非全部。
3. ~~搜索预算差距~~ **已排除**（①臂：sims×4 全平）。
4. ~~policy target 过期~~ **已反证**（③臂：ROSMO 现刷有害、存量 MCTS target 更好）。
5. 温度/探索调度细节差异（参照 KL 自适应 lr + 温度常 1.0）。

> **实现正确性外部背书（2026-07-05）**：GPT-5.5 静态审查确认双人 negamax 主链路
> （backup/select/completedQ/n-step/D4 增广/真规则树/ROSMO 翻转）无符号级 bug——
> 墙是算法/信号层现象，非实现错误。

## 四、后续裁决入口（不阻塞 M4；2026-07-05 与用户定稿为**前台主线**，按序执行）

- [x] **① sims 100→400 臂** ✅ 2026-07-05 已裁决：**平**——naive0 = 0.05/0.15/0.05
      （中位 0.05，预注册线 ≥0.3 远未达；vs random 0.950 / vs 快照中位 0.900 护栏正常）。
      **假设 2（搜索预算差距）排除**：4× 模拟预算对战术视野零改善，墙不在搜索侧。
      日志 `.bench/gomoku_naive0_sims400_20260705.log`，数字入[账本](../../examples/my_zero/gomoku/README.md)
- [x] **② 树内真规则诊断臂** ✅ 2026-07-05 已裁决：**方向性正信号、未坐实**——
      5-seed（42–46，首跑 3-seed 后按哨兵复裁先例扩展）naive0 = 0.10/0.00/0.45/0.20/0.20
      （中位 **0.20** vs base 0.10；max 0.45 = 全批次历史最强单点；naive1 2/5 非零），
      未达预注册 ≥0.3 部分坐实线、亦非 ≈0.1 排除形态。判读：真规则树解锁 acting 侧
      上限（「必挡」树内可见），但训练侧 value/policy 信号撑不住稳定复现 → 瓶颈主体
      在**训练信号**而非树内转移；战略分叉（松开万金油铁律）**不触发**、留观察。
      实现为可插拔 `true_rules_tree` 开关（默认关；纯 Rust `RulesBoard` 规则层 +
      Python 板逐步对照金测试锁死等价性）。日志 `.bench/gomoku_naive0_true_rules{,_ext}_20260705.log`
- [x] **③ replay32 × ROSMO 刷新臂** ✅ 2026-07-05 已裁决：**有害（护栏崩塌）**——
      vs random 0.425/0.625/0.475（replay32 对照 0.875、base 1.000）、naive0 全 0。
      判读：弱模型下一步 look-ahead 的 adv ≈ 噪声，`prior×exp(adv)` 现算 target 自指
      反馈污染 policy 学习；同时**反证「policy target 过期」嫌疑**（存量 MCTS target
      显著优于现刷 target）。ROSMO 棋盘格 ❌（开关 `rosmo_refresh` 留库默认关，
      实现含 negamax q 翻转 + 合法掩码 + top-16 剪枝，GPT-5.5 审查确认语义正确——
      负结果是机制不适配而非实现 bug）。日志 `.bench/gomoku_naive0_rr32_rosmo_20260705.log`
- [ ] 增广弱阳性复核：更大预算（2000 局）× 增广，若显著则 promote 进棋盘 recipe
- [ ] recon 棋盘 coef 重标（{1,4} pilot）——现 16 为 CartPole 域值，偏害嫌疑

## 五、复现

```bash
cargo test --release --features blas-mkl gomoku_m3_<arm> -- --ignored --nocapture --test-threads=1
# arm ∈ {gumbel, base_s16, gumbel_s16, cons, recon, cnn, budget, replay32, lr3e3, augment, combo}
```

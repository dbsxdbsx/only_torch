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
| ④ batch 16→256（2026-07-05） | 0.925 | 0.700 | **0.00/0.00/0.15（中位 0.00）** | **平——梯度噪声嫌疑排除**（每步梯度噪声 ÷16 对战术视野零改善；护栏在改判线内，不触发 lr 失配条款）。判读 = 方差瓶颈不在梯度聚合层，在 target 信息分辨率（整局共享 ±1）；gating 走低 ~0.2 = 大 batch 同 lr 下 SGD 噪声正则减弱的已知形态。嫌疑收敛探索覆盖。日志 `.bench/gomoku_naive0_batch256_20260705.log` |
| ⑤ G1 组合臂：真规则×2000 局×增广×温度全程 1.0×buffer 300（2026-07-05，诊断性破例） | **1.000（3/3）** | 0.925 | **0.00/0.20/0.45（中位 0.20）**，naive1–3 全 0 | **未翻墙——免费配方叠加零增益**（中位与 ②臂真规则单臂持平；护栏史上最佳）。判读 = 温度/预算/增广不是缺失项；seed 方差三度复现（同配方 0.00 vs 0.45）坐实「战术局面 = self-play 抽签」→ 定向战术开局课程对症首选，G2 训练强度包（KL lr×batch512×epoch 对齐，40× 每局梯度功差距）次之。墙钟 ~3–4 min/seed。日志 `.bench/gomoku_naive0_combo_ceiling_20260705.log` |
| ⑥ 定向战术开局课程：p=0.25 ×②臂底座（true_rules、400 局）（2026-07-05） | 1.000（3/3） | 0.600 | **0.60/0.40/0.35（中位 0.40）** | **强正信号——全批次首个推动中位的臂**（②臂底座 2×，seed 42 破 0.5 翻墙线；护栏全正常）。判读 =「战术局面 = self-play 抽签」病理对症起效：`tactical_opening`（一步胜威胁构造式生成，前缀 replay 不入训练记录）把必挡局面确定性注入训练分布。机制可插拔（p=0.0 默认路径逐 bit 不变），Leela 开局库/KataGo forced openings 同族。→ ⑦ 终局配方复核。日志 `.bench/gomoku_naive0_tactical_openings_20260705.log` |
| **⑦ 终局配方复核：G1 组合底座 × 课程 p=0.25（2026-07-05）** | **1.000（3/3）** | 0.550 | **0.80/0.20/0.70（中位 0.70）+ naive1 首次非零 0.30 + naive2 首次非零 0.10（均 seed 44）** | **✅ 翻墙坐实**——naive0 中位 0.10（base）→ 0.70，「真规则树 × 战术开局课程」组合击破战术墙；gating 走低 = ②臂同款测量形态（双方同真规则树）。seed 43（0.20）示方差仍在，中位裁决稳。→ 课程 promote 讨论 + G3 四档验收开启（naive1–3 需课程扩展：「活三/进攻」课题）。日志 `.bench/gomoku_naive0_final_recipe_20260705.log` |
| ⑧/⑧b 活三课题混入：50/50 混配、增量混配（2026-07-05） | 1.000 | 0.575/0.550 | 中位 **0.30**（50/50）/ **0.45**（增量） | **负——弃活三课题**：50/50 稀释必挡剂量（25%→12.5%）命中「攻挤防」；增量混配保持必挡 25% 仍拉低 ⑦ 的 0.70 → 活三是**主动干扰**非稀释（嫌疑 = 活三局结局噪声大/攻防语义混淆 value）。课程维持纯必挡。日志 `.bench/gomoku_naive0_open_three{,_additive}_20260705.log` |
| ⑨ G2 训练强度包：batch512 × trains5 × KL 自适应 lr（2026-07-05） | 1.000（3/3） | 0.650 | 中位 **0.50**，naive1 中位 0.05 | **无增益——训练强度非当前瓶颈**（40× 每局梯度功无梯队抬升，墙钟 ×3）。KL-lr 机制无害验证通过（护栏全绿、自动配平 batch 512），`kl_adaptive_lr` 开关留库。日志 `.bench/gomoku_naive0_g2_intensity_20260705.log` |
| **⑩ G3 四档验收：⑦ 配方 × 40 局/档（2026-07-05）** | **1.000（3/3）** | 0.550 | **naive0 = 0.62/0.17/0.77（中位 0.62）· naive1 中位 0.05 · naive2/3 ≈ 0** | **未达标**（验收线：四档 ≥0.75 全达 / 部分达标 = naive0 ≥0.75 且 naive1 ≥0.5）。缺口归因：① seed 方差（0.17–0.77，五度复现）是达标第一障碍；② naive1+ 缺口已排除课程配比（⑧）与训练强度（⑨），剩余嫌疑 = **网络容量**（CNN 在课程底座未测）与**预算量级**。日志 `.bench/gomoku_naive0_g3_acceptance_20260705.log` |
| ⑪ CNN × ⑦ 配方（2026-07-05） | 1.000（3/3） | 0.350 | naive0 中位 **0.55** · naive1 中位 **0.15**（3×）· naive2 单点 **0.35**（史上最高） | **未达 promote 线，容量假设按线排除**（弱阳纹理留档：naive1/2 有抬升方向、需配预算放大再裁）；seed 方差六度复现（0.20–0.75，掉队者换 seed 44）坐实「seed 抽签 > 组件效应」。墙钟 ~3.5×。缺口终局收敛 = **预算量级 ×（可选 CNN）+ seed 方差治理**。日志 `.bench/gomoku_naive0_cnn_recipe_20260705.log` |
| **⑫ 纯 self-play 上限标定：真规则 × CNN × 5000 局 × 无课程（2026-07-05 晚，「纯 self-play 万金油」哲学修正首臂）** | **1.000（3/3）** | 0.900 | **naive0 = 0.90/0.05/1.00（中位 0.90，seed 44 满分史上首个）· naive1 = 0.10/0.00/0.40（无课程首次非零，0.40 史上最高）· naive2/3 = 0** | **✅ 纯 self-play 可行坐实（≥0.75 预注册线）——课程非必需**：⑦ 课程配方（0.70）被无课程配置超越，课程实为「预算/容量不足下的拐杖」——预算 2.5× + CNN 后自然 self-play 分布自行覆盖战术局面。seed 43 崩 0.05 = 方差第七度复现（0.05–1.00 史上最宽散布，中位裁决稳）。判读用绝对线（conv2d 优化后 CNN 数值流漂移，不与 ⑪ 精细比差）。→ ⑬ learned dynamics 同配置（规则学习税定量账）。墙钟 38–55 min/seed。日志 `.bench/gomoku_pure_selfplay_truerules_20260705.log` |
| **⑬ learned dynamics 同配置：⑫ 唯一差树内转移（2026-07-07）** | 0.925（0.900/0.925/0.925） | 0.387 | **naive0 = 0.10/0.15/0.10（中位 0.10 vs ⑫ 的 0.90）· naive1 ≈ 0 · naive2/3 = 0** | **❌ 档位级落后——规则学习税坐实 0.8 档**（命中预注册「⑫ ≥0.75 而 ⑬ <0.5」条款）。gating 全 <0.5（终局赢不了半程快照）+ vs random 掉出满分。**关键读数：learned dynamics 对预算/容量钝感**——M3 base 400 局 0.10 → 预算×5 0.10–0.15 → 本臂 12.5× 预算 + CNN + 增广仍 0.10；墙不在数据量在 dynamics 学不会「五连终局」，树在幻觉模型里搜索。附带：墙钟 822–1404s/seed 反比 ⑫ 快 2–3×（learned 树仅根过 CNN）。→ ⑭ PER 改在 ⑬ 底座测；真规则 vs 万金油战略取舍进入有定量账阶段。日志 `.bench/gomoku_pure_selfplay_learned_20260705.log` |

### 结论（M3/M4 收口）

1. **recipe = base 定型**：九臂无一显著抬升主指标，「变慢 ≠ 失败」但「不动 = 不 promote」。
2. **naive0 战术墙为结构性遗留**：头号假设 = MuZero 规则学习税（参照 AlphaZero 实现树内真规则、零规则学习负担）；后续裁决入口（sims 400 臂 / 树内真规则诊断臂 / 增广大预算复核）见 [issue](../../../.issue/items/gomoku_naive0_tactical_wall.md)，不阻塞支柱。
3. **Gumbel / completedQ 负结果 issue 终局归档**：|A|≫sims（s16）复裁中性、无灾难，CartPole 灾难确系 n≫|A| regime + 前置双 bug（greedy 去噪 + q_range）所致；棋盘域两组件按中性处置（不 promote、代码保留）。

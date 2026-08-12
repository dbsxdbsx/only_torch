# MyZero · Gomoku 9×9（棋盘账本）

> [← 返回 MyZero 总览](../README.md)
>
> **本文件是棋盘（self-play）benchmark 数字的唯一账本（owner）**：状态总览 / issue / AGENTS 一律链到这里取数字。每行实测带口径（profile / BLAS / seeds / 日期）。战略见 [RL 状态总览](../../../.doc/design/rl_myzero_status.md)。
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

## 命名词典

- **Phase**：RL 路线图的大阶段；Gomoku 属于 Phase 2。
- **M0–M4**：棋盘支柱建设里程碑——M0 性能风险 spike、M1 最小训练闭环、
  M2 预注册基线立柱、M3 组件消融、M4 recipe/smoke/账本工程收口。
- **G1–G3**：M4 后战术墙攻坚关卡——G1 上限组合、G2 训练强度包、G3 赢家
  配方四档终局验收；G3 不是第三代网络架构。
- **①②③…**：单个实验臂的时间顺序；**seed** 是同一实验的独立重复。

历史原因：`gomoku_m3_bench.rs` 后来继续承载了 M4 后的裁决臂；文件名只表示
它由 M3 手动 benchmark 载体演化而来，不代表其中所有实验仍属于 M3。

## 当前 recipe（2026-07-05 定型）

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
| ⑭ PER 优先回放：⑬ 底座 × `per=true` 唯一新变量（2026-07-07，seed 并行跑法首战） | **1.000（3/3，⑬ 的 0.900–0.925 → 满分恢复）** | 0.350 | **naive0 = 0.15/0.15/0.05（中位 0.15 vs ⑬ 的 0.10）· naive1 中位 0.05 · naive2 单点 0.10** | **带内持平——按预注册排除：分布覆盖瓶颈不在消费端**（抬升 0.05 <0.15 正信号线；⑬ 对照 5-seed 重锚 0.15 后 = 零抬升，裁决加固）。判读 = PER 把 \|ν−z\| 最大的终局矛盾局面顶到队列前，但 dynamics 此预算/容量下连被集中喂也学不会终局规则；课程 ⑥⑦ 有效因改**生成**分布（注入新数据），PER 只重排**已有**数据——「课程的零领域知识版」棋盘域不成立。附带：护栏满分 + 零发散 = PER 无害，`PerPriorities` 留库默认关；剩余杠杆收敛 = 监督端（recon 重标 / consistency 家族）× 预算量级。墙钟 1490–1521s/seed（3 进程并行，臂总 ~25 min）。日志 `.bench/gomoku_pure_selfplay_per_20260707.log` |
| ⑬-ext 对照 seed 扩展 45/46（2026-07-07，⑮ 协议修订②配对对照件） | 1.000（2/2） | 0.600/0.800（**双过 0.55 线**） | **naive0 = 0.35/0.35 · naive1 = 0.30/0.05** | **⑬ 对照 5-seed 重锚：naive0 中位 0.10 → 0.15（散布 0.10–0.35，seed 方差第八度复现）**——⑮⑯ 正信号线随锚上调 ≥0.30。连带修正：⑬「gating 全 <0.5」在扩展 seed 不成立、「预算钝感」修正为「中位钝感、上限有方差」；⑭ 对新锚零抬升加固排除。墙钟 2205/2286s（12 进程竞争膨胀 ~1.6–2.8×）。载体 `gomoku_pure_selfplay_ctrlext`，日志 `.bench/gomoku_pure_selfplay_ctrlext_20260707.log` |
| **⑮ recon 棋盘重标：⑬ 底座 × `reconstruction` coef {1,4}（2026-07-07，监督端首臂，5-seed 发现集）** | coef=1：**0.975–1.000 全绿**；coef=4：最低 0.900 压线 | coef=1：0.475（中位）；coef=4：0.075–0.800 散布极宽 | **coef=1：naive0 = 0.40/0.35/0.30/0.20/0.40（中位 0.35，配对 4/5 抬升）· naive1 中位 0.10**；coef=4：naive0 中位 0.20 | **coef=1 发现集正信号（0.35 ≥0.30），但 promote 资格后来被⑱未见 seed 终审否决**；coef=4 带内排除。历史价值 = 证明直接监督存在可爬坡纹理并给出候选剂量，不等于可外推结论。墙钟 2364–2509s/seed（12 进程档）。载体 `gomoku_pure_selfplay_recon{1,4}`，日志 `.bench/gomoku_pure_selfplay_recon{1,4}_20260707.log` |
| ⑯ consistency：⑬ 底座 × `consistency`（coef 2.0，2026-07-07，监督端次臂，5-seed） | 0.975–1.000 全绿 | 0.325–0.700（中位 0.550） | **naive0 = 0.20/0.25/0.40/0.45/0.25（中位 0.25，配对 4/5 抬升）· naive1 中位 0.00** | **带内未越线（弱阳纹理留档）——按预注册不 promote**（0.25 <0.30 重锚线）。判读 = 同底座同 seed 集直接对照 recon coef=1（0.35 越线）：**「冻结真值 > 自指目标」信息量排序实测坐实**——SimSiam target 是网络自己正在学的 `repr(next_obs)`，上限被 h 质量卡住；方向与家族一致（4/5 弱阳）但强度不过线。M3 旧「中性」初裁修正为「CNN 底座带内弱阳」；EfficientZero 图像域主证据不受否定，图像线复测入口保留。墙钟 1549–1604s/seed（5 进程档）。载体 `gomoku_pure_selfplay_cons`，日志 `.bench/gomoku_pure_selfplay_cons_20260707.log` |
| ⑰ recon1 × PER：⑮ 赢家剂量复叠 ⑭ 无害件（2026-07-07，5-seed） | **1.000（5/5）** | 0.375–0.900（中位 0.500） | **naive0 = 0.15/0.15/0.20/0.15/0.15（中位 0.15；对 recon1 单药 5/5 回落）· naive1 中位 0.10** | **无复叠增益——PER 消费端排除维持，赢家锁定 recon1 单药**（0.15 远低于 ≥0.50 线，也低于单药 0.35）。recon 修复监督供给后，旧模型 `|ν−z|` 优先级仍未兑现；消费加权反而破坏 recon 对全棋盘转移的均匀稠密监督。不调 α/IS/刷新，避免元过拟合。墙钟含机器休眠，不作性能数据。载体 `gomoku_pure_selfplay_reconper`，日志 `.bench/gomoku_pure_selfplay_reconper_20260707.log` |
| **⑱ G3 新-seed 配对终审 + selected holdout（2026-07-10）** | primary：recon1 0.950–1.000 / control 0.975–1.000；holdout：1.000 | primary 中位：recon1 **0.650** / control 0.475；holdout：0.875 | primary control naive0 中位 **0.28**；recon1 四档中位 **0.20/0.03/0.00/0.00**。预注册选 seed48 后，独立 100 局/档 holdout = **0.19/0.09/0.07/0.00** | **❌ 科学与交付双失败**：recon1 未见 seed naive0 中位差 −0.08（要求 ≥+0.15），配对仅 1/5 不劣 → 不 promote；G3 未达部分线。selected holdout 无一档达到 0.675，7/100 的 naive2 纹理不代表已学会。棋盘 recipe 维持 base；1.9 MiB `.otm` 仅作失败候选复现（完整契约、真实加载冒烟通过，不进 Git/release）。日志 `.bench/gomoku_g3_{control,recon1,holdout}_20260710.log` |

### 结论（M3/M4 收口）

1. **recipe = base 定型**：九臂无一显著抬升主指标，「变慢 ≠ 失败」但「不动 = 不 promote」。
2. **naive0 战术墙为结构性遗留**：头号假设 = MuZero 规则学习税（参照 AlphaZero 实现树内真规则、零规则学习负担）；后续裁决入口（sims 400 臂 / 树内真规则诊断臂 / 增广大预算复核）见 [issue](../../../.issue/items/gomoku_naive0_tactical_wall.md)，不阻塞支柱。
3. **Gumbel / completedQ 负结果 issue 终局归档**：|A|≫sims（s16）复裁中性、无灾难，CartPole 灾难确系 n≫|A| regime + 前置双 bug（greedy 去噪 + q_range）所致；棋盘域两组件按中性处置（不 promote、代码保留）。
4. **监督端战役 ⑮–⑱ 收口**：recon1 发现集正信号未在未见 seed 复现，consistency 仅带内弱阳、PER 单药/复叠均无增益；G3 selected holdout 亦未通过。当前预算/容量下无通用组件赢家，下一步进入预算 / 真规则 / 转图像线的战略复盘点。

## Phase 3A0 · 主动数据误差 proxy 审计（2026-07-12）

**问题**：在实现 ErrorQ / Collector 前，先验证真实 transition 上的 model-error proxy
是否同时满足「任务相关」与「新增数据可稳定降低」。

**协议**：

- 所有 audit episode 从真实空盘 `reset()` 开始；behavior lane 复用训练口径
  `temperature=1 + root Dirichlet`，diagnostic lane 才使用 greedy/no-noise。
- true-rules 只生成 reference-policy 与一步胜威胁局面标签，不生成训练 target。
- seeds 52–54；每个 fixed block 约 800 条多样化 transition。首轮 greedy audit
  因 20 局完全重复而整体作废；新增最小守门测试，锁死两个不同 seed lane 的单局
  action 序列不得完全相同。
- 可学习性做两层检查：同一 early future block 在 400/2000/5000 局 checkpoint
  重评分；以及 2000 局 checkpoint 新增 30 局真实 game、按原 MuZero loss 更新
  500 次，再评估剔除 bitwise-exact 重复 state-action 的未来 20 局 holdout。
  本轮未做 D4 等价状态 canonical 去重，列为结论边界。

| seed | fixed raw reward CE* | fixed continuation | 干预后 raw reward CE* | 干预后 continuation |
|---|---:|---:|---:|---:|
| 52 | −3.2% | +1.3% | −6.15% | −0.51% |
| 53 | +3.7% | +2.2% | +12.57% | +11.64% |
| 54 | −5.8% | −1.5% | +6.03% | +7.11% |

\* 首轮实验记录的是 two-hot reward target 的 raw cross-entropy；它含类别相关的
target entropy 下界，不能据此声称 reward 模型误差与战术相关。保留该列仅用于同一
fixed block / holdout 的 before-after 方向；代码终态已改为
`KL(target || prediction) = CE - H(target)`，后续复用不会再混入该下界。

**任务相关性成立的无混淆证据**：late checkpoint 上 continuation Brier 对
reference-policy JSD 的 Spearman 为 `0.54 / 0.52 / 0.69`，tactical-position
top-decile lift 为 `2.23 / 2.29 / 2.40`。

**❌ 最终裁决**：当前协议未通过“新增数据后稳定降低”门槛。fixed block 的
continuation 只有 seed54 小幅下降；新增 30 局真实 game 干预后，也仅 seed52 小幅改善，
seed53/54 分别回归约 11.6% / 7.1%。因此没有证据支持当前 proxy 在现有
MuZero loss、容量、500-update 干预下可跨 seed 稳定降低，主动追逐它存在放大干扰的
风险。该结论不外推为“proxy 本质不可学习”：实验仅 3 seeds、局内 transition 相关，
且干预重新初始化 Adam、没有 matched old-replay control。按预注册仍应停止 ErrorQ、
Collector、H=K 与 5+5 seed 战役，不调 proxy 权重、KL 或 SAC/WGAN 补救。代码只保留
test-only 3A0 诊断与可复现审计载体。

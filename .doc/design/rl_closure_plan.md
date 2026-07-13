# RL 全面收口规划（v0.26 → v0.28）

> **定位**：把「所有 RL 基准 + MyZero 万金油算法」收口到可验收终态的**战役次序文档**——回答"先做什么、后做什么、为什么这个顺序、做到什么算过关"。
> 当前状态与验收协议见 [rl_roadmap.md](./rl_roadmap.md)；战术细节各自链出（Simulus 消融见 [my_zero_simulus_ablation_plan.md](./my_zero_simulus_ablation_plan.md)）；实测数字按环境分账：[CartPole](../../examples/my_zero/cartpole/README.md) / [Gomoku](../../examples/my_zero/gomoku/README.md)。
> **创建**：2026-07-02 · 贯穿纪律沿用 [roadmap §3.2](./rl_roadmap.md#32-改动纪律搬运--改进沿用)：一次一项、预注册协议、3-seed 消融、数字只进账本；性能优化 measure-first、挂实测需求（内核级优化路线已盖棺，见 [optimization_candidates](../performance/optimization_candidates.md)）。

---

## 0. 终态验收（"收口"的操作定义，四条全绿才算完成）

1. **组件×环境裁决矩阵零 ⏳**：[组件矩阵](../../examples/my_zero/README.md#内部组件进展团队--promote-时改-recipers)每格 ✅/❌/⏸ 有据，❌ 有 issue、✅ 有账本数字。
2. **四类环境支柱各有 promoted recipe + 3-seed 账本数字**：低维离散（CartPole，已有）· 图像离散（待建）· 棋盘 self-play（**✅ Gomoku 已立**，2026-07-05，[账本](../../examples/my_zero/gomoku/README.md)）· 连续（Pendulum 终审后两出口，见 §5）。
3. **全部 RL open issue 归档**（映射见 §6）。
4. **发版关卡覆盖全部支柱**：`smoke-rl` 扩容纳入图像 + Gomoku（**✅ Gomoku 已入**，`smoke-my-zero-gomoku`），三层验收协议不变。

「万金油」的操作定义 = 第 2 条：`MyZero::new(env_id)` 四类环境开箱即用、recipe 有实测背书。

## 1. Phase 0 · 训练信号收口（v0.26 上半，进行中）

**为什么最先**：图像/self-play 全在这套 loss 栈上训练，汇率错一个量级会污染所有后续消融；且 CartPole 迭代最便宜。

- ✅ 系数消融 4+1 臂 + recon 5-seed 复裁（已收口：recon_coef 1→16 promote，哨兵中位 66.2k→~9.8k；recon=1 实测有害、悬案闭合——见账本与 CHANGELOG 2026-07-02 条目）
- ✅ 梯度流审计（2026-07-02 闭环：现状完全 canonical——cons target 已 sg、×0.5/×1/K 缩放齐备、recon 回流为设计本意；**sg 解耦两臂均不追加**——(b) 与 t1 数据经验冗余、(c) 反价值等价原则且前提不成立；流向图与裁决细节见 [Simulus 计划 A2 审计结论](./my_zero_simulus_ablation_plan.md#a2-审计结论2026-07-02--已闭环现状-canonicalbc-两臂均不追加)，复活触发留 Phase 1 干扰症状）
- ✅ **HL-Gauss 编码消融**（= Simulus 计划 A1；2026-07-02 闭环：**负结果**——中位 9.8k→27.6k 显著劣化，窄 support 低噪声下 two-hot 尖标签是信息优势；回退 two-hot、`hl_gauss` 开关留库，Phase 1 图像域 native 复测，见下节复裁条目）
- ✅ **obs 无量纲化（symlog）消融**（2026-07-02 闭环：**负结果**——三臂 symlog × recon {16,4,1} 中位 12.9k / 19.7k / 39.4k，「系数向 1 回移」完全未发生（仍 16 最优且单调恶化），symlog+16 亦无增益；由此裁决 **recon_coef=16 本质是自监督话语权旋钮而非单位换算**——CartPole obs 本就小量纲、无量纲可撤。`obs_symlog` 开关留库、默认恒关，触发条件回归 Simulus 计划 §3（连续特征范围失控）；图像线走 [0,1] 像素归一 + 该域重标系数）
- ✅ **退出判据达成（2026-07-02）**：v0.26 recipe 定稿 = **系数 recon16 + two-hot 编码 + raw obs + canonical 梯度流**（四者全部有据，两项负结果证明现状即最优）、账本回填完毕；**CartPole 自此冻结为纯回归哨兵**（条款二生效，哨兵 12,519 / 8,643 / 9,826 · 中位 ~9.8k）。

## 2. Phase 1 · 图像线立柱 + 一级风险压测（v0.26 下半）

**为什么第二**：唯一可能否定整条路线的风险要最早、用最便宜的实验裁决；同时这是 consistency/reconstruction 的 native 场景（CartPole 上的暧昧表现到此才见分晓）。

> **⚠️ 次序修订（2026-07-04）**：S0 ✅（spike GO，一级风险裁决已完成——本阶段排第二的首要理由已兑现）、S1 ✅；但 **S2 基准两跑皆 0/3 平直**（旧/新数值流各一次），**S3 三臂诊断全平**（recon pilot {1,4,16} / cons-off / HL-Gauss 均无差异）→ 组件层排除，嫌疑收敛到「训练预算差 2 个数量级 + 稀疏 reward」（[负结果 issue](../../.issue/items/my_zero_pong_image_flat_negative.md)）。
> 裁决：**图像线降级为后台预算标定**（replay ratio ↑ × ROSMO 刷新 / lr 扫描 / DIAG，吃机器时间不吃注意力），**Phase 2 Gomoku 提前为当前主线**（§3；棋盘域是本派系最强场、直接服务象棋战略目标、对图像线仅依赖已闭环的 S1 CNN）。图像支柱的退出判据不变、账本债保留，标定臂出证据后回头收口。
> reanalyze 阶梯一变更：**机制已进库**（`rosmo.rs`，2026-07-04；CartPole 回归闸门 3/3 绿、裁定旧灾难为机制病理而非实现 bug），图像域价值裁决随图像基线一并顺延。
>
> **⚠️ 预算再定价 + pilot-first 修订（2026-07-05，外证 [EfficientZero Remastered](https://www.gigglebit.net/blog/efficientzero)）**：
> 第三方复现实测 = A100 + 20–32 核 **1–2 天/模型**（论文口径 7 小时，分数亦低于论文）且训练管线是
> CPU 大户——据此重新定价：CPU-only 上把 Pong 类基线喂到参照量级（~12 万 updates @ batch256 vs
> 我们 S2 的 9.6k @ batch16）是**以周计**的承诺，原「后台预算标定」低估了量级。修订三条：
> ① **pilot-first**：预算标定臂改「轻量图像载体（MinAtar 类 / 更狠降采样）+ 单 seed + 判停协议
> （中间检查点 greedy 曲线仍平即停）」，先证明「管线能学」再谈 Atari 类支柱验收；
> ② 预算与门槛一律以**第三方复现数据**为锚（条款三，见 §7）；
> ③ S0 spike 的教训入档：pilot 必须覆盖「学得动吗」（训练预算），不只「跑得动吗」（推理 wall-clock）。
> 同文外证：EfficientZero 三组件中 **consistency 是核心**、value_prefix / off-policy correction
> 疑为 Atari-100k 特调——与 value_prefix ❌ CartPole 实测互证，图像 recipe consistency ON 维持。

- **风险 spike 先行**：最小 CNN 栈 + 假图像输入，实测「CNN 前向 × sims」单步 wall-clock，对照 [CPU 风险 issue](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) 的触发线，产出去/留/改道决策
- 图像 obs 管线（复用 `src/vision/preprocess`：降采样/灰度/帧堆叠）→ CNN representation 进 `network.rs`（按 recipe 注入）→ **预注册基准门槛**（Pong 类；验收 = 3-seed 可复现学习曲线 + 预注册分数，**非 SOTA**）
- native 域复裁：consistency / reconstruction / HL-Gauss 各一次 A/B；recon_coef=16 为 CartPole 域标定值，图像 obs 走 [0,1] 像素归一并**在该域重标**（Phase 0 symlog 负结果已裁决：该系数是权衡旋钮而非单位换算，跨量纲不免重调）
- **reanalyze 复活 · ROSMO-first 两级阶梯（2026-07-03 从 Phase 3 提前，图像基线立柱后串行）**：
  - **提前理由**：Phase 3 原排期依据是"等样本贵的验证场"——图像线立柱后该前提即满足；且 reanalyze 若有效，Phase 1 之后每轮图像实验都直接省采样成本（战略组件早验收早受益）。
  - **阶梯一（默认档）**：**ROSMO 式一步 target 刷新**（Xiao et al., ICLR 2023 · [arXiv:2210.05980](https://arxiv.org/abs/2210.05980)，开源 sail-sg/rosmo）——学习模型上一步 look-ahead 构造 advantage/target + 行为正则，不重跑整棵树。选它做第一臂的三重理由：① 旧 reanalyze 在 CartPole 的失败模式（弱网全树重搜写回投毒）与 ROSMO 诊断 MuZero Unplugged 失效的病理同源（旧数据上深搜复合模型误差）；② 重刷成本 ≈ 一次前向，正面绕开 CPU-only 一级风险（旧全树版每局 wall-clock ×10+）；③ 一步展开误差暴露面小，弱网期鲁棒。
  - **阶梯二（升级档）**：阶梯一实测有增益后，再单变量消融**全树 MCTS reanalyze**（在线 off-policy 设定下网络变强、数据变新后深搜红利可能回归——ROSMO 结论出自纯离线设定，不自动外推）。
  - **同批**：loss 优先回放（[Simulus 计划 B1](./my_zero_simulus_ablation_plan.md#b1--loss-优先回放随-reanalyze-复活挂-phase-1-图像线尾段)）。
  - **纪律**：必须先立**无 reanalyze** 的图像基线（预注册门槛），阶梯一/二各为单变量 A/B；CartPole 只做回归覆盖（条款二：只答"崩没崩"），价值裁决全在图像域。
- Simulus 暂缓项触发点：若环境奖励稀疏且探索成实测瓶颈，ensemble JSD 内在奖励在此转正（唯一入口）
- **退出判据**：图像支柱 3-seed 账本 + CPU 风险 issue 第一格勾选 + reanalyze 阶梯一裁决入账本（[reanalyze issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) 可归档或降级为全树版残留项）。

## 3. Phase 2 · self-play 线：Gomoku 踏脚石（v0.27 上半 → **2026-07-04 提前启动，当前主线**）

**为什么第三**：象棋战略目标的既定踏脚石（地基已备：`SelfPlayGame` / negamax backup / legal_mask / `python/gym_env/gomoku/`）；且 **Gumbel/completedQ 两个负结果 issue 的关闭条件就是在 |A|≫sims 场景复测**——Gomoku |A|=225 正是。

> **提前启动口径（2026-07-04，随 §2 修订）**：
> - **万金油铁律不破**：树内全程 learned dynamics（canonical MuZero 口径）；真环境仅用于
>   self-play/eval 走子、**根节点** legal_mask、终局判定三处。
> - **纪律**：最小 base recipe 先过预注册门槛（见下），之后**一次一臂**消融；结论只填
>   组件矩阵的**棋盘列**（域裁决不迁移，图像列的债不因此注销）。

> **✅ Phase 2 主体已收口（2026-07-05，M0–M4 全程闭环）**：
> M1 训练闭环 ✅ → M2 预注册双门槛 **3/3 达标**（vs random 中位 1.000 / gating 0.950）→
> M3 九臂消融全跑完（全中性/弱阳，无组件过 promote 线）→ M4 工程收口
> （recipe=base 进 `recipe.rs`、`smoke-rl` 纳入 `smoke-my-zero-gomoku`、
> [棋盘账本](../../examples/my_zero/gomoku/README.md) 落地、Gumbel/completedQ
> 负结果 issue 终局归档、三件套分层沉淀 [vision §5.4](./my_zero_algorithm_vision.md#54-三件套正交分层sampled--gumbel--completedq2026-07-05-定稿)）。
> 退出判据三项全兑现。遗留 = naive0 战术墙（结构性，[issue](../../.issue/items/gomoku_naive0_tactical_wall.md)，不阻塞支柱）；
> 15×15 / 象棋扩展随战略推进另立。
>
> 以下为原规划内容（留档）：

- **Gumbel 复裁前置修复（必做，否则复裁无效）**：
  - ① greedy eval 去噪 bug：`gumbel.rs::final_recommendation` 无视 `temperature=0` 仍用 Gumbel 噪声打分，eval 分数被污染（疑似"Gumbel 未收敛"负结果真因）——详见[负结果 issue §七](../../.issue/_archive/my_zero_gumbel_completedq_cartpole_negative.md)
  - ② `q_range` 局部归一化同源 bug（completedQ 已修 tree-level，Gumbel 侧未修，issue §六留档）
- Gomoku 训练闭环（棋盘 2D 表征复用 Phase 1 CNN）→ **预注册棋盘账本口径**（建议 vs random ≥95% + vs 旧 checkpoint ≥55%）→ Gumbel-root / completedQ 终局复裁（兼 P2 少 sim acting 复测）
- 条件项：self-play 对弈吞吐成实测瓶颈时，兑现 `predict_batch` 批量叶子评估接缝（挂需求、不主动做；缓解杠杆背景见 [CPU 风险 issue](../../.issue/items/cpu_only_mcts_image_realtime_risk.md)）
- 裁决后据实沉淀「三件套正交分层」文档至 [vision](./my_zero_algorithm_vision.md) §5.4（Sampled=候选层 / Gumbel=根层 / completedQ=目标层 + 融合契约——旧规划遗留文档债）
- **退出判据**：胜率门槛达成 + 两个负结果 issue 终局归档 + 棋盘 recipe 进 `recipe.rs`。

## 4. Phase 3 · 样本效率纵深（v0.27 下半）

**为什么第四**：target_net / SVE 同属 value target 质量域，一批清账最经济。
**范围变更（2026-07-03）**：reanalyze 复活 + loss 优先回放已提前至 Phase 1（ROSMO-first 阶梯，见 §2）；本阶段收窄为 value target 质量清账 + acting/reanalyze 解耦的工程落地。

> **⏸ 暂缓注记（2026-07-05，Phase 2 收口后裁决）**：Phase 2 主体已闭环，但**本阶段不顺位开启**——
> 三项内容当前均无裁决场：① target_net / SVE 治理的是 **bootstrap target 新鲜度**，需要
> 「value 靠 bootstrap 且基线在学习」的环境出裁决，而 CartPole 已冻结（条款二禁价值判断）、
> 图像基线 0/3 平直（无信号，S3 已演示消融臂全平不可判读）、棋盘 value = negamax MC 终局事实
> **不 bootstrap**（target_net/SVE 在棋盘域接近 ⏸）；② acting/reanalyze 解耦显式依赖
> 「Phase 1 裁决出的阶梯档位」，该裁决已随图像基线顺延。
> **重启触发条件** = 图像基线出信号（正负终审皆可）：若预算标定救活基线 → 按原范围开启；
> 若终审负向触发条款一改道（载体收缩棋盘单支柱）→ 本阶段范围重审，target_net/SVE
> 大概率直接裁决 ⏸/删除、解耦架构按「acting 期真规则」诊断结论重设计。
> **当前实际次序（终态快照，2026-07-10；过程档案一律见
> [naive0 issue §四](../../.issue/items/gomoku_naive0_tactical_wall.md) 与[棋盘账本](../../examples/my_zero/gomoku/README.md)，本文不再复述逐臂经过）**：
> - **Gomoku 纯 self-play 战役 ①–⑱ 全线收官**：真规则树上限 0.90 证明任务可学；
>   learned dynamics 规则学习税主导且对 12.5× 预算/CNN/增广钝感；生成端（课程，
>   降级诊断脚手架）、消费端（PER 单药 ⑭ + 复叠 ⑰）、监督端（recon ⑮⑱ /
>   consistency ⑯）三端系统性排除——recon 发现集正信号被未见-seed 配对终审否决，
>   G3 selected holdout 未达标，**棋盘 recipe 维持 base**。并行轨 KL-lr：CartPole
>   ❌ 不 promote、棋盘无害留库。
> - **战略复盘已定档（2026-07-10）**：生产搜索保持全程 learned dynamics；
>   true-rules / env snapshot 只作 reference diagnostic。路线不再按载体三选一，改为
>   “一个入口 + 稳定契约 + 可退化模块族”。前两阶段地基已完成：
>   `ObservationSchema / ActionSchema+Codec / LatentState / WorldModel`，以及矩形图像、
>   Image+Dense Dict、token、MultiDiscrete、2D continuous、Platform Hybrid 最小纵切。
>   详见[通用 world model 地基](./my_zero_world_model_foundation.md)。
> - **主动数据 3A0 已负裁（2026-07-12）**：continuation Brier 与任务关键
>   局面相关，但新增真实 game 按原 MuZero loss 更新后仅 1/3 seed 改善、2/3 回归；
>   未进入 ErrorQ/Collector/H=K，不用 SAC/WGAN 补救。下一步顺位改为 recurrent
>   posterior / sequence burn-in，再做 stochastic chance nodes；均不得破坏已验收的
>   确定性完全可观测 MDP 轻路径。
> - **后台 = Phase 1 轻量图像 pilot**（§2 修订注记：单 seed + 判停协议；
>   预注册见[图像负结果 issue](../../.issue/items/my_zero_pong_image_flat_negative.md)）。
> - **ROSMO「哨兵基础组件」提案已否决（2026-07-05）**：CartPole 叠加 29.6k vs 哨兵
>   ~8.7k env-steps 无增益；棋盘域 ③臂有害。
> - **Pendulum 不提前（2026-07-05 复议维持压轴）**：连续域不在战略轴，压轴免费吃
>   前序修复红利（§5）。

- target_net 接入训练循环 + 消融（staleness 治理；若 Phase 1 阶梯一实测需要 target 网 bootstrap 则相应前移）→ acting/reanalyze 解耦落地（[纲领 §2.3](./my_zero_algorithm_vision.md#23-战略目标与优先轴2026-07-01-定稿) 战略架构：实时轻 acting + 离线重刷新，刷新引擎 = Phase 1 裁决出的阶梯档位）→ SVE 接入消融或删除（不留 ⏳）
- **退出判据**：target_net/SVE 零 ⏳、解耦架构跑通并入账本；[reanalyze issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) 若 Phase 1 未终局则在此归档。

## 5. Phase 4 · 总收口（v0.28）

**为什么压轴**：收口 = 汇总前四阶段证据；Pendulum 特意最后测——Phase 0 新 loss 栈 + Phase 3 target_net 都可能改变其诊断前提，压轴等于免费获得两轮修复红利。

- **Pendulum 固定预算终审**（假设清单见[诊断 issue](../../.issue/items/pendulum_failure_diagnosis.md)）：修好 → 连续支柱成立（Sampled B=7 真实子采样一并裁决）；修不好 → 正式 known-limitation。**两个出口都是收口。**
- **value_prefix 复测前置**（矩阵清零该行前必做，若 Phase 1 提前复测则随 Phase 1）：补 `use_value_prefix=true` 的逐样本 vs batch 等价测试 + `ValuePrefixLstm` 专项测试——现状 `batch_train_equivalence.rs` 显式 `vp=false`，LSTM 路径零测试覆盖（v0.24 遗留测试债，v0.25 batch 化后扩大）。
  **⚠️ 断链发现（2026-07-05 GPT-5.5 静态审查）**：`value_prefix=true` 时训练走 LSTM prefix 头（`network.rs` unroll loss），但**搜索期 `Dynamics::recurrent` 仍从普通 reward head 解码**（该头此时不被训练）——树内 reward 静默变糊。复测前必须先修「搜索期 reward = prefix 增量」接线（`MctsModel::State` 携带 LSTM hidden 的忠实版接缝已留，见 `mcts/traits.rs` doc）；此断链亦为 CartPole value_prefix ❌ 负结果的候补解释，修复后该格裁决需重跑
- **reward_scale 去旋钮化**（旧 completedq_norm_reshape 规划 B1 遗留：`config.rs` 用户旋钮、Pendulum override 0.1；自动 support 半宽或内部归一化；若 Phase 1 图像 reward 尺度先成障碍则提前）
- **Sampled 小动作空间自动短路裁决**（认领账本 v0.25 结论 3「留 v0.26 评估」的悬空项）：纯离散 K_eff=N 全枚举时 Sampled 与 PUCT 已实证逐步等价、仅余 +~26% wall-clock 簿记开销——recipe 是否自动短路在此终审，做/不做皆可、裁决回注账本（等价性优化，非行为改动）
- 矩阵清零（逐格补测或裁决 ⏸）→ smoke-rl 扩容 + `baseline_matrix_bench` 增量链重定档 → 文档终审（roadmap 重写为收口版，v0.26–v0.27 归档）
- **Platform 混合动作正式不在收口范围**（v0.25 旧规划 G3 硬承诺作废，roadmap 已降级留 backlog，§0 终态验收不含它）

## 6. Issue → 阶段映射 · 版本映射

| Open issue | 裁决阶段 |
|---|---|
| [CPU-only × 图像 × MCTS 一级风险](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) | Phase 1（spike 数据裁决） |
| [Gumbel / completedQ 负结果](../../.issue/_archive/my_zero_gumbel_completedq_cartpole_negative.md) | Phase 2（前置修复 + native 复测） |
| [CartPole 哨兵红灯（ndarray 漂移）](../../.issue/_archive/cartpole_sentinel_red_ndarray_drift.md) | ✅ **已收口（2026-07-04）**：新数值流预注册复裁两臂均 5/5，recon=16 维持 promote、recipe 零变更，官方 3-seed 哨兵回绿（中位 ~8.7k）；Phase 1 前置阻断解除 |
| [reanalyze 回归](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) | Phase 1（ROSMO-first 阶梯复活，2026-07-03 自 Phase 3 提前） |
| [Pendulum 诊断](../../.issue/items/pendulum_failure_diagnosis.md) | Phase 4（固定预算终审） |
| [Sampled 决策备忘](../../.issue/items/my_zero_action_space_sampled_policy.md) | Phase 4（随 Pendulum B=7 落地） |

版本映射：Phase 0+1 → **v0.26**；Phase 2+3 → **v0.27**；Phase 4 → **v0.28**。每版发版关卡照旧三层协议。
> **映射修订注记（2026-07-05）**：实际交付次序已偏离该映射（Phase 1 降级后台、Phase 2 提前收口）——
> 版本号按**实际交付节奏**在发版点定档（CHANGELOG 唯一权威），本映射降级为初始意向参考；
> push/同步与定版解耦，不因未发版而阻塞推送。

### 6b. Simulus 组件吸纳对照（arXiv 2502.11537 → 本规划落点）

HL-Gauss 编码（A1）→ Phase 0 + Phase 1 复测 · 梯度流审计/sg 解耦（A2）→ Phase 0 · loss 优先回放（B1）→ Phase 3 · ensemble JSD 内在奖励（暂缓）→ Phase 1 稀疏奖励触发点 · planning-free 实证 → 条款一退路依据（[风险 issue §三b](../../.issue/items/cpu_only_mcts_image_realtime_risk.md)）· tokenizer/RetNet 不采纳。细节见 [Simulus 消融计划](./my_zero_simulus_ablation_plan.md)。

### 6c. 旧规划清账对照（2026-07-02 审计，零散 plan 全部吸收或作废，不必再翻）

- **v0.25 实施 plan**：M1/M2/M5 与旧目录删除已完成；S3 target_net / S4 SVE → Phase 3；Pendulum G2 → Phase 4；Platform G3 硬承诺作废（§5 声明）。
- **gumbel_sampled_concept_freeze plan**：completedQ 三项（vmix / tree-level σ / never-worse）已完成入负结果 issue §六；Gumbel eval 去噪 + q_range 两 bug 与三件套分层文档 → Phase 2 吸收。
- **completedq_norm_reshape plan**：线 A（A1–A6）已完成；B1 reward_scale 去旋钮化 → Phase 4；B2 Pendulum smoke 已在 smoke-rl 关卡。
- **rl_审计优化 plan 及更早**（ez-v2 / muzero_* / pendulum_收敛诊断等）：被 v0.25 收口与现 roadmap 全覆盖，无遗留。
- **v0.24 EZ-V2 组件对账笔记**：各组件去向均已覆盖（target_net 在 v0.25 统一时从"已生效"退回"待接"，矩阵如实、Phase 3 接回）；唯一活债 = value_prefix LSTM 测试缺失（→ Phase 4 复测前置）。

## 7. 制度化条款

- **条款一（风险改道）**：Phase 1 spike 是唯一可改写后续阶段的节点——若实测撞死，载体收缩为 Gomoku 单支柱、planning-free 退路评估插入 v0.27；其余阶段顺序不变。
- **条款二（哨兵铁律）**：Phase 0 之后 CartPole 只回答"崩没崩"，不回答"好不好"——任何组件价值判断到 native 环境验证，防 t1/t2 悬案泥潭复发。
- **条款三（复现差距纪律，2026-07-05 增补）**：预注册门槛与预算估计一律以**第三方复现数据**为锚，不直接采信论文原文数字——已两次吃亏：EfficientZero 论文 7 小时 vs 复现 1–2 天（[外证](https://www.gigglebit.net/blog/efficientzero)）、Gomoku M3 参照实现每局训练强度差 40–50× 才在消融后期发现。无第三方复现可查时，按论文数字 ×3–10 悲观折算并在预注册协议里写明折算依据。

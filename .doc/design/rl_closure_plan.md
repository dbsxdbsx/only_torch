# RL 全面收口规划（v0.26 → v0.28）

> **定位**：把「所有 RL 基准 + MyZero 万金油算法」收口到可验收终态的**战役次序文档**——回答"先做什么、后做什么、为什么这个顺序、做到什么算过关"。
> 当前状态与验收协议见 [rl_roadmap.md](./rl_roadmap.md)；战术细节各自链出（Simulus 消融见 [my_zero_simulus_ablation_plan.md](./my_zero_simulus_ablation_plan.md)）；实测数字唯一账本仍是 [cartpole README](../../examples/my_zero/cartpole/README.md)。
> **创建**：2026-07-02 · 贯穿纪律沿用 [roadmap §3.2](./rl_roadmap.md#32-改动纪律搬运--改进沿用)：一次一项、预注册协议、3-seed 消融、数字只进账本。

---

## 0. 终态验收（"收口"的操作定义，四条全绿才算完成）

1. **组件×环境裁决矩阵零 ⏳**：[组件矩阵](../../examples/my_zero/README.md#内部组件进展团队--promote-时改-recipers)每格 ✅/❌/⏸ 有据，❌ 有 issue、✅ 有账本数字。
2. **四类环境支柱各有 promoted recipe + 3-seed 账本数字**：低维离散（CartPole，已有）· 图像离散（待建）· 棋盘 self-play（Gomoku，待建）· 连续（Pendulum 终审后两出口，见 §5）。
3. **全部 RL open issue 归档**（映射见 §6）。
4. **发版关卡覆盖全部支柱**：`smoke-rl` 扩容纳入图像 + Gomoku，三层验收协议不变。

「万金油」的操作定义 = 第 2 条：`MyZero::new(env_id)` 四类环境开箱即用、recipe 有实测背书。

## 1. Phase 0 · 训练信号收口（v0.26 上半，进行中）

**为什么最先**：图像/self-play 全在这套 loss 栈上训练，汇率错一个量级会污染所有后续消融；且 CartPole 迭代最便宜。

- ✅ 系数消融 4+1 臂 + recon 5-seed 复裁（已收口：recon_coef 1→16 promote，哨兵中位 66.2k→~9.8k；recon=1 实测有害、悬案闭合——见账本与 CHANGELOG 2026-07-02 条目）
- ⏳ 梯度流审计（读码产出 cons/recon 对 representation/dynamics 的回流现状图，零风险先行）→ 视结果追加 **sg 解耦臂**（= Simulus 计划 A2）
- ⏳ **HL-Gauss 编码消融**（= Simulus 计划 A1：two-hot → 高斯软标签，改动集中在 `value_encoding.rs`）
- **退出判据**：v0.26 recipe 定稿（系数 + 编码 + 梯度流结构三者有据）、账本回填；**此后 CartPole 冻结为纯回归哨兵**（见 §7 条款二）。

## 2. Phase 1 · 图像线立柱 + 一级风险压测（v0.26 下半）

**为什么第二**：唯一可能否定整条路线的风险要最早、用最便宜的实验裁决；同时这是 consistency/reconstruction 的 native 场景（CartPole 上的暧昧表现到此才见分晓）。

- **风险 spike 先行**：最小 CNN 栈 + 假图像输入，实测「CNN 前向 × sims」单步 wall-clock，对照 [CPU 风险 issue](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) 的触发线，产出去/留/改道决策
- 图像 obs 管线（复用 `src/vision/preprocess`：降采样/灰度/帧堆叠）→ CNN representation 进 `network.rs`（按 recipe 注入）→ **预注册基准门槛**（Pong 类；验收 = 3-seed 可复现学习曲线 + 预注册分数，**非 SOTA**）
- native 域复裁：consistency / reconstruction / HL-Gauss 各一次 A/B；recon_coef=16 为 CartPole 临时值，在此复验
- Simulus 暂缓项触发点：若环境奖励稀疏且探索成实测瓶颈，ensemble JSD 内在奖励在此转正（唯一入口）
- **退出判据**：图像支柱 3-seed 账本 + CPU 风险 issue 第一格勾选。

## 3. Phase 2 · self-play 线：Gomoku 踏脚石（v0.27 上半）

**为什么第三**：象棋战略目标的既定踏脚石（地基已备：`SelfPlayGame` / negamax backup / legal_mask / `python/gym_env/gomoku/`）；且 **Gumbel/completedQ 两个负结果 issue 的关闭条件就是在 |A|≫sims 场景复测**——Gomoku |A|=225 正是。

- **Gumbel 复裁前置修复（必做，否则复裁无效）**：
  - ① greedy eval 去噪 bug：`gumbel.rs::final_recommendation` 无视 `temperature=0` 仍用 Gumbel 噪声打分，eval 分数被污染（疑似"Gumbel 未收敛"负结果真因）——详见[负结果 issue §七](../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md)
  - ② `q_range` 局部归一化同源 bug（completedQ 已修 tree-level，Gumbel 侧未修，issue §六留档）
- Gomoku 训练闭环（棋盘 2D 表征复用 Phase 1 CNN）→ **预注册棋盘账本口径**（建议 vs random ≥95% + vs 旧 checkpoint ≥55%）→ Gumbel-root / completedQ 终局复裁（兼 P2 少 sim acting 复测）
- 裁决后据实沉淀「三件套正交分层」文档至 [vision](./my_zero_algorithm_vision.md) §5.4（Sampled=候选层 / Gumbel=根层 / completedQ=目标层 + 融合契约——旧规划遗留文档债）
- **退出判据**：胜率门槛达成 + 两个负结果 issue 终局归档 + 棋盘 recipe 进 `recipe.rs`。

## 4. Phase 3 · 样本效率纵深（v0.27 下半）

**为什么第四**：reanalyze 在 CartPole 负结果的根因是"样本便宜喂不出价值"，必须等 Phase 1/2 提供样本贵的验证场；target_net / SVE 同属 value target 质量域，一批清账最经济。

- loss 优先回放（= Simulus 计划 B1：per-sample loss + α=0.3 混合采样 + ν₀ 初始值，单参数）→ target_net 接入训练循环 + 消融（staleness 治理，reanalyze 前置）→ reanalyze 复活 + acting/reanalyze 解耦落地（[纲领 §2.3](./my_zero_algorithm_vision.md#23-战略目标与优先轴2026-07-01-定稿) 战略架构）→ SVE 接入消融或删除（不留 ⏳）
- **退出判据**：[reanalyze issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) 终局归档、target_net/SVE 零 ⏳、with/without reanalyze 样本效率对照入账本。

## 5. Phase 4 · 总收口（v0.28）

**为什么压轴**：收口 = 汇总前四阶段证据；Pendulum 特意最后测——Phase 0 新 loss 栈 + Phase 3 target_net 都可能改变其诊断前提，压轴等于免费获得两轮修复红利。

- **Pendulum 固定预算终审**（假设清单见[诊断 issue](../../.issue/items/pendulum_failure_diagnosis.md)）：修好 → 连续支柱成立（Sampled B=7 真实子采样一并裁决）；修不好 → 正式 known-limitation。**两个出口都是收口。**
- **value_prefix 复测前置**（矩阵清零该行前必做，若 Phase 1 提前复测则随 Phase 1）：补 `use_value_prefix=true` 的逐样本 vs batch 等价测试 + `ValuePrefixLstm` 专项测试——现状 `batch_train_equivalence.rs` 显式 `vp=false`，LSTM 路径零测试覆盖（v0.24 遗留测试债，v0.25 batch 化后扩大）
- **reward_scale 去旋钮化**（旧 completedq_norm_reshape 规划 B1 遗留：`config.rs` 用户旋钮、Pendulum override 0.1；自动 support 半宽或内部归一化；若 Phase 1 图像 reward 尺度先成障碍则提前）
- 矩阵清零（逐格补测或裁决 ⏸）→ smoke-rl 扩容 + `baseline_matrix_bench` 增量链重定档 → 文档终审（roadmap 重写为收口版，v0.26–v0.27 归档）
- **Platform 混合动作正式不在收口范围**（v0.25 旧规划 G3 硬承诺作废，roadmap 已降级留 backlog，§0 终态验收不含它）

## 6. Issue → 阶段映射 · 版本映射

| Open issue | 裁决阶段 |
|---|---|
| [CPU-only × 图像 × MCTS 一级风险](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) | Phase 1（spike 数据裁决） |
| [Gumbel / completedQ 负结果](../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md) | Phase 2（前置修复 + native 复测） |
| [reanalyze 回归](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) | Phase 3（样本贵环境复活） |
| [Pendulum 诊断](../../.issue/items/pendulum_failure_diagnosis.md) | Phase 4（固定预算终审） |
| [Sampled 决策备忘](../../.issue/items/my_zero_action_space_sampled_policy.md) | Phase 4（随 Pendulum B=7 落地） |

版本映射：Phase 0+1 → **v0.26**；Phase 2+3 → **v0.27**；Phase 4 → **v0.28**。每版发版关卡照旧三层协议。

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

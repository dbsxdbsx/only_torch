# Simulus 组件采纳消融计划（v0.26 挂靠）

> **来源**：Simulus（arXiv 2502.11537，Cohen et al.，世界模型版 Rainbow）——四个"各自验证过但没组合过"的改进拼进 token-based WM，三基准 planning-free SOTA，逐组件消融证明贡献。判定背景见[论文阅读日志](../paper/reading_log.md)。
> **定位**：本文只回答"哪些组件、以什么顺序、按什么验收进 MyZero"。纪律沿用 [rl_roadmap §3.2](./rl_roadmap.md#32-改动纪律搬运--改进沿用)：**一次一项、3-seed 消融、过 CartPole 哨兵**；数字回填唯一账本。
> **创建**：2026-07-02

---

## 1. 组件 → MyZero 映射总表

| Simulus 组件 | MyZero 现状 | 判定 | 挂靠 |
|---|---|---|---|
| RaC value/reward 头（symexp 分箱 + **HL-Gauss** 软标签 CE） | **已有** categorical 头 + two-hot（`value_encoding.rs`，MuZero 附录 F 口径）——采纳成本骤降为"two-hot → HL-Gauss" | ✅ 采纳 | P0 批次（A1） |
| 表征/动力学 **stop-gradient 解耦**（论文附录 B.1：联合优化目标互相干扰的实证） | consistency / reconstruction 辅助 loss 与主干联合训练；`loss.rs` 无显式 detach | ✅ 采纳（作为实验而非直接改行为） | P0 批次（A2） |
| **loss 优先回放**（per-sample 模型 loss + α 混合采样 + 高初始值 ν₀） | buffer 均匀采样 | ✅ 采纳 | P1 reanalyze 复活时（B1） |
| ensemble 分歧（JSD）内在奖励 | 无；探索靠 MCTS + Dirichlet 噪声 | ⏳ 暂缓 | 触发条件见 §3 |
| 多模态 tokenizer / RetNet + POP | MyZero 走连续 latent，非 token 家族 | ❌ 不采纳 | 认知储备 |
| （副产品）planning-free 路线实证 | — | 已录为 CPU-only 风险退路 | [risk issue §三b](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) |

## 2. 消融项详单

### A1 · value/reward 编码 two-hot → HL-Gauss（P0 批次）

- **改动**：`value_encoding.rs` 增加 `scalar_to_hl_gauss(x, cfg, σ)`——软标签由"相邻两原子线性插值"改为"高斯 CDF 差分摊到全 support"（Farebrother et al. 2024 实证 HL-Gauss > two-hot > MSE）；解码端不变（期望 → h⁻¹）。网络头与 CE loss 完全复用，改动集中在编码函数 + 一个 σ 超参（论文用 σ ≈ 0.75 × bin 宽）。
- **前置**：无框架级前置（CE 已支持 soft target——two-hot 本身就是 soft 的）。
- **消融**：recipe 不动，仅切编码；3-seed vs 现哨兵（以[账本](../../examples/my_zero/cartpole/README.md)当前 promoted 行为准，不在本文维护数字）。
- **预期与判读**：CartPole value 范围窄，预期持平——**持平即通过**（哨兵证"不崩"），真正收益押在图像线大 value 噪声场景；若显著劣化则回退并记录。

### A2 · 辅助 loss stop-gradient 解耦实验（P0 批次，与 loss 系数重标定同批）

- **动机**：P0 loss 系数重标定与本项是同一个问题的两把刀——"辅助 loss（cons/recon）与动力学主干怎么相互作用"。Simulus B.1 显示解耦（动力学入口对表征输出 sg）能显著降重建 loss 且回报不降，说明有些干扰**调系数调不掉**。
- **第一步（改动前必做）**：审计 `loss.rs` / `network.rs` 现状——consistency 的 target 分支是否已按 EfficientZero 惯例 stop-grad、reconstruction 梯度是否回流 representation。**先画清梯度流向图再定实验矩阵**。
- **实验矩阵**（每项独立 3-seed）：(a) 现状；(b) recon 梯度只训 decoder 不回流 encoder；(c) dynamics 入口 sg（Simulus B.1 式）。与系数消融正交组合时，优先跑单变量。
- **判读**：per-loss 曲线（recon/cons/policy/value 分项）+ env-steps 中位；关注"某项 loss 降但收敛变慢"的干扰特征。

### B1 · loss 优先回放（P1，随 reanalyze 复活）

- **改动**：buffer 每样本存最近一次训练 loss（新样本初始 ν₀=10 保证必被光顾）；采样时 α=0.3 按 softmax(loss) 抽、其余均匀；训练后回写。单参数 α，无 PER 的 IS 权重复杂度（Simulus 实测此简化方案稳健）。
- **挂靠理由**：reanalyze 的价值 = 榨旧样本；优先回放决定"榨哪些"——两者天然一批验收，也共享"CartPole 负结果不构成否定"的复测前提（[reanalyze issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md)）。
- **消融**：α ∈ {0（对照）, 0.3}，3-seed。

## 3. 暂缓项与触发条件

| 项 | 触发条件 |
|---|---|
| ensemble JSD 内在奖励 | 图像线遇到**稀疏奖励**环境（商业游戏类）且探索成为实测瓶颈；实现时用 stop-grad 输出上的多头（开销小），奖励只注入想象/搜索期 |
| symlog 观测归一化 | **已升级为主动项（2026-07-02）**：排入[收口规划 Phase 0](./rl_closure_plan.md)「obs 无量纲化（symlog）消融」，不再等触发；协议以该处为准 |

## 4. 执行顺序与验收

1. **A2 第一步审计**先行（零风险、纯读码，产出梯度流向现状图）→ 决定 A2 实验矩阵。
2. A1（编码切换）与 P0 系数重标定**串行**执行（同为"改行为"，禁止混批）。
3. B1 等 reanalyze 复活排期，不提前。
4. 每项验收 = [rl_roadmap §3](./rl_roadmap.md#3-验收协议三层发版固定关卡) 三层协议；数字只进[基准账本](../../examples/my_zero/cartpole/README.md)；负结果照常记 `.issue`（参照 completedQ/Gumbel 先例）。

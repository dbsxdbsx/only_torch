# MyZero — 统一 Model-Based RL 算法

> only_torch 的终极强化学习算法：learned-model MCTS，按环境内置已验收组件组合，持续迭代。  
> 战略层（做/不做、文献谱系、战略目标）：[MyZero 算法纲领](../../.doc/design/my_zero_algorithm_vision.md) · 当前状态与方向：[RL 路线图](../../.doc/design/rl_roadmap.md) · **实测数字按环境分账**：[CartPole](cartpole/README.md) / [Gomoku](gomoku/README.md)

## 设计理念

- **一个算法，持续迭代**：维护不断进化的 MyZero，不再为每篇论文建独立实现
- **用户零组件概念**：`MyZero::new(env_id)` 自动套用库内按环境配置的组件组合；团队 promote 组件时只改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)
- **优先轴（2026-07-10 修订）**：生产统一为 learned-world-model 模块族；先完成稳定 schema 地基，再分别验证主动数据、POMDP 与 stochastic。实现边界见[地基设计](../../.doc/design/my_zero_world_model_foundation.md)

## 环境 × 状态

| 环境 | 动作类型 | 门禁 | 状态 |
|------|---------|------|------|
| [**CartPole-v1**](cartpole/README.md) | 离散（2） | greedy eval ≥ 475 | ✅ 回归哨兵（官方口径 3-seed 中位 env-steps，数字见[账本](cartpole/README.md)） |
| [**Pendulum-v1**](pendulum/README.md) | 纯连续（1） | return ≥ -200 | 诊断中·已降级（当前 recipe 复用 CartPole 栈作诊断，不代表组件已裁决；[issue](../../.issue/items/pendulum_failure_diagnosis.md)） |
| 图像离散（Atari-100k 类） | 离散 | 任务指标 | 后台预算标定中（S2 基准 0/3 平直，[负结果 issue](../../.issue/items/my_zero_pong_image_flat_negative.md)） |
| [**Gomoku 9×9**](gomoku/README.md)（→ 象棋） | 离散棋盘（81） | vs random ≥0.95 + gating ≥0.55 | ✅ 支柱已立（M2 双门槛 3/3 · recipe=base；naive0 战术墙留 [issue](../../.issue/items/gomoku_naive0_tactical_wall.md)） |
| Schema toys | MultiDiscrete / 2D continuous / token / Image+Dense | loss 有限、管线通 | ✅ 接口纵切；只证明骨架，不作可学习性声明 |
| Platform-v0 | 固定混合 Tuple | return 趋势上升 | ✅ MyZero smoke 接通；统计价值未裁决 |

## 内部组件进展（团队 · promote 时改 [`recipe.rs`](../../src/rl/algo/my_zero/recipe.rs)）

> **裁决矩阵**：一行一个组件、一列一个环境；表示「该组件在该环境是否已完成效果裁决」，不等同于「当前诊断栈是否临时启用」。
> 图例：`✅ 已验收` · `❌ 实测有害/无增益` · `⏳ 待测` · `⏸ 此环境不适用` · `— 未实现`
> 用户 API **不暴露**组件开关；实测数字一律见[基准账本](cartpole/README.md)，本表只记裁决结论。

| 组件         | [CartPole-v1](cartpole/README.md) | [Pendulum-v1](pendulum/README.md) | 图像 | [Gomoku](gomoku/README.md) | 备注 |
| ------------ | :-------------------------------: | :-------------------------------: | :--: | :---: | --------------------------- |
| consistency  |                ✅                 |                 ⏳                 |  ⏳   | ⚠️ 带内弱阳（⑯ CNN 底座：naive0 中位 0.15→0.25，4/5 seed 抬升但 <0.30 线，不 promote） | SimSiam 一致性 loss；棋盘两裁：M3 MLP 底座中性 → ⑯ CNN 底座带内弱阳（同底座对照 recon=1 越线 = 冻结真值 > 自指目标实测，[账本](gomoku/README.md)） |
| reconstruction |              ✅※                |                 ⏳                 |  ⏳   | ❌ **新-seed 未复现**（⑮ 发现集 0.15→0.35；⑱ 未见 seed 配对 0.28→0.20，仅 1/5 不劣，不 promote） | ※ autograd 修复后 3-seed 中位不再显示增益（方差大、未回滚），v0.26 P0 重标定 loss 系数后复裁，见[账本结论](cartpole/README.md#结论v025-收口)；棋盘 coef=1 的发现集正信号保留为探索证据，但终审否决外推（[账本](gomoku/README.md)） |
| Sampled（联合候选采样） |       ✅ 接入不回归        |                 ⏳                 |  ⏸   |  ⏸   | MultiDiscrete([4,4,16])、2D continuous 与 Platform Hybrid 接缝已通；CartPole K=N 退化全枚举，native 统计价值仍须单独裁决 |
| HL-Gauss value/reward 编码 | ❌ 显著劣化（中位 9.8k→27.6k） | ⏸ | ⏳ **native 场景** | ⏸ | 窄 support 低噪声下 two-hot 尖标签是信息优势；开关留库，Phase 1 图像域复测（[账本](cartpole/README.md#v026-phase-0编码--量纲消融2026-07-02)）；图像域初裁中性 |
| obs symlog 无量纲化 | ❌ 无增益且系数不回移 | ⏳ | ⏸ 图像走 [0,1] 归一 | ⏸ 0/1 平面 | CartPole obs 本就小量纲；开关留库，触发条件回归 [Simulus 计划 §3](../../.doc/design/my_zero_simulus_ablation_plan.md)（连续特征范围失控时） |
| reanalyze    | ❌ 当前实现有害（[issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md)） | ⏳ | **v0.26 战略组件** | ⏸ | 「实时轻 acting + 离线重 reanalyze」解耦，图像线优先验证；棋盘 target = 终局事实不过期，非 native 场景 |
| ROSMO 一步刷新 | ❌ 无增益（叠加中位 29.6k vs 哨兵 ~8.7k，慢 ~3.4×；「哨兵基础组件」提案否决 2026-07-05） | ⏳ | ⏳ 价值裁决待图像基线 | ❌ 有害（rr32×刷新护栏崩塌，vs random ~0.5；弱模型 adv 噪声自指反馈，[账本](gomoku/README.md)） | `.rosmo(true)` / 棋盘 `rosmo_refresh` 消融开关，recipe 默认关 |
| PER 位置级优先回放 | — | — | — | ❌ 无增益（⑭ 单药 0.15；⑰ 与 recon1 复叠仍 0.15，且对 recon1 单药 5/5 回落） | `PerPriorities` 伴生采样器（buffer 层通用件）留库默认关；棋盘两裁均排除消费端——即使 recon 修复监督供给，`|ν−z|` 重排仍破坏 recon 的均匀稠密监督；不调 α/IS/刷新（[账本](gomoku/README.md)） |
| value_prefix |                ❌※2               |                 ⏳                 |  ⏳   |  ⏸ 无中间 reward  | ※2 CartPole 有害，但 2026-07-05 审查发现**训练/搜索断链**（训 LSTM prefix 头、搜索仍读普通 reward head）——修复接线后该裁决需重跑，见[收口规划 §5](../../.doc/design/rl_closure_plan.md) |
| target_net   |                 ⏳                 |                 ⏳                 |  ⏳   |  ⏳   | 已入库，训练循环待接 |
| SVE          |                 ⏳                 |                 ⏳                 |  ⏳   |  ⏳   | 已入库，训练循环待接；🔲 改进：固定权重 → 自适应 mixed target |
| completedQ   | ❌ 系统性慢于 visit（[issue](../../.issue/_archive/my_zero_gumbel_completedq_cartpole_negative.md)） |                 ⏳                 |  ⏳   | ❌ 中性（s100+s16 复裁） | `\|A\|≫sims` 复裁已完成（棋盘 s16），无灾难亦无增益 |
| Gumbel-root  | ❌ 未收敛（同上 issue） |                 ⏳                 | ⏳ 少 sim acting 候选 | ❌ 中性（同左） | 前置双修复（greedy 去噪 + tree-level q_range）后棋盘复裁中性；少 sim acting 可用降档 |
| recurrent posterior (POMDP-lite) | ❌ masked 未达标（ON ~45 vs OFF ~40，0/3；[issue](../../.issue/items/my_zero_pomdp_lite_posterior_negative.md)） | ⏳ | ⏳ | ⏸ 完全可观测 | GRU 状态估计 + burn-in；代码完整，归因容量/预算不足、架构未否定；复活条件见 issue §六 |
| **Stochastic MuZero (always-on)** | ✅ 3/3 达标（K=8，26k，1.7x） | ⏳ | ⏳ | ⏳ | **默认 K=8**：afterstate + chance encoder + KL(q‖p) + chance-in-edge 搜索；确定性环境自动退化；StochasticCartPole K=8 3/3 vs K=1 2/3 |
| KL 自适应 lr | ❌ **灾难级**（0/3，greedy 钉死 ~9.2；lr 死亡螺旋：小 batch 下探针 KL ≈ 噪声 → 乘子棘轮顶 10× 上限 → lr 0.2 发散，诊断轨迹 `.bench/cartpole_kl_lr_diag_20260705.log`） | ⏸ | ⏸ | ✅ 无害（⑨臂 batch 512 自动配平，护栏全绿） | 「去旋钮」组件，`kl_adaptive_lr` 开关默认关；域适配前提 = 每局训练样本量大且探针与训练分布相关（棋盘成立、CartPole batch 8×buffer 1000 脱钩）；重试条件 = 改「刚训 minibatch 上测 KL」（参照实现原口径）再过闸门 |

> 论文全称与 arXiv：[算法纲领 §4.1 — 组件文献对照](../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)

### 处方表（跨域路由规则，2026-07-07 立）

> 上方矩阵记**结果**（哪格测过、结论如何），本表记**规则**（为什么起效、什么环境该开/禁用、剂量怎么定）——「万金油 = 带适应症表的医生，不是包治百病的药」的落地件（反元过拟合纪律，2026-07-07 与用户定稿）。用法：新环境**先查表路由，再一次预注册 A/B 确认**，不允许网格搜索；组件若在新域必须重搜参数才起效，即为「非万金油」证伪，按纪律出库/降级。剂量判据：**宽平台 = 真机制签名，刀锋最优点 = 过拟合签名**；魔法数字的升级方向 = 自归一比例（如辅助 loss 梯度模长钉在任务 loss 固定百分比）。

| 组件 | 机制（为什么起效） | 适应症（可观测环境特征） | 禁忌症（已知失败边界） | 剂量规则 |
|------|------------------|----------------------|--------------------|---------|
| consistency | 潜状态对齐真实下一帧编码（SimSiam stop-grad）——给 dynamics 加**直接监督**，治 value-equivalent 训练「dynamics 只吃间接梯度」的结构性饥饿（EfficientZero 主药方）；**target 自指**（`repr(next_obs)` = 网络自己正在学的输出），信号上限被 h 质量卡住 | 任务信号稀疏且 dynamics 为瓶颈；图像域 native（EfficientZero 100k 帧实证）；表征需先有结构（CNN/增广底座）target 才有信息量 | 暂无害例；棋盘两裁均未过线（MLP 底座中性 → CNN 底座带内弱阳 0.25<0.30）——**同底座 recon=1 越线：目标可冻结真值时优先 recon，cons 是 obs 空间不可回归时的替代品**（图像域两者角色互换） | coef=2.0（EfficientZero 对齐），暂无跨域重标证据 |
| reconstruction | latent 经 decoder 对齐真实 obs——**稠密地面真值副监督**（每步整张 obs vs 每局个位数 bit 任务信号）；unroll 槽位梯度穿 dynamics（与 consistency 同族，绕道 obs 空间） | 任务信号稀疏 × obs 比特基本全任务相关（低维状态向量 / 棋盘平面）；CartPole 有实测 promote。棋盘发现集曾命中该预测（⑮ 0.15→0.35），但未见 seed 终审未复现，故不能把机制故事当跨 seed 适应症证据 | 高维自然图像含任务无关比特（干扰物挤占表征）；**seed/训练轨迹交互可压过平均药效**（⑱：control 0.28、recon1 0.20，仅 1/5 不劣）；系数跨域不迁移（Gomoku 16 偏害，1 仅发现集越线） | CartPole 标定 16；Gomoku 候选 1（4/16 更差）但**不 promote**。跨域重标后仍必须用未见 seed 配对终审，禁止在发现 seed 上宣告剂量成立 |
| PER 优先回放 | 改数据**消费**分布（课程的零领域知识版）：`p=\|ν−z\|^α` 把「整机判断离现实最远」的局面顶到队列前——ν 是搜索总产出，任何头生病都汇集于它（复合症状排序，分诊交给梯度） | 分布覆盖不足（关键局面在 buffer 稀有）× value target 为冻结事实 × **优先级与当前训练病灶同构**（只重排真正缺练且可学的样本） | 高结果噪声域（翻盘局被误优先）；勿在线自评刷新（③臂已证弱网自评是噪声源）；病灶不在消费端时无效（⑭）；**辅助监督需要全局均匀覆盖时禁用**（⑰：recon1×PER 中位 0.15，较 recon1 单药 0.35 且 5/5 回落——`|ν−z|` 与 recon 逐比特误差不同构） | α=0.6、无 IS 仅为 MuZero 原口径；棋盘双裁 ❌ 后不再调 α/IS/刷新，留库默认关 |
| HL-Gauss | value/reward 分类标签高斯软化，高噪声宽 support 下抗过拟合 | 图像域 native（大 value 噪声） | 窄 support 低噪声域（CartPole 9.8k→27.6k 显著劣化：two-hot 尖标签是信息优势） | σ 见 `value_encoding` 常量 |
| obs symlog | 连续特征无量纲化（DreamerV3 口径），治特征范围失控 | 连续特征范围跨数量级 / 失控时 | obs 本就小量纲（CartPole 无增益且系数不回移）；图像走 [0,1] 归一、棋盘 0/1 平面均不适用 | 无参数 |
| reanalyze | 存量轨迹全树重搜刷新 target，治 target 陈旧 | 数据陈旧 × 高 replay ratio × 离线算力富余 | 新鲜数据低 replay（CartPole 当前实现有害，issue 在案）；棋盘 value = 终局事实不过期（非 native） | 待图像域裁决 |
| ROSMO 一步刷新 | 现算一步 look-ahead 替代存量 MCTS target（reanalyze 轻量阶梯） | 陈旧数据 × 高 replay ratio × **模型自评质量足够** | 弱模型自评（棋盘 rr32 护栏崩塌：adv ≈ 噪声自指反馈）；新鲜数据低 replay（CartPole 无益且慢 3.4×） | α=0.2（论文 Atari 口径）+ prior top-16 剪枝 |
| completedQ / Gumbel-root | 少 sim 下 completedQ 目标 / 根置信序贯减半提升根决策质量 | 理论适应症 = sims ≪ \|A\|；**实测未证实**（棋盘 s16 正是该 regime，复裁中性）——仅余「少 sim acting 降档」候选资格，全域 recipe 关 | sims ≫ \|A\|（CartPole 该 regime 系统性劣化） | c_visit=50 / c_scale=1.0（tree-level 归一化后棋类口径通用） |
| KL 自适应 lr | 每次更新测 policy KL 位移自动调 lr（「用户不调 lr」去旋钮件） | 每局训练样本量大 × 探针分布与训练分布同源（棋盘 batch 512 配平实测无害） | 小 batch × 大 buffer 脱钩形态（CartPole 灾难级：探针 KL ≈ 噪声 → 乘子棘轮 10× → lr 发散） | kl_targ=0.02（参照口径）；重试条件 = 改「刚训 minibatch 上测 KL」再过闸门 |
| Sampled | 大 / 连续动作空间采 K 候选 + π̂_β 先验校正 | 连续 / 超大离散动作空间（K_eff < N 真实子采样） | 小离散空间 K=N 退化全枚举（仅 +~26% 簿记开销；自动短路裁决挂 Phase 4） | K 按 N、sims 公式自动解析（`sampled_params`） |

> 未列组件（value_prefix / target_net / SVE）：接线未完成或断链在修（[收口规划 §5](../../.doc/design/rl_closure_plan.md)），证据不足以开处方，不写推测行。课程 / 真规则树为**诊断脚手架**非常驻组件（哲学修正 2026-07-05），不进本表。

**CartPole-v1 当前内置**：base + **consistency + reconstruction + Sampled + Stochastic K=8**（PUCT · sims=20 · td=5 · continuation 二值门）。

**Pendulum-v1 当前内置**：复用 CartPole 栈做**诊断**（B=7 · sims=20 · `reward_scale(0.1)`），仍在失败区间；不要据此给组件下 ✅/❌ 裁决。

**Gomoku 当前内置**：**base 全关**（Flat MLP · negamax MC target · sims=100 · D4 增广关）；M3 九臂消融裁决与数字见[棋盘账本](gomoku/README.md)。

## 快速开始

```bash
cargo run --example my_zero_cartpole --release
```

```rust
use only_torch::rl::algo::my_zero::MyZero;

let mz = MyZero::new("CartPole-v1")
    .solved(475.0)
    .max_episodes(2000)
    // 多 seed 时库内自动插入 `seed_{seed}/` 子目录
    .save_model_when_eval("models/my_zero/CartPole-v1/best")
    .train()?;

// 训后加载用本次实际落盘路径（TrainReport::model_path），避免多 seed 加载错位
let best = mz.train_report().and_then(|r| r.model_path.clone());
let mz = match best {
    Some(p) => mz.load_model_if_exists(p)?,
    None => mz,
};
mz.eval(10)?;
```

## 训练与推理生命周期

| 时机 | 权重 | 说明 |
|------|------|------|
| `.train()` 返回 / 训后直接 eval | **latest** | 训末权重 |
| periodic greedy eval 创新高 | 写入 `{path}.otm` | 须 `.save_model_when_eval(path)` |
| `.load_model_if_exists(path)` 后 eval | **best** | 部署 / 演示用 |

`TrainReport`：`final_greedy` = latest；`best_greedy` / `model_path` = 训练期 periodic eval 最优（有落盘时）。

## 评判口径

| 维度 | 操作定义 |
|------|---------|
| **好** | greedy(temp=0) eval 达门槛（唯一成功判据） |
| **快** | env-steps-to-solved 为主（官方口径 **3-seed 中位 + 达标率**），wall-clock 为辅 |

CartPole 是 sanity/regression **哨兵**（叠组件不崩的证明），不是组件价值证明台；不拿 wall-clock 判生死（[纲领 §2.2/§2.3](../../.doc/design/my_zero_algorithm_vision.md#22-首要评价指标)）。

## 算法核心

MuZero 三网络（Representation / Dynamics / Prediction）+ latent MCTS + K 步 unroll + categorical value/reward。Replay / n-step 显式区分 `terminated / truncated / continuation`：真终止 `continuation=0`，普通步与 time-limit truncation `continuation=1`。Dynamics 同时学习 continuation；MCTS imagined edge 的 discount 用 **binary gate `gamma * (1 - done)`**（`done` 由 continuation 头阈值化），与 n-step value target 的二值 continuation 口径一致——**不**用 soft `gamma * predicted_continuation` 对每条健康边连续衰减（确定性/无终止环境下软折扣只注方差，见账本结论）。该基础语义不列入上方消融组件矩阵。母论文见 [算法纲领 §4.1 — base](../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)。

## 代码组织

- `src/rl/algo/my_zero/`：自包含 MyZero 库（**唯一** `*Zero` 实现）
- `recipe.rs`：按 `env_id` 内置组件开关（内部维护，文件名保留 recipe）
- `tests/baseline_matrix_bench.rs`：发版基线增量链（base → +cons → +recon → promoted，`--ignored` 手动）
- `examples/my_zero/*/main.rs`：薄示例（env + 训练契约 + opt-in 落盘）

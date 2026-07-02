# Phase 1 · 图像线立柱 实施报告（草稿,待用户检查,未提交）

> **状态**:进行中——spike 与基础设施已闭环,基准/消融数字待回填。
> **协议**:[Phase 1 计划](./rl_phase1_image_plan.md)(预注册)· 账本:[pong README](../../examples/my_zero/pong/README.md)。

## 一、S0 风险 spike:裁决 **GO(绿)**

载体 `src/rl/algo/my_zero/tests/spike_cnn_mcts_bench.rs`(手动档 bench,`just spike-cnn-mcts`;假图像纯计时,release+MKL,warmup 后中位数):

| 测项 | 实测 | 对照 |
|------|------|------|
| CNN root 前向 84²×4(~255k 参数) | 2.4 ms | — |
| CNN root 前向 42²×4 | 0.4 ms | — |
| MLP recurrent(g+f,latent 64) | 0.03 ms | — |
| conv recurrent(悲观臂 32×6×6) | 0.10 ms | — |
| **完整 acting 单步 flat 臂** sims=2/4/20/50 | **1.9 / 2.0 / 2.3 / 3.9 ms** | 预算 16–33 ms |
| 完整 acting 单步 conv 臂 sims=20/50 | 3.5 / 6.6 ms | 预算 16–33 ms |
| 训练 batch=8 fwd+bwd+step 84²/42² | 37.7 / 6.8 ms | 120k steps ≈ 1.3h |

**结论**:「CNN×sims」一级风险在本框架形态下不成立——CNN 只进 representation(每步 1 次),sims 放大的是轻量 recurrent。全部档位余量 8–14×,条款一不触发。真实瓶颈移位为**训练吞吐 × 消融轮次**。已回填 [CPU 风险 issue §四/§五](../../.issue/items/cpu_only_mcts_image_realtime_risk.md)。

## 二、S1 图像管线 + CNN 表征(已闭环,11 个新单测全绿)

- **GymEnv**:`ALE/*` 自动 `register_envs(ale_py)`;图像 obs 快路径(numpy astype+tobytes 整块拷贝,免 10 万元素逐个 extract)。
- **`obs_pipeline.rs`(新)**:BT.601 灰度 → 双线性 84² → [0,1] → 4 帧堆叠;`ObsAdapter`(与 `ActionAdapter` 对偶)自动检测图像 env;**内存纪律**:buffer 只存单帧(28KB/步),堆叠在 acting 滑窗 / 训练组装两处按需拼(episode 起点前向填充,两处语义逐 bit 一致)。
- **`network.rs`**:`ObsSpec::{Flat,Image}` + `ConvRepresentationNet`(stride-2 3×3 栈,空间压到 ≤7,min-max 归一同口径)+ `ReprNet` 枚举;`MyZeroModel::new` 旧签名零破坏。
- **`recipe.rs`**:`ALE/*` → image base 栈 = consistency ON + recon OFF + two-hot + raw obs(预注册,未 promote)。
- **顺手的必要修复**:非 reanalyze 训练路径改零克隆采样(`PreparedBatch::Borrowed`,RNG 序逐 bit 兼容)——图像单局数十 MB 时旧的整局 clone 是实测主瓶颈(6 局 profile:batch_prepare 67.7s + writeback 32.9s → 全归零,单局 wall 33s→16s)。
- 回归:`just test-filter rl` 全绿;CartPole 3-seed 哨兵 **12,519 / 8,643 / 9,826(中位 ~9.8k,3/3 达标)与冻结基线逐 bit 一致**——图像线改动对哨兵零扰动。

## 三、S2 基准(进行中)

- 预注册口径已锁(pong README):150 局/seed,门槛 3-seed 中位 best greedy ≥ −18。
- **事故记录**:首次 3-seed 跑 seed42 在 Ep53 崩溃(`Tensor::new` 形状不匹配,ndarray OutOfBounds)——复现跑(RUST_BACKTRACE=full)进行中,已加强 panic 消息(打印 data 长度与 shape)。
- （数字待回填）

## 四、S3 消融(载体就绪,待基准裁决后跑)

`pong_image_ablation_bench.rs`:recon(RECON_COEF pilot {1,4,16} → 3-seed)/ cons-off / hl-gauss 三臂,半额预算 75 局,协议已预注册。

## 四b、附带产出:商业实时游戏目标画像登记(2026-07-02)

以真实商业标的(匿名化)对表 Phase 1 产物,新开 [目标画像 issue](../../.issue/items/commercial_realtime_game_target_profile.md):
16:9 非方形输入 / 堆叠数可配 / Dict obs 双分支 / MultiDiscrete 适配 / POMDP 验证五项缺口登记(均不进 v0.26 关键路径);
关键洞察——私有侧「MCTS 实时延迟硬伤」旧否决基于 Python 栈算术,被本框架 spike 实测推翻,已链入纲领 §2.3。

## 五、待办

- [ ] Ep53 崩溃根因 + 修复
- [ ] 3-seed 基准数字 + 学习曲线 → 账本
- [ ] 三臂消融 → 账本
- [ ] 每局耗时随 buffer 占用增长(25s→68s 平台期)的定性(疑内存压力,measure-first 后决定是否挂优化需求)
- [ ] 文档同步(roadmap/AGENTS/CHANGELOG)——**全部改动不提交,待用户检查**

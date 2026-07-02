# Phase 1 · 图像线立柱实施计划(预注册协议)

> **定位**:[收口规划 §2](./rl_closure_plan.md#2-phase-1--图像线立柱--一级风险压测v026-下半) 的战术展开——spike 协议、基准门槛、A/B 协议在跑之前全部预注册,防事后挑数。
> **创建**:2026-07-02 · 纪律沿用 [roadmap §3.2](./rl_roadmap.md#32-改动纪律搬运--改进沿用):一次一项、3-seed、数字只进账本。
> **关联**:[CPU 风险 issue](../../.issue/items/cpu_only_mcts_image_realtime_risk.md)(spike 结果回填其 §四第一格)。

---

## S0 · 风险 spike(先行,唯一可改道节点,条款一)

**目的**:实测「CNN 前向 × sims」单步 wall-clock,对照实时预算触发线,产出**去/留/改道**三态裁决。

### 架构事实(决定测什么)

MyZero 现架构 = flat latent(`latent_dim=64`)+ MLP dynamics/prediction。CNN 只进 representation h:
- **acting 单步成本 = 1 × CNN root 前向 + sims × MLP recurrent 前向**(CNN 不被 sims 放大);
- 悲观路径(EfficientZero 忠实版空间 latent + conv dynamics)才是「CNN × sims」:**sims × conv recurrent**。
两条都测,风险上下界一次卡死。

### 测量矩阵(release + MKL,warmup ≥20 次后取 ≥100 次中位数)

| 项 | 配置 |
|---|---|
| CNN root 前向(h+f) | 输入 `[1,4,84,84]` 与 `[1,4,42,42]` 两档;栈 = conv3x3s2(4→32)-ReLU → conv3x3s2(32→64)-ReLU → conv3x3s2(64→64)-ReLU → flatten → linear→latent64(EfficientZero-lite,无残差) |
| MLP recurrent(g+f) | 现库口径:latent 64、action 6(Pong)、hidden 128 |
| conv recurrent(悲观臂) | 空间 latent `[1,32,6,6]`,dynamics = conv3x3s1(32+action平面→32)×2 + reward/continuation head |
| 组合单步 | `1×CNN + sims×MLP` 与 `sims×conv`,sims ∈ {2, 4, 20, 50} |
| 训练吞吐 | batch=8 CNN 前向+反向 一次 optimizer step wall-clock → 换算 100k env-steps 实验总时长 |

### 预注册判决规则(触发线 = 风险 issue 的 16–33ms/步)

- **去(绿)**:flat-latent 组合 @ sims=20 ≤ 33ms → 图像线全速推进,实时口径无需妥协。
- **留(黄)**:sims=20 超,但 ① sims≤4(Gumbel 少 sim acting)≤ 33ms,或 ② 实时不达但离线训练吞吐可行(100k env-steps 单实验 ≤ 24h)→ 推进,实时预算按 acting/reanalyze 解耦架构记账。
- **改道(红)**:sims=2 + 42×42 仍 > 33ms **且** 训练吞吐 > 数天/实验 → 触发条款一(载体收缩 Gomoku、planning-free 退路评估插入 v0.27)。

载体:`src/rl/algo/my_zero/tests/spike_cnn_mcts_bench.rs`(手动档 bench,假图像输入纯计时,不训练、不接 env;`just spike-cnn-mcts`)。

## S1 · 图像 obs 管线 + CNN representation(spike 绿/黄后)

1. **预处理**(复用/扩展 `src/vision/preprocess`):HWC f32 → 灰度(luminance)→ 双线性降采样 → [0,1] 归一 → **k=4 帧堆叠** → CHW 扁平。
2. **buffer 内存纪律**:图像 obs 逐帧存 **u8 单帧**(不存堆叠),采样期组装 stack(84×84×4 f32 直存 ≈ 113KB/position,100k steps ≈ 11GB,不可接受)。
3. **CNN representation 进 `network.rs`**:`RepresentationNet` 增加 CNN 变体(S0 同款栈),按 recipe 注入;dynamics/prediction 维持 flat-latent MLP(悲观臂仅当 S2 学不动时才考虑)。
4. **recipe**:`recipe.rs` 增图像 env 条目(见 S2 base 臂);`ModelSettings` 承载 obs 编码器选择。
5. 单测:CNN repr 前向/反向 shape、batch 等价、checkpoint 往返、预处理数值(灰度/降采样/堆叠各自 + 组合)。

## S2 · 预注册基准(Pong 类)

- **环境**:`ALE/Pong-v5`(gymnasium 默认口径:frameskip=4、sticky 0.25),预处理 = S1 管线(84×84 灰度 ×4 帧;若 spike 判黄则降 42×42)。
- **预算**:每 seed **≤ 120k env-steps**(Atari-100k 量级)或 wall-clock 触顶(单 seed ≤ 24h)先到为准;seeds = {42, 43, 44}。
- **门槛(预注册,非 SOTA)**:3-seed 中位 best greedy(10 局)≥ **−18**(随机 ≈ −20.7,−18 = 显著脱离随机),且学习曲线(≥5 个 eval 点)单调趋势上升。达标 = 图像支柱成立;不达标但曲线明确上升 = 记部分信号、按曲线裁决下一步;平直 = 记负结果 issue。
- **口径**:reward 不缩放(Pong ∈ [−21,21],support ±420 足够);账本新开 `examples/my_zero/pong/README.md`(或并入 my_zero README),口径列齐全。

## S3 · native 域复裁 A/B(基准通过后)

**image base 臂(预注册)**:consistency ON + reconstruction OFF + two-hot + raw [0,1] obs。

| 臂 | 变更(一次一项) | 备注 |
|---|---|---|
| A/B-1 recon | recon ON;coef 先 1-seed pilot {1, 4, 16} 挑最优,再 3-seed 定裁 | 图像域重标(Phase 0 已裁决:系数是话语权旋钮,跨量纲必重调);decoder 重建 **4 帧堆叠 obs**(28224 维,与库内 recon 通路一致;单帧重建留作后续变体) |
| A/B-2 consistency | consistency OFF | 图像是 cons 的 native 场景,CartPole 暧昧表现到此见分晓 |
| A/B-3 HL-Gauss | `hl_gauss` ON | Phase 0 CartPole 负结果的图像域复测(Simulus 计划 A1 复裁条款) |

A/B 预算 = 基准预算的**半额**(60k env-steps/seed)× 3 seeds;判读 = 中位曲线 + best greedy,数字进账本。

**Simulus 暂缓项触发点**(留观,不主动做):若 Pong 探索成实测瓶颈(曲线平直且诊断指向探索),ensemble JSD 内在奖励在此转正——唯一入口。

## S4 · 收尾

- 回填 [CPU 风险 issue §四](../../.issue/items/cpu_only_mcts_image_realtime_risk.md) 第一格(spike 数据 + 裁决)。
- 图像支柱账本落盘;组件矩阵图像列更新;`smoke-rl` 扩容(图像 smoke)留待发版关卡一并做。
- Phase 1 报告(本轮**不提交**,待检查)。

## 退出判据(= 收口规划 §2 原文)

图像支柱 3-seed 账本 + CPU 风险 issue 第一格勾选。

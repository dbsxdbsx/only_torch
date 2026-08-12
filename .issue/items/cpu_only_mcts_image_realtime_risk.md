---
status: active
created: 2026-07-02
updated: 2026-07-02 (CNN×MCTS spike 实测回填，一级风险裁决 GO)
owners: []
reviewers: []
---

# 一级风险：CPU-only × 图像 CNN × MCTS × 实时的结构性冲突

> **性质**：这不是 bug，是**路线级结构性风险**——MyZero 路线上唯一可能真正致命的约束冲突，显式记录、持续重估，不许藏在「感觉慢」里。
> **关联**：[RL 状态总览](../../.doc/design/rl_myzero_status.md)（战略目标、图像线 backlog、acting/reanalyze 解耦）

---

## 一、冲突是什么

项目铁约束 **CPU-only**（AGENTS.md 项目定位），而 v0.26 起的主推方向是**图像观测的商业游戏**（实时、环境不等 agent）。四个因素相乘：

| 因素 | 成本量级 |
|------|----------|
| CNN 表征前向（图像 obs） | 每次推理毫秒级起，远重于当前低维 MLP |
| MCTS 每步 sims 次 recurrent 前向 | ×20（当前默认 sims）|
| 实时 acting 预算 | 商业游戏一帧 ~16–33ms，环境不暂停 |
| 训练吞吐 | reanalyze / batch 训练同样吃 CNN 前向 |

学界治 MCTS wall-clock 的方案（TransZero ~11×、SpeedyZero、ReZero）**全部依赖 GPU 并行**——CPU-only 下不可搬运。

## 二、为什么现在不阻塞

1. **象棋线不受此约束**：盘面是低维离散张量，MLP/小 CNN + MCTS 在 CPU 上可行（AlphaZero 系树搜索本身是 CPU 友好的串行逻辑）。
2. **acting / reanalyze 解耦**已定为战略架构（纲领 §2.3）：实时侧用 policy 先验或 Gumbel 少 sim（论文保证 sims=2 仍有 policy improvement），重搜索全部离线化——把实时预算问题转化为离线吞吐问题。
3. **规划的主要价值在训练期**（Hamrick 2020 / De Vries 2023）：部署期砍掉搜索损失有限。

## 三、缓解手段清单（推进图像线时逐项压测）

- Gumbel 少 sim acting（sims=2–4）+ 完整搜索仅离线 reanalyze
- 小 CNN（EfficientZero Atari 用的也只是小卷积栈）+ 帧降采样 / 灰度 / 帧堆叠
- `predict_batch` 批量叶子评估摊销前向（接缝已留）
- 训练/部署非对称：训练期不限 wall-clock，部署期只跑 policy 网络
- BLAS（MKL）已接通；必要时评估 int8 / 量化推理

## 三b、有据可查的路线级退路：planning-free 世界模型（2026-07-02 补）

若上述缓解逐项压测后「图像 + 实时 + MCTS」仍不可行，**免规划世界模型家族是实证退路**：Simulus（arXiv 2502.11537，[阅读日志](../../.doc/paper/reading_log.md)）证明 planning-free WM 在 Atari-100K 达人类水平 IQM（该口径首个），样本效率打平多数带规划方法；acting 期只跑 policy 网络、零 MCTS 开销，正对本 issue 的实时预算约束。且其可拆组件（RaC、loss 优先回放，见[RL 状态总览 · 处方表](../../.doc/design/rl_myzero_status.md#34-处方表跨域路由规则)）在 MCTS / planning-free 两条路线上通用——现在做不白做，切换成本被压低。

## 四、触发重估的条件

- [x] **图像 acting spike（2026-07-02 已裁决：GO 绿）**：`src/rl/algo/my_zero/tests/spike_cnn_mcts_bench.rs`（手动档 bench，`just spike-cnn-mcts`；release+MKL，warmup 后中位数，协议见该文件头注释）实测：
  - 组件：CNN root（84²×4，EfficientZero-lite 栈，~255k 参数）**2.4ms**；42² 降档 **0.4ms**；MLP recurrent **0.03ms**；conv recurrent（悲观臂空间 latent 32×6×6）**0.10ms**
  - 完整 acting 单步（真实 `mcts_search` 含树簿记，flat-latent 臂）：sims=2 → **1.9ms** · sims=20 → **2.3ms** · sims=50 → **3.9ms**；悲观 conv-recurrent 臂 sims=20 → 3.5ms、sims=50 → 6.6ms——**两臂全部 sims 档位均远低于 16–33ms 实时预算线**
  - 训练吞吐：batch=8 CNN fwd+bwd+Adam step 84² **37.7ms** / 42² 6.8ms → 120k env-steps 单实验（1 train/step 悲观口径）≈ **1.3h**，离线吞吐无忧
  - **裁决：GO**——「CNN 前向 × sims」在本框架架构下不成立为一级风险：CNN 只进 representation（每步 1 次），sims 放大的是轻量 recurrent；条款一改道预案不触发，图像线全速推进。实测中 wall-clock 主耗在**训练步**（37.7ms × replay ratio）而非 acting——风险从「实时预算」移位为「训练吞吐 × 消融轮次」，量级可控
- [ ] 若离线训练吞吐使一次实验 > 数天：重估「CPU-only 是否对图像线豁免」（例如允许可选 GPU feature，项目其余部分保持 CPU-only）——按 spike 数据当前不触发
- [ ] 象棋线不受本 issue 约束，正常推进

## 五、诚实结论（2026-07-02 spike 后更新）

原判断「CPU-only 与图像 + 实时 + MCTS 同时全量成立的概率不高」经 spike 实测**证伪于本框架的具体形态**：flat-latent MyZero（CNN 仅在 h，g/f 为 MLP）下单步 acting 仅 ~2–4ms，距 16–33ms 预算余量 8–14×。该结论的适用边界：latent 64 / sims ≤ 50 / 84² 灰度输入 / 小卷积栈；若未来上大 CNN（残差塔）或空间 latent conv dynamics，需按本 issue §三缓解清单重新压测（悲观臂 6.6ms @ sims=50 仍留有余量）。学习效果（能否在此预算下学会 Pong）由 [pong 账本](../../examples/my_zero/pong/README.md) 另行裁决——本 issue 只管 wall-clock 结构性风险。

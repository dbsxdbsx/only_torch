---
status: active
created: 2026-07-02
updated: 2026-07-02
owners: []
reviewers: []
---

# 一级风险：CPU-only × 图像 CNN × MCTS × 实时的结构性冲突

> **性质**：这不是 bug，是**路线级结构性风险**——MyZero 路线上唯一可能真正致命的约束冲突，显式记录、持续重估，不许藏在「感觉慢」里。
> **关联**：[算法纲领 §2.3](../../.doc/design/my_zero_algorithm_vision.md#23-战略目标与优先轴2026-07-01-定稿)（战略目标与 acting/reanalyze 解耦）· [RL 路线图 §5](../../.doc/design/rl_roadmap.md#5-v026-方向2026-07-01-战略转向定稿)（v0.26 P0：CNN 图像表征）

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

## 四、触发重估的条件

- [ ] v0.26 图像基准（Atari-100k 类）首个 spike：实测「CNN 前向 × sims」单步 wall-clock，对照 16–33ms 实时预算
- [ ] 若离线训练吞吐使一次实验 > 数天：重估「CPU-only 是否对图像线豁免」（例如允许可选 GPU feature，项目其余部分保持 CPU-only）
- [ ] 象棋线不受本 issue 约束，正常推进

## 五、诚实结论（当前）

CPU-only 与「图像 + 实时 + MCTS」**同时全量成立的概率不高**；本 issue 的目的是让将来做取舍时（砍实时性 / 砍图像分辨率 / 放宽 CPU-only）有据可查，而不是走到跟前才发现。

# MyZero · Pendulum-v1

> [← 返回 MyZero 总览](../README.md)｜组件裁决见总览「组件 × 环境 效果矩阵」

- **规格**：纯连续（1 维）· 门禁 greedy return ≥ -200 · 当前诊断栈 B=7 连续候选 · 默认 `gamma=0.997`，`sims=20`
- **状态**：**诊断中**——当前 best greedy 约 −942，仍在失败区间（门禁 −200），属「还没学会」。先查可学习性，暂不对组件下裁决
- **定位**：**判别环境**——CartPole 分辨不出的组件（value_prefix / completedQ / Gumbel-root）在此见真章

## 运行

```bash
cargo run --example my_zero_pendulum --release

# SMOKE 管线验证
SMOKE=1 cargo run --example my_zero_pendulum

# P0 value / target / search 诊断（训练结束后追加诊断输出）
DIAG=1 cargo run --example my_zero_pendulum --release

# 阶段耗时汇总（self-play / batch prepare / train step / eval）
PROFILE=1 cargo run --example my_zero_pendulum --release

# 临时覆盖 n-step bootstrap（诊断用）
TD_STEPS=5 cargo run --example my_zero_pendulum --release
```

算法配方由库内 `recipe.rs` 注入。Pendulum 当前复用 CartPole 的 `consistency + reconstruction + Sampled` 作为**诊断栈**（不是已验收裁决）；示例只显式写 Pendulum 的 `reward_scale(0.1)` 与训练契约，连续动作 B=7 由库内默认动作方案注入。

基础 transition 语义：Pendulum-v1 通常以 time-limit truncation 结束，当前不会把 200 步截断误当 terminal；truncation 仍 `continuation=1` 并 bootstrap，只有 MDP 真终止才 `continuation=0`。

## 实测（seed=42，门禁 −200）

> ⚠️ 下列数字都落在**失败区间**（远未达 −200），所以「观察」列只是诊断记录，**不是对组件的 ✅/➖/❌ 裁决**——在还没学会的任务上比组件好坏没有判别力。

### sims=50

| 配置 | greedy eval（终） | env-steps | wall | 观察（非裁决） |
|------|------------------|-----------|------|------|
| +consistency（Step 0，ep~350 停） | 最好 −353（ep325，方差大）；终 −828 | ~70k | — | 失败区间，仅偶发尖峰 |
| +consistency +completedQ（Step 1，满 600 ep） | **−866.6** | 120k | 1496s | 失败区间，无判别力 |

### sims=16（低 sims A/B）

| 配置 | greedy eval（终） | env-steps | wall | 观察（非裁决） |
|------|------------------|-----------|------|------|
| +consistency | **−1287.1** | 120k | 704s | 失败区间 |
| +consistency +completedQ | −1440.5 | 120k | 710s | 失败区间 |

> **解读**：四组都在失败区间，**不能据此给 consistency / completedQ 下中性裁决**（那是无信号，不是中性）。CartPole 的 ✅ 不受影响。下一步先做可学习性诊断（见下方 sweep），确认能不能把分数拉出失败区间，再决定是继续离散调参还是提前上 Gumbel-root / 连续候选。

### 当前诊断栈（consistency + reconstruction + Sampled，B=7，sims=20）

| 配置 | best greedy eval | env-steps | wall | 观察（非裁决） |
|------|------------------|-----------|------|------|
| +cons+recon+Sampled | **−942.2** @ ep575 | 120k | ~495s | loss 能降到 ~5–7，但 greedy 策略仍未入门 |
| +cons+recon+Sampled · `TD_STEPS=5` · continuation backbone | **−1085.2** @ ep200 | 120k | 496.5s | 仍失败；final greedy −1252.2，DIAG 显示 value 链路继续压扁 |

这组与 CartPole 已验收栈保持一致，用于确认 Sampled 机制在连续动作路径上的 plumbing；它仍在失败区间，不能裁决 consistency / reconstruction / Sampled 对 Pendulum 是否有效。

value_prefix 也将在此**重测**：CartPole 上 ❌ 是因 reward 恒 +1 退化为步数计数器，
而 Pendulum 的连续 reward 才是它的判别场（见 [CartPole 详情](../cartpole/README.md)）。

## 失败诊断 sweep（2026-06-17，"先查清再改"）

> **动机**：上述 sims=50/16 的实测全部落在 −353 ~ −1445 区间，门禁 −200，**所有配置都在"失败区间"**。在模型根本没学会的任务上做组件消融，➖ 裁决全是噪声——不足以回答"consistency 是不是中性"。本 sweep 的唯一目标：用最小代价回答 **"MyZero 的离散化方案到底能不能学会 Pendulum？"**，并把根因锁定到具体旋钮。
>
> **只诊断，不动算法语义**（不加 Gumbel、不改 PUCT、不改 consistency 的 stop-grad 实现）。

### 判准门限（看 greedy eval 轨迹最高点，排除 avg_R/loss）

| 轨迹最高点 | 含义 | 后续动作 |
|-----------|------|---------|
| **−800 ≈ "质的飞跃"** | 该旋钮**解锁学习**，方法通 | 进 A/B 正式定性（base vs 该旋钮）|
| **−600** | 接近门禁一半，强解锁 | 同上 |
| **持续在 −1000 附近徘徊** | 失败区间，该旋钮无效 | 排除该旋钮 |

### 决策树

```
所有臂都在 [-1000 附近徘徊]?
├─ 是 → 确认"纯离散化 MCTS 在 Pendulum 上、sims≤50 约束下走不通"
│       → 结论：必须提前上 Gumbel-root / 连续采样候选（矩阵 Gumbel 行升级为关键路径）
└─ 否，某臂触达 [-800, -600]：
        该臂 vs base 的唯一差异旋钮 = 头号根因
        → 在该旋钮上做 base vs 该旋钮的 A/B，正式定性 consistency / CQ 的 ✅/➖/❌
```

### 已排除的怀疑点（读代码 + 数值校验）

| 怀疑点 | 状态 | 依据 |
|--------|------|------|
| value support 溢出 | ✅ **排除** | support 变换域 [−20,20] 覆盖原始 value ±440；Pendulum 最差 value target ≈ −316（`h=−17.0`），不溢出 |
| reward head 分辨率（RSCALE）| ❌ **已证伪** | 臂 A（RSCALE 0.5）反而更差 + dynamics 诊断显示 reward head 健康（预测 std 1.86 ≈ 真实 1.87）→ 非病根 |
| sims 不够（9 动作 × 50 sims）| ⏸ **硬约束不碰** | MuZero/EZ 上限就是 50 sims；用户领域知识：不靠加算力解决 |
| CONSISTENCY 实现（同 projector + 无 stop-grad）| ⚠️ **环境敏感** | 偏离 SimSiam 设计有塌缩风险，但 CartPole 开 cons 反而学得好（loss 9.6→0.7）→ 非致命 bug，须 base 对照定性 |
| **value head 坍缩** | ❗ **确认根因（2026-06-18）** | dynamics 诊断：value 预测 std 26 vs 真实 MC return std 175，坍缩成常数 → MCTS Q 无区分 → 搜索失效、loss 卡 17。详见 [失败诊断 issue](../../../.issue/items/pendulum_failure_diagnosis.md) |

### 实测（seed=42，MAX_EP=300，单臂对 base 只动一个变量）

| 臂 | 唯一差异 | 假设 | eval 轨迹最高点 | 观察 |
|----|---------|------|----------------|------|
| **base** | 全关 | 失败区间基线 | −929.6（env_steps≈80k） | 失败区间，loss 长期 ~17 不降 |
| A: +RSCALE | `RSCALE=0.5` | reward 分辨率 5× | −1134.7（env_steps≈55k） | 失败区间，比 base 略差 → reward 分辨率非主因 |
| B: +NUM_ACTIONS | `NUM_ACTIONS=25` | 0 附近加密 | 未跑 | — |
| C: +consistency | `CONSISTENCY=1` | 隔离 cons 真实作用 | 未跑 | — |

> **读法**：每臂只相对 base 改一个变量，触达 [−800,−600] 的那个臂的差异旋钮就是头号根因。原始训练日志不入库，关键数值已摘录于上（与 [失败诊断 issue](../../../.issue/items/pendulum_failure_diagnosis.md) 一致）。


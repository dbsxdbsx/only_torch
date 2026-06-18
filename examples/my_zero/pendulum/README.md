# MyZero · Pendulum-v1

> [← 返回 MyZero 总览](../README.md)｜组件裁决见总览「组件 × 环境 效果矩阵」（含 ➖ 中性）

- **规格**：纯连续（1 维）· 门禁 greedy return ≥ -200 · 9 档离散化 ∈[-2, 2] · 默认 `gamma=0.99`，`sims=50`
- **状态**：sims=50 单 seed 未达标（➖）；**低 sims=16 A/B 进行中**（对齐 CartPole completedQ 判别实验）
- **定位**：**判别环境**——CartPole 分辨不出的组件（value_prefix / completedQ / Gumbel-root）在此见真章

## 运行

```bash
# +consistency（当前 Pendulum 基线栈）
EZ_CONS=1 cargo run --example my_zero_pendulum --release

# +consistency +completedQ
EZ_CONS=1 CQ=1 cargo run --example my_zero_pendulum --release

# 低 sims A/B（对齐 CartPole completedQ 判别点）
EZ_CONS=1 SIMS=16 cargo run --example my_zero_pendulum --release
EZ_CONS=1 CQ=1 SIMS=16 cargo run --example my_zero_pendulum --release

# 多 seed（seed 42/43/44 取中位数）
SEEDS=3 EZ_CONS=1 CQ=1 SIMS=16 cargo run --example my_zero_pendulum --release

# SMOKE 管线验证
SMOKE=1 cargo run --example my_zero_pendulum
```

> 支持：`EZ_CONS / EZ_VP / EZ_TARGET / EZ_SVE / CQ / CQ_SCALE / SIMS / SEEDS / SMOKE / GAMMA / MAX_EP`

## 实测（seed=42，门禁 −200）

### sims=50

| 配置 | greedy eval（终） | env-steps | wall | 裁决 |
|------|------------------|-----------|------|------|
| +consistency（Step 0，ep~350 停） | 最好 −353（ep325，方差大）；终 −828 | ~70k | — | ➖ 未达标、无害 |
| +consistency +completedQ（Step 1，满 600 ep） | **−866.6** | 120k | 1496s | ➖ 相对 cons-only 无可见增益 |

### sims=16（低 sims A/B，对齐 CartPole completedQ 判别点）

| 配置 | greedy eval（终） | env-steps | wall | 裁决 |
|------|------------------|-----------|------|------|
| +consistency | **−1287.1** | 120k | 704s | ➖ 未达标 |
| +consistency +completedQ | −1440.5 | 120k | 710s | ➖ 无增益（略差，非 ❌） |

> **解读**：Pendulum 上 completedQ **在 sims=50 与 sims=16 均未带来增益**（➖），与 CartPole 低 sims 大胜形成对比——**completedQ 不能毕业为全局默认**，CartPole 保留 ✅，Pendulum 标 ➖。瓶颈更可能在离散化 / dynamics / 连续控制整体，下一步考虑 Gumbel-root、加大离散档数或 value_prefix 翻案。

value_prefix 也将在此**重测**：CartPole 上 ❌ 是因 reward 恒 +1 退化为步数计数器，
而 Pendulum 的连续 reward 才是它的判别场（见 [CartPole 详情](../cartpole/README.md)）。

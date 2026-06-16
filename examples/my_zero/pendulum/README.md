# MyZero · Pendulum-v1

> [← 返回 MyZero 总览](../README.md)｜组件裁决（✅/❌/⏳）见总览的「组件 × 环境 效果矩阵」

- **规格**：纯连续（1 维）· 门禁 return ≥ -200 · 离散化候选（骨架）→ Gumbel 连续搜索（目标）
- **状态**：骨架已建 + 管线通；待提升至达标
- **定位**：当前的**判别环境**——CartPole 分辨不出的组件（value_prefix / completedQ / Gumbel-root）在这里见真章

## 运行

```bash
# 连续动作 → 离散化候选（骨架，待 Gumbel 提升）
EZ_CONS=1 cargo run --example my_zero_pendulum --release

# 多 seed（seed 42/43/44 取中位数）
SEEDS=3 EZ_CONS=1 cargo run --example my_zero_pendulum --release

# SMOKE 管线验证
SMOKE=1 cargo run --example my_zero_pendulum
```

> 注：Pendulum 示例当前支持 `EZ_CONS / EZ_VP / EZ_TARGET / EZ_SVE / SIMS / SEEDS / SMOKE`；
> `CQ`（completedQ）尚未接入本示例（故总览矩阵 Pendulum 列为 ⏳）。

## 实测

待补——completedQ / Gumbel-root 在 Pendulum 的低 sims A/B 与多 seed 结果将记录于此。

value_prefix 也将在此**重测**：CartPole 上 ❌ 是因 reward 恒 +1 退化为步数计数器，
而 Pendulum 的连续 reward 才是它的判别场（见 [CartPole 详情](../cartpole/README.md)）。

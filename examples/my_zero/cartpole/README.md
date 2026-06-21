# MyZero · CartPole-v1

> [← 返回 MyZero 总览](../README.md)

离散 2 动作 · 门禁 **greedy eval ≥ 475** · seed=42 · sims=50 · γ=0.997

组件组合（consistency、reconstruction 等）由库内 [`recipe.rs`](../../../src/rl/algo/my_zero/recipe.rs) 按 `CartPole-v1` 自动注入；示例只写训练契约。论文全称见 [总览 · 组件文献对照](../README.md#组件文献对照读全称用此表)。

## 运行

```bash
cargo run --example my_zero_cartpole --release
```

训练日志：**`len`** = 本局步数；**`total_env_steps`** = 累计真实环境交互（首要评价指标）。

---

## 消融结论（seed=42 · release）

**判据**：greedy(temp=0) eval 均值 ≥ 475；`avg_R` 仅作学习进度参考，不作成功判据。

| 配置 | avg_R @ep250 | greedy 达标 | total_env_steps | wall-clock | 备注 |
|------|-------------|------------|-----------------|------------|------|
| base（组件全关） | 80.3 | 未在 ep250 达标 | — | — | 2026-06-16 |
| +consistency | 111.6 | **500.0** @ ep325 | **28,996** | 541s | 2026-06-20 复测 ✅ |
| **+consistency +reconstruction**（**当前内置**） | **50.8** | **500.0** @ ep275 | **11,682** | **183.9s** | 2026-06-21 ✅；训后 eval×10 mean=500 |
| +cons+recon · **sims=20** · visit | — | **500.0** @ ep250 | **12,186** | 80s | 2026-06-21；低 sim 基线几乎不降 |
| +cons+recon · **sims=50** · +completedQ | ~80 @ ep400 | **500.0** @ ep575 | **34,490** | 381s | 2026-06-21；vπ 已修；≈基线 **3×** steps |
| +cons+recon · **sims=20** · +completedQ | — | **500.0** @ ep450 | **30,409** | 180s | 2026-06-21；仍 ≈同 sim visit 基线 **2.5×**；**无低 sim 红利** |
| +consistency +reanalyze +写回 | ~12 | **9.4**（ep200 仍随机） | 未达标 | — | 2026-06-20 ❌ 见 [issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) |

**结论**：

- CartPole 当前内置 **consistency + reconstruction**（文献见总览对照表：EfficientZero 系 SimSiam 一致性 + Scholz et al. 2021 观测重建 loss）。
- 相对仅 consistency：env-steps **28,996 → 11,682（−60%）**，wall-clock **541s → 184s**，greedy 仍满分、无回归。
- **completedQ 2×2 消融（seed=42 · release）**：visit 目标在 sims=50/20 均 ~**12k** steps 达标；+completedQ 在 sims=50/20 分别 ~**34.5k / 30.4k** steps（均能学会，但样本效率稳定差 **2.5–3×**）。**低 sim 未缩小差距** → CartPole **不 promote completedQ**；组件保留供 Pendulum / Gumbel-root 判别。
- reanalyze 写回已入库但 **暂不开启**。

---

## 默认超参

`sims=50` · `gamma=0.997` · `k_unroll=5` · `td_steps=50` · `lr=0.02` · `train_batch_size=8` · `trains_per_episode=8`

组件 loss 权重（写死在 `loss.rs` / `runner.rs`，非用户可调）：consistency coef **2.0** · reconstruction coef **1.0**（Scholz et al. 2021 默认 \(l_g\) 权重）。

## 参照（跨算法）

| 算法 | CartPole-v1 到 greedy 500 约需 env-steps |
|------|------------------------------------------|
| **MyZero**（consistency + reconstruction） | **~11.7k** |
| MyZero +consistency only | ~29k |
| PPO | ~82k |
| SAC | ~105k |

CartPole 上 model-based 样本效率领先 model-free 基线。

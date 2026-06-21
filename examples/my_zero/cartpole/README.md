# MyZero · CartPole-v1

> [← 返回 MyZero 总览](../README.md)

离散 2 动作 · 门禁 **greedy eval ≥ 475** · seed=42 · **sims=20**（默认）· γ=0.997

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
| **+consistency +reconstruction**（**当前内置 · sims=20**） | — | **500.0** @ ep250 | **12,186** | **80s** | 2026-06-21 ✅；默认 sim 自 v0.25 起为 20 |
| +cons+recon · sims=10 | — | **500.0** @ ep875 | **16,152** | **~125s** | 2026-06-21；样本效率差于 sims=20，不 promote |
| +cons+recon · sims=15 | — | **500.0** @ ep500 | **26,306** | **~167s** | 2026-06-21；比 sims=10/20 更差，不 promote |
| +cons+recon · sims=50（旧默认） | **50.8** | **500.0** @ ep275 | **11,682** | **183.9s** | 2026-06-21；env-steps 略优，wall-clock 约 2.3× |
| +cons+recon · sims=50 · +completedQ | ~80 @ ep400 | **500.0** @ ep575 | **34,490** | 381s | CartPole ❌ promote |
| +cons+recon · sims=20 · +completedQ | — | **500.0** @ ep450 | **30,409** | 180s | CartPole ❌ promote |
| +consistency +reanalyze +写回 | ~12 | **9.4**（ep200 仍随机） | 未达标 | — | 2026-06-20 ❌ 见 [issue](../../.issue/items/my_zero_reanalyze_cartpole_regression.md) |

**结论**：

- CartPole 当前内置 **consistency + reconstruction**，**默认 sims=20**：env-steps ~**12.2k**（较 sims=50 多 ~4%），wall-clock **~80s**（约 **−56%**）；**sims=10/15** 扫参均明显差于 20（~16.2k / ~26.3k steps），仅作参考。
- 相对仅 consistency：env-steps **28,996 → 12,186（−58%）**，greedy 仍满分。
- **completedQ** 2×2 消融：visit ~12k steps；+completedQ ~30–34k steps → CartPole **不 promote**。
- reanalyze 写回已入库但 **暂不开启**。

---

## 默认超参

`sims=20` · `gamma=0.997` · `k_unroll=5` · `td_steps=50` · `lr=0.02` · `train_batch_size=8` · `trains_per_episode=8`

组件 loss 权重（写死在 `loss.rs` / `runner.rs`，非用户可调）：consistency coef **2.0** · reconstruction coef **1.0**（Scholz et al. 2021 默认 \(l_g\) 权重）。

## 参照（跨算法）

| 算法 | CartPole-v1 到 greedy 500 约需 env-steps |
|------|------------------------------------------|
| **MyZero**（consistency + reconstruction · sims=20） | **~12.2k** |
| MyZero +consistency only | ~29k |
| PPO | ~82k |
| SAC | ~105k |

CartPole 上 model-based 样本效率领先 model-free 基线。

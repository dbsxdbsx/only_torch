# MyZero · CartPole-v1

> [← 返回 MyZero 总览](../README.md)

离散 2 动作 · 门禁 **greedy eval ≥ 475** · seed=42 · sims=50 · γ=0.997

算法配方（consistency 等）由库内 [`recipe.rs`](../../../src/rl/algo/my_zero/recipe.rs) 按 `CartPole-v1` 自动注入；示例只写训练契约。

## 运行

```bash
cargo run --example my_zero_cartpole --release
```

训练日志：**`len`** = 本局步数；**`total_env_steps`** = 累计真实环境交互（North Star 指标）。

---

## 消融结论（2026-06-20，seed=42）

**判据**：greedy(temp=0) eval 均值 ≥ 475；`avg_R` 仅作学习进度参考，不作成功判据。

| 配置 | avg_R @ep250 | greedy 终值 | 达标 total_env_steps | 备注 |
|------|-------------|------------|---------------------|------|
| base（组件全关） | 80.3 | — | 未在 ep250 达标 | 2026-06-16 |
| **+consistency**（当前 recipe） | **111.6** | **500.0** | **28,996**（ep325，380s） | 2026-06-20 复测 ✅ |

**结论**：CartPole recipe 仅 promote **consistency**；其余组件在内部消融，未写入 recipe。

---

## 默认超参

`sims=50` · `gamma=0.997` · `k_unroll=5` · `td_steps=50` · `lr=0.02` · `batch_games=8` · `trains_per_episode=8`

## 参照（跨算法，2026-06-16）

MyZero base ~17k env-steps 到 500；PPO ~82k、SAC ~105k（model-free）。CartPole 上 model-based 样本效率领先。

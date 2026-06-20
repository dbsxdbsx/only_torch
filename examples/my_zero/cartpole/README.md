# MyZero · CartPole-v1

> [← 返回 MyZero 总览](../README.md)

离散 2 动作 · 门禁 **greedy eval ≥ 475** · seed=42 · sims=50 · γ=0.997

**示例默认**：`main.rs` 已开 **+consistency**（库默认仍是 base，组件全关）。

## 运行

```bash
# 当前示例（+consistency，sim=50）
cargo run --example my_zero_cartpole --release

# base 消融：临时去掉 main.rs 的 `.consistency()` 再跑
```

训练日志：**`len`** = 本局步数；**`total_env_steps`** = 累计真实环境交互（North Star 指标）。

---

## 消融结论（2026-06-20，seed=42）

**判据**：greedy(temp=0) eval 均值 ≥ 475；`avg_R` 仅作学习进度参考，不作成功判据。

### consistency（主测项）

| 配置 | avg_R @ep250 | greedy 终值 | 达标 total_env_steps | 备注 |
|------|-------------|------------|---------------------|------|
| base（组件全关） | 80.3 | — | 未在 ep250 达标 | 2026-06-16 |
| **+consistency** | **111.6** | **500.0** | **28,996**（ep325，380s） | 2026-06-20 复测 ✅ |

**结论**：+consistency 明显加速早期学习（ep250：80 → 112），最终 greedy 打满 500。样本效率有 run 间方差（同 seed 历史报 ~18k steps），待多 seed 再定稿。

### 其他旋钮（仅粗测，未写入示例默认）

在 **+consistency、SIMS=16、CQ 开** 下试过不同 `c_scale`（0.02 / 0.5 / 1.0），**均未在合理步数内稳定达标**（例如 0.5 超 8 万步、1.0 超 10 万步 best greedy 仍 <475）。CartPole 上 **consistency 单独已够**，其余组件待 clean A/B 后再写进文档。

---

## 默认超参

`sims=50` · `gamma=0.997` · `k_unroll=5` · `td_steps=50` · `lr=0.02` · `batch_games=8` · `trains_per_episode=8`

## 参照（跨算法，2026-06-16）

MyZero base ~17k env-steps 到 500；PPO ~82k、SAC ~105k（model-free）。CartPole 上 model-based 样本效率领先。

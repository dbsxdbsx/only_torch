# MyZero · CartPole-v1

> [← 返回 MyZero 总览](../README.md)｜组件裁决见总览「组件 × 环境 效果矩阵」

- **规格**：离散（2 动作）· 门禁 greedy eval ≥ 475 · 默认 `num_simulations=50`，`gamma=0.997`
- **状态**：✅ +consistency +completedQ 回归哨兵（greedy 满分）；下一步 Gumbel-root（判别环境验证）

## 运行

```bash
# base（sim=50）
cargo run --example my_zero_cartpole --release

# +consistency
CONSISTENCY=1 cargo run --example my_zero_cartpole --release

# +consistency +completedQ · sims=16（推荐回归命令）
CONSISTENCY=1 CQ=1 SIMS=16 cargo run --example my_zero_cartpole --release

# 多 seed（42/43/44 取中位数）
SEEDS=3 CONSISTENCY=1 cargo run --example my_zero_cartpole --release
```

落盘 opt-in：`.save_model_when_eval(path)` 或 `SAVE_MODEL=path`（示例 `main.rs` 已配置）。API 细节见 [总览 · 训练与推理](../README.md#训练与推理生命周期)。

## 关键超参（默认）

| 参数 | 值 |
|------|-----|
| `num_simulations` | 50（+CQ 可降至 16） |
| `gamma` | 0.997 |
| `k_unroll` / `td_steps` | 5 / 50 |
| `lr` | 0.02 |
| `batch_games` / `trains_per_episode` | 8 / 8 |

## 跨算法 Benchmark（2026-06-16）

| 算法 | greedy | env-steps | wall | 备注 |
|------|--------|-----------|------|------|
| PPO | 484.6 | 81,920 | 107s | model-free |
| SAC | 487.0 | 104,982 | 186s | model-free |
| MyZero base（sim=50） | 500.0 | 17,260 | 287s | model-based，env-step 最少 |

## 组件消融（seed=42）

**Ep250 快照（sim=50）**

| 配置 | avg_R @ep250 | 观察 |
|------|-------------|------|
| base | 80.3 | 未达标 |
| **+consistency** | **97.1** | ✅ 显著加速 |
| +consistency +value_prefix | 15.5 | ❌ CartPole 有害（≠ 组件全局坏，见总览脚注 ᵃ） |

**+consistency +completedQ · sims=16**

| 配置 | greedy | env-steps to 475 | wall | 结论 |
|------|--------|------------------|------|------|
| visit-count | 299.5 | 未达标 | 248s | ❌ |
| CQ `c_scale=0.1` | 138.5 | 未达标 | 242s | ❌ 过锐 |
| **CQ `c_scale=0.02`** | **500.0** | **57,420** | **257s** | ✅ 2026-06-20 单 seed 复测 |
| visit-count @ sims=50 | 500.0 | ~17k | ~287s | 参照基线 |

> 2026-06-16 曾报 3-seed 中位 **5,141 steps / 39.6s**，与本机 2026-06-20 复测偏差大；**greedy 仍稳定 500**，样本效率待 `SEEDS=3` 重测后再定稿。

**+consistency · sim=50 · 多 seed**

| seed | greedy | env-steps to 500 |
|------|--------|------------------|
| 42 | 500.0 | 18,159 |
| 43 | 500.0 | 13,684 |
| 中位 | **500.0** | **~16k** |

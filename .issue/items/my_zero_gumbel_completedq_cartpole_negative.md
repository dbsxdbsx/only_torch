---
status: active
created: 2026-06-21
updated: 2026-06-21
owners: []
reviewers: []
---

# MyZero · completedQ / Gumbel-root：CartPole 消融失败（暂缓 promote）

> **状态**：active —— 库内已实现 `completed_q_target` / `GumbelPolicy` + bench 用例；CartPole recipe **保持关**，当前基线为 **consistency + reconstruction · PUCT · sims=20**（~12.2k env-steps）。
> **关联**：[CartPole README](../../examples/my_zero/cartpole/README.md) · [MyZero 总览](../../examples/my_zero/README.md) · [算法纲领 §5.1](../../.doc/design/my_zero_algorithm_vision.md#51-组件与方向)
> **代码**：`src/rl/mcts/gumbel.rs` · `search_policy.rs` · `target.rs`（completedQ）· `tests/completed_q_cartpole_bench.rs`

---

## 一、现象（seed=42 · release · cons+recon）

**判据**：greedy(temp=0) eval ≥ 475；对照基线 visit target + PUCT @ sims=20 → **12,186 env-steps @ ep250**。

### completedQ（训练侧策略 target）

| sims | target | 达标 | total_env_steps | 结论 |
|------|--------|------|-----------------|------|
| 20 | visit（基线） | ep250 | **12,186** | ✅ |
| 20 | +completedQ | ep450 | **30,409** | ❌ ~2.5× 更慢 |
| 50 | visit | ep275 | 11,682 | ✅ |
| 50 | +completedQ | ep575 | **34,490** | ❌ ~3× 更慢 |

### Gumbel-root（搜索侧 · visit target）

| sims | search | 跑至 | greedy 峰值 | total_env_steps @ 终止 | 结论 |
|------|--------|------|-------------|------------------------|------|
| 20 | Gumbel | ep1725（手动停） | **123.0** | ~142k | ❌ 未收敛 |
| 10 | Gumbel | ep1800+（仍在跑时可停） | **154.0** | ~101k | ❌ 未收敛 |

同配置 **PUCT + visit @ sims=10** 仍可在 **16,152 env-steps @ ep875** 达标 → Gumbel 不是「sim 少所以慢」，而是**搜索机制在该环境/regime 下有害**。

**结论**：CartPole 上 **completedQ 与 Gumbel-root 均不 promote**；与 reanalyze 类似，**代码保留、recipe 关闭**。

---

## 二、机制假设（待更大动作空间验证）

Gumbel MuZero（Danihelka et al. 2022）主要解决 **`|A| > n`**（动作多、模拟少）时根节点 coverage 与 policy improvement。CartPole **`|A|=2`，`n=10/20`** → **`n ≫ |A|`**：

- Gumbel-Top-k / Sequential Halving **无筛选/淘汰**（两动作始终在 active set）；
- 根 sim 分配与出动作规则 **偏离** 已调通的 PUCT + visit target；
- completedQ 与 visit 行为 **不对齐**，在 trivial 2 动作 bandit 上放大样本浪费。

**不据此否定实现正确性**；应在 **`|A| ≫ n`** 的环境（Pendulum 离散化、Platform、Atari 类）再验收。

---

## 三、已实现（勿删）

| 项 | 状态 |
|----|------|
| `GumbelPolicy` + `RootScheduler::on_search_start` | ✅ |
| `MyZeroSearchPolicy` 接入 self-play / eval / reanalyze | ✅ |
| builder `.gumbel()` / `.gumbel_standard()` | ✅ |
| completedQ target（Eq.10–12）+ reanalyze 对齐 | ✅ |
| bench `cartpole_bench_s*_visit/completed_q/gumbel_*` | ✅ manual |
| CartPole recipe promote | ❌ 暂缓 |

---

## 四、后续（非 CartPole 主线）

1. CartPole 回归哨兵固定 **cons+recon · PUCT · sims=20**，不再叠 Gumbel / completedQ。
2. 复测时机：**Pendulum** 或 **`|A| > n`** 矩阵格；必要时扫 `n ∈ {2,4,8}` × `|A|`。
3. 若 Gumbel standard（+ completedQ）在其它 env 仍失败，再拆「搜索 vs target」归因。

---

## 五、复现

```bash
# 基线（应 ~12.2k steps）
cargo test --release cartpole_bench_s20_visit --features blas-mkl --lib -- --ignored --nocapture

# 失败对照
cargo test --release cartpole_bench_s20_completed_q --features blas-mkl --lib -- --ignored --nocapture
cargo test --release cartpole_bench_s20_gumbel_visit --features blas-mkl --lib -- --ignored --nocapture
cargo test --release cartpole_bench_s10_gumbel_visit --features blas-mkl --lib -- --ignored --nocapture
```

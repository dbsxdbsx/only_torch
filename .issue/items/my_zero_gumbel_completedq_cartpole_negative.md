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
> **tree-level σ 归一化修复实测（2026-06-25）**：定位并修复 completedQ σ 归一化的 `|A|=2` 退化 bug（局部 over-children min-max → tree-level 全局 Q range），默认回 `50/1.0`。CartPole 3-seed：seed42 **16.7k**、seed43 **95k** 达标，**seed44 232k+ 仍未达标（手动停）** vs visit 13.1k/55.9k/11.7k——修复让旧版「全 seed 灾难」改善到「两 seed 达标」，但**仍系统性慢于 visit、未达 never-worse**。代码保留（对 `|A|≫n` 环境有益），CartPole 仍不 promote。详见 §六。

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

---

## 六、tree-level σ 归一化修复（2026-06-25，代码保留）

### 根因定位
completedQ 的 σ 归一化（[target.rs](../../src/rl/algo/my_zero/target.rs) `completed_q_policy_target`）原先用「只在当前节点几个动作之间」的局部 min-max。`|A|=2` 时两动作归一化后**恒为 {0,1}**，无论真实 Q 差多小，`σ=(c_visit+max_n)·c_scale·norm_q` 退化成与 Q 差无关的「符号开关」→ near-one-hot 目标污染训练。Gumbel 的 `gumbel.rs::q_range`（L315）同源，本次**未改**，留待 Gumbel 排期。

### 代码改动（保留，勿删）
- [min_max.rs](../../src/rl/mcts/min_max.rs)：新增 `MinMaxStats::range() -> Option<(f32,f32)>`，暴露 tree-level Q 极值。
- [types.rs](../../src/rl/mcts/types.rs)：`SearchResult.q_range: Option<(f32,f32)>`；[search.rs](../../src/rl/mcts/search.rs) 两个构造点填入（正常用搜索维护的 `min_max`，空候选 `None`）。
- [target.rs](../../src/rl/algo/my_zero/target.rs)：`completed_q_policy_target` 加 `q_range` 参数，σ 归一化优先用全局 range，`None`/退化时 fallback 局部 min-max；`mcts_policy_target` 透传 `result.q_range`。
- [component.rs](../../src/rl/algo/my_zero/component.rs)：默认 `cq_c_scale` 0.1 → **1.0**（论文棋类口径）。
- 离线 target-shape 单测（target.rs `mod tests` 的 `tree_range_*`）：`|A|=2` 小 Q 差不再 one-hot（实测 0.53 vs 局部 min-max 的 0.99）、随 Q 差单调、同比例缩放不变、退化 fallback 不 panic——**单元层面修复已验证**。

### 实测（seed 42/43/44 · release · sims=20 · cons+recon+Sampled · 50/1.0）
- seed 42：completedQ **16,669 达标** vs visit 13,115（1.27×）
- seed 43：completedQ **95,038 达标** vs visit 55,872（1.70×）
- seed 44：completedQ **>232k 未达标（手动停）** vs visit 11,678（灾难）
- 对照旧版（裸 vπ + 局部 min-max + `c_scale=0.1`）：3 seed 全失败。

### 结论
1. **归一化确是真 bug**：tree-level range 把旧版「全 seed 灾难」改善到「seed42/43 达标」，方向正确。
2. **但 CartPole 仍未达 never-worse**：每个达标 seed 都明显慢于 visit，seed44 直接退化到学不会。
3. **根因假设（待 `|A|≫n` 验证）**：CartPole 健康期搜索树各节点 Q 都很接近 → tree-level range 仍偏窄 → σ 仍偏尖 → 高方差 + 偏慢；叠加 `|A|=2`、sims=20 时 visit target 已足够密，completedQ 无增益空间。
4. **never-worse 口径澄清**：论文（Danihelka 2022）的 never-worse 是**单步 policy improvement** 理论（π' 不比先验差），**不保证**端到端训练样本效率追平 visit-count 目标。不能据此期待 CartPole 训练曲线必然追平。

### 决策
- 代码**保留**（tree-level range 是正确方向，对 `|A|≫n` 环境有益）；CartPole recipe **仍不 promote**。
- 复测时机：Pendulum 离散化（|A|=9/25）/ Platform / Atari 类 `|A|≫n`；届时顺带修 `gumbel.rs::q_range` 同源 bug。

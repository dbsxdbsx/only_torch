---
status: archived
created: 2026-06-21
updated: 2026-07-05
owners: []
reviewers: []
---

# MyZero · completedQ / Gumbel-root：CartPole 消融失败（暂缓 promote）

> **状态**：**archived（2026-07-05 终局归档）**——关闭条件「Gomoku |A|≫sims 场景复裁出结果」已达成，见 §八。
> 裁决：棋盘域两组件**中性**（无 CartPole 式灾难、亦无增益）→ 全部 recipe 保持关、代码保留；
> CartPole 灾难归因闭合 = n≫|A| regime 无筛选空间 + 前置双 bug（§七，已修）。
> 数字唯一账本：[Gomoku 棋盘账本](../../examples/my_zero/gomoku/README.md)。
>
> 原始状态：库内已实现 `completed_q_target` / `GumbelPolicy` + bench 用例；CartPole recipe **保持关**，当时 recipe 为 **consistency + reconstruction + Sampled · PUCT · sims=20 · td=5**（基线数字见 [CartPole 基准账本](../../examples/my_zero/cartpole/README.md)）。
> **⚠️ 口径提示（2026-07-02）**：下文全部实测数字为 **pre-autograd-fix 旧口径**（MSE 系 loss 反向忽略 `upstream_grad` 时代 + 纯 Rust BLAS + 单 seed 为主）。v0.25 修复后官方口径改 **3-seed 中位（release + MKL）** 且全量重测（见账本）；本文数字仅保留**方向性结论**（completedQ / Gumbel 系统性慢于 visit），复测（v0.26+，`|A| > sims` 或低延迟 acting 场景）时须按新口径重跑。
> **关联**：[CartPole README](../../examples/my_zero/cartpole/README.md) · [MyZero 总览](../../examples/my_zero/README.md) · [RL 状态总览](../../.doc/design/rl_myzero_status.md)
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

---

## 七、Gumbel 复裁前置修复清单（2026-07-02 补记，防丢失）

> 复测排期已定：[RL 状态总览 · backlog](../../.doc/design/rl_myzero_status.md#5-未完成事项收口时的-backlog)（Gomoku，|A|=225 ≫ sims）。**下述两项不修，复裁结果无效。**
>
> **✅ 双修复已落地（2026-07-04，Gomoku M3 前置）**：
> ① `temperature=0` 时 Gumbel 噪声向量置零（= mctx `gumbel_scale=0` 评测口径），
> greedy 路径 Top-m / halving / 终选全程确定，self-play（temp>0）噪声保留；
> ② `gumbel_halving_score` σ 归一化改用 `MinMaxStats::range()` 的 tree-level q_range
> （`RootScheduler` trait 增 `q_range` 参数透传），`None`/退化 fallback 局部 min-max。
> 回归测试 3 项新增于 `src/rl/tests/mcts_gumbel.rs`（greedy 跨 rng 确定且选 Q 最优、
> self-play 保留噪声多样性、tree-level 范围下微小 Q 差不压倒 prior）。
> 本 issue 关闭条件不变：Gomoku |A|=81/225 ≫ sims 场景复裁出结果。

### 7.1 greedy eval 注入 Gumbel 噪声 bug（已核实存在，无修复）

[gumbel.rs](../../src/rl/mcts/gumbel.rs) 的 `GumbelRootScheduler::final_recommendation` 用 `gumbel_halving_score(..., &self.gumbel, ...)` 打分出动作——`self.gumbel` 是本次搜索**现采的 Gumbel(0,1) 噪声**，且该函数**完全无视 `temperature=0`**。greedy eval 路径（`runner.rs::greedy_one_episode` → `MyZeroSearchPolicy` → `final_recommendation`）因此被注入探索噪声：CartPole `|A|=2` 时 σ 项 ≈ 1.2 而 Gumbel 噪声 std ≈ 1.28，**噪声尺度不小于 Q 信号差**——所谓 greedy eval 近半随机。这几乎就是上文「Gumbel greedy 峰值 123/154、从未收敛」的疑似真因（Gumbel 噪声本应只在 self-play 探索用）。

**修法**：eval（temperature=0）路径退回无噪 `argmax(visit)` 或无噪 `argmax(logits + σ(q̂))`；self-play 保留噪声。补单测：固定 children stats 下 greedy 推荐必须确定（不随 rng 变）。测试落点 `src/rl/tests/mcts_gumbel.rs`（现有 `final_recommendation_is_deterministic_with_seed` 只验"同 seed 同结果"，遮不住本 bug）。

### 7.2 `q_range` 局部归一化同源 bug

§六已留档：completedQ 侧已改 tree-level range，`gumbel.rs::q_range` 仍是局部 over-children min-max（`|A|=2` 退化同病）。修复时直接复用 `SearchResult.q_range`。

---

## 八、Gomoku 终局复裁（2026-07-05，关闭条件达成 → 归档）

> 载体 = Gomoku M3 消融（9×9 · |A|=81 · 3-seed · 同 M2 载体），前置双修复（§七）已落地。
> 完整表与口径见[棋盘账本](../../examples/my_zero/gomoku/README.md)。

| 臂 | vs random（中位） | vs 快照（中位） | naive0 | 判读 |
|---|---|---|---|---|
| base s100（对照） | 1.000 | 0.950 | 0.00–0.15 | — |
| Gumbel+completedQ s100 | 1.000 | 0.875 | 0.05–0.10 | 中性（n=100 > \|A\|=81，仍属 n>\|A\| regime） |
| base s16（对照） | 0.925 | 0.925 | ≈0 | — |
| **Gumbel+completedQ s16** | 0.975 | 0.825 | ≈0 | **中性**（\|A\|≫sims native 场景，关闭条件所指） |

**终局裁决**：

1. **无灾难**：CartPole 式「未收敛/系统性 2–3× 慢」在棋盘域未复现——支持归因 = CartPole 的 n≫\|A\| regime（无筛选空间放大样本浪费）+ 双 bug（greedy 噪声污染 eval + q_range 局部退化，均已修）。
2. **无增益**：\|A\|≫sims 的 native 场景下亦未跑赢 PUCT+visit 对照 → 两组件全域 recipe 保持关、代码保留（少 sim acting 时 Gumbel 降档可用：s16 对 s100 战力损失有限）。
3. 「三件套正交分层」文档债已沉淀 [RL 状态总览](../../.doc/design/rl_myzero_status.md)。

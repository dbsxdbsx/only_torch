# MyZero · CartPole-v1（基准账本）

> [← 返回 MyZero 总览](../README.md)
>
> **本文件是全项目 RL benchmark 数字的唯一账本（owner）**：vision / roadmap / AGENTS / issue 一律链到这里，不各自维护数字。每行实测必须带口径（profile / BLAS / seeds / 日期）。

离散 2 动作 · 门禁 **greedy eval ≥ 475** · **官方口径：3-seed（42/43/44）中位 env-steps-to-solved + 达标率**

组件组合由库内 [`recipe.rs`](../../../src/rl/algo/my_zero/recipe.rs) 按 `CartPole-v1` 自动注入（当前：consistency + reconstruction + Sampled · PUCT · sims=20 · td=5）；示例只写训练契约。论文全称见 [算法纲领 §4.1](../../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)。

## 运行

```bash
cargo run --example my_zero_cartpole --release

# 官方哨兵口径：多 seed 统计（打印中位 env-steps 与达标率）
SEEDS=3 cargo run --example my_zero_cartpole --release

# 临时覆盖 n-step bootstrap（默认 td_steps=5）
TD_STEPS=50 cargo run --example my_zero_cartpole --release

# 发版基线增量链（base → +cons → +cons+recon → promoted，各 3 seeds）
cargo test --release --features blas-mkl cartpole_baseline_t3_promoted -- --ignored --nocapture
cargo test --release --features blas-mkl cartpole_baseline_t2_cons_recon -- --ignored --nocapture
cargo test --release --features blas-mkl cartpole_baseline_t1_cons -- --ignored --nocapture
cargo test --release --features blas-mkl cartpole_baseline_t0_base -- --ignored --nocapture
```

训练日志：**`len`** = 本局步数；**`total_env_steps`** = 累计真实环境交互（首要评价指标）。

---

## 基线（v0.25 收口 · 官方口径）

**口径**：`release`（thin LTO + cg16）+ **Intel MKL** + seeds 42/43/44 · 2026-07-02 · autograd `upstream_grad` 修复后 + batch-native 训练 + MCTS 单趟前向。判据：greedy(temp=0) eval ≥ 475；env-steps 为「首次达标时累计真实交互」。

### MyZero 增量链（`baseline_matrix_bench`，各档 3 seeds）

| 档 | 配置 | seed42 / 43 / 44 env-steps | **中位** | 达标率 | wall（中位/seed） |
|----|------|----------------------------|----------|--------|-------------------|
| t0 | base（组件全关，≤1000 局） | 未达标（greedy 187.7）/ 未达标（142.0）/ 15,373 | — | **1/3** | 141.5s |
| t1 | +consistency | 3,536 / 151,362 / 17,535 | **17,535** | 3/3 | 53.3s |
| t2 | +consistency +reconstruction | 45,308 / 82,720 / 66,166 | **66,166** | 3/3 | 104.4s |
| t3 | **promoted**（+Sampled，= 当前 recipe） | 45,308 / 82,720 / 66,166 | **66,166** | 3/3 | 131.3s |

### 跨算法对照（同口径重测）

| 算法 | seed42 / 43 / 44 env-steps | **中位** | 达标率 |
|------|----------------------------|----------|--------|
| **MyZero promoted** | 45,308 / 82,720 / 66,166 | **66,166** | 3/3 |
| PPO（`SEED=…` 重测） | 81,920 / 81,920 / 102,400 | **81,920** | 3/3 |
| SAC（`SEED=…` 重测） | 115,906 / 152,150 / 159,408 | **152,150** | 3/3 |

### 结论（v0.25 收口）

1. **哨兵健康**：promoted 栈 3/3 达标、中位 greedy 500，官方哨兵数字定为 **中位 ~66.2k env-steps**。样本效率仍领先 model-free（PPO 1.2×、SAC 2.3×），但领先幅度较旧口径（bug 时代宣称 6–8×）大幅收窄——旧数字部分依赖 autograd bug 放大辅助 loss，本表为诚实基线。
2. **base 负对照成立**：组件全关 1000 局内仅 1/3 达标（两个失败 seed greedy 停在 142–188）——自监督组件对**达标率**的贡献仍是结构性的（+consistency 即 3/3），「组件有效」的证据链在新口径下依旧闭合。
3. **Sampled 退化等价实证**：t2 与 t3 三个 seed 的 env-steps **完全一致**——CartPole `N=2`、`K_eff=2` 全枚举时 Sampled 路径与标准 PUCT 逐步等价（π̂_β 修复后的自洽性证据）；代价是 wall-clock +~26%（纯簿记开销）。是否在纯离散小动作空间自动短路 Sampled，留 v0.26 评估。
4. **reconstruction 增益在新口径下存疑**：t1 中位（17.5k）优于 t2（66.2k），但 t1 的 seed 间方差极大（3.5k–151k），3 seeds 不足以下「reconstruction 有害」的裁决——旧结论「cons→cons+recon −58%」在修复 autograd 后**不再成立**。组件排序复裁排入 v0.26 P0（loss 系数重标定 + 更多 seeds），在此之前 recipe 保持现状不动（收口不做行为改动）。
5. **PPO / SAC 均正常收敛**，无 known-fail；旧参考值（PPO ~82k 单 seed）与新中位吻合，SAC 新中位（152k）高于旧单 seed 值（~105k，pre-autograd 时代 + 不同 BLAS 后端），符合「变慢 ≠ 失败、新数字即新基线」原则。

---

## 口径变更史（读旧数字前必看）

> **哨兵口径变更（batch-native + autograd 修复后，2026-07-01）**：修复了 MSE/MAE/BCE/Huber 反向忽略 `upstream_grad` 的框架 bug（作中间 loss 时丢链式缩放因子），训练同时改 batch-native（与逐样本数学等价）。梯度归约顺序改变 → env-steps 不再逐 bit 复现，验收改**统计口径**（3-seed 中位 + 达标率）。旧的 ~10–13k env-steps 部分依赖该 bug 使 continuation/reconstruction 辅助 loss 偏强；修复后辅助 loss 回到正确量级。若要收回样本效率，正道是显式调大 `RECONSTRUCTION_LOSS_COEF` / `CONTINUATION_LOSS_COEF`（v0.26 P0 消融）。
>
> **BLAS 口径变更（2026-07-02）**：`just` 的 MyZero/PPO 目标此前漏传 `{{_blas_flag}}`，历史 wall-clock 与 env-steps 均为**纯 Rust（matrixmultiply）口径**；现统一为自动检测注入（本机 Intel MKL）。GEMM 浮点累加顺序不同会使轨迹漂移，属统计口径已覆盖的扰动，但与历史行对比时需注意后端差异。
>
> **Release profile 口径变更（2026-07-02）**：`[profile.release]` 从 fat LTO + cg=1 放宽为 thin LTO + cg=16 + 增量编译（重编译 ~1m30s → ~13s，运行时约 +5~10%）；Criterion / 宏基准钉死旧配置于 `[profile.bench]`。与历史行对比 wall-clock 时注意此差异；极限速度长跑可手动 `cargo run --profile bench`。

---

## 历史消融表（pre-autograd-fix · seed=42 单点 · 纯 Rust BLAS · fat-LTO release）

> ⚠️ **仅方向性参考**：下表全部数字测于 autograd bug 修复前，绝对值与相对排序均不可与上方新基线直接比较；组件的**机制性结论**（如 soft 折扣注方差、completedQ 慢于 visit）仍有效，**数值结论**（如 −58%）已失效。

| 配置 | greedy 达标 | total_env_steps | wall-clock | 备注 |
|------|------------|-----------------|------------|------|
| base(组件全关) | 未在 ep250 达标 | — | — | 2026-06-16 |
| +consistency | **500.0** @ ep325 | 28,996 | 541s | 2026-06-20 复测 |
| +consistency +reconstruction · sims=20 | **500.0** @ ep250 | 12,186 | 80s | 2026-06-21 |
| +cons+recon + Sampled · sims=20 | **491.6** @ ep300 | 15,193 | 109s | 2026-06-22；N=2 K_eff=2 退化全枚举（旧实现路径差，π̂_β 修复后已逐步等价，见新基线结论 2） |
| +cons+recon+Sampled · td=5 · continuation soft `γ·c`（已废） | **500.0** @ ep375 | 30,158 | 185.6s | 2026-06-25；软折扣系统性压低好状态 value |
| +cons+recon+Sampled · td=5 · continuation 二值门 | **484.9** @ ep275 | 13,115 | 132s | 2026-06-25；binary `γ·(1−done)` 从 30.2k 修回 |
| 同上 · `TD_STEPS=50` | **500.0** @ ep225 | 10,317 | 88s | 2026-06-25；大-n 在确定性 reward 下略优，非稳健默认 |
| +cons+recon · sims=10 | **500.0** @ ep875 | 16,152 | ~125s | 2026-06-21 |
| +cons+recon · sims=15 | **500.0** @ ep500 | 26,306 | ~167s | 2026-06-21 |
| +cons+recon · sims=50（旧默认） | **500.0** @ ep275 | 11,682 | 183.9s | 2026-06-21；wall ~2.3× |
| +cons+recon · sims=50 · +completedQ | **500.0** @ ep575 | 34,490 | 381s | ❌ 不 promote |
| +cons+recon · sims=20 · +completedQ | **500.0** @ ep450 | 30,409 | 180s | ❌ [issue](../../../.issue/items/my_zero_gumbel_completedq_cartpole_negative.md) |
| +cons+recon · sims=20 · Gumbel-root | 峰值 **123** @ ep750+ | ~142k 手动停 | — | ❌ 同上 issue |
| +cons+recon · sims=10 · Gumbel-root | 峰值 **154** @ ep1800+ | ~101k+ 未达标 | — | ❌ 同上 issue |
| +consistency +reanalyze +写回 | **9.4**（ep200 仍随机） | 未达标 | — | ❌ [issue](../../../.issue/items/my_zero_reanalyze_cartpole_regression.md) |

**仍有效的机制性结论**：

- continuation search-discount 必须用 binary gate `γ·(1−done)`，soft `γ·c` 在确定性终止/无终止环境注方差并系统性压低好状态 value（基础语义，不列消融矩阵）。
- `td_steps` 默认 5：对齐 canonical MuZero/EZ（与 `k_unroll=5` 一致）、低方差；50 是旧「no-terminal 价值膨胀」时代遗留，终止已由 continuation/absorbing 接管。
- completedQ / Gumbel-root / reanalyze 在 CartPole regime（`sims ≫ |A|`、数据不受限）无增益或有害；复测留 `|A| > sims`、低延迟 acting、数据受限（图像）场景。

---

## 默认超参

`sims=20` · `gamma=0.997` · `k_unroll=5` · `td_steps=5` · `lr=0.02` · `train_batch_size=8` · `trains_per_episode=8`

- `k_unroll=5` 是 **dynamics 想象空间** 的 unroll 深度；`td_steps=5` 是 value target 在 **真实环境轨迹** 上的 n-step 步数——两者正交。
- 基础 transition 语义：真终止（杆倒）后 `continuation=0`，time-limit truncation 仍 `continuation=1` 并 bootstrap。
- 组件 loss 权重（写死在 `loss.rs` / `runner.rs`，非用户可调）：consistency coef **2.0** · reconstruction coef **1.0**（v0.26 P0 将重标定并暴露消融）。

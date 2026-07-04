# MyZero · CartPole-v1（基准账本）

> [← 返回 MyZero 总览](../README.md)
>
> **本文件是全项目 RL benchmark 数字的唯一账本（owner）**：vision / roadmap / AGENTS / issue 一律链到这里，不各自维护数字。每行实测必须带口径（profile / BLAS / seeds / 日期）。

离散 2 动作 · 门禁 **greedy eval ≥ 475** · **官方口径：3-seed（42/43/44）中位 env-steps-to-solved + 达标率**

组件组合由库内 [`recipe.rs`](../../../src/rl/algo/my_zero/recipe.rs) 按 `CartPole-v1` 自动注入（当前：consistency(coef 2) + reconstruction(**coef 16**，v0.26 P0 重标定) + Sampled · PUCT · sims=20 · td=5）；示例只写训练契约。论文全称见 [算法纲领 §4.1](../../../.doc/design/my_zero_algorithm_vision.md#41-组件文献对照单一事实源)。

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

# v0.26 P0 loss 系数重标定消融（预注册协议见测试文件 doc 注释）
cargo test --release --features blas-mkl cartpole_coef -- --ignored --nocapture --test-threads=1
cargo test --release --features blas-mkl cartpole_seedext -- --ignored --nocapture --test-threads=1

# 哨兵红灯复裁（2026-07-04 已收口；recon {4,16} × 5-seed）
cargo test --release --features blas-mkl cartpole_recal -- --ignored --nocapture --test-threads=1

# v0.26 Phase 0 编码 / 量纲消融（预注册协议见各测试文件 doc 注释）
cargo test --release --features blas-mkl cartpole_hlgauss -- --ignored --nocapture --test-threads=1
cargo test --release --features blas-mkl cartpole_symlog -- --ignored --nocapture --test-threads=1
```

训练日志：**`len`** = 本局步数；**`total_env_steps`** = 累计真实环境交互（首要评价指标）。

---

## 哨兵复裁收口（2026-07-04 · **当前官方哨兵**）

**口径**：`release`（thin LTO + cg16）+ Intel MKL · ndarray 0.17.2 · **最终数值流**（优化战报 P/Q/C1/R 全部落地后的 HEAD）· recipe 零变更（recon=16）。预注册协议（r1/r2 两臂、裁决规则）见 [`loss_coef_ablation_bench.rs`](../../../src/rl/algo/my_zero/tests/loss_coef_ablation_bench.rs) doc 注释「哨兵红灯复裁」节。

### 复裁臂（seeds 42–46，promoted recipe 为底、单变量）

| 臂 | seed42 / 43 / 44 / 45 / 46 env-steps | **中位** | 达标率 | range |
|----|----------------------------------------|----------|--------|-------|
| recon=4 | 11,419 / 15,527 / 13,556 / 9,520 / 14,954 | 13,556 | 5/5 | 9.5k–15.5k |
| **recon=16（现 recipe）** | **8,741 / 71,969 / 6,744 / 9,765 / 11,772** | **9,765** | **5/5** | 6.7k–72.0k |

### 裁决（按预注册规则）

1. **recon=16 维持 promote，recipe 零变更**：两臂均 5/5 达标，预注册规则「达标率 ≥ 4/5 的臂中取中位更优者」→ 16（9,765 < 13,556）。
2. **备注（未进裁决、留档供参考）**：recon=4 的 range 显著更紧（9.5k–15.5k，无长尾），recon=16 有一个 71,969 长尾 seed（43）；达标率打平时预注册规则以中位裁决，不事后改规则——若未来哨兵再现临界震荡，recon=4 是现成的稳健后备档。
3. **红灯根因复核闭合**：红灯 1/3 实测发生在战报 P 数值定稿**之前**；P（`sum_to_shape` 单趟 / `min_max_normalize` 广播）落地后轨迹再次漂移，当前流上 recipe 零变更即回绿——进一步支持「达标率对具体浮点轨迹敏感、非逻辑回归」的判读。哨兵继续只回答「崩没崩」（条款二），env-steps 绝对值跨数值流不可比。

### 官方哨兵定格（`SEEDS=3`，example + recipe 路径）

**8,741 / 71,969 / 6,744，中位 ~8.7k，3/3 达标**（与消融臂 42/43/44 逐 bit 一致，example 路径 ≡ bench 路径实证）。

### 跨算法对照（同口径同日重测，应本批框架优化后轨迹漂移全量刷新）

| 算法 | seed42 / 43 / 44 env-steps | **中位** | 达标率 | 备注 |
|------|----------------------------|----------|--------|------|
| **MyZero promoted** | 8,741 / 71,969 / 6,744 | **8,741** | 3/3 | 官方哨兵 |
| PPO | 61,440 / 122,880 / 122,880 | **122,880** | 3/3 | 粒度受 rollout(2048)×eval 节奏量化 |
| SAC | 127,751 / 136,130 / 122,157 | **127,751** | 3/3 | |

样本效率领先：PPO **14.1×** · SAC **14.6×**（中位对中位；PPO 粒度粗，倍数仅方向性）。

---

## 📦 哨兵红灯（2026-07-03 · ndarray 升级后实测 · **已收口，见上节**）

> **✅ 已收口（2026-07-04）**：上方「哨兵复裁收口」节为现行基线；本节保留为红灯现场历史记录。
> issue 已归档：`.issue/_archive/cartpole_sentinel_red_ndarray_drift.md`。

**口径**：`release`（thin LTO + cg16）+ Intel MKL · **ndarray 0.17.2 + pyo3/numpy 0.29 新依赖栈** ·
recipe 与超参零变更（recon=16）。两轮实测：

| 轮次 | 代码状态 | seed 结果（env-steps / 未达标附 greedy） | 达标率 |
|---|---|---|---|
| 5-seed 探测 | 优化 N 后、逻辑修复批次前 | 42 未达标(259.3) / 43 = 11,181 / 44 = 46,678 / 45 未达标(165.3) / 46 = 29,415 | **3/5** |
| 3-seed 官方口径 | 逻辑修复批次后（per-seed env 流 + 常数温度调度） | 42 未达标(229.6) / 43 = 12,874 / 44 未达标(185.6) | **1/3** |

**判读**：
1. **不是新逻辑 bug**：两轮独立代码审查（backward 拓扑 / 优化器融合 / Add 分流 / BN 公式 / pool 布局守卫 /
   MCTS 解码 / StoredObs / owned 入图）全部带证据排除；seed 42 逐局对比确认与修复批次前
   前 ~1170 局逐 bit 相同（分叉仅温度表达式的浮点 ulp 差）。
2. **失败模式是「临界震荡」非「不学习」**：失败 seed 的 self-play 均值稳定在 ~200，greedy 反复冲到
   430–475 区间但 2000 局内跨不过 475（修复批次前 seed 42 峰值曾达 474.7）。
3. **根因指向**：BLAS 全布局派发（ndarray #1419）改变 GEMM 浮点累加顺序 → 轨迹漂移把训练踢出了
   recon=16 在旧数值流上的舒适区；v0.26 P0 的 5/5 快照对具体浮点轨迹存在过拟合成分。
4. **两轮达标率差异**（3/5 vs 1/3）不构成修复批次回归证据：seed 43/44 的 env 初始状态流在 per-seed
   派生修复后本来就换了（statistical 独立性修复，口径变更史有档）；样本量下 3/5 与 1/3 同为
   「达标率显著低于旧 5/5」的一个证据。
5. **下一步**：不能只「重定数字」——达标率跌破门槛属于「预算内不收敛」，须按纪律在新数值流上
   **复裁 loss 系数**（P0 消融矩阵重跑，重点 recon {4, 16} × 更多 seeds），必要时连同温度调度常数一起消融。
   在此之前下方 v0.26 P0 哨兵数字（~9.8k）仅为**旧依赖栈历史记录**。

---

## v0.26 P0：loss 系数重标定（2026-07-02 · ⚠️ 旧依赖栈口径，见上方红灯记录）

**口径**：`release`（thin LTO + cg16）+ Intel MKL · 2026-07-02 · 与 v0.25 收口口径同批代码 + 系数旋钮重构（行为零变化已由 101 单元测试 + seed42 逐 bit 复现 45,308 验证）。预注册协议（档位、裁决规则）见 [`loss_coef_ablation_bench.rs`](../../../src/rl/algo/my_zero/tests/loss_coef_ablation_bench.rs) doc 注释。

**背景**：autograd `upstream_grad` 修复只影响 MSE 系反向 → MyZero 中仅 **reconstruction / continuation** 曾被隐式放大（等效 ≈ K×B = 40）；consistency（余弦）与 policy/value/reward（CE）一直正确，无重标定理由。档位取对数网格 {1, 4, 16}（锚点 = 论文默认 1 ↔ bug 等效 ~40），16 触边后补 64 边界核查。

### 消融臂（promoted recipe 为底，单变量，seeds 42/43/44）

| 臂 | seed42 / 43 / 44 env-steps | **中位** | 达标率 | 裁决（预注册规则） |
|----|----------------------------|----------|--------|--------------------|
| baseline（recon=1 · cont=1，= v0.25 t3） | 45,308 / 82,720 / 66,166 | 66,166 | 3/3 | 对照 |
| recon=4 | 11,020 / 57,613 / 21,196 | 21,196 | 3/3 | 与 baseline range 轻微重叠，单独不 promote |
| **recon=16** | **12,519 / 8,643 / 9,826** | **9,826** | **3/3** | **range 完全不重叠且全面更优 → promote** |
| recon=64（边界核查） | 12,193 / 未达标 / 14,948 | — | 2/3 | 过冲（方差+失败 seed 回归）→ 16 为平台点 |
| cont=4 | 6,499 / 92,407 / 未达标 | — | 2/3 | 达标率不足 → 保持 1.0 |
| cont=16 | 15,385 / 32,385 / 180,847 | 32,385 | 3/3 | range 与 baseline 重叠 → 无差异，保持 1.0 |

### 5-seed 扩展（补 seeds 45/46，与上表 42/43/44 合并；recon 去留复裁）

| 配置 | 5-seed env-steps（升序） | **中位** | 达标率 |
|------|--------------------------|----------|--------|
| t1 +consistency | 3,536 / 17,535 / 18,573 / 22,982 / 151,362 | 18,573 | 5/5 |
| t3 promoted · recon=1 | 9,793 / 45,308 / 66,166 / 82,720 / 未达标(s45) | 55,737 | **4/5** |
| **promoted · recon=16（新 recipe）** | 8,643 / 9,826 / 12,519 / 44,694 / 79,466 | **12,519** | **5/5** |

### 哨兵定格（recipe 默认已改 `RECONSTRUCTION_LOSS_COEF = 16.0`）

`SEEDS=3` 官方哨兵（example + recipe 路径）逐 bit 复现消融臂：**12,519 / 8,643 / 9,826，中位 ~9.8k，3/3 达标**。对照同口径 model-free：领先 PPO（81.9k）**8.3×**、SAC（152.2k）**15.5×**。

### 结论

1. **recon 系数是真信号、已 promote**：1 → 4 → 16 单调改善（66.2k → 21.2k → 9.8k），16 处 seed 方差极小；64 过冲（2/3）。逐 seed 配对（vs recon=1）：42/43/44/45 四胜（含救活 s45 不达标）、仅 s46 反转。中位超过 bug 时代 13.1k——丢失的样本效率已收回且有富余。
2. **修复后的 recon=1 实测有害**：5-seed 4/5 达标、中位 55.7k，比纯 consistency（18.6k、5/5）差——弱 recon 梯度不足以塑造 latent、只在共享表征上制造干扰；v0.25 的"t1 优于 t2 悬案"两侧同时解释闭合。
3. **cont 系数保持 1.0**：两档均无稳定收益且注入方差（cont=4 一个 seed 不收敛）；binary gate 语义不需要更大权重。
4. **recon(16) vs 纯 consistency 在 CartPole 分不出高下**（中位 12.5k vs 18.6k，逐 seed 2/5，尾部更好 79k vs 151k）——哨兵分辨率极限；recon 的最终价值裁决留给图像环境（自监督本命田），CartPole 保留其回归覆盖。
5. **临时值声明**：recon_coef=16 为 CartPole 单环境证据 + 框架级放大倍数推导（K×B）联合支持的临时默认；图像线 obs 归一化后量纲变化，须复验后定稿。

---

## v0.26 Phase 0：编码 / 量纲消融（2026-07-02）

**口径**：`release`（thin LTO + cg16）+ Intel MKL · seeds 42/43/44 · 3-seed 中位 env-steps + 达标率；
baseline = 上节哨兵（12,519 / 8,643 / 9,826，中位 ~9.8k，range 8.6k–12.5k），不重跑。
预注册协议见 [`hl_gauss_ablation_bench.rs`](../../../src/rl/algo/my_zero/tests/hl_gauss_ablation_bench.rs) 与 [`obs_symlog_ablation_bench.rs`](../../../src/rl/algo/my_zero/tests/obs_symlog_ablation_bench.rs) doc 注释。

### HL-Gauss value/reward 编码（two-hot → 高斯软标签，σ=0.75×bin）

| 臂 | seed42 / 43 / 44 env-steps | **中位** | 达标率 | 裁决 |
|----|----------------------------|----------|--------|------|
| baseline（two-hot，= 哨兵） | 12,519 / 8,643 / 9,826 | 9,826 | 3/3 | 对照 |
| HL-Gauss | 14,480 / 27,975 / 27,602 | 27,602 | 3/3 | **range 完全更差不重叠（14.5k–28.0k vs 8.6k–12.5k）→ 负结果，回退 two-hot** |

**结论**：CartPole 上 HL-Gauss 显著劣化（中位 9.8k → 27.6k，~2.8×），非文献预期的「持平」。机制解释：CartPole value 目标经 h 变换后范围窄（≈0–17.6）且低噪声，two-hot 的"尖"标签在此是**信息优势**（目标可精确定位到相邻两原子）；HL-Gauss 把质量摊到 ~±3 原子等于主动注入标签模糊，其鲁棒性收益只在大 value 噪声场景（图像/大规模）才能兑现——正合 Farebrother et al. 的适用前提。**默认保持 two-hot；`hl_gauss` 开关与实现保留在库，Phase 1 图像域（native 场景）复测**（收口规划 §2 已有该复裁条目）。

### obs symlog 无量纲化（symlog × recon_coef 三臂）

| 臂 | seed42 / 43 / 44 env-steps | **中位** | 达标率 | 裁决（预注册规则） |
|----|----------------------------|----------|--------|--------------------|
| baseline（raw obs · recon=16，= 哨兵） | 12,519 / 8,643 / 9,826 | 9,826 | 3/3 | 对照 |
| symlog + recon=16 | 41,081 / 10,623 / 12,864 | 12,864 | 3/3 | 与哨兵 range 重叠但中位 +31%、尾部 41k；无增益 |
| symlog + recon=4 | 19,710 / 19,596 / 39,985 | 19,710 | 3/3 | 差于 symlog+16 → 系数未回移 |
| symlog + recon=1 | 39,399 / 187,827 / 16,352 | 39,399 | 3/3 | 最差 + 187.8k 极端尾部 → promote 条件不满足 |

**结论（负结果，保持 raw obs）**：

1. **「最优系数向 1 回移」完全没有发生**——symlog 下仍是 16 最优且单调恶化（16 → 4 → 1 = 12.9k → 19.7k → 39.4k）。预注册 promote 条件（小系数与 16 打平）不满足，且 symlog+16 相对哨兵无任何改善（中位 +31%、尾部方差放大）。
2. **机制解释**：CartPole obs 本来就是 O(0.1–1) 小量纲（angle ±0.42、实际 pos/vel ±3 内），symlog 在 |x|<1 近恒等——**无量纲可撤**，只轻微非线性挤压了 obs 的可分辨结构。由此裁决 v0.25 复盘的悬案：**recon_coef=16 在 CartPole 上的本质是「自监督话语权」权衡旋钮，不是单位换算**（若是单位换算，撤走量纲后系数应回归 1）。
3. **对 Phase 1 的指导**：低维小量纲环境不需要 obs 归一化；图像线走 [0,1] 像素归一 + 该域重标 recon 系数即可。`obs_symlog` 开关保留库内（未来大范围连续特征环境可按 [Simulus 计划 §3](../../../.doc/design/my_zero_simulus_ablation_plan.md) 原触发条件启用），recipe 默认恒关。

### Phase 0 收口定格（2026-07-02 · CartPole 自此冻结为纯回归哨兵）

两项消融均负结果 → **recipe 零变更**；两个开关（`hl_gauss` / `obs_symlog`）默认关落地后，
`SEEDS=3` 官方哨兵**逐 bit 复现** promoted 数字：**12,519 / 8,643 / 9,826，中位 ~9.8k，3/3 达标**
（行为零变化的实证）。v0.26 recipe 定稿 = cons(2) + recon(**16**) + two-hot + raw obs + canonical 梯度流
（[收口规划 §1](../../../.doc/design/rl_closure_plan.md) 四者有据）；此后 CartPole 只回答「崩没崩」（条款二）。

---

## 基线（v0.25 收口 · 官方口径）

> ⚠️ 本节 promoted/t2/t3 行为 **recon_coef=1 时代**数字，已被上节 v0.26 P0 哨兵（中位 ~9.8k）取代；t0/t1 与跨算法对照仍有效。

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

> **「逐 bit 复现」声明的有效域**：batch-native 训练只保证与逐样本数学等价（浮点归约顺序不同），
> BLAS 后端的 GEMM 累加顺序又随依赖版本 / 派发路径变化——因此本文件中任何「逐 bit 复现」
> 均只在**同一二进制 + 同一依赖栈**内成立，跨版本对比一律用统计口径（3~5 seed 中位 + 达标率）。

> **多 seed 独立性 + 温度调度口径变更（2026-07-03）**：① self-play / eval 的 `env.reset` 种子改为
> **per-seed 派生**（旧实现所有 seed 共用 base seed 的同一条 env 初始状态流，多 seed 统计独立性打折）；
> 单 seed 行为不变，多 seed 的 seed 43+ 轨迹自此变化。② 温度退火从「按 `max_episodes` 比例」改为
> **显式常数调度**（hold 1000 局恒 1.0 → 1000 局线性退火到 0.25；`TrainSettings::temp_hold/decay_episodes`）——
> 官方 CartPole 口径（2000 局预算）下逐点等价（有单测锤），但改预算不再静默改变探索行为。
>
> **ndarray 0.16.1/0.17.2 升级致轨迹漂移（2026-07-03）**：ndarray 0.15.6 → 0.16.1（同日连升 0.17.2）后，BLAS 全布局派发（#1419）使转置视图 GEMM 改走 MKL 转置标志路径，浮点累加顺序变化 → 「~9.8k 逐 bit 复现」**自此不再可复现**。正式重测已执行（见「哨兵红灯」历史节）：达标率跌破门槛，非孤例——升级为「新数值流上复裁系数」待办，**已于 2026-07-04 复裁收口**（见顶部「哨兵复裁收口」节，recipe 零变更回绿）；旧数值流的 env-steps 绝对值仅作方向性参考。0.16.1 → 0.17.2 无 BLAS 改动，数值与 0.16.1 一致。
>
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
| +cons+recon · sims=20 · +completedQ | **500.0** @ ep450 | 30,409 | 180s | ❌ [issue](../../../.issue/_archive/my_zero_gumbel_completedq_cartpole_negative.md) |
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
- 组件 loss 权重（默认在 `loss.rs`，经 `Components` 系数字段传导，用户 API 不暴露）：consistency coef **2.0** · reconstruction coef **16.0**（v0.26 P0 重标定，见上方消融）· continuation coef **1.0**。

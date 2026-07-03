# MyZero · ALE/Pong-v5（图像离散支柱）

> **定位**：v0.26 Phase 1 图像线立柱的**唯一账本**（[收口规划 §2](../../../.doc/design/rl_closure_plan.md)、[Phase 1 计划](../../../.doc/design/rl_phase1_image_plan.md)）。
> 其余文档引用本页数字，不另抄。

## 预注册口径（2026-07-02 定稿，先于任何正式跑）

| 项 | 值 |
|----|-----|
| 环境 | `ALE/Pong-v5`（gymnasium 默认：frameskip=4、sticky 0.25、Discrete(6)） |
| obs 管线 | 210×160×3 → BT.601 灰度 → 双线性 84² → **u8 量化存储**（round，2026-07-03 起）→ 读取反量化 [0,1] → 4 帧堆叠（库内自动；量化误差 ≤ 0.5/255，见[优化战报 M](../../../.doc/performance/optimization_log.md)） |
| 网络 | ConvRepr（conv3×3s2 ×4：4→32→64→64→64 → fc→latent64）+ MLP dynamics/prediction（CartPole 同构） |
| 组件 | image base = **consistency ON** + reconstruction OFF + two-hot + raw obs（`recipe.rs`） |
| 训练 | γ=0.997 · K=5 · td=5 · sims=20 · lr=0.003 · batch=16 · 64 trains/ep · buffer 32 局 |
| 预算 | 每 seed ≤150 局（≈120k env-steps）或 24h 先到为准；seeds {42,43,44} |
| **门槛** | 3-seed 中位 **best greedy(10 局) ≥ −18**（随机 ≈ −20.7；非 SOTA 口径），且学习曲线呈上升趋势 |
| 判读 | 达标 = 图像支柱成立；不达标但曲线上升 = 部分信号按曲线裁决；平直 = 负结果 issue |

## 运行

```bash
SMOKE=1 cargo run --example my_zero_pong                                  # 管线验证（3 局）
cargo run --example my_zero_pong --release --features blas-mkl            # 单 seed
SEEDS=3 cargo run --example my_zero_pong --release --features blas-mkl    # 官方 3-seed
```

## 账本（口径：release + MKL，best greedy 10 局）

| 日期 | 配置 | seeds | best greedy（各 seed） | 中位 | 裁决 |
|------|------|-------|------------------------|------|------|
| （待跑） | image base（预注册栈） | 42/43/44 | — | — | — |

### 工程基线（非学习指标）

| 日期 | 存储 | 逐局 wall（release+MKL，64 trains/ep） | 备注 |
|------|------|------|------|
| 2026-07-02 | f32 帧 | Ep1 2.4s → 随 buffer 占用爬升 → 满（Ep32）后平台 65~110s | Run A/B + 3-seed 首跑一致复现 |
| 2026-07-03 | **u8 量化帧** | **Ep5~Ep60 全程 ~10s 平坦，无增长** | 60 局 716.8s（含 eval）；归因与 PROFILE 见[优化战报 M](../../../.doc/performance/optimization_log.md) |

## A/B 消融（§S3，基准裁决后半额预算 60k env-steps/seed × 3 seeds）

| 臂 | 变更 | 状态 |
|----|------|------|
| recon | reconstruction ON（coef 先 1-seed pilot {1,4,16}） | 待跑 |
| cons-off | consistency OFF | 待跑 |
| hl-gauss | HL-Gauss 编码 ON（CartPole 负结果的图像域复测） | 待跑 |

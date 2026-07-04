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
| 2026-07-03 | image base（预注册栈）· **旧数值流 ⚠️** | 42/43/44 | −20.4 / −21.0 / −21.0 | **−21.0** | **未达标，曲线平直** → 按预注册判读记负结果，见 [issue](../../../.issue/items/my_zero_pong_image_flat_negative.md)；14 个 eval 点 greedy 恒 −21 级（仅偶发 −20.x），self-play avg 恒 −20~−21，loss 下降但不转化为策略改进；每 seed 满 150 局 ≈134k env-steps，wall ~26min/seed；450 局全程零 panic。**代码基线 `b697e95`，先于 `adfc02f` 框架修复批（per-seed reset 派生 / GroupNorm 梯度 / 温度调度显式化等）——数字仅作历史参考，图像支柱裁决以新数值流复跑为准** |
| 2026-07-04 | image base（同栈）· **新数值流复跑**（哨兵复裁收口后 HEAD，per-seed env 流独立已生效） | 42/43/44 | −21.0 / −20.3 / −20.6 | **−21.0** | **0/3，曲线仍平直 → 负结果确认**（排除"旧数值流 / seed 共享 reset 序列"两个嫌疑）；self-play 偶见 −17~−18 但 greedy 恒 −20.3 以下、无上升趋势；每 seed 满 150 局 ≈134k env-steps，wall ~30min/seed。下一步按负结果 issue 预注册：S3 三臂兼当诊断（首查 recon ON 臂——CartPole 域最大杠杆组件在 base 臂恰好关闭）；ROSMO 阶梯一已进库（`.rosmo(true)`，2026-07-04）可作"训练量不足"嫌疑的后续单变量臂 |

### 工程基线（非学习指标）

| 日期 | 存储 | 逐局 wall（release+MKL，64 trains/ep） | 备注 |
|------|------|------|------|
| 2026-07-02 | f32 帧 | Ep1 2.4s → 随 buffer 占用爬升 → 满（Ep32）后平台 65~110s | Run A/B + 3-seed 首跑一致复现 |
| 2026-07-03 | **u8 量化帧** | **Ep5~Ep60 全程 ~10s 平坦，无增长** | 60 局 716.8s（含 eval）；归因与 PROFILE 见[优化战报 M](../../../.doc/performance/optimization_log.md) |

## A/B 消融（§S3，半额预算 75 局/seed；2026-07-04 实跑，兼当 S2 平直诊断）

| 臂 | 变更 | 结果（best greedy） | 判读 |
|----|------|------|------|
| recon pilot | reconstruction ON，coef {1,4,16} × 1-seed | −21.0 / −20.4 / −21.0 | **全平**，无 coef 拉起曲线 → 嫌疑 3（recon OFF 是主因）排除 |
| cons-off | consistency OFF × 3-seed | −21.0 / −20.7 / −21.0，中位 −21.0，0/3 | **平**（与 base 无差异）→ 嫌疑 2（cons 图像域反噬）排除 |
| hl-gauss | HL-Gauss ON × 3-seed | −21.0 / −21.0 / −21.0，中位 −21.0，0/3 | **平** → 编码不是瓶颈；Simulus A1 图像复测记「无差异」 |

**S3 诊断裁决（2026-07-04）**：三臂 + base 全平 → 组件层排除，**嫌疑 1（训练量不足：9.6k updates/batch16 vs EfficientZero 参考 ~120k/batch256，差两个数量级）与嫌疑 4（稀疏 reward + γ=0.997）上位**。
按 2026-07-04 规划修订：**Gomoku（Phase 2）提前为主线**，图像线降级为后台预算标定
（replay ratio ↑ × ROSMO 现算刷新 / lr 扫描 / DIAG，见负结果 issue「下一步」）。
日志：`.bench/pong_s3_*.log`。

---
status: active
created: 2026-07-03
updated: 2026-07-03
owners: []
reviewers: []
---

# CartPole 哨兵红灯：ndarray 升级轨迹漂移后达标率跌破门槛（需新数值流复裁系数）

> **状态**：active —— 已实测确认非孤例、非新逻辑 bug；数字与完整判读见
> [cartpole 账本「哨兵红灯」节](../../examples/my_zero/cartpole/README.md)（唯一账本，此处不重复数字）。
> **关联**：CHANGELOG deps 条目（ndarray 0.16/0.17 升级）· `.doc/design/rl_closure_plan.md`（CartPole 纯回归哨兵定位）

## 一、现象（摘要）

ndarray 0.15.6 → 0.17.2 升级（BLAS 全布局派发 #1419，GEMM 浮点累加顺序变化）后，
官方哨兵（recipe 零变更，recon=16）达标率从 5/5 跌至 **3/5（5-seed）/ 1/3（3-seed 官方口径）**；
失败 seed 模式 = self-play 均值 ~200、greedy 反复冲 430–475 但 2000 局预算内跨不过门槛（临界震荡，非不学习）。

## 二、已排除

1. **新逻辑 bug**：两轮独立只读审查（优化 J/K/L/M/N + Add 分流 + ndarray 布局守卫横切面）逐项带证据排除；
2. **修复批次（2026-07-03 六项改良）引入回归**：seed 42 与修复前逐局对比，前 ~1170 局逐 bit 相同
   （分叉点在温度退火段，纯浮点 ulp 差异）；
3. **环境噪声孤例**：env-steps 对给定 seed 确定性，两轮独立跑均复现失败。

## 三、当前假设

recon_coef=16（v0.26 P0 在旧数值流上标定）对具体浮点轨迹存在过拟合成分；
BLAS 累加顺序漂移把训练踢出其舒适区 → 临界 seed 翻车。

## 四、待办（按改动纪律）

1. 安静窗口在**新数值流**上重跑 P0 系数消融矩阵（重点 recon {4, 16}，seeds ≥ 5），重新 promote；
   —— **数值流已于 2026-07-03 最终定稿**（优化战报 P：`sum_to_shape` 单趟归约 + MyZero
   `min_max_normalize` 去 repeat 走广播，均可能 ulp 级漂移；刻意赶在复裁前落地，使复裁
   一次性覆盖），**复裁跑完并收口前不得再改任何数值路径**；
   —— **预注册臂已备（2026-07-03）**：`cartpole_recal_r1_recon4` / `r2_recon16`（各 5-seed 42–46，
   裁决规则见 `loss_coef_ablation_bench.rs` doc 注释）；当晚首轮启动后为让位其他优化工作**中止**（未产出可用数字），待新安静窗口重跑；
2. 可选臂：温度调度常数（hold/decay）消融——失败模式（高温期反复摸门槛）与探索调度直接相关；
3. 新基线落账本后，本 issue 收口；在此之前 CartPole 哨兵不能作为「崩没崩」的绿灯依据，
   Phase 1 图像线行为改动需以单测 + smoke + Pong 曲线自行兜底。

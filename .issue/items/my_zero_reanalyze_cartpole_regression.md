---
status: active
created: 2026-06-20
updated: 2026-07-03
owners: []
reviewers: []
---

# MyZero · Reanalyze + 写回：CartPole 学习失效（暂缓 promote）

> **状态**：active —— 写回闭环已实现并单测覆盖；CartPole recipe **已关** `reanalyze`，当前 recipe 为 **consistency + reconstruction + Sampled**（基线数字见 [CartPole 基准账本](../../examples/my_zero/cartpole/README.md)）。
> **⚠️ 口径提示（2026-07-02）**：下文实测数字为 pre-autograd-fix 旧口径，仅保留方向性结论（当前 reanalyze 实现在 CartPole 上有害而非单纯变慢）。
> **战略升级（2026-07-01，纲领 §2.3）**：reanalyze 已定为 v0.26 **战略组件**（「实时轻 acting + 离线重 reanalyze」解耦是商业游戏路线核心）；本 issue 的负结果不构成否定——复活时先查 §三假设（早期弱网写回污染 / partial window 覆盖 / 缺 target net），并按新口径重测。
> **关联**：[CartPole README](../../examples/my_zero/cartpole/README.md) · [MyZero 总览](../../examples/my_zero/README.md) · [RL 路线图](../../.doc/design/rl_roadmap.md)
> **代码**：`src/rl/algo/my_zero/reanalyze.rs` · `runner.rs`（`prepare_train_batch` / `writeback_reanalyzed_samples`）· `buffer/replay.rs`（`sample_indexed` / `update_at`）
> **日志**：`.bench/my_zero_cartpole_cons_only.log` · `.bench/my_zero_cartpole_cons_reanalyze.log`

---

## 一、现象

seed=42 · release · CartPole-v1 · recipe **consistency + reanalyze + 写回**：

| 指标 | 结果 |
|------|------|
| greedy eval @ ep25–200 | **钉死在 9.4**（≈ 随机） |
| avg_R @ ep200 | ~11.6 |
| 每局 wall-clock（ep10+） | ~2.5–3.7s（无 reanalyze ~0.17–0.2s） |

同配置关 `reanalyze` 后（**仅 consistency**，写回代码仍在、不触发）：

| 指标 | 结果 |
|------|------|
| greedy eval 轨迹 | 9.2 → 15 → 143 → 309 → 407 → **500** |
| 达标 | **ep325**，**28,996 env-steps**，540.8s |
| 与 2026-06-20 基线 | **无回归**（env-steps 一致） |

**结论**：CartPole 上当前 reanalyze（position 级 MCTS + buffer 写回）**有害**，不是单纯变慢。

---

## 二、已实现（勿删）

| 项 | 状态 |
|----|------|
| position 级 `reanalyze_unroll_window` | ✅ |
| `sample_indexed` → reanalyze → train → `update_at` 写回 | ✅ |
| 单测 `tests/reanalyze_writeback.rs`（4 项） | ✅ |
| CartPole recipe promote reanalyze | ❌ 暂缓 |

---

## 三、待查假设（下次接手）

1. **早期弱网 + 写回污染 buffer**：reanalyze 用当前差网络重刷 `policy_target` / `root_value` 并持久化，是否比 stale self-play 标签更差？
2. **partial window 写回整局**：只刷 `[start,start+K]`，但 `update_at` 写整局；同 batch 重复 idx「后者覆盖」是否丢刷新？
3. **缺 target net**：reanalyze bootstrap 是否应走 target 网（`compute_n_step_target_with` 已预留）？
4. **CartPole 不适用**：数据不受限时 reanalyze 增益本就不确定（见 `post_ez_v2_research_backlog.md`）；失效是预期还是实现 bug 待分。

---

## 四、恢复条件（promote 前）

> **复活路径重构（2026-07-03）：ROSMO-first 两级阶梯，裁决场 = Phase 1 图像线**（详见[收口规划 Phase 1](../../.doc/design/rl_closure_plan.md#2-phase-1--图像线立柱--一级风险压测v026-下半)）。
> 关键洞察：本 issue §三假设 1（弱网全树重搜写回投毒）与 ROSMO（Xiao et al., ICLR 2023 ·
> [arXiv:2210.05980](https://arxiv.org/abs/2210.05980)，开源 sail-sg/rosmo）诊断 MuZero Unplugged
> 失效的病理**同源**——旧数据上搜得越深模型误差复合越狠。故复活第一臂改为 **ROSMO 式一步
> target 刷新**（一步 look-ahead + 行为正则，不重跑整棵树；重刷成本 ≈ 一次前向，CPU 友好、
> 弱网鲁棒），全树 MCTS reanalyze 降为阶梯二（一步版有增益后再单变量消融——在线 off-policy
> 下深搜红利可能回归，ROSMO 纯离线结论不自动外推）。

- [x] 阶梯一**实现已进库**（2026-07-04）：`src/rl/algo/my_zero/rosmo.rs`（一步 look-ahead
      改进分布 + 现算 n-step bootstrap + 优势过滤行为正则 α=0.2；**不写回**——写回嫌疑
      （§三假设 2）随架构消灭；bootstrap 暂用 online 网络，target 网接线仍留条件项）。
      builder `.rosmo(true)`、recipe 默认关；7 项单测 + CartPole 回归闸门载体
      `rosmo_cartpole_bench.rs`（含预注册判读，兼作实现 bug 探测器——ROSMO 弱网鲁棒，
      若 CartPole 上仍灾难性失败则强指向管线 bug，可裁 §三假设 4）
- [x] **CartPole 回归闸门实测绿（2026-07-04）**：promoted recipe + `.rosmo(true)` × seeds
      42/43/44 = **3/3 达标**（101,657 / 27,447 / 29,585，中位 **29,585**，greedy 500/486/500），
      在预注册绿灯带内（≤ 哨兵基线 ~8.7k 的 5×；比基线慢 ~3.4× 属预期——样本免费环境
      吃不到省样本红利，一步 target 弱于新鲜数据自带的 MCTS target）。
      **§三假设 4 裁决**：同一管线底座换一步刷新即正常收敛 → 旧灾难（greedy 钉死 9.4）
      主因是**机制病理**（弱网全树重搜 + 写回持久化投毒），非底座实现 bug。
      日志 `.bench/rosmo_gate_20260704.log`；CartPole recipe 零变更（条款二）
- [ ] 阶梯一价值裁决：图像域（Phase 1）ROSMO 式一步刷新 vs 无 reanalyze 基线，3-seed 样本效率有增益
      （前置：S2 基线须先立起来——2026-07-04 复跑仍 0/3 平直，见 Pong 负结果 issue）
- [ ] 阶梯二（条件项）：一步版增益确认后，全树 reanalyze+写回单变量复测（§三假设 2/3 随此臂验证）
- [ ] CartPole 仅回归覆盖（条款二），不再作为价值裁决场
- [ ] 若接 target net：训练循环接线 + 与刷新引擎联调（Phase 3 target_net 项可能前移）

---

## 五、当前决策

- `recipe.rs`：`CartPole-v1` = **consistency + reconstruction + Sampled**（内置），`reanalyze = false`
- 写回路径保留；`Components.reanalyze` 仍可用于内部消融 / 其他 env
- 不在此 issue 内改 train_batch 真 batched unroll 或 Rayon 并行（正交）

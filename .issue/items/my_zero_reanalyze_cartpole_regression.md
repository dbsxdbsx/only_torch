---
status: active
created: 2026-06-20
updated: 2026-06-20
owners: []
reviewers: []
---

# MyZero · Reanalyze + 写回：CartPole 学习失效（暂缓 promote）

> **状态**：active —— 写回闭环已实现并单测覆盖；CartPole recipe **已关** `reanalyze`，consistency-only 回归正常。
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

- [ ] 至少一条：CartPole greedy ≥475 且 env-steps 不劣于 consistency-only 基线
- [ ] 或：Atari / 数据受限 env 上证明 reanalyze+写回有增益（CartPole 可永久 ⏸）
- [ ] 若接 target net：训练循环接线 + 与 reanalyze 联调

---

## 五、当前决策

- `recipe.rs`：`CartPole-v1` = **consistency only**，`reanalyze = false`
- 写回路径保留；`Components.reanalyze` 仍可用于内部消融 / 其他 env
- 不在此 issue 内改 train_batch 真 batched unroll 或 Rayon 并行（正交）

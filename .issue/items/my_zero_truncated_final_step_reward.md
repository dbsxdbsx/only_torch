---
status: active
created: 2026-07-03
updated: 2026-07-03
owners: []
reviewers: []
---

# MyZero · truncated 局末步 reward 无监督（与 terminated 局轻微不对称）

> **状态**：active —— 已在代码处显式登记语义（`runner.rs` `unroll_len_at` doc 注释），本 issue 记录完整分析与修复方案，等训练器支持逐位置 mask 后收口。
> **来源**：2026-07-03 逻辑审查（哨兵退化排查附带发现，非退化根因）。
> **代码**：`src/rl/algo/my_zero/runner.rs`（`unroll_len_at` / `train_batch` 目标构造）

---

## 一、现象

`unroll_len_at` 对 episode 末端的处理不对称：

| 局结局 | actual_k | 末步 reward 是否被监督 |
|---|---|---|
| terminated（真终止） | 恒 `k_unroll`（越界走 absorbing padding：reward=0 / cont=0 / uniform policy） | ✅ |
| truncated（time-limit）/ in-progress | `k_unroll.min(len - 1 - start)` | ❌（`start = len-1` 时 `actual_k = 0`） |

CartPole 满 500 步的成功局都是 truncated——最后一个 transition 的 reward 永远不进训练目标。

## 二、为什么当前不能直接纳入

若把 truncated 局的 unroll 放宽到 `len - start`，unroll 末端位置的 value / policy 目标
对应「truncation 之后的状态」：该状态**非终止**，正确目标应为 n-step bootstrap
（`compute_n_step_target` 对越界位置返回 0，语义是 absorbing，不适用），uniform policy
padding 同理。当前 batch-native 训练器（`train_batch` 按 `(actual_k, next_obs 步数)`
分组堆叠）**没有逐位置 loss mask**，无法「只训 reward、屏蔽 value/policy/consistency」。
强行纳入 = 用 absorbing 假目标污染非终止状态的 value，弊大于利（v0.25 曾修过同族问题：
continuation soft 折扣系统性压低好状态 value）。

## 三、影响评估

- 每条 truncated 局仅损失 **1 个 transition 的 reward 监督**（CartPole 500 步局 ≈ 0.2% 数据）；
- CartPole reward 恒为 1，该步信息量几乎为零——对哨兵数字无可测影响；
- reward 稀疏且末步关键的环境（如整局只有终局得分的棋类）才需要认真对待。

## 四、修复方案（等训练器能力）

1. `UnrollItem` 增加逐位置 loss mask（reward / value / policy / consistency 各自独立）；
2. truncated 局放宽 `actual_k` 到 `len - start`，末位置只开 reward + continuation mask，
   value / policy / next-obs 系全屏蔽；
3. 守门测试：构造 truncated final-step 样本，断言 reward head 收到监督且 value head 无假目标。

触发条件：图像线 / 棋类（Gomoku self-play）出现「末步 reward 关键」的环境时实施。

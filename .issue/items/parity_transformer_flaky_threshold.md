---
status: active
created: 2026-07-04
updated: 2026-07-04
owners: []
reviewers: []
---

# parity_transformer_var_len 示例：训练准确率贴门槛 flaky（68.5% vs 70%）

> **状态**：active —— 存量问题，与任何近期改动无关（2026-07-04 在干净 master
> `4ec2eb0` 上复现同值 68.5%）。非关键路径：该示例本就用于展示 transformer 在
> parity 这类 counter 任务上的**固有限制**（O(1/n) sensitivity，见示例 README），
> 70% 门槛只是「学到一点但学不好」的证据线。

## 现象

- `just example-parity-transformer-var-len`（debug 构建）以
  `ComputationError("训练最佳准确率 68.5% 未达到目标 70%")` 退出码 1 失败。
- 2026-07-04 两次运行（含 RNN 改动工作区 + stash 后干净 master）均为 **68.5%**，
  数值稳定复现——更像「当前默认 seed/数值流下就是 68.5%」而非 run-to-run 抖动。
- 该示例是 `just examples-memory-unit` 聚合关卡的一环，会让聚合命令整体报红。

## 疑似诱因（未验证）

- 示例上一次全绿验证在 ndarray 0.16/0.17 升级 + BLAS 轨迹漂移（见
  `cartpole_sentinel_red_ndarray_drift.md`）之前；transformer 对初始化/数值流
  敏感，漂移后贴线掉到门槛下方是合理假设。
- 与 2026-07-04 RNN 输入投影批量化无关（transformer 路径未动，干净 master 复现）。

## 候选处理方向（择一，待安静窗口）

1. 换 seed / 微调超参（lr、epoch 数）让它回到门槛上方——最便宜。
2. 门槛从 70% 降到 65%（与「固有限制」叙事一致，README 同步说明）。
3. 若 ndarray 漂移复裁（哨兵 issue）后数值流再变，先复跑本例再决定。

## 相关

- 示例：`examples/traditional/parity_transformer_var_len/`
- 聚合命令：`justfile` → `examples-memory-unit`
- 关联 issue：`.issue/items/cartpole_sentinel_red_ndarray_drift.md`（数值流漂移背景）

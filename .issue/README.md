# Issue Records

`.issue/` 是项目内的**未闭环问题日志**：保留暂时无法收尾、但不能丢失上下文的现场，避免下次继续时重复排查。

它不是任务清单，也不是长期知识库：

- 普通待办放在 `README.md` TODOs、项目管理工具或当前任务计划。
- 已验证、可复用的架构 / 配置 / 排障经验沉淀到 [`.doc/`](../.doc/README.md)。
- 多轮调试后仍未闭环、需要暂缓的问题记录到 `.issue/items/`。

设计参考：[Architecture Decision Records](https://adr.github.io/) 的目录与文件约定、Diátaxis 的 reference 分类。

## Directory Contract

```text
.issue/
├── README.md                 # 本文件：契约 + 模板
├── items/                    # 当前仍需关注的问题条目（平铺）
├── assets/                   # 当前问题附件，按条目 stem 分组（若有）
└── _archive/                 # 已闭环 / 已替代的历史条目（本仓库平铺 .md）
    └── assets/               # 归档附件（若有）
```

约定：

- `items/` 只放**当前仍需关注**的条目；不要把条目直接放在 `.issue/` 根。
- 附件目录名与条目文件名（去掉 `.md`）一致；条目内用相对路径引用。
- 本仓库历史归档目前平铺在 `_archive/*.md`（未强制 `_archive/items/` 子目录）；新建归档可继续平铺，或按需迁入 `_archive/items/`，但须同步修正相对链接。
- 归档前须将 frontmatter `status` 改为 `resolved` 或 `superseded`，刷新 `updated`，并在正文补结论 / 根因 / 修复或替代方式 / 验证。

## Naming

- 条目文件名：优先 `YYYY-MM-DD_<topic>.md`（英文 `snake_case`）。历史条目存在无日期前缀的短名（如 `cpu_only_mcts_image_realtime_risk.md`），保留不强制改名。
- 一个条目只记录一个未闭环问题。

## Entry Template

```markdown
---
status: suspended
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# 问题标题

## 背景

## 现象 / 影响

## 已尝试

## 当前卡点

## 暂缓原因

## 下次恢复条件

## 下一步建议

## 相关文件 / 命令 / 对话
```

字段：

- `status`：`suspended`｜`blocked`｜`resolved`｜`superseded`（另有历史值如 `active`，新条目请用上表）
- `created` / `updated`：ISO 日期 `YYYY-MM-DD`

## When to Add / Archive

适合新增：多轮未解的 bug / 构建 / 依赖 / 环境问题；缺权限或决策必须暂停；已有尝试路径需保留以免重踩。

不适合：立即可做的小 TODO；已有明确修复方案的问题；纯想法；已可写入 `.doc/` 的稳定结论。

适合归档：已解决并写清验证；被新条目 / PR / `.doc` 替代且标注 `superseded`；不再影响当前开发但仍有追溯价值。

## RL 相关条目

强化学习战略与组件裁决以 [`.doc/design/rl_myzero_status.md`](../.doc/design/rl_myzero_status.md) 为准。`.issue/items/` 下 `my_zero_*` / `pendulum_*` / `pong_*` / `gomoku_*` / `cpu_only_*` / `commercial_*` 等只记**现场与负结果**，不另立路线图；关联链接应指向状态总览或各环境数字账本。

## Current Items（索引）

| 条目 | 主题 |
|------|------|
| [cpu_only_mcts_image_realtime_risk.md](items/cpu_only_mcts_image_realtime_risk.md) | CPU × 图像 × MCTS × 实时一级风险 |
| [commercial_realtime_game_target_profile.md](items/commercial_realtime_game_target_profile.md) | 商业图像游戏目标画像 |
| [gomoku_naive0_tactical_wall.md](items/gomoku_naive0_tactical_wall.md) | Gomoku naive0 战术墙 |
| [pendulum_failure_diagnosis.md](items/pendulum_failure_diagnosis.md) | Pendulum 失败诊断 |
| [my_zero_reanalyze_cartpole_regression.md](items/my_zero_reanalyze_cartpole_regression.md) | reanalyze CartPole 回归 |
| [my_zero_pong_image_flat_negative.md](items/my_zero_pong_image_flat_negative.md) | Pong 图像平直负结果 |
| [my_zero_pomdp_lite_posterior_negative.md](items/my_zero_pomdp_lite_posterior_negative.md) | POMDP-lite posterior 负裁 |
| [my_zero_action_space_sampled_policy.md](items/my_zero_action_space_sampled_policy.md) | Sampled 动作空间 |
| [my_zero_truncated_final_step_reward.md](items/my_zero_truncated_final_step_reward.md) | truncated 终步 reward |
| [pong_ep53_panic_investigation.md](items/pong_ep53_panic_investigation.md) | Pong Ep53 panic |
| [parity_transformer_flaky_threshold.md](items/parity_transformer_flaky_threshold.md) | Transformer parity flaky |
| [proc_macro_error2_future_incompat.md](items/proc_macro_error2_future_incompat.md) | proc-macro-error2 告警 |
| [trivial.md](items/trivial.md) | 杂项占位 |

归档条目见 [`_archive/`](_archive/)。

---
status: suspended
created: 2026-08-14
updated: 2026-08-14
---

# Gomoku bench 文件名仍用 M1 / M3 / P3A0 里程碑号

## 背景

棋盘立柱期（2026-07）用 M0–M4、G1–G3、Phase 3A0 当建设里程碑。发版后活文档已少写这套番号，但手动档载体文件名没改。2026-08-14 讨论：职责还在，番号像过渡期记号，想到再改。

## 现象 / 影响

四份载体的**职责**仍清楚，**文件名**已经对不上现在怎么说话：

| 文件 | 历史号 | 实际职责 |
|------|--------|----------|
| `gomoku_m1_bench.rs` | M1 | `just smoke-my-zero-gomoku` 的 3 局全路径冒烟 |
| `gomoku_m2_bench.rs` | M2 | 棋盘官方 3-seed 基线（双门槛） |
| `gomoku_m3_bench.rs` | M3 | 先是九臂消融，后来塞进 ①–⑱ / G1–G3，名不副实 |
| `gomoku_p3a0_bench.rs` | Phase 3A0 | model-error proxy 审计；主动数据已负裁，只留可复现 |

新人会以为「还在做 M3」。改名会牵动 `just`、账本命令、`.bench/gomoku_m3_*` 日志、issue 链接、`M3_SEEDS`。

## 已尝试

- 口头对齐：留文件、说话改用「冒烟 / 基线 / 消融本 / model-error 审计」。
- 未改文件名，避免和旧数字对不上。

## 当前卡点

没有执行改名。账本 [命名词典](../../examples/my_zero/gomoku/README.md#命名词典) 仍在解释 M0–M4。

## 暂缓原因

RL 主线暂缓；改名是文档/仓库卫生，不挡训练。现在改成本高于收益。

## 下次恢复条件

用户明确要收拾 Gomoku bench 命名，或棋盘线重新开工、准备动这些文件时。

## 下一步建议

按**职责**改，不要换一套新番号：

- `gomoku_m1_*` → `gomoku_smoke`
- `gomoku_m2_*` → `gomoku_baseline`
- `gomoku_m3_bench.rs` 拆开，或至少改成 `gomoku_ablation.rs`（臂名已有 `pure_selfplay_*`）
- `gomoku_p3a0_*` → `gomoku_model_error_audit`

改时同步：`justfile`、`tests/mod.rs`、账本运行命令、`.issue` 链接、`M3_SEEDS`（或改名并留一版别名）。旧 `.bench` 日志文件名可保留，账本用一句话指向新载体。

## 相关文件 / 命令 / 对话

- 载体：`src/rl/algo/my_zero/tests/gomoku_{m1,m2,m3,p3a0}_bench.rs`
- 冒烟：`just smoke-my-zero-gomoku` → `gomoku_m1_smoke`
- 账本：[examples/my_zero/gomoku/README.md](../../examples/my_zero/gomoku/README.md)
- 3A0 负裁见账本「Phase 3A0」节；战术墙现场仍是 [gomoku_naive0_tactical_wall.md](gomoku_naive0_tactical_wall.md)

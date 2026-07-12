# 性能基准历史存档（.bench/）

Criterion 的原始数据在 `target/criterion/`，属于**构建产物**（`/target` 已 gitignore），
`just clean` 会整块删除、也不入仓——因此长期趋势会随清理丢失（曾丢过 4 月的
`pre-execution-context` / `post-mode-refactor` 两个基线）。

本目录把每次值得留存的 baseline **快照入仓**，用于看「整个历史长河」的性能趋势
（到底是持续优化还是出现回归）。

## 目录结构

```
.bench/history/<时间戳>-<baseline名>[-<label>]/
├── meta.txt                       # 环境快照（见下）
├── <case_1>/estimates.json        # Criterion 均值 / 置信区间（核心指标）
├── <case_1>/benchmark.json        # case 元信息（分组名等）
└── ...
```

每个 case 仅存 `estimates.json`（均值、中位、置信区间）+ `benchmark.json`，
一次全量快照约几百 KB。不存原始采样点（体积大且无长期趋势价值）。
因此入仓快照适合查看历史趋势，**不能还原 Criterion 的跨版本统计检验**；工具链
A/B 必须在清理 `target/criterion` 前完成，或另行保留完整本地报告。

## meta.txt 字段

记录归档动作发生时的仓库与环境状态，用于**跨环境点的可比性判断**：

- `git_commit` / `git_dirty`：基线数据对应的代码状态（dirty=yes 表示有未提交改动）。
- `rustc` / `rustc_host` / `llvm` / `cargo`：完整工具链指纹。
- `blas` / `blas_feature` / `profile` / `cargo_target_dir`：后端、Cargo feature、
  测量 profile 与 Criterion 数据来源目录。
- `os` / `cpu`：操作系统与处理器型号。
- `rustflags` / `*_num_threads`：会改变代码生成或线程调度的关键环境变量。

`schema: 2` 起记录上述完整字段；旧快照没有的字段视为 `unknown`。**换了编译器、
后端、profile、机器或线程配置的点不能直接比百分比**——趋势线上出现跳变时，
先查 meta 是不是环境变了，再判定是否代码回归。

## 用法

```bash
# 跑基线并保存到 Criterion（改动前）
just bench-save pre-hotpath-opt

# 归档该基线到本目录（可选 label 便于识别）
just bench-archive pre-hotpath-opt before-hotpath

# 改动后同法归档，或直接对比
just bench-compare pre-hotpath-opt
```

## 与 .doc/performance/ 的分工

- [`.doc/performance/optimization_log.md`](../.doc/performance/optimization_log.md)：人工账本（战报），写 before/after 摘要数字 + 结论；流程与 baseline 台账见同目录 [`benchmark_workflow.md`](../.doc/performance/benchmark_workflow.md)。
- `.bench/history/`：机器原始数据，供脚本画趋势线或精确回溯。

两者互补：账本讲「为什么、结论如何」，存档留「精确数值、可复算」。

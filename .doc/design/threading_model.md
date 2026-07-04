# 线程模型与 Rayon 定位

> **定位**：回答三个问题——Rayon 在本框架中扮演什么角色、与用户自己的并行代码如何共存、什么时候并行什么时候串行。
> 创建：2026-07-04（热路径审计 Reviewer 方向 2 收口时定稿）。

## 1. 总原则

| 层 | 并行方式 | 理由 |
|---|---|---|
| **op 级（intra-op）** | rayon **全局线程池**，per-sample / per-row 数据并行，带小任务阈值分流（§3） | 与用户侧 rayon 共享同一池、天然可组合；绝不自建常驻池 |
| **演化级（inter-candidate）** | 显式 scoped `ThreadPool` + `pool.install`（`evolution/mod.rs::evaluate_batch`） | 评估并行度需独立可配；`install` 内的 op 级并行自动继承该池、并行度有界 |
| **BLAS（GEMM 内部）** | **单线程**（MKL `seq` 配置） | 避免 MKL 线程 × rayon 线程过订阅；多核利用交给 per-sample rayon |
| **RL 主流程** | 单线程（`Python::attach` 内） | pyo3 桥约束；profiling 桶因此可用 thread_local |

## 2. 与用户侧 Rayon 的共存（嵌套安全性）

用户在自己的高层项目里用 rayon、内部再调 only_torch，是**安全且高效**的组合：

- **全进程只有一个线程池**：cargo 将同大版本 rayon 统一为唯一 crate 实例（本库 rayon 1.x），
  用户与本库共享同一全局池——不存在"两套池、2×核数线程"的爆炸。
- **嵌套并行是 rayon 的一等公民**：用户 `par_iter` 内调本库 op，内层 `par_iter` 不建新线程,
  只把任务切片喂给同一工作窃取池；池饱和时内层优雅退化为"基本在本线程串行 + 偶尔被窃取"。
- **死锁面已被类型系统封死**：op 并行区只写裸 buffer、不持锁；`Graph` 本身 `!Sync`,
  用户无法把同一张图塞进多线程。
- **用户可控**：`RAYON_NUM_THREADS` 环境变量配置全局池；或用自己的
  `ThreadPool::install` 包住调用，本库内层并行自动继承该池。

真正的开销场景只有一种：外层池已饱和时，内层 par 的切分调度是纯开销——
这正是 §3 阈值分流要消的（小任务本来就该串行）。

已知边界（预期行为，非 bug）：
- 依赖树若出现两个 rayon **大版本**共存（0.x + 1.x）会有两个池——`Cargo.lock` 目前唯一 1.x。
- 用户把 `RAYON_NUM_THREADS` 设小，本库 op 吞吐随之下降。

## 3. 小任务阈值分流（`utils::parallel`）

**问题**：per-sample 并行在 batch=1 或小张量时，rayon fork-join 调度开销（µs 级）盖过收益。
**定标**（`benches/rayon_threshold.rs`，2026-07-04，release+MKL）：

| 总工作量（元素×每元素运算） | 串行 vs 并行 |
|---|---|
| ~128（2×64） | 串行快 ~9×（0.5µs vs 4.8µs） |
| ~8k | 串行快 2~4× |
| ~33k（8×~4k） | 交叉点附近 |
| ≥65k | 并行稳定胜 1.5~8× |

**落地**：`PAR_MIN_WORK = 32768`；统一入口 `for_each_chunk_mut` / `for_each_chunk_mut2` /
`map_indexed`（`src/utils/parallel.rs`），串行/并行执行体为同一闭包，产物**逐 bit 一致**。
接入点：conv2d（pad/前向/1x1/dX/dK）、conv_transpose2d（前向/dX/dK）、max/avg_pool2d、
upsample2d、batch_norm dx、softmax/log_softmax 反向。
`batched_mat_mul` 特殊：按**单 GEMM flops** 判断（多而微的 GEMM 形态并行实测回归 +20~40%,
attention bench 定标），单 GEMM ≥ 阈值才并行。

**适用边界（铁律）**：阈值分流**仅限 map-only 场景**（各 chunk 独立写、无跨 chunk 归约）。
跨样本归约（CE 前向 par sum 等）的串行/并行切换会改变 f32 累加顺序——RL 数值冻结期间不动,
解冻后若要动按"一次一项"走消融。

## 4. f32 可复现性纪律

**禁用 rayon `reduce` / `sum` 做 f32 跨样本归约**：其合并分组随工作窃取漂移,
同输入两次运行结果可能不逐 bit 相同,破坏"逐 bit 金测试"方法论与 3-seed 复现口径。
既定写法：并行 `map` 收集（`collect` 保序）→ **按 batch 序串行累加**
（conv2d / conv_transpose2d 的 dK 已按此修复,2026-07-04）。
遗留：`softmax_cross_entropy` 前向 loss 标量仍是 par `sum()`（梯度路径不经过它,
只影响日志/收敛判据的潜在 run-to-run 漂移）——挂候选表,数值冻结解除后修复。

## 5. 分配画像（诊断工具）

`alloc-profile` feature（默认关）：全局计数 allocator（只做两个 atomic 计数,
见 `utils/alloc_profile.rs`）+ `PROFILE` 命名桶 delta 归因（`rl/profiling.rs`）。
用途：为候选表 #6（小张量 Arc 分配）/#7（视图算子）的"profiler 证明进入热点"
触发条件提供客观证据。用法：

```bash
PROFILE=1 cargo run --example my_zero_cartpole --release --features "blas-mkl alloc-profile"
```

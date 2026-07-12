# Benchmark 工作流与基线台账

> 本文档是性能工作的**流程契约**：验证流程、测量纪律、当前 baseline 台账、benchmark 基础设施。
> 文档分工：候选与否决看 [optimization_candidates.md](./optimization_candidates.md) · 已实施战报看 [optimization_log.md](./optimization_log.md) · 全局策略与架构约束看 [optimization_strategy.md](./optimization_strategy.md)。
> 最后更新: 2026-07-12

---

## 性能验证标准流程

任何声称"优化"或"重构"的改动，必须按以下流程验证：

1. **保存 baseline**：`just bench-save before-change`（在改动前的 commit / 工作树上）
2. **执行改动**
3. **正确性验证**：优先 `just test-filter <pattern>`，影响面大时跑 `just test`
4. **快速性能回归检查**：`just bench-smoke`（目标整组约 30 秒内）
5. **对比关键路径**：`just bench-compare before-change`（查看 Criterion 输出与 HTML report）
6. **Macro 验证**：`just bench-macro`（缺少本地 chess ONNX 模型时先跑 `just bench-macro-core`）
7. **回填战报**：在 [optimization_log.md](./optimization_log.md) 新增条目，写明 before / after 数据、命令与环境

宏基准用于真实用户路径参考，不作为默认门禁；不要把不同 CPU、不同 BLAS 后端或不同模型文件的百分比直接混为同一条趋势。

## 测量纪律（历次战役沉淀，出处见战报 J/K）

- **基线录制与对比跑尽量同为安静窗口**；白天对比只做方向参考，跨时段存在 ~10% 系统性环境偏移。
- **白天窗口跑 bench 前先跑一个未触碰组做金丝雀**（如 `tensor_clone`、纯 MKL `tensor_matmul`）；金丝雀齐涨 = 环境污染，数字不可用。
- **初跑异常项必须复测**：离群回归经常复测翻转（先例：+226% → No change）。
- **归因存疑时用对照实验锤死**：stash 旧代码同窗复跑、或 worktree 检出基线 commit 复跑（战报 J/K/L 与依赖升级条目均有先例）。
- 跨环境 / 跨基线的百分比不混为同一条趋势。

## Rust 工具链升级 A/B

编译器升级必须把两个问题分开：

1. **迁移裁决**：旧 / 新完整工具链在同一源码上的正确性与性能差异。
2. **单项归因**：例如 symbol mangling；只有迁移裁决发现异常后，才用同一编译器切
   单个选项做控制实验，不能把 `rustc A → B` 的全部差异归给某一 release note。

以 Rust 1.95.0 → 1.97.0 为例：

1. 在更新 `stable` 前显式安装两端版本：`rustup toolchain install 1.95.0 1.97.0`。
2. 固定同一 clean commit、BLAS feature、线程环境、电源模式与安静窗口；先跑
   `just blas-status` 并记录环境。
3. 新工具链先过正确性门禁：
   `rustup run 1.97.0 just check` → `just test` → `just lint`。命令中的后两项也应
   放在同一个 `rustup run 1.97.0 ...` 上下文中执行。
4. Criterion 使用同一专用 `CARGO_TARGET_DIR` 保留原始样本，按 **A-B-B-A**：

   ```bash
   # A1：旧工具链保存 baseline
   CARGO_TARGET_DIR=target/toolchain-runtime cargo +1.95.0 bench \
     --bench smoke --features blas-mkl -- --save-baseline rust-1.95
   # B1：新工具链与 A1 比较
   CARGO_TARGET_DIR=target/toolchain-runtime cargo +1.97.0 bench \
     --bench smoke --features blas-mkl -- --baseline rust-1.95
   # B2：保存新工具链 baseline
   CARGO_TARGET_DIR=target/toolchain-runtime cargo +1.97.0 bench \
     --bench smoke --features blas-mkl -- --save-baseline rust-1.97
   # A2：旧工具链反向比较，排除时段漂移
   CARGO_TARGET_DIR=target/toolchain-runtime cargo +1.95.0 bench \
     --bench smoke --features blas-mkl -- --baseline rust-1.97
   ```

   代表集合为 `smoke`、`my_zero_forward`、Attention 大形状、
   Conv2d board/Atari 真形状与一个 `end_to_end` 训练步；先按工具链聚合运行，
   避免每个 case 来回切换 rustc 触发重复 LTO。
5. Criterion 固定走 `[profile.bench]`，但 RL 长跑走 `[profile.release]`；因此再用两端
   工具链各跑一次 `just spike-cnn-mcts`，覆盖真实 thin-LTO release 路径。
6. 编译 / 链接性能使用**各自独立且起始不存在**的 `CARGO_TARGET_DIR` 做 clean build；
   不与 Criterion 的共享目录混用。记录总耗时与代表性产物体积。
7. 检查新工具链的 linker 输出、panic backtrace，以及日常 debugger / profiler 能否
   解析符号。仓库脚本若不读取符号名，无需为了工具链升级临时引入新 profiler。
8. A/B 完成前不要清理 `target/toolchain-runtime/criterion`；`.bench/history/` 只保存
   estimates，不含原始采样，不能还原 Criterion 的统计比较。

切换项目 `rust-toolchain.toml` 后，终端中的 `cargo +<version>` / `rustup run` 都会启动
新进程，**无需关闭 IDE 才能让 benchmark 生效**；只需重启 rust-analyzer（或 Reload
Window）刷新编辑器诊断。只有修改了 Windows 持久环境变量且旧进程必须继承时，才需要
新开终端或重启 IDE。

## 当前 baseline

| baseline | 日期 | 命令 | 后端 | 用途 |
|---|---|---|---|---|
| `pre-execution-context` | 2026-04-29 | `just bench-save pre-execution-context` | `blas-mkl` | `Mode` 重构前的 Criterion 对照基线 |
| `post-mode-refactor` | 2026-04-29 | `just bench-save post-mode-refactor` | `blas-mkl` | `Mode` 重构完成后的新命名完整基线，后续性能回归从这里继续比较 |
| `pre-hotpath-opt` | 2026-07-03 | `just bench-save pre-hotpath-opt` | `blas-mkl` | 热路径分配/拷贝消除批量优化（见[战报 J](./optimization_log.md)）前的完整基线；已快照入仓 `.bench/history/20260703-000547-pre-hotpath-opt-before-hotpath/`（104 case） |
| `post-hotpath-opt` | 2026-07-03 | `just bench-save post-hotpath-opt` | `blas-mkl` | 优化 J 全部落地（含 Add 分流修复 `3027ac0`）后的新基线；已快照入仓 `.bench/history/20260703-093545-post-hotpath-opt-after-hotpath/`（104 case，conv2d 前向 / max_pool 前向组经安静复测覆盖首跑受扰值）。注意与 pre 基线跑于不同时段，存在 ~10% 环境偏移（见战报 J 噪声鉴定），跨基线百分比勿直接混读 |
| `post-c1-20260704` | 2026-07-04 | `just bench-save post-c1-20260704` | `blas-mkl` | 战报 N/O/P/Q/R + C1（3D batched MatMul attention 重写）全部落地后的完整基线（133 case），**后续性能回归从这里比较**；已快照入仓 `.bench/history/20260704-142938-post-c1-20260704-full-baseline/`。⚠️ 白天窗口录制（13:21–14:29）且金丝雀鉴定显示环境污染（纯 MKL `tensor_matmul/32x32` vs 07-03 晨间基线 +105%、`tensor_negate` +95% 齐涨），**绝对值整体偏高**——用于对比前先复跑金丝雀组换算，或安静窗口重录覆盖。**forward-only bench 测量语义修复同日落地**：conv2d/pool/norm 前向组（含 smoke 同名 case、`two_layer_cnn`）此前只建图不求值（框架惰性求值），实测的是「入图拷贝 + 建图」，战报 O Arc/CoW 后更塌缩成无信息量常数；已补显式 `out.forward()` 强制求值并同批重跑覆盖本基线，历史快照中这些组的数字与本基线**不可比**。另注：`group_norm_backward` 绝对值 ~10ms 是 2026-07-03 梯度断流修复后首次真正计算全图梯度（修 bug 的代价，非性能回归） |
| `pre_lowering` | 2026-07-05 | `cargo bench --bench conv2d/end_to_end -- --save-baseline pre_lowering` | `blas-mkl` | Conv2d 隐式 lowering（[战报 S](./optimization_log.md)）前的 conv2d + end_to_end(cnn) 局部基线，含真形状裁决器新 case（board15/19、atari84/42、b128 组）在旧实现上的首录。⚠️ 白天窗口，同窗 A/B 复核发现未变路径 case 漂移 ±40~80%，百分比只看方向；conv2d 节点级最稳测量仪改用 `conv2d_timing_probe`（`#[ignore]` 手动档，建图一次循环重执行） |
| `rust-1.95` | 2026-07-12 | 工具链 A/B 代表集合 A 侧 | `blas-mkl` | Rust 1.95.0、LLVM 22.1.2；18 case（smoke 7 + MyZero 3 + Attention 2 + Conv2d 真形状 6），已入仓 `.bench/history/20260712-123811-rust-1.95-rust-1.97-ab-old/` |
| `rust-1.97` | 2026-07-12 | 工具链 A/B 代表集合 B 侧 | `blas-mkl` | Rust 1.97.0、LLVM 22.1.6；同 18 case，已入仓 `.bench/history/20260712-123819-rust-1.97-rust-1.97-ab-new/`。仅作工具链迁移裁决，不替代下一次优化战役的完整 baseline |

说明：baseline 保存在 Criterion 的 `target/criterion` 报告目录中，属于本地构建产物，不入仓。`pre-execution-context` 仅用于解释 `Mode` 重构前后差异；`smoke_conv2d_eval_1x1_b1` 已随语义改名为 `smoke_conv2d_inference_1x1_b1`，后续不再为重构前 baseline 兼容名称，统一从 `post-mode-refactor` 继续对比。

> **`post-hotpath-opt` 对新依赖栈仍有效（2026-07-03 裁定）**：ndarray 0.17.2 + pyo3/numpy
> 0.29 升级经三层归因（噪声金丝雀 / 复测翻转 / worktree 控制实验，详见[战报「依赖升级」条目](./optimization_log.md)）判定性能中性，
> 该基线无需因依赖升级重录；下次优化战役开跑时照常 `just bench-save pre-<名>` 即自然获得
> 新栈起点。唯一纪律不变：**基线录制与对比跑尽量同为安静窗口**，白天对比只做方向参考。

### Rust 1.97.0 升级裁决（2026-07-12）

- **正确性**：`just check`、`just test`、`just lint` 全绿；主测试 3421 passed，
  `just smoke-rl` 全绿。未出现 `linker_messages`；唯一提示是间接依赖
  `proc-macro-error2 2.0.1` 的 future-incompat lint，当前不阻断 1.97。
- **Criterion 运行时**：18 个代表 case 按 A-B-B-A 跑完；首轮同时出现改善与回归，
  聚焦 50-sample 复测中 CNN / GroupNorm 又随顺序反转，确认机器存在明显时段漂移。
  未发现跨两轮、跨组一致的整体回归，因此不把任一单点百分比归因给 symbol mangling。
- **真实 release 路径**：CNN × MCTS spike 两版重复运行均为 `GO`。最后一个紧邻
  A/B 对中，核心 `flat_s20` 为 1.648 → 1.410 ms，84×84 train step 为
  45.10 → 41.33 ms；其他亚毫秒组件有小幅双向变化，只裁定“无实质回归”，不宣称提速。
- **clean bench build**：两种顺序均显示 1.97 稳定慢约 4–5%
  （A→B：2m51s → 2m59s；B→A 独立目录：2m46s vs 2m39s）。这是完整 rustc /
  Cargo / LLVM 升级的编译期开销，不能单独归因于 v0 mangling；幅度可接受。
- **产物**：代表 smoke EXE 11,259,392 → 11,235,328 bytes（-0.21%），PDB
  2,150,400 → 2,158,592 bytes（+0.38%），总体中性。
- **符号工具**：`RUST_BACKTRACE=1` 的 1.97 panic 栈可正确 demangle 项目函数。
- **结论**：升级通过，项目由 `rust-toolchain.toml` 固定到 1.97.0。IDE 无需关闭；
  切换后重启 rust-analyzer / Reload Window 即可刷新编辑器侧状态。

---

## Benchmark 基础设施

已引入 `criterion` 框架，15 个 benchmark 文件覆盖各层面：

| 文件 | 覆盖范围 | 场景 |
|------|---------|------|
| `benches/tensor_ops.rs` | Tensor 底层操作 | clone/add/mul/negate/matmul/where（7 组） |
| `benches/backward.rs` | 节点反向传播 | Add/Negate/Subtract 链路 + MLP backward（4 组） |
| `benches/conv2d.rs` | Conv2d 卷积 | forward/full_step/two_layer_cnn（3 组） |
| `benches/end_to_end.rs` | 端到端训练步 | MLP(XOR/MNIST)/CNN(MNIST) × 多 batch_size（2 组） |
| `benches/smoke.rs` | 快速性能回归 | Tensor / Conv2d / MLP / CNN / Add backward 主链路 |
| `benches/pool2d.rs` | Pool2d | MaxPool2d / AvgPool2d forward + backward |
| `benches/optimizer.rs` | 优化器 | SGD / Adam step |
| `benches/normalization.rs` | 归一化层 | BatchNorm / LayerNorm / RMSNorm / GroupNorm |
| `benches/loss.rs` | Loss | MSE / CrossEntropy / BCE / Huber forward + backward |
| `benches/rnn.rs` | 循环层 | RNN / LSTM / GRU 小规模序列 forward + backward |
| `benches/attention.rs` | Attention | MultiHeadAttention self-attention / cross-attention forward + backward |
| `benches/rayon_threshold.rs` | 并行阈值 | 串行 / Rayon 在不同 chunk 网格下的交叉点 |
| `benches/my_zero_forward.rs` | MyZero 搜索前向 | root initial_state + recurrent（离散 / MultiDiscrete） |
| `benches/my_zero_train_batch.rs` | MyZero 批训练 | per-sample 与 batched 的 x8 / x32 对照 |
| `benches/obs_batch_assembly.rs` | 图像观测组装 | Pong history=1/8 的旧/新组装、入图与端到端 |

```bash
# 运行所有 benchmark
just bench

# 运行特定 benchmark
just bench-backward

# 保存基准线并对比
just bench-save before
just bench-compare before
```

报告输出在 `target/criterion/` 目录。与 `.bench/`（入仓快照）的分工见 [.bench/README.md](../../.bench/README.md)。

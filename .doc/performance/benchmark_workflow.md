# Benchmark 工作流与基线台账

> 本文档是性能工作的**流程契约**：验证流程、测量纪律、当前 baseline 台账、benchmark 基础设施。
> 文档分工：候选与否决看 [optimization_candidates.md](./optimization_candidates.md) · 已实施战报看 [optimization_log.md](./optimization_log.md) · 全局策略与架构约束看 [optimization_strategy.md](./optimization_strategy.md)。
> 最后更新: 2026-07-04

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

## 当前 baseline

| baseline | 日期 | 命令 | 后端 | 用途 |
|---|---|---|---|---|
| `pre-execution-context` | 2026-04-29 | `just bench-save pre-execution-context` | `blas-mkl` | `Mode` 重构前的 Criterion 对照基线 |
| `post-mode-refactor` | 2026-04-29 | `just bench-save post-mode-refactor` | `blas-mkl` | `Mode` 重构完成后的新命名完整基线，后续性能回归从这里继续比较 |
| `pre-hotpath-opt` | 2026-07-03 | `just bench-save pre-hotpath-opt` | `blas-mkl` | 热路径分配/拷贝消除批量优化（见[战报 J](./optimization_log.md)）前的完整基线；已快照入仓 `.bench/history/20260703-000547-pre-hotpath-opt-before-hotpath/`（104 case） |
| `post-hotpath-opt` | 2026-07-03 | `just bench-save post-hotpath-opt` | `blas-mkl` | 优化 J 全部落地（含 Add 分流修复 `3027ac0`）后的新基线；已快照入仓 `.bench/history/20260703-093545-post-hotpath-opt-after-hotpath/`（104 case，conv2d 前向 / max_pool 前向组经安静复测覆盖首跑受扰值）。注意与 pre 基线跑于不同时段，存在 ~10% 环境偏移（见战报 J 噪声鉴定），跨基线百分比勿直接混读 |
| `post-c1-20260704` | 2026-07-04 | `just bench-save post-c1-20260704` | `blas-mkl` | 战报 N/O/P/Q/R + C1（3D batched MatMul attention 重写）全部落地后的完整基线（133 case），**后续性能回归从这里比较**；已快照入仓 `.bench/history/20260704-142938-post-c1-20260704-full-baseline/`。⚠️ 白天窗口录制（13:21–14:29）且金丝雀鉴定显示环境污染（纯 MKL `tensor_matmul/32x32` vs 07-03 晨间基线 +105%、`tensor_negate` +95% 齐涨），**绝对值整体偏高**——用于对比前先复跑金丝雀组换算，或安静窗口重录覆盖。**forward-only bench 测量语义修复同日落地**：conv2d/pool/norm 前向组（含 smoke 同名 case、`two_layer_cnn`）此前只建图不求值（框架惰性求值），实测的是「入图拷贝 + 建图」，战报 O Arc/CoW 后更塌缩成无信息量常数；已补显式 `out.forward()` 强制求值并同批重跑覆盖本基线，历史快照中这些组的数字与本基线**不可比**。另注：`group_norm_backward` 绝对值 ~10ms 是 2026-07-03 梯度断流修复后首次真正计算全图梯度（修 bug 的代价，非性能回归） |
| `pre_lowering` | 2026-07-05 | `cargo bench --bench conv2d/end_to_end -- --save-baseline pre_lowering` | `blas-mkl` | Conv2d 隐式 lowering（[战报 S](./optimization_log.md)）前的 conv2d + end_to_end(cnn) 局部基线，含真形状裁决器新 case（board15/19、atari84/42、b128 组）在旧实现上的首录。⚠️ 白天窗口，同窗 A/B 复核发现未变路径 case 漂移 ±40~80%，百分比只看方向；conv2d 节点级最稳测量仪改用 `conv2d_timing_probe`（`#[ignore]` 手动档，建图一次循环重执行） |

说明：baseline 保存在 Criterion 的 `target/criterion` 报告目录中，属于本地构建产物，不入仓。`pre-execution-context` 仅用于解释 `Mode` 重构前后差异；`smoke_conv2d_eval_1x1_b1` 已随语义改名为 `smoke_conv2d_inference_1x1_b1`，后续不再为重构前 baseline 兼容名称，统一从 `post-mode-refactor` 继续对比。

> **`post-hotpath-opt` 对新依赖栈仍有效（2026-07-03 裁定）**：ndarray 0.17.2 + pyo3/numpy
> 0.29 升级经三层归因（噪声金丝雀 / 复测翻转 / worktree 控制实验，详见[战报「依赖升级」条目](./optimization_log.md)）判定性能中性，
> 该基线无需因依赖升级重录；下次优化战役开跑时照常 `just bench-save pre-<名>` 即自然获得
> 新栈起点。唯一纪律不变：**基线录制与对比跑尽量同为安静窗口**，白天对比只做方向参考。

---

## Benchmark 基础设施

已引入 `criterion` 框架，11 个 benchmark 文件覆盖各层面：

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

# 性能优化候选项

> 本文档记录性能优化的候选项、已实施项和已否决项。
> 最后更新: 2026-04-29

---

## 性能验证标准流程

任何声称"优化"或"重构"的改动，必须按以下流程验证：

1. **保存 baseline**：`just bench-save before-change`（在改动前的 commit / 工作树上）
2. **执行改动**
3. **正确性验证**：优先 `just test-filter <pattern>`，影响面大时跑 `just test`
4. **快速性能回归检查**：`just bench-smoke`（目标整组约 30 秒内）
5. **对比关键路径**：`just bench-compare before-change`（查看 Criterion 输出与 HTML report）
6. **Macro 验证**：`just bench-macro`（缺少本地 chess ONNX 模型时先跑 `just bench-macro-core`）
7. **回填本文档**：在对应优化段写明 before / after 数据、命令与环境

宏基准用于真实用户路径参考，不作为默认门禁；不要把不同 CPU、不同 BLAS 后端或不同模型文件的百分比直接混为同一条趋势。

### 当前 baseline

| baseline | 日期 | 命令 | 后端 | 用途 |
|---|---|---|---|---|
| `pre-execution-context` | 2026-04-29 | `just bench-save pre-execution-context` | `blas-mkl` | `Mode` 重构前的 Criterion 对照基线 |
| `post-mode-refactor` | 2026-04-29 | `just bench-save post-mode-refactor` | `blas-mkl` | `Mode` 重构完成后的新命名完整基线，后续性能回归从这里继续比较 |

说明：baseline 保存在 Criterion 的 `target/criterion` 报告目录中，属于本地构建产物，不入仓。`pre-execution-context` 仅用于解释 `Mode` 重构前后差异；`smoke_conv2d_eval_1x1_b1` 已随语义改名为 `smoke_conv2d_inference_1x1_b1`，后续不再为重构前 baseline 兼容名称，统一从 `post-mode-refactor` 继续对比。

### Mode 重构对比结果

- `just bench-compare pre-execution-context` 跑到 `smoke` 时因 benchmark case 改名中断；中断前 `tensor_ops`、`conv2d_forward`、`backward`、`end_to_end` 多个分组相对重构前 baseline 已显示明显改善，但该 baseline 不再作为后续门禁基线。
- 已复跑 `cargo bench --bench smoke --features blas-mkl -- --baseline pre-execution-context`；6 个 smoke 项全部显著改善，关键链路未复现回归。
- 已保存 `post-mode-refactor` 完整新基线，并补跑 `smoke` 到 `attention` 后续分组；新命名 benchmark 全部跑通。
- 行为回归已用 invariants、梯度流、模型加载、BatchNorm / Conv2d 节点测试，以及 MNIST / MNIST GAN / CartPole SAC / chess YOLO ONNX example 验证；CartPole SAC 训练达到单回合 200，但三次测试平均 185.7，低于示例目标 190，仍为随机训练波动范围内。

---

## 待优化项

### 1. RNN 场景 `select` + `set_value` 二次复制问题

**位置**：`src/nn/layer/rnn.rs` → `forward`

**现状**：每个时间步两次数据复制。当前规模开销可忽略（~1.2 MB 总冗余）。

**推荐方案**：方案 A（使用已有的 `set_value_owned` 方法），最小改动。

**状态**：暂缓（YAGNI），等 RNN 处理大规模数据时再实施

---

### 2. RNN 场景更多优化

参见 #1，等 RNN 处理大规模数据时统一评估

---

## 已否决项

### `&[&NodeHandle]` vs `&[NodeHandle]`

**结论**：`&[&NodeHandle]` **不推荐**。双重间接引用增加指针追踪开销、缓存局部性差。

### 前向传播 clone NodeHandle 问题

**原始问题**：`NodeHandle.clone()` 深拷贝内部 `Tensor`。

**结论**：v2 动态图架构已彻底消除此问题。`NodeInner` 由 `Rc` 管理，前向传播通过 `borrow()` 零拷贝借用父节点值，无需 clone `NodeHandle`。

---

## 已实施的优化

### A. 赋值算子减少 clone（早期）

| 位置 | 改动 | 效果 |
|------|------|------|
| `GradientAccumulator::get_average_gradient` | `gradient.clone() / scalar` → `gradient / scalar` | 少一次 Tensor clone |
| `graph.rs` 梯度累加 | `current + contribution` → `current += &contribution` | 避免临时张量分配 |
| 反向传播 | clone `upstream_grad` → 借用引用 | 避免大 Tensor 拷贝 |

### B. Conv2d im2col + GEMM 优化（2026-02-14）

| 改动 | 效果 |
|------|------|
| 前向/反向卷积从嵌套循环改为 im2col + ndarray `.dot()` | 完整训练步 2.6-4.4x 加速 |
| 利用 ndarray 底层 matrixmultiply 库的 AVX2 自动向量化 | 无需引入外部 BLAS |

### C. 反向传播全局优化（2026-02-14）

| 改动 | 效果 |
|------|------|
| 全部 59 个节点实现 `grad_mut()`，梯度累加改为原地 `+=` | 消除每次累加的临时 Tensor 分配 |
| ReLU 反向融合 mask + multiply 为单次 `where_with_tensor` | 2 次 Tensor 分配 → 1 次 |
| MaxPool2d 反向用 `par_chunks_mut` 预分配 buffer | 消除 Vec<Vec> + flatten 双重分配 |

**综合效果（vs 优化前 baseline，release benchmark）**：

| 场景 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 完整训练步 单样本 | 493 us | 110 us | 4.5x |
| 完整训练步 batch32 | 12.3 ms | 4.4 ms | 2.8x |
| 完整训练步 batch64 | 26.3 ms | 5.8 ms | 4.5x |

### D. GradResult 零拷贝梯度传递（2026-02-15）

**原始问题**：`calc_grad_to_parent` 返回 `Result<Tensor>`，Add/Identity 等节点被迫 clone `upstream_grad`。Profile 显示 Add 反向占 12%，其中大部分是 clone。

**解决方案**：引入 `GradResult` 枚举替代裸 `Tensor` 返回：

```rust
pub(in crate::nn) enum GradResult {
    PassThrough,       // 零拷贝，直接用 upstream_grad 累加
    Negated,           // 零分配，累加时原地 -=
    Computed(Tensor),  // 新计算的梯度
}
```

| 节点 | 变体 | 效果 |
|------|------|------|
| Add（无广播）、Identity、Subtract（第一父节点）、Dropout（eval） | `PassThrough` | 零 clone |
| Negate、Subtract（第二父节点） | `Negated` | 零分配（`accumulate_grad_negated` 原地 `-=`） |
| 其余 53 个节点 | `Computed` | 行为不变 |

**影响范围**：全部 59 个节点 + trait 签名 + `propagate_grad_to_parents` 调用方

### E. Conv2d 反向传播并行策略（2026-02-15）

**原始问题**：反向传播中每个 batch 样本独立做 `im2col + GEMM`，N 次小矩阵乘法。

**尝试方案**：`batch_im2col` 将所有样本拼成大矩阵做单次 GEMM。

**最终结论**：批量化方案被**撤回**。因为 MKL 配置为 `seq`（单线程，避免与 Rayon 冲突），
单次大 GEMM 只用一个核心，反而不如 per-sample Rayon 并行（多核各跑一个小 GEMM）。
参考 Burn/Candle/PyTorch 的做法后统一为 per-sample Rayon 路径。
启用 BLAS 后 `dot()` 内部自动用 MKL 加速，代码无需区分。

### F. 优化器 set_value_owned + Adam 中间变量优化（2026-02-15）

| 改动 | 效果 |
|------|------|
| `TraitNode` 新增 `set_value_owned(Tensor)` | 优化器更新参数时零拷贝（消除 `set_value(Some(&val))` 的 clone） |
| SGD/Adam `step()` 改用 `set_value_owned` | 每个参数更新省一次完整 Tensor clone |
| Adam 偏差修正因子外提到循环外 | 所有参数共享 `bc1`/`bc2`，省去重复计算 |
| Adam `grad_sq *= (1-β2)` 原地操作 | 省去 `scaled_grad_squared` 临时 Tensor |
| Adam `denom += ε` 原地操作 | 省去 `&v_sqrt + eps` 临时 Tensor |

### G. BLAS 可选支持 — Intel MKL / OpenBLAS（2026-02-15）

**位置**：`Cargo.toml` features + `lib.rs`

**解决方案**：通过 feature flag 启用 BLAS 后端，`ndarray::dot()` 自动路由到 MKL/OpenBLAS：

```toml
[features]
blas-mkl     = ["ndarray/blas", "dep:intel-mkl-src"]     # Intel CPU 推荐
blas-openblas = ["ndarray/blas", "dep:openblas-src"]      # 跨平台备选
```

配置选择 `mkl-static-lp64-seq`（lp64 = 与 cblas-sys 兼容；seq = 避免与 Rayon 线程冲突）。

**实测效果**（Chinese Chess CNN，debug 模式，50 epoch）：

| 指标 | 无 BLAS | 有 MKL | 提升 |
|------|---------|--------|------|
| 训练总耗时 | 43.1s | **36.7s** | **14.9%** |
| 推理 batch=1 | 1.5ms | 1.4ms | 6.7% |
| 推理 batch=256 | 70.2ms | 68.9ms | 1.9% |

**设计决策**：无需条件编译不同代码路径。MKL 加速完全透明（在 `dot()` 内部），
per-sample Rayon 并行策略在有无 BLAS 时完全一致。

### H. 节点 cache clone 消除（2026-02-15）

| 节点 | 优化前 | 优化后 |
|------|--------|--------|
| Conv2d `padded_input` | `Some(padded.clone())` 缓存 | `Some(self.pad_input(input))` 直接 move |
| LeakyReLU | 缓存完整 `parent_value` | 不再缓存，反向时用 `value`（输出）判断区域，数学等价 |
| ChannelBiasAdd | `let mut result = input.clone()` | 节点已删除，由通用 Add + 广播替代 |

### I. Conv2d Inference 推理快路径（2026-04-29）

**原始问题**：YOLOv5 TinyChess 在 Debug 模式下单图检测约 2.0s，其中 forward 约 1.87s；decode + NMS 仅约 0.4ms，不是瓶颈。`Conv2d` 原本不区分训练 / 推理，`1x1` 卷积也走通用 `im2col + GEMM`，并保存 backward 需要的 `im2col_cache`。

**解决方案**：

| 改动 | 效果 |
|------|------|
| `Conv2d` 感知 `eval` 模式 | 推理时可跳过 backward 缓存 |
| `1x1 stride=1 padding=0 dilation=1` 卷积走直接 GEMM 快路径 | 避免为每个空间位置构造等价的 `im2col` 矩阵 |
| padding / `im2col` 热循环改用连续 slice 索引 | 避免 Debug 模式下多维动态索引开销 |

**实测效果**（`chinese_chess_yolov5_onnx_recognize_fen`，Debug + MKL，单图）：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| forward | 1871 ms | 596 ms | 约 3.1x |
| 总耗时 | 2030 ms | 745 ms | 约 2.7x |

**设计结论**：卷积的数学前向在 Train / Inference 下相同，但执行引擎需要区分“是否要为 backward 保存缓存”。后续新增重算代价高、缓存占用大的节点时，应同时设计训练路径和推理路径，避免推理承担训练负担。

---

## Benchmark 基础设施（✅ 已搭建）

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

报告输出在 `target/criterion/` 目录。

### I. Mode 执行上下文统一（2026-04-29）

**原始问题**：旧 `is_eval_mode` 同时承担层行为与 backward 缓存控制；中间过渡设计 `ExecutionContext { training, grad_enabled }` 把两者拆成两个正交字段。对 only_torch 当前训练 / 验证 / 推理 / 演化评估目标来说，"训练分支 + 不缓存"或"推理分支 + 仍缓存"不是核心用例，继续保留会让节点接入和测试覆盖度爆炸。

**解决方案**：用单枚举 `Mode { Train, Inference }` 统一三件事：层行为切换、backward 缓存策略、`backward()` 是否被允许（详见 [`mode_design.md`](design/mode_design.md)）。`Graph::load_model()` / `Graph::from_onnx()` 默认进入 `Mode::Inference`，即推理分支 + 不缓存 backward + `backward()` 直接报错。已接入 mode 的重缓存节点：Dropout、BatchNorm、Conv2d、Softmax、LogSoftmax、LayerNorm、RMSNorm、Abs、Square、Pow、Clip、Reciprocal、Ln、Log2、Log10。重构后的后续性能回归统一沿用 `post-mode-refactor` baseline 对比，验证 `Mode::Train` 路径无回归 + `Mode::Inference` 路径节省内存。

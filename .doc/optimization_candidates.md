# 性能优化候选项

> 本文档记录性能优化的候选项、已实施项和已否决项。
> 最后更新: 2026-02-15

---

## 待优化项

### 1. RNN 场景 `select` + `set_value` 二次复制问题

**位置**：`src/nn/layer/rnn.rs` → `forward`

**现状**：每个时间步两次数据复制。当前规模开销可忽略（~1.2 MB 总冗余）。

**推荐方案**：方案 A（使用已有的 `set_value_owned` 方法），最小改动。

**状态**：暂缓（YAGNI），等 RNN 处理大规模数据时再实施

---

### 2. BLAS 可选支持（Phase 6）

**位置**：`Cargo.toml` features

**现状**：已在 `Cargo.toml` 中定义 `blas-mkl` / `blas-openblas` feature flag，但尚未进行 benchmark 对比和文档完善。

**收益预估**：matmul ~1.3-1.5x（小矩阵收益有限）

**状态**：feature flag 已定义，待 benchmark 验证和文档说明

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

### E. Conv2d 反向 im2col 批量化（2026-02-15）

**原始问题**：反向传播中每个 batch 样本独立做 `im2col + GEMM`，N 次小矩阵乘法。

**解决方案**：

| 梯度 | 优化前 | 优化后 |
|------|--------|--------|
| dL/dK（权重） | N 次 `im2col` + N 次小 GEMM + reduce | 1 次 `batch_im2col` 垂直拼接 + 1 次大 GEMM（自然求和） |
| dL/dX（输入） | N 次小 GEMM + N 次 `col2im` | 1 次大 GEMM + 并行 `col2im` |

### F. 优化器 set_value_owned + Adam 中间变量优化（2026-02-15）

| 改动 | 效果 |
|------|------|
| `TraitNode` 新增 `set_value_owned(Tensor)` | 优化器更新参数时零拷贝（消除 `set_value(Some(&val))` 的 clone） |
| SGD/Adam `step()` 改用 `set_value_owned` | 每个参数更新省一次完整 Tensor clone |
| Adam 偏差修正因子外提到循环外 | 所有参数共享 `bc1`/`bc2`，省去重复计算 |
| Adam `grad_sq *= (1-β2)` 原地操作 | 省去 `scaled_grad_squared` 临时 Tensor |
| Adam `denom += ε` 原地操作 | 省去 `&v_sqrt + eps` 临时 Tensor |

### G. 节点 cache clone 消除（2026-02-15）

| 节点 | 优化前 | 优化后 |
|------|--------|--------|
| Conv2d `padded_input` | `Some(padded.clone())` 缓存 | `Some(self.pad_input(input))` 直接 move |
| LeakyReLU | 缓存完整 `parent_value` | 不再缓存，反向时用 `value`（输出）判断区域，数学等价 |
| ChannelBiasAdd | `let mut result = input.clone()` | 节点已删除，由通用 Add + 广播替代 |

---

## Benchmark 基础设施（✅ 已搭建）

已引入 `criterion` 框架，4 个 benchmark 文件覆盖各层面：

| 文件 | 覆盖范围 | 场景 |
|------|---------|------|
| `benches/tensor_ops.rs` | Tensor 底层操作 | clone/add/mul/negate/matmul/where（7 组） |
| `benches/backward.rs` | 节点反向传播 | Add/Negate/Subtract 链路 + MLP backward（4 组） |
| `benches/conv2d.rs` | Conv2d 卷积 | forward/full_step/two_layer_cnn（3 组） |
| `benches/end_to_end.rs` | 端到端训练步 | MLP(XOR/MNIST)/CNN(MNIST) × 多 batch_size（2 组） |

```bash
# 运行所有 benchmark
cargo bench

# 运行特定 benchmark
cargo bench --bench backward

# 保存基准线并对比
cargo bench -- --save-baseline before
cargo bench -- --baseline before
```

报告输出在 `target/criterion/` 目录。

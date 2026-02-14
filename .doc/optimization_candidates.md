# 性能优化候选项

> 本文档记录待验证的性能优化点，需 benchmark 数据支持后再决定是否实施。
> 最后更新: 2026-02-14

---

## 待优化项

### 1. `calc_grad_to_parent` 返回 owned Tensor 问题（架构级）

**位置**：`TraitNode` trait → `calc_grad_to_parent`

**现状**：
```rust
fn calc_grad_to_parent(..., upstream_grad: &Tensor) -> Result<Tensor, GraphError>
```

**问题**：trait 签名要求返回 owned `Tensor`，即使是 Add 这种"直接传递 upstream_grad"的节点也必须 clone。Profile 数据显示 Add 反向占 12%，其中大部分是 clone 开销。

**可能的方案**：
- 方案 A：改为 `&mut Tensor` 输出参数，避免分配：`fn calc_grad_to_parent(..., output: &mut Tensor)`
- 方案 B：引入 `Cow<Tensor>` 返回类型，允许借用或拥有
- 方案 C：为 identity 梯度（Add、Identity 等）添加特殊标记，跳过计算

**影响范围**：所有 59 个节点的 `calc_grad_to_parent` 实现
**收益预估**：反向传播整体 ~10-15%（消除 Add/Identity 的 clone）
**状态**：待设计，需权衡 API 复杂度

---

### 2. 前向传播 clone NodeHandle 问题

**位置**：`src/nn/graph.rs` → `forward_node_internal`

**现状**：
```rust
let parent_nodes = parents_ids
    .iter()
    .map(|id| self.get_node(*id).unwrap().clone())
    .collect::<Vec<NodeHandle>>();
```

**问题**：`NodeHandle.clone()` 会深拷贝内部的 `Tensor`，在大网络/大 batch 下可能有显著开销。

**可能的解法**：`Rc<Tensor>` 或 `Arc<Tensor>` 共享 Tensor 数据（需较大架构变动）

**状态**：待 benchmark 验证（当前 profile 显示前向只占 <1%，不紧急）

---

### 3. 节点 cache 的 Tensor clone

**位置**：多个节点的 `calc_value_by_parents` 实现

**示例**：
```rust
// src/nn/nodes/raw_node/ops/conv2d.rs
self.padded_input = Some(padded.clone());
```

**问题**：反向传播缓存需要 clone，但某些场景可重新计算或用 `Rc<Tensor>` 共享。

**状态**：待分析必要性（Conv2d 的 padded_input clone 在 profile 中不显著）

---

### 4. RNN 场景 `select` + `set_value` 二次复制问题

**位置**：`src/nn/layer/rnn.rs` → `forward`

**现状**：每个时间步两次数据复制。当前规模开销可忽略（~1.2 MB 总冗余）。

**推荐方案**：方案 A（新增 `set_value_owned` 方法），最小改动。

**状态**：暂缓（YAGNI），等 RNN 处理大规模数据时再实施

---

### 5. Conv2d 反向 im2col 批量化

**位置**：`src/nn/nodes/raw_node/ops/conv2d.rs` → `calc_grad_to_parent`

**现状**：每个 batch 样本独立做 im2col + GEMM，Rayon 在 batch 维度并行。

**可能的优化**：将所有 batch 的 im2col 合并为一次大矩阵乘法（减少 GEMM 调用次数，利用更大矩阵的 SIMD 效率）。

**收益预估**：Conv2d 反向 ~10-20%（release 模式下收益有限，debug 模式收益更大）
**状态**：待实现

---

## 已否决项

### `&[&NodeHandle]` vs `&[NodeHandle]`

**结论**：`&[&NodeHandle]` **不推荐**。双重间接引用增加指针追踪开销、缓存局部性差。

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

---

## Benchmark 基础设施（✅ 已搭建）

已引入 `criterion` 框架，benchmark 文件：`benches/conv2d.rs`

```bash
# 运行 benchmark 并生成报告
cargo bench --bench conv2d

# 保存基准线
cargo bench --bench conv2d -- --save-baseline before

# 对比
cargo bench --bench conv2d -- --baseline before
```

报告输出在 `target/criterion/` 目录。

### 测试场景（已实现）

| 组 | 场景 | 说明 |
|---|------|------|
| conv2d_forward | 4 种配置 | 隔离前向卷积性能 |
| conv2d_full_step | 3 种配置 | 前向+反向+优化器完整步 |
| two_layer_cnn | 3 种配置 | 模拟真实 MNIST/象棋 CNN |


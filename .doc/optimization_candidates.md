# 性能优化候选项

> 本文档记录待验证的性能优化点，需 benchmark 数据支持后再决定是否实施。

## 1. 前向传播 clone NodeHandle 问题

**位置**：`src/nn/graph.rs` → `forward_node_internal`

**现状**：
```rust
let parent_nodes = parents_ids
    .iter()
    .map(|id| self.get_node(*id).unwrap().clone())
    .collect::<Vec<NodeHandle>>();
```

**问题**：`NodeHandle.clone()` 会深拷贝内部的 `Tensor`（因为 `NodeType` 包含 `Option<Tensor>` 字段），在大网络/大 batch 下可能有显著开销。

**曾尝试的方案**：
- 改接口为 `&[&NodeHandle]` → 引入双重间接引用开销，**不一定更优**
- 临时 `remove` 当前节点 → 仍需收集 `Vec<&NodeHandle>`

**可能的真正解法**：
- 使用 `Rc<Tensor>` 或 `Arc<Tensor>` 共享 Tensor 数据，避免深拷贝
- 但需要较大架构变动，风险较高

**状态**：待 benchmark 验证

---

## 2. 节点 cache 的 Tensor clone

**位置**：多个节点的 `calc_value_by_parents` 实现

**示例**：
```rust
// src/nn/nodes/raw_node/ops/leaky_relu.rs
self.parent_value = Some(parent_value.clone());

// src/nn/nodes/raw_node/ops/channel_bias_add.rs
let mut result = input.clone();

// src/nn/nodes/raw_node/ops/conv2d.rs
self.padded_input = Some(padded.clone());
```

**问题**：这些 cache 用于反向传播，clone 有开销。

**可能的优化**：
- 某些场景下 cache 可能不必要（如果可以重新计算）
- 用 `Rc<Tensor>` 共享引用

**状态**：待分析必要性 + benchmark

---

## 3. `&[&NodeHandle]` vs `&[NodeHandle]` 分析

**结论**：`&[&NodeHandle]` **不推荐**

**原因**：
1. 双重间接引用增加指针追踪开销
2. 缓存局部性差
3. 需要额外构造 `Vec<&NodeHandle>`

**参考**：
- [Rust Users: Converting &[T] to &[&T]](https://users.rust-lang.org/t/converting-t-t/82276)
- [StackOverflow: Rust multiple levels of indirection performance impact](https://stackoverflow.com/questions/73610826/rust-multiple-levels-of-indirections-performance-impact)

---

## 4. RNN 场景 `select` + `set_value` 二次复制问题

**位置**：`src/nn/layer/rnn.rs` → `forward`

**现状**：
```rust
for t in 0..seq_len {
    let x_t = x.select(1, t);           // ① 复制：index_axis → to_owned()
    self.input_node.set_value(&x_t)?;   // ② 复制：value.cloned()
}
```

**问题**：每个时间步有两次数据复制，实际上只需一次。

**影响评估**（以 parity 示例为例）：
| 指标 | 数值 |
|------|------|
| 每次 select 复制 | batch_size × input_size = 32 × 1 = 32 floats |
| 调用次数 | 8 步 × ~31 batch × ~40 epoch ≈ 10K 次 |
| 总冗余复制量 | ~1.2 MB |

**结论**：对于当前规模，开销可忽略（真正瓶颈在矩阵乘法）。

**可能的优化方案**：

### 方案 A：新增 `set_value_owned` 方法

```rust
// 保留原有方法
pub fn set_value(&self, value: &Tensor) -> Result<(), GraphError>

// 新增：接受所有权，避免二次 clone
pub fn set_value_owned(&self, value: Tensor) -> Result<(), GraphError>

// 使用
let x_t = x.select(1, t);              // 复制 ①
self.input_node.set_value_owned(x_t)?;  // 转移所有权，无复制
```

**优点**：向后兼容，现有 253 处 `set_value` 调用不受影响
**缺点**：API 冗余

### 方案 B：新增 `select_view` + `set_value_from_view`

```rust
// 新增视图方法
pub fn select_view(&self, axis: usize, index: usize) -> ArrayViewD<'_, f32>
pub fn set_value_from_view(&self, view: ArrayViewD<'_, f32>) -> Result<...>

// 使用
let view = x.select_view(1, t);              // 零拷贝
self.input_node.set_value_from_view(view)?;  // 复制 ①
```

**优点**：对只读场景也有收益
**缺点**：引入新类型（或暴露 ndarray 的 `ArrayViewD`），增加 API 复杂度

### 方案 C：引入 `TensorView<'a>` 包装类型

```rust
pub struct TensorView<'a> {
    view: ArrayViewD<'a, f32>,
}

impl Tensor {
    pub fn select_view(&self, axis: usize, index: usize) -> TensorView<'_>
}

// 让 set_value 通过泛型或 trait 接受 TensorView
```

**优点**：类型封装更优雅
**缺点**：实现复杂度高，需要处理生命周期

**状态**：暂缓实施（YAGNI），等性能分析证明是瓶颈再优化

**备注**：
- 核心限制：`ArrayViewD` 和 `Tensor` 是不同类型，无法让 `select` 返回 `&Tensor`
- 如果未来 RNN 处理更大规模数据，可优先考虑方案 A（最小改动）

---

## 已实施的优化（已验证有效）

以下优化已实施并通过测试：

### A. 赋值算子减少 clone

| 位置 | 改动 | 效果 |
|------|------|------|
| `GradientAccumulator::get_average_gradient` | `gradient.clone() / scalar` → `gradient / scalar` | 少一次 Tensor clone |
| `graph.rs` 梯度累加 | `current + contribution` → `current += &contribution` | 避免临时张量分配 |
| 反向传播 | clone `upstream_grad` → 借用引用 | 避免大 Tensor 拷贝 |

---

## Benchmark 基础设施

### 推荐工具：criterion

```toml
# Cargo.toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[[bench]]
name = "forward_backward"
harness = false
```

### 示例 benchmark 代码

```rust
// benches/forward_backward.rs
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use only_torch::prelude::*;

fn bench_forward(c: &mut Criterion) {
    let mut graph = Graph::new();
    // ... 构建网络 ...

    c.bench_function("forward_mnist_linear", |b| {
        b.iter(|| graph.forward(loss_id).unwrap())
    });
}

fn bench_backward(c: &mut Criterion) {
    // ...
    c.bench_function("backward_mnist_linear", |b| {
        b.iter(|| graph.backward(loss_id).unwrap())
    });
}

criterion_group!(benches, bench_forward, bench_backward);
criterion_main!(benches);
```

### 使用流程

```bash
# 运行 benchmark 并生成 HTML 报告
cargo bench

# 保存当前性能作为基准
cargo bench -- --save-baseline before

# 修改代码后，对比性能变化
cargo bench -- --baseline before
```

报告输出在 `target/criterion/` 目录。

### 测试场景

| 场景 | 网络规模 | Batch 大小 | 代表用例 |
|------|----------|------------|----------|
| 小 | ~10 节点 | 1-8 | Adaline |
| 中 | ~100 节点 | 32-64 | MNIST Linear |
| 大 | ~1000 节点 | 64-128 | MNIST CNN |

### 测量指标

- 单次 forward 耗时
- 单次 backward 耗时
- 内存峰值（可用 `peak_alloc` crate 辅助）


# Batch Forward/Backward 机制设计

> 最后更新: 2025-12-21
> 状态: **已完成** ✅
> 优先级: ⭐⭐⭐⭐（高）

---

## 1. 背景与动机

### 1.1 问题现状

当前框架采用**逐样本处理**模式：

```rust
// 当前：循环处理每个样本
for i in 0..num_samples {
    graph.set_node_value(x, Some(&single_sample))?;  // [1, 784]
    graph.forward_node(loss)?;
    graph.backward_nodes(&trainable, loss)?;
    // 手动累积梯度...
}
```

**问题**：
- CPU 无法利用 SIMD 向量化
- 循环开销大
- 缓存命中率低
- MNIST 训练耗时过长（参考 MatrixSlow：30 epochs 需 5-15 分钟）

### 1.2 预期收益

根据 `optimization_strategy.md` 分析，Batch 机制预期带来 **3-10x** 加速：

| 优化来源 | 预期收益 |
|----------|----------|
| SIMD 向量化 | 2-4x |
| 缓存命中率提升 | 1.5-2x |
| 循环开销消除 | 1.2-1.5x |
| **综合** | **3-10x** |

---

## 2. 核心设计决策

### 2.1 Jacobi vs Gradient

**关键决策**：Batch 模式采用 **Gradient-based** 反向传播，而非完整 Jacobi 矩阵。

| 对比 | Jacobi（当前单样本） | Gradient（Batch） |
|------|---------------------|-------------------|
| **存储内容** | 完整 `∂y/∂x` 矩阵 | 最终 `∂L/∂x` 梯度 |
| **Shape** | `[dim_y, dim_x]` | `= param.shape` |
| **内存占用** | O(dim_y × dim_x) | O(param_size) |
| **Batch 扩展** | 内存爆炸 | 线性增长 |
| **信息量** | 完整（可分析） | 压缩（仅训练用） |

**原因**：
- Jacobi 对 batch 内存不可接受：`[batch × dim_y, dim_x]`
- Gradient 是所有现代框架（PyTorch/JAX/TensorFlow）的做法
- 训练只需要 `∂L/∂param`，不需要完整 Jacobi

### 2.2 保留两套机制

**决策**：保留单样本 Jacobi API，新增 Batch Gradient API。

| 机制 | 适用场景 |
|------|----------|
| **单样本 Jacobi** | NEAT 进化（结构不同）、调试、研究、二阶优化 |
| **Batch Gradient** | 固定结构训练、推理、生产环境 |

**架构优势**：
- 向后兼容，现有测试不受影响
- 单样本可作为 Batch 的正确性基准
- 符合 NEAT + 传统训练混合的项目定位

---

## 3. API 设计

### 3.1 阶段一：分离式 API（过渡方案）

```rust
impl Graph {
    // ===== 现有 API（单样本，Jacobi-based）=====
    pub fn forward_node(&mut self, node_id: NodeId) -> Result<(), GraphError>;
    pub fn backward_nodes(&mut self, targets: &[NodeId], result: NodeId) -> Result<(), GraphError>;

    // ===== 新增 API（Batch，Gradient-based）=====
    /// Batch 前向传播
    /// 输入节点的 value shape 应为 [batch_size, ...]
    pub fn forward_batch(&mut self, node_id: NodeId) -> Result<(), GraphError>;

    /// Batch 反向传播
    /// 自动计算所有可训练参数的梯度（已对 batch 求平均）
    pub fn backward_batch(&mut self, loss_id: NodeId) -> Result<(), GraphError>;

    /// 获取参数的梯度（Batch 模式）
    pub fn get_node_grad_batch(&self, node_id: NodeId) -> Result<Option<&Tensor>, GraphError>;
}
```

### 3.2 阶段二：统一式 API（最终目标）

```rust
impl Graph {
    /// 统一前向传播（自动处理 batch_size=1 或 batch_size>1）
    pub fn forward(&mut self, node_id: NodeId) -> Result<(), GraphError>;

    /// 统一反向传播
    pub fn backward(&mut self, loss_id: NodeId) -> Result<(), GraphError>;

    /// 获取梯度（统一接口）
    pub fn get_grad(&self, node_id: NodeId) -> Result<Option<&Tensor>, GraphError>;
}
```

**迁移路径**：
```
阶段一: forward_node() + forward_batch()  →  验证机制正确性
阶段二: 统一为 forward()                  →  PyTorch 风格体验
```

---

## 4. 节点层设计

### 4.1 Trait 扩展

```rust
pub(in crate::nn::nodes) trait TraitNode {
    // ===== 现有（单样本）=====
    fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError>;
    fn calc_jacobi_to_a_parent(&self, parent: &NodeHandle, assistant: Option<&NodeHandle>)
        -> Result<Tensor, GraphError>;

    // ===== 新增（Batch）=====
    /// Batch 前向计算（默认实现可复用 calc_value_by_parents）
    fn calc_value_batch(&mut self, parents: &[NodeHandle]) -> Result<(), GraphError> {
        // 大多数节点的 element-wise 操作天然支持 batch
        self.calc_value_by_parents(parents)
    }

    /// 计算梯度（非完整 Jacobi）
    /// upstream_grad: 从下游传来的梯度，shape 与本节点 value 相同
    /// 返回: 对 parent 的梯度，shape 与 parent.value 相同
    fn calc_grad_to_parent(
        &self,
        parent: &NodeHandle,
        upstream_grad: &Tensor,
        assistant: Option<&NodeHandle>,
    ) -> Result<Tensor, GraphError>;
}
```

### 4.2 关键节点实现要点

#### MatMul（最关键）

```rust
// Forward: [batch, n, m] @ [m, k] → [batch, n, k]
//   使用 broadcasting 或 batch matmul

// Backward:
//   对左父节点: dL/dA = upstream_grad @ B^T        → [batch, n, m]
//   对右父节点: dL/dB = sum_batch(A^T @ upstream_grad) → [m, k]
//                      ↑ 对 batch 维度求和
```

#### Add

```rust
// Forward: element-wise，天然支持 batch
// Backward: upstream_grad 直接传递（可能需要 reduce 处理广播）
```

#### 激活函数（Sigmoid/Tanh/ReLU）

```rust
// Forward: element-wise，天然支持 batch
// Backward: upstream_grad * local_derivative（element-wise）
```

#### SoftmaxCrossEntropy

```rust
// Forward: 对每个样本计算 loss，然后求平均
//   loss = mean(cross_entropy(softmax(logits), labels))  → scalar [1, 1]

// Backward:
//   dL/d_logits = (softmax - labels) / batch_size  → [batch, num_classes]
```

---

## 5. Shape 约定

### 5.1 Batch 维度约定

| 节点类型 | 单样本 Shape | Batch Shape |
|----------|--------------|-------------|
| Input | `[1, 784]` | `[batch, 784]` |
| Parameter (weights) | `[784, 128]` | `[784, 128]`（不变） |
| Parameter (bias) | `[1, 128]` | `[1, 128]`（广播） |
| Hidden activation | `[1, 128]` | `[batch, 128]` |
| Output | `[1, 10]` | `[batch, 10]` |
| Loss | `[1, 1]` | `[1, 1]`（标量，已聚合） |

### 5.2 关键原则

1. **Batch 维度始终在第一维**（dim 0）
2. **参数节点不含 batch 维度**（所有样本共享）
3. **损失输出为标量**（batch 内已求平均）
4. **梯度 shape = 参数 shape**（batch 梯度已聚合）

---

## 6. 实现路线图

### Phase 1: 基础设施 ✅

- [x] 定义 `calc_grad_to_parent` trait 方法
- [x] 实现 `Graph::forward_batch` 框架
- [x] 实现 `Graph::backward_batch` 框架
- [x] 添加 batch 梯度存储字段

### Phase 2: 核心节点实现 ✅

- [x] MatMul batch forward/backward
- [x] Add batch backward
- [x] Sigmoid/Tanh batch backward
- [x] SoftmaxCrossEntropy batch forward/backward

### Phase 3: 优化器适配 ✅

- [x] SGD 适配 batch 梯度
- [x] Adam 适配 batch 梯度
- [x] 验证梯度聚合正确性

### Phase 4: 集成测试 ✅

- [x] 新增 `test_mnist_batch.rs`
- [x] 对比单样本 vs Batch 结果一致性（`batch_mechanism.rs`）
- [x] MNIST 90% 准确率验证

### Phase 5: API 统一（后续）

- [ ] 合并 `forward_batch` 到 `forward`
- [ ] 更新文档和示例
- [ ] 废弃旧 API（可选）

---

## 7. 测试策略

### 7.1 正确性验证

```rust
#[test]
fn test_batch_gradient_equals_accumulated_single() {
    // 1. Batch 模式计算梯度
    let batch_grad = compute_batch_gradient(&batch_input);

    // 2. 单样本累积梯度
    let mut accumulated_grad = zeros_like(&param);
    for sample in batch_input.iter_samples() {
        accumulated_grad += compute_single_gradient(&sample);
    }
    accumulated_grad /= batch_size;

    // 3. 验证一致性
    assert_tensors_close(&batch_grad, &accumulated_grad, 1e-5);
}
```

### 7.2 性能基准

```rust
#[test]
fn benchmark_batch_vs_single() {
    let batch_time = measure(|| train_batch(epochs));
    let single_time = measure(|| train_single(epochs));

    let speedup = single_time / batch_time;
    println!("Speedup: {:.2}x", speedup);
    assert!(speedup > 3.0, "Batch 应至少提速 3x");
}
```

---

## 8. 与现有架构的关系

### 8.1 依赖关系

```
batch_mechanism_design.md（本文档）
    ├── 依赖: optimization_strategy.md（优化策略）
    ├── 依赖: gradient_clear_and_accumulation_design.md（梯度累积）
    ├── 影响: data_loader_design.md（DataLoader 返回 batch）
    └── 影响: optimizer_architecture_design.md（优化器适配）
```

### 8.2 与 NEAT 的兼容性

| 阶段 | 推荐 API | 原因 |
|------|----------|------|
| **结构进化** | 单样本 API | 每个个体结构不同，无法 batch |
| **权重训练** | Batch API | 结构固定，最大化效率 |
| **适应度评估** | 视情况 | 小 batch 或单样本均可 |

---

## 9. 参考资料

- PyTorch autograd 机制: https://pytorch.org/docs/stable/autograd.html
- JAX vmap 设计: https://jax.readthedocs.io/en/latest/jax-101/03-vectorization.html
- 项目内部: `optimization_strategy.md`, `broadcast_mechanism_design.md`

---

## 10. 变更记录

| 日期 | 变更内容 |
|------|----------|
| 2025-12-21 | 初始版本，定义核心设计决策和实现路线图 |
| 2025-12-21 | Phase 1-4 全部完成，MNIST batch 测试达到 90% 准确率 |


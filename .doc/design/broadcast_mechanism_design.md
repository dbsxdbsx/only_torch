# 广播机制设计决策

> 最后更新: 2025-12-20
> 状态: **已确定**
> 影响范围: Tensor 层、Graph 层、高层 API、NEAT 演化

---

## 核心决策

**Only Torch 采用"显式节点广播"策略，而非 PyTorch/NumPy 风格的隐式广播。**

---

## 设计原则

```
显式优于隐式 (Explicit is better than implicit)
```

在 NEAT 演化场景下，网络结构的透明性至关重要。隐式广播会模糊计算图的真实拓扑，阻碍结构优化和可解释性。

---

## 目标场景：混合进化-训练循环

Only Torch 的核心使用场景是**结构进化与权重训练的交替循环**：

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   NEAT 进化结构 ──→ 固定结构 ──→ Batch 梯度训练 ──→ 瓶颈？──┤
│         ↑                                          │    │
│         └──────────── 微调结构 ←───────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

| 阶段         | 网络结构 | 训练方式        |  广播需求   |
| :----------- | :------- | :-------------- | :---------: |
| 结构进化     | 动态变化 | 逐样本/小 batch |     低      |
| **权重训练** | **固定** | **大 batch**    | **✅ 需要** |
| 结构微调     | 局部变化 | 逐样本/小 batch |     低      |

**关键洞察**：大部分训练时间花在"权重训练"阶段，这正是 batch 效率最重要的地方。ScalarMultiply 节点正是为此设计。

---

## 两层广播策略

| 层次          | 支持的广播  | 实现方式                          |
| :------------ | :---------- | :-------------------------------- |
| **Tensor 层** | 标量 ↔ 张量 | Rust 运算符重载（`f32 + Tensor`） |
| **Graph 层**  | 标量 → 矩阵 | `ScalarMultiply`节点              |

### Tensor 层（底层计算）

```rust
// ✅ 支持：标量与张量的运算
let result = 2.0 * tensor;        // f32 * Tensor
let result = tensor + 1.0;        // Tensor + f32

// ❌ 不支持：不同形状张量的隐式广播
let a = Tensor::new(&[1.0], &[1, 1]);
let b = Tensor::new(&[1.0; 10], &[10, 1]);
let result = a + b;  // panic! 形状不匹配
```

### Graph 层（计算图）

```rust
// ✅ 正确做法：使用 ScalarMultiply 节点
let b = graph.new_parameter_node(&[1, 1], Some("b"))?;
let ones = graph.new_input_node(&[batch_size, 1], Some("ones"))?;
let bias = graph.new_scalar_multiply_node(b, ones, None)?;  // [1,1] → [batch,1]
let output = graph.new_add_node(&[matmul, bias], None)?;

// ❌ 错误做法：期望 Add 节点自动广播
let output = graph.new_add_node(&[matmul, b], None)?;  // Error! 形状不匹配
```

---

## 为什么 ScalarMultiply 而非隐式广播？

### 1. 雅可比矩阵计算复杂度

**隐式广播**：

```
forward:  [1,1] + [10,3] → [10,3]
backward: ∂L/∂a 需要 sum reduce [10,3] → [1,1]
jacobi:   不再是简单的单位矩阵，需要处理维度归约
```

**显式节点**：

```
forward:  ScalarMultiply(a, ones) → [10,3], 然后 Add
backward: 梯度直接通过节点链传播
jacobi:   每个节点的雅可比矩阵形式明确
```

### 2. NEAT 演化友好性

| 特性       | 隐式广播 | 显式节点 |
| :--------- | :------: | :------: |
| 结构透明性 |    ❌    |    ✅    |
| 可进化性   |    ❌    |    ✅    |
| 基因表示   |   复杂   |   直接   |

隐式广播将"形状适配"逻辑隐藏在运算符内部，NEAT 无法感知和优化这部分结构。显式节点使每个计算步骤都成为可进化的基因单元。

### 3. CPU 优化解耦

广播的 SIMD 优化应该在 Tensor 层（通过 ndarray/BLAS）实现，而非在 Graph 层。这样：

- Graph 层专注于拓扑和梯度流
- Tensor 层专注于计算效率
- 关注点分离，便于独立优化

---

## Batch 机制的价值：技术效率视角

### 为什么 Batch 比 Rayon 逐样本并行更高效？

在 CPU 上，batch 矩阵运算的效率显著优于 Rayon 并行处理单个样本：

```
方案 A：Rayon 逐样本并行
─────────────────────────
Core0: sample0 → [1,3]×[3,1] → result0
Core1: sample1 → [1,3]×[3,1] → result1
Core2: sample2 → [1,3]×[3,1] → result2
...
开销：线程调度 × N 样本

方案 B：Batch 矩阵运算（SIMD）
─────────────────────────
Single Core: [batch,3]×[3,1] → [batch,1]
             └── SIMD: 8 个 float 同时计算（AVX2）
开销：一次矩阵乘法
```

### 效率对比（预估经验值，待实测）

| 操作                                  | 逐样本循环 | Batch 矩阵 | 加速比      |
| :------------------------------------ | :--------: | :--------: | :---------- |
| `[1,3]×[3,1]` × 1000 次               |    1.0x    |    3-5x    | SIMD + 缓存 |
| `[1,784]×[784,128]` × 1000 次         |    1.0x    |   10-50x   | BLAS 优化   |
| `[1,3]×[3,1]` × 1000 次 + Rayon(8 核) |    6-7x    |     -      | 并行开销    |

### Batch 胜出的原因

1. **SIMD 向量化**：

   ```
   逐样本：3 次乘法 × 1000 样本 = 3000 条 MUL 指令
   Batch： AVX2 一次处理 8 个 float → ~375 条 VMUL 指令
   ```

2. **缓存命中率**：

   ```
   逐样本：每次从内存取权重矩阵 → 缓存失效
   Batch： 权重矩阵常驻 L1 缓存 → 极高命中率
   ```

3. **指令流水线**：
   ```
   逐样本：循环开销、分支预测
   Batch： 连续内存访问，预取有效
   ```

### 最优策略：Batch + Rayon 组合

```rust
// 最高效的方式：batch 内 SIMD，batch 间 Rayon
data.par_chunks(batch_size)      // Rayon 并行处理多个 batch
    .for_each(|batch| {
        let result = weights.dot(&batch);  // SIMD 矩阵运算
    });
```

| 层级     | 优化方式                           | 收益             |
| :------- | :--------------------------------- | :--------------- |
| 单个运算 | SIMD（自动，由 ndarray/BLAS 提供） | 4-8x             |
| Batch 内 | 矩阵化（ScalarMultiply 等）        | 3-10x            |
| Batch 间 | Rayon 并行                         | 核心数 x         |
| **组合** | **Batch × Rayon**                  | **可达 50-100x** |

---

## Batch 机制的价值：用户体验视角

### 用户期望的使用方式

```rust
// 1. 传统深度学习（PyTorch 风格）
let model = Linear::new(784, 10);
for batch in data_loader {
    let loss = model.forward(&batch)?;
    optimizer.step()?;
}

// 2. 进化网络结构（NEAT 风格）
let evolved_model = neat.evolve(|model| {
    evaluate_fitness(model, &validation_data)
})?;

// 3. 混合模式（Only Torch 的核心场景）
loop {
    // 固定结构，batch 训练
    train_with_batch(&model, &data, epochs)?;

    if is_bottleneck(&model) {
        // 微调结构
        model = neat.mutate_structure(&model)?;
    }
}
```

### 高层 API 封装策略

高层 API（如 `Linear`、`Conv2d`）应在内部封装广播逻辑，对用户透明：

```rust
impl Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, GraphError> {
        let matmul = self.graph.new_mat_mul_node(self.weight, input)?;

        // 封装层自动处理批量 vs 单样本
        let output = if is_batch(input) {
            let bias_broadcast = self.broadcast_bias(input.shape()[0])?;
            self.graph.new_add_node(&[matmul, bias_broadcast])?
        } else {
            self.graph.new_add_node(&[matmul, self.bias])?
        };

        Ok(output)
    }
}
```

用户代码：

```rust
let output = linear.forward(&batch_input)?;  // 用户无需关心广播细节
```

**ScalarMultiply 节点是实现这种用户体验的基础设施。**

---

## 常见场景的处理方式

### 全连接层（单样本）

```python
# MatrixSlow 设计：形状天然匹配，无需广播
weights = Variable((out, in), ...)      # [out, in]
bias = Variable((out, 1), ...)          # [out, 1]
output = Add(MatMul(weights, input), bias)  # [out,1] + [out,1] ✓
```

### 批量训练

```python
# 需要 ScalarMultiply 广播 bias
b = Variable((1, 1), ...)
ones = Variable((batch, 1), ...)
bias = ScalarMultiply(b, ones)          # [1,1] → [batch,1]
output = Add(MatMul(X, w), bias)        # [batch,1] + [batch,1] ✓
```

### 卷积层

```python
# 同样使用 ScalarMultiply
ones = Variable(input_shape, ...)
bias = ScalarMultiply(Variable((1,1), ...), ones)  # [1,1] → [H,W]
output = Add(conv_result, bias)
```

---

## 扩展路径

如果未来需要更复杂的广播模式，采用**渐进式添加专用节点**：

| 阶段 | 新增节点             | 覆盖场景              |
| :--- | :------------------- | :-------------------- |
| MVP  | `ScalarMultiply`     | 标量 × 矩阵           |
| 按需 | `VectorBroadcastAdd` | 向量 + 矩阵（沿某轴） |
| 按需 | `BatchNormBroadcast` | BN 专用广播           |

**不建议**实现通用的 `BroadcastAdd`，保持"一种模式一个节点"的原则。

---

## 总结

### 设计决策

| 决策      | 选择                 | 原因                |
| :-------- | :------------------- | :------------------ |
| 广播策略  | 显式节点             | NEAT 友好、梯度清晰 |
| Tensor 层 | 仅标量广播           | 简单高效            |
| Graph 层  | `ScalarMultiply`节点 | 覆盖主要场景        |
| 高层 API  | 封装隐藏             | 用户体验            |
| 扩展方式  | 渐进添加节点         | 可控复杂度          |

### ScalarMultiply 的双重价值

| 维度         | 价值                                                           |
| :----------- | :------------------------------------------------------------- |
| **技术效率** | 使 batch 训练能利用 SIMD 向量化，比 Rayon 逐样本并行快 3-10 倍 |
| **用户体验** | 使高层 API（Linear 等）能自动处理 batch，用户无需手动广播      |

### 核心权衡

**牺牲少量 API 便利性，换取结构透明性和演化能力。对于以 NEAT 为远期目标的框架，这是正确的取舍。**

同时，通过高层 API 封装，用户仍然可以享受 PyTorch 风格的简洁体验，而底层保持 NEAT 友好的显式节点设计。

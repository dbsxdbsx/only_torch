# 自动微分统一设计：从 Jacobian 到 VJP

> **状态**：设计稿 (v1.0)
> **作者**：架构评审
> **创建日期**：2026-01-02
> **背景**：基于对 Candle、Burn、Neuronika、PyTorch、JAX 等主流框架的深度调研，决定将 only_torch 的自动微分从 Jacobian 模式迁移到 VJP 模式，同时统一单样本/批量 API

---

## 1. 设计摘要

本文档描述 only_torch 自动微分系统的重大重构：

1. **从 Jacobian 模式迁移到 VJP 模式**：去除 `jacobi` 字段，统一使用 `grad` 字段
2. **统一单样本与批量 API**：用户无需区分 `backward_nodes` / `backward_batch`，底层透明处理
3. **渐进式迁移**：通过 `#[deprecated]` 标记平滑过渡，最终**完全删除**旧 API

### 核心变更

| 维度 | 旧设计 | 新设计 |
|------|--------|--------|
| 反向传播模式 | Jacobian + VJP 双轨 | 仅 VJP |
| 梯度存储 | `jacobi` + `grad` 两个字段 | 仅 `grad` |
| 单样本 API | `backward_nodes()` | `backward()` |
| 批量 API | `backward_batch()` | `backward()` |
| 清梯度 | `clear_jacobi()` + `clear_grad()` | `zero_grad()` |
| 优化器 | `one_step()` / `one_step_batch()` | `step()` |

---

## 2. 历史背景：为什么最初选择 Jacobian？

### 2.1 MatrixSlow 的设计哲学

only_torch 的早期设计参考了 [MatrixSlow](./MatrixSlow) 项目。MatrixSlow 是一个**教学导向**的深度学习框架，其核心目标是：

- **数学透明性**：让用户能够"看到"反向传播的每一步
- **概念清晰**：Jacobian 矩阵是微积分中的标准概念，易于理解
- **推导可验证**：显式 Jacobian 便于手工验证梯度正确性

MatrixSlow 的反向传播实现（参考 `MatrixSlow/matrixslow/core/node.py`）：

```python
def backward(self):
    """反向传播：计算结果节点对本节点的雅可比矩阵"""
    if self.jacobi is None:
        # 从子节点收集雅可比矩阵，通过链式法则累加
        for child in self.children:
            if child.jacobi is None:
                child.backward()
            # jacobi = jacobi + child.jacobi @ child.get_jacobi(self)
            self.jacobi = self.jacobi + child.jacobi.dot(child.get_jacobi(self))
```

### 2.2 Jacobian 模式的优点（教学场景）

| 优点 | 说明 |
|------|------|
| **概念直观** | Jacobian 矩阵 $J = \partial y / \partial x$ 是标准数学概念 |
| **链式法则透明** | $J_{total} = J_n \cdot J_{n-1} \cdots J_1$ 清晰可见 |
| **易于调试** | 可以打印中间 Jacobian 矩阵检查正确性 |
| **完整信息** | 保留了所有偏导数信息，不仅仅是标量梯度 |

### 2.3 为什么 Jacobian 模式不适合生产环境？

**核心问题：空间复杂度爆炸**

对于节点值形状为 `[n]` 的情况：
- Jacobian 矩阵形状：`[n, n]`
- 空间复杂度：**O(n²)**

实际训练中的数字：

| 场景 | 节点形状 | Jacobian 大小 | 内存占用 |
|------|---------|--------------|---------|
| 小模型 | `[64, 128]` = 8K | 8K × 8K = 64M | **256 MB** |
| 中模型 | `[64, 512]` = 32K | 32K × 32K = 1B | **4 GB** |
| 大模型 | `[64, 2048]` = 128K | 128K × 128K = 16B | **64 GB** |

**单个中间节点**的 Jacobian 就可能占用数 GB 内存，而网络中有数十甚至数百个节点。

**时间复杂度问题**

以 MatMul 为例（A: `[m, n]`, B: `[n, p]`, C: `[m, p]`）：

| 模式 | 操作 | 复杂度 |
|------|------|--------|
| Jacobian | 构造 dC/dA 矩阵 `[m*p, m*n]` | O(m²np) |
| Jacobian | upstream @ Jacobian 乘法 | O(m³p²n) 最坏 |
| **VJP** | upstream_grad @ B^T | **O(mnp)**（与前向同量级） |

**关键洞察**：VJP 的复杂度与**前向传播相同量级**，而 Jacobian 会额外引入一个维度。对于逐元素操作（Add、ReLU 等），VJP 是 O(n) 而 Jacobian 需要 O(n²)。

---

## 3. VJP 模式详解

### 3.1 什么是 VJP？

**VJP = Vector-Jacobian Product（向量-雅可比积）**

对于函数 $y = f(x)$，VJP 计算的是：

$$
\bar{x} = \bar{y}^T \cdot J = \bar{y}^T \cdot \frac{\partial y}{\partial x}
$$

其中 $\bar{y}$ 是"上游梯度"（upstream gradient），$\bar{x}$ 是"下游梯度"。

**关键洞察**：我们不需要显式构造 Jacobian 矩阵，只需要计算它与向量的乘积！

### 3.2 VJP vs Jacobian：以 MatMul 为例

对于 $C = A \times B$（A: `[m, n]`, B: `[n, p]`, C: `[m, p]`）

**Jacobian 模式**（当前 `calc_jacobi_to_a_parent`）：

```rust
// 构造 dC/dA 的 Jacobian，形状 [m*p, m*n]
let mut jacobi = Tensor::zeros(&[m * p, m * n]);
for i in 0..m {
    for j in 0..p {
        for k in 0..n {
            jacobi[[i * p + j, i * n + k]] = b_value[[k, j]];
        }
    }
}
// 然后做矩阵乘法：upstream_jacobi @ jacobi
```

**VJP 模式**（当前 `calc_grad_to_parent`）：

```rust
// 直接计算 dL/dA = upstream_grad @ B^T
// 无需构造中间 Jacobian！
Ok(upstream_grad.mat_mul(&b_value.transpose()))
```

### 3.3 数学等价性证明

对于标量 loss $L$，有：

$$
\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot \frac{\partial C}{\partial A}
$$

- Jacobian 模式：先构造 $\frac{\partial C}{\partial A}$（巨大矩阵），再乘以 $\frac{\partial L}{\partial C}$
- VJP 模式：直接计算乘积结果

**两者数学上完全等价**，但 VJP 避免了中间矩阵的分配。

### 3.4 为什么所有主流框架都用 VJP？

| 框架 | 反向传播模式 | 梯度存储 |
|------|-------------|---------|
| **PyTorch** | VJP | `tensor.grad` |
| **TensorFlow** | VJP | `GradientTape` |
| **JAX** | VJP (default) | 函数返回值 |
| **Candle** | VJP | `GradStore` |
| **Burn** | VJP | `Gradients` |
| **Neuronika** | VJP | `grad` 字段 |

**零例外**——没有任何生产级框架使用显式 Jacobian。

---

## 4. Loss 函数的 Reduction 语义

### 4.1 PyTorch 的设计

PyTorch 的 loss 函数支持 `reduction` 参数：

```python
loss = nn.CrossEntropyLoss(reduction='mean')  # 默认：对 batch 取平均
loss = nn.CrossEntropyLoss(reduction='sum')   # 可选：对 batch 求和
loss = nn.CrossEntropyLoss(reduction='none')  # 可选：不聚合，返回每个样本的 loss
```

### 4.2 各模式的使用场景

| 模式 | 使用频率 | 典型场景 |
|------|---------|---------|
| **Mean** | ~90% | 标准训练；学习率与 batch size 解耦 |
| None | ~8% | 样本加权、难样本挖掘、Curriculum Learning |
| Sum | ~2% | 强化学习、特殊优化算法 |

### 4.3 only_torch 的设计决策

**我们采用 PyTorch 风格，默认且目前仅支持 Mean 模式**：

1. **Loss 函数**负责对 batch 取平均（`calc_grad_to_parent` 中 `/ batch_size`）
2. **`backward()`** 梯度累加到 `.grad`，不自动清零、不额外平均
3. **`optimizer.step()`** 直接使用 `.grad`，不除以任何东西
4. **用户**负责调用 `zero_grad()`

**理由**：
- Mean 覆盖 90% 使用场景
- 学习率与 batch size 解耦，换 batch size 不用调学习率
- 实现简单，无需追踪累积次数
- 与 PyTorch 行为一致，用户熟悉

**后续扩展**：如有需求，可为 loss 函数添加 `reduction` 参数（参考 `MSELoss` 已有的实现）。

### 4.4 现有 Loss 函数状态

| Loss 函数 | 输出形状 | Batch 平均 | VJP 实现 |
|-----------|---------|-----------|---------|
| MSELoss | 标量 `[1,1]` | ✅ 已实现 | ✅ `calc_grad_to_parent` |
| SoftmaxCrossEntropy | 标量 `[1,1]` | ✅ 已实现 | ✅ `calc_grad_to_parent` |
> - 如需 element-wise 损失分布，可通过 `ReLU(-x)` 组合实现

### 4.5 形状验证策略（广播语义）

**设计决策：不支持隐式广播，形状不匹配时显式报错**

```rust
// 形状验证示例（所有二元操作节点）
if input_shape != target_shape {
    return Err(GraphError::ShapeMismatch {
        expected: input_shape.to_vec(),
        got: target_shape.to_vec(),
        message: "形状必须完全相同".to_string(),
    });
}
```

**理由**：
- "Explicit is better than implicit"——显式报错更安全，调试更容易
- 与 Candle、Neuronika 等 Rust 框架保持一致
- 避免隐式广播隐藏形状 bug
- 实现简单，无需处理复杂的广播规则

---

## 5. 单样本与批量的统一

### 5.1 当前双轨设计的问题

**现状**：

```rust
// 单样本训练循环
for sample in dataset {
    graph.set_value(input, &sample)?;
    graph.forward_node(loss)?;
    graph.backward_nodes(&params, loss)?;  // ← 单样本 API
    // ... 更新参数 ...
    graph.clear_jacobi()?;                  // ← 单样本清理
}

// 批量训练循环
for batch in dataloader {
    graph.set_value(input, &batch)?;
    graph.forward_batch(loss)?;
    graph.backward_batch(loss, Some(&params))?;  // ← 批量 API
    // ... 更新参数 ...
    graph.clear_grad()?;                          // ← 批量清理
}
```

**问题**：

| 问题 | 影响 |
|------|------|
| **API 分裂** | 用户需要记住两套 API |
| **语义不一致** | `jacobi` vs `grad`，`clear_jacobi` vs `clear_grad` |
| **代码重复** | 优化器需要 `one_step()` 和 `one_step_batch()` |
| **迁移困难** | 从单样本切换到批量需要大量代码改动 |

### 5.2 统一设计

**核心理念**：单样本是 batch_size=1 的特例

```rust
// 统一后的训练循环（适用于任何 batch_size）
for batch in dataloader {  // batch_size=1 也是有效的 batch
    graph.set_value(input, &batch)?;
    graph.zero_grad()?;        // 统一的清梯度
    graph.forward(loss)?;      // 统一的前向
    graph.backward(loss)?;     // 统一的反向
    optimizer.step()?;         // 统一的更新
}
```

### 5.3 统一 API 设计

```rust
impl Graph {
    /// 前向传播（单样本和批量统一）
    ///
    /// 输入遵循 Batch-First 格式：
    /// - FC 层：`[batch, features]`（单样本用 `[1, 784]`）
    /// - CNN 层：`[batch, C, H, W]` 或 `[C, H, W]`（3D 无 batch）
    pub fn forward(&mut self, target: NodeId) -> Result<(), GraphError>;

    /// 反向传播（VJP 模式，单样本和批量统一）
    ///
    /// 这是 `backward_ex(loss, false)` 的简写，覆盖 90% 的训练场景。
    ///
    /// # 语义
    /// - 计算 loss 对**所有** requires_grad 参数的梯度
    /// - 梯度存储在节点的 `grad` 字段
    /// - 梯度会累积（需要先调用 `zero_grad()` 清零）
    /// - 返回 loss 的标量值（方便用户打印）
    ///
    /// # 梯度隔离
    /// GAN 等场景的梯度隔离通过 `detach()` 实现，而非 `target_params`。
    /// 详见 [梯度流控制设计 - 附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params)。
    ///
    /// # 关于 requires_grad / 冻结
    /// 默认所有参数节点 `requires_grad=true`。如需冻结部分参数（如迁移学习），
    /// 可通过 optimizer 选择性绑定参数，或参考 [梯度流控制设计 - 附录 B](gradient_flow_control_design.md#附录-brequires_grad--冻结机制可选功能)
    /// 了解可选的 `requires_grad` / 冻结机制（Optional TODO）。
    pub fn backward(&mut self, loss: NodeId) -> Result<f32, GraphError>;

    /// 反向传播（扩展版本，支持 retain_graph）
    ///
    /// # 参数
    /// - `loss`: 损失节点 ID
    /// - `retain_graph`: 是否保留计算图（用于多次 backward）
    ///
    /// # 使用场景
    /// | 场景 | `retain_graph` |
    /// |------|----------------|
    /// | 标准训练 | `false` |
    /// | 多任务学习（多 loss 累加） | `true` |
    ///
    /// # 设计决策：移除 `target_params`
    /// PyTorch 的 `backward()` 没有 `target_params` 参数。
    /// GAN 等场景的梯度隔离通过 `detach()` 实现，语义更清晰、性能更优。
    /// 详见 [梯度流控制设计 - 附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params)。
    pub fn backward_ex(
        &mut self,
        loss: NodeId,
        retain_graph: bool,
    ) -> Result<f32, GraphError>;

    /// 清零所有参数的梯度（PyTorch 风格）
    pub fn zero_grad(&mut self) -> Result<(), GraphError>;
}

impl Optimizer {
    /// 更新参数（单样本和批量统一）
    pub fn step(&mut self) -> Result<(), GraphError>;

    /// 清零梯度
    pub fn zero_grad(&mut self) -> Result<(), GraphError>;
}
```

### 5.4 废弃并最终删除旧 API

**过渡策略**：先标记 `#[deprecated]` 发出编译警告，待用户迁移后在后续版本**完全删除**。

**转发逻辑**：旧 API 内部转发到新实现，**忽略 `target_params` 参数**。

> **设计决策**：`target_params` 功能已移除，GAN 等场景改用 `detach()` 实现梯度隔离。
> 详见 [梯度流控制设计 - 附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params)。

```rust
impl Graph {
    /// @deprecated 使用 `backward()` 替代
    #[deprecated(since = "0.x.0", note = "使用 backward() 替代，target_params 功能已移除，改用 detach() 控制梯度流")]
    pub fn backward_nodes(
        &mut self,
        _target_nodes_ids: &[NodeId],  // ⚠️ 已忽略，仅保留签名兼容
        result_node_id: NodeId,
    ) -> Result<f32, GraphError> {
        // 忽略 target_params，转发到新实现
        // 如需梯度隔离，请在计算图中使用 detach()
        self.backward(result_node_id)
    }

    /// @deprecated 使用 `backward()` 替代
    #[deprecated(since = "0.x.0", note = "使用 backward() 替代，target_params 功能已移除，改用 detach() 控制梯度流")]
    pub fn backward_batch(
        &mut self,
        loss_id: NodeId,
        _target_params: Option<&[NodeId]>,  // ⚠️ 已忽略，仅保留签名兼容
    ) -> Result<f32, GraphError> {
        // 忽略 target_params，转发到新实现
        // 如需梯度隔离，请在计算图中使用 detach()
        self.backward(loss_id)
    }

    /// @deprecated 使用 `zero_grad()` 替代
    #[deprecated(since = "0.x.0", note = "使用 zero_grad() 替代")]
    pub fn clear_jacobi(&mut self) -> Result<(), GraphError>;

    /// @deprecated 使用 `zero_grad()` 替代
    #[deprecated(since = "0.x.0", note = "使用 zero_grad() 替代")]
    pub fn clear_grad(&mut self) -> Result<(), GraphError>;
}
```

> **注意**：
> 1. `#[deprecated]` 仅是过渡手段，最终目标是**完全删除**这些旧 API
> 2. 使用 `target_params` 的现有代码需要迁移到 `detach()` 方式
> 3. 相关测试（如 `test_backward_batch_target_params_only_computes_specified`）需要重写为 PyTorch 风格

---

## 6. 节点层变更

### 6.1 TraitNode 接口变更

**删除**：
```rust
// 删除 Jacobian 相关方法
fn calc_jacobi_to_a_parent(...) -> Result<Tensor, GraphError>;
fn jacobi(&self) -> Option<&Tensor>;
fn set_jacobi(&mut self, jacobi: Option<&Tensor>) -> Result<(), GraphError>;
fn clear_jacobi(&mut self) -> Result<(), GraphError>;
```

**保留/增强**：
```rust
// VJP 相关方法（已存在，继续使用）
fn calc_grad_to_parent(
    &self,
    target_parent: &NodeHandle,
    upstream_grad: &Tensor,
    assistant_parent: Option<&NodeHandle>,
) -> Result<Tensor, GraphError>;

fn grad(&self) -> Option<&Tensor>;
fn set_grad(&mut self, grad: Option<&Tensor>) -> Result<(), GraphError>;
fn clear_grad(&mut self) -> Result<(), GraphError>;
```

### 6.2 节点结构体变更

```rust
// 旧设计
pub(crate) struct MatMul {
    // ...
    jacobi: Option<Tensor>,  // ← 删除
    grad: Option<Tensor>,    // ← 保留
}

// 新设计
pub(crate) struct MatMul {
    // ...
    grad: Option<Tensor>,    // 唯一的梯度存储
}
```

### 6.3 不可微节点的 VJP 处理

部分激活函数在数学上不可微（导数不存在或不连续），需要特殊处理：

| 节点 | 数学特性 | VJP 策略 | PyTorch 参考 |
|------|---------|---------|-------------|
| **Step** | x=0 处不连续 | 返回 0 | `torch.heaviside` 梯度为 0 |
| **Sign** | x=0 处不可微 | 返回 0 | `torch.sign` 梯度为 0 |

**实现示例**（以 Step 为例）：

```rust
fn calc_grad_to_parent(
    &self,
    _target_parent: &NodeHandle,
    upstream_grad: &Tensor,
    _assistant_parent: Option<&NodeHandle>,
) -> Result<Tensor, GraphError> {
    // Step 函数不可微，梯度恒为 0
    // 这意味着梯度不会流经 Step 节点
    Ok(Tensor::zeros(upstream_grad.shape()))
}
```

**设计理由**：
- 与 PyTorch 行为一致，用户迁移无障碍
- 保守策略：宁可梯度为 0，也不传播错误的梯度
- 这些节点通常用于推理（如二值化），训练时应使用可微替代（如 Sigmoid、SoftPlus）

---

## 7. 对 NEAT 的影响分析

### 6.1 NEAT 核心操作与梯度的关系

| NEAT 操作 | 是否需要梯度？ | 说明 |
|-----------|--------------|------|
| 添加节点 | ❌ | 拓扑变更，不涉及导数 |
| 添加连接 | ❌ | 拓扑变更 |
| 权重变异 | ❌ | 随机扰动，不用梯度 |
| fitness 评估 | ❌ | 只需 forward |
| 交叉/选择 | ❌ | 基因操作 |
| 混合训练（可选） | ✅ | NEAT + 梯度微调 |

**结论**：去除 `jacobi` 对 NEAT 核心功能**零影响**。

### 6.2 `on_topology_changed()` 调整

```rust
// 旧实现
pub fn on_topology_changed(&mut self) {
    for node in self.nodes.values_mut() {
        let _ = node.clear_jacobi();  // ← 删除
        node.set_last_backward_pass_id(0);
    }
}

// 新实现
pub fn on_topology_changed(&mut self) {
    for node in self.nodes.values_mut() {
        let _ = node.clear_grad();  // ← 改为清 grad
        node.set_last_backward_pass_id(0);
    }
}
```

---

## 8. 如果需要完整 Jacobian？（暂不实现）

> **状态**：此功能**暂不实现**，不在当前迁移计划内。后期如有明确需求再添加。

### 7.1 潜在场景

- 梯度检查（gradient checking）
- 科研分析
- 特殊优化算法（如自然梯度）

### 7.2 工具函数设计（参考）

**复杂度警告**：
- **时间复杂度**：O(output_size × VJP_cost)，需要对输出的每个元素执行一次完整的反向传播
- **空间复杂度**：O(output_size × input_size)，即 O(n²)

对于 output_size = 1000、input_size = 10000 的场景：
- 需要执行 **1000 次**完整反向传播
- 存储 **10M 个浮点数**（约 40 MB）

**因此此函数仅适用于小规模调试，绝不应在训练主循环中使用。**

```rust
/// 通过多次 VJP 构造完整 Jacobian（仅用于调试/研究）
///
/// # 警告
/// - 时间复杂度：O(output_size × backward_cost)
/// - 空间复杂度：O(output_size × input_size)
/// - 仅适用于小规模调试，不要在训练主循环中使用！
///
/// # 示例
/// ```rust
/// let jacobian = compute_full_jacobian(&graph, output, input)?;
/// // jacobian.shape() = [output_size, input_size]
/// ```
pub fn compute_full_jacobian(
    graph: &Graph,
    output: NodeId,
    input: NodeId,
) -> Result<Tensor, GraphError> {
    let output_shape = graph.get_node_value_shape(output)?;
    let input_shape = graph.get_node_value_shape(input)?;
    let output_size = output_shape.iter().product::<usize>();
    let input_size = input_shape.iter().product::<usize>();

    let mut jacobian = Tensor::zeros(&[output_size, input_size]);

    for i in 0..output_size {
        // 构造单位向量 seed
        let mut seed = Tensor::zeros(&output_shape);
        seed.flat_set(i, 1.0);

        // 做一次 VJP
        graph.zero_grad()?;
        graph.backward_with_seed(output, &seed)?;

        // 提取梯度作为 Jacobian 的第 i 行
        let grad = graph.get_node_grad(input)?.flatten();
        for j in 0..input_size {
            jacobian[[i, j]] = grad[j];
        }
    }

    Ok(jacobian)
}
```

---

## 9. 迁移计划

### 测试策略：从简单到复杂

迁移过程遵循**先单元测试、后集成测试**的原则：

1. **先从最简单的单元测试开始**：选择一个确定受影响的操作节点（如 MatMul），验证 VJP 实现正确性
2. **逐步扩展**：逐个节点/模块更新，每完成一个就确保其单元测试通过
3. **最后全面更新集成测试**：所有单元测试通过后，再迁移端到端的训练测试

---

### Phase 1：统一反向传播入口（1-2 周）

**1.1 实现新 API**
- [x] 实现统一的 `graph.backward(loss)`（内部调用现有 VJP 逻辑）
- [x] 实现统一的 `graph.zero_grad()`

**1.2 旧 API 标记废弃**
- [x] 标记旧 API 为 `#[deprecated]`
- ~~将 `backward_nodes()` 内部改为转发到新的 VJP 实现~~ → 合并到 Phase 2
- ~~将 `backward_batch()` 内部改为转发到新的 VJP 实现~~ → 合并到 Phase 2

**1.3 单元测试验证** → 合并到 Phase 2
- ~~选择一个简单操作节点（如 Add/MatMul），验证新旧 API 结果一致~~
- ~~逐步扩展到其他操作节点的单元测试~~

> **调整说明**：Phase 1.2 的转发和 1.3 的测试翻新合并到 Phase 2 一起做，避免大量测试同时失败的风险。

### Phase 2：清理 Jacobian 代码 + 测试翻新（1-2 周）

> 本阶段同时完成：删除 Jacobian → 旧 API 转发到 VJP → 翻新节点单元测试

- [x] 将 `backward_nodes()` / `backward_batch()` 转发到 `backward()`
- [x] 删除 `TraitNode` 中的 `jacobi` 相关方法
- [x] 删除所有节点结构体的 `jacobi` 字段
- [x] 删除 `calc_jacobi_to_a_parent` 实现
- [x] 更新 `on_topology_changed()` 清理 `grad` 而非 `jacobi`
- [x] 翻新所有操作节点的单元测试（使用新 API）
  - [x] 核心图测试：`graph_backward.rs`、`graph_dynamic.rs`
  - [x] 梯度控制测试：`gradient_flow_control.rs`（`detach`、`retain_graph`、`no_grad`）
  - [x] 批处理测试：`batch_mechanism.rs`
  - [x] 状态节点测试：`node_state.rs`（BPTT 支持）
  - [x] 各操作节点的梯度累积测试（14 个 `_gradient_accumulation` 测试）

### Phase 3：统一优化器 & API 重命名（1 周） ✅

- [x] 实现统一的 `optimizer.step()`
- [x] 将 `one_step()` / `one_step_batch()` 保留实现但标记废弃
- [x] 标记旧优化器方法为 `#[deprecated]`
- [x] 优化器单元测试全部通过（29 tests）
- [x] 将 `forward_node()` 重命名为 `forward()`（API 对称性）
- [x] 修复 `test_adaline_batch` 集成测试（移除手动 mean_loss，使用新 API）

### Phase 4：彻底删除废弃代码（1 周） ✅

> **调整说明**：原 Phase 4/5 顺序互换。单元测试已全部通过，先清理废弃代码再写集成测试更高效。

- [x] 删除所有 `#[deprecated]` 标记的旧 API
  - [x] `Graph::backward_nodes_ex()` / `backward_nodes()` / `backward_batch()`
  - [x] `Graph::clear_jacobi()` / `get_node_jacobi()` / `get_node_jacobi_shape()` / `get_node_jacobi_size()`
  - [x] `Optimizer::one_step()` / `update()` / `one_step_batch()` / `update_batch()`
  - [x] `OptimizerState::GradientAccumulator` 及相关方法
- [x] 删除 `jacobi` 相关的所有残留代码
- [x] 清理测试中对旧 API 的引用
  - [x] `src/nn/tests/optimizer/basic.rs`（已删除，测试 GradientAccumulator）
  - [x] 更新 `src/nn/tests/optimizer/sgd.rs`、`adam.rs`、`trait_tests.rs`
  - [x] 更新 `src/nn/tests/node_mse_loss.rs`、`node_softplus.rs`
- [x] 更新集成测试使用 `backward()` 返回的 loss 值（而非调用后再 `get_node_value()`）
  - [x] `test_california_housing_price.rs`
  - [x] `test_mnist.rs`、`test_mnist_batch.rs`、`test_mnist_linear.rs`、`test_mnist_cnn.rs`
  - [x] `test_mnist_gan.rs`
  - [x] `test_simple_regression_full_batch.rs`

### Phase 5：集成测试 & 文档（1 周） ✅

**5.1 集成测试迁移**
- [x] 更新所有集成测试使用新 API（在 Phase 4 已完成）
- [x] 验证端到端训练流程正确性（单样本 & 批量）
  - [x] `test_mnist_batch`: 91% 准确率 ✅
  - [x] `test_california_housing_regression`: 70% R² ✅
  - [x] `test_mnist_gan`: 多 Loss + detach 正常 ✅
- [x] 清理过渡期测试文件 `test_unified_backward_api.rs`（功能已被集成测试覆盖）

**5.2 文档更新**
- [x] 更新 README（标记 API 统一已完成）
- [ ] 编写迁移指南（如有外部用户）← 当前无外部用户，暂不需要

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 现有测试失败 | 中 | 短期保留废弃 API 作为过渡；逐步迁移测试后彻底删除 |
| 数值精度变化 | 低 | VJP 与 Jacobian 数学等价；但浮点运算顺序可能略有不同 |
| 用户代码兼容性 | 中 | `#[deprecated]` 警告 + 迁移指南；给用户 1 个版本周期适应 |
| 遗漏某些节点的 `calc_grad_to_parent` | 中 | 编译期检查 trait 实现完整性 |

---

## 11. 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-01-02 | 从 Jacobian 迁移到 VJP | 空间 O(n²)→O(n)，时间与前向同量级；与主流框架一致 |
| 2026-01-02 | 统一单样本与批量 API | 简化用户体验；单样本是 batch_size=1 的特例 |
| 2026-01-02 | 采用 PyTorch 风格 `zero_grad()` | 显式清梯度更安全；支持梯度累积场景 |
| 2026-01-02 | 废弃 API 最终完全删除 | `#[deprecated]` 仅是过渡手段，避免技术债务长期累积 |
| 2026-01-06 | Loss 函数默认使用 Mean reduction | 覆盖 90% 场景；学习率与 batch size 解耦；与 PyTorch 一致 |
| 2026-01-06 | 不可微节点（Step/Sign）的 VJP 返回 0 | 与 PyTorch 行为一致；保守策略，梯度不流经这些节点 |
| 2026-01-06 | 提供 `backward()` + `backward_ex()` 双 API | 简单场景用 `backward()`；多任务学习用 `backward_ex(loss, retain_graph=true)` |
| 2026-01-06 | 移除 `target_params` 参数，改用 `detach()` | PyTorch 风格：语义更清晰、性能更优、与主流框架一致。详见 [附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params) |
| 2026-01-06 | `backward()` 返回 `f32`（loss 值） | 方便用户打印 loss，与 `architecture_v2_design.md` 保持一致 |
| 2026-01-20 | 移除 PerceptionLoss 节点 | 不常用，可用 CrossEntropy 替代 |
| 2026-01-06 | 不支持隐式广播，形状不匹配时显式报错 | "Explicit is better than implicit"；与 Candle/Neuronika 一致；调试更容易 |
| 2026-01-06 | `compute_full_jacobian` 暂不实现 | 当前无明确需求；后期如有需要再添加 |
| 2026-01-06 | `requires_grad` / 冻结机制列为 Optional TODO | `detach` 已覆盖 99% 场景；optimizer 选择性绑定也可实现部分训练；详见 [梯度流控制设计 - 附录 B](gradient_flow_control_design.md#附录-brequires_grad--冻结机制可选功能) |

---

## 附录 A：框架调研详情

### Candle (Hugging Face)

```rust
// candle-core/src/backprop.rs
pub fn backward(&self) -> Result<GradStore> {
    let sorted_nodes = self.sorted_nodes();
    let mut grads = GradStore::new();
    grads.insert(self, self.ones_like()?);

    for node in sorted_nodes.iter() {
        let grad = grads.remove(node).expect("grad not populated");
        // 对每个操作类型，计算 VJP 并累加到父节点
        match op {
            Op::Binary(lhs, rhs, BinaryOp::Add) => {
                *grads.or_insert(lhs)? = grads[lhs].add(&grad)?;
                *grads.or_insert(rhs)? = grads[rhs].add(&grad)?;
            }
            // ... 其他操作
        }
    }
}
```

### Neuronika

```rust
// neuronika-variable/src/vardiff.rs
pub fn backward(&self, seed: f32) {
    self.grad_mut().fill(seed);
    self.history.buffer().iter().rev()
        .for_each(|(op, _)| op.backward());
}

// neuronika-variable/src/node/matrix_matrix_mul/mod.rs
impl Backward for MatrixMatrixMulBackwardLeft {
    fn backward(&self) {
        // dL/dA = upstream_grad @ B^T
        general_mat_mul(1., &*self.gradient.borrow(), &self.right_data.borrow().t(),
                        1., &mut *self.left_gradient.borrow_mut());
    }
}
```

### Burn

```rust
// burn-autodiff/src/ops/backward.rs
pub fn binary<B, FLhs, FRhs>(
    parents: [Option<NodeRef>; 2],
    node: NodeRef,
    grads: &mut Gradients,
    func_lhs: FLhs,
    func_rhs: FRhs,
) {
    let [grad_4lhs, grad_4rhs] = duplicate(&parents, Some(grads.consume::<B>(&node)));
    if let Some(node) = node_lhs {
        let grad = func_lhs(grad_4lhs.unwrap());
        grads.register::<B>(node.id, grad)
    }
    // ...
}
```

---

## 附录 B：API 迁移对照表

| 旧 API | 新 API | 说明 |
|--------|--------|------|
| `graph.forward_node(target)` | `graph.forward(target)` | 重命名 |
| `graph.forward_batch(target)` | `graph.forward(target)` | 统一 |
| `graph.backward_nodes(&params, loss)` | `graph.backward(loss)` | ⚠️ `params` 已忽略，改用 `detach()` 控制梯度流 |
| `graph.backward_batch(loss, None)` | `graph.backward(loss)` | 简化版（计算所有梯度） |
| `graph.backward_batch(loss, Some(&params))` | `graph.backward(loss)` | ⚠️ `params` 已忽略，改用 `detach()` 控制梯度流 |
| `graph.backward_nodes_ex(&params, loss, retain)` | `graph.backward_ex(loss, retain)` | ⚠️ `params` 已移除，改用 `detach()` |
| `graph.clear_jacobi()` | `graph.zero_grad()` | 重命名 + 语义统一 |
| `graph.clear_grad()` | `graph.zero_grad()` | 重命名 |
| `graph.get_node_jacobi(id)` | `graph.get_node_grad(id)` | 统一到 grad |
| `graph.get_node_grad(id)` | `graph.get_node_grad(id)` | 保留 |
| `optimizer.one_step(&mut graph, loss)` | `optimizer.step()` | 简化；loss.backward() 分开调用 |
| `optimizer.one_step_batch(&mut graph, loss)` | `optimizer.step()` | 统一 |
| `optimizer.update(&mut graph)` | `optimizer.step()` | 重命名 |
| `optimizer.update_batch(&mut graph)` | `optimizer.step()` | 统一 |

> **迁移注意**：使用 `target_params` 的代码需要重构为 `detach()` 方式。
> 详见 [梯度流控制设计 - 附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params)。

---

*本文档是 only_torch 自动微分系统重构的完整设计参考。核心变更是从教学导向的 Jacobian 模式迁移到工业标准的 VJP 模式，同时统一单样本与批量 API，提供 PyTorch 级用户体验。*


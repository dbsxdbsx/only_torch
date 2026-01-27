# Node vs Layer 架构设计

> 最后更新: 2025-12-22
> 状态: **设计确定**

## 概述

本项目采用**两层抽象**架构，参考 MatrixSlow 设计并优化为 PyTorch 风格：

```
┌─────────────────────────────────────────────────────────┐
│            Layer (便捷函数)                               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                  |
│  │ linear()│  │  conv() │  │pooling()│   ...            |
│  └────┬────┘  └────┬────┘  └────┬────┘                  |
│       │            │            │                        |
│       ▼            ▼            ▼                        |
│  创建并组合多个 Node，返回输出 NodeId                     |
│  Batch-First 格式: [batch, features...] 或 [batch, C, H, W] |
└─────────────────────────────────────────────────────────┘
                          |
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Node (原子操作)                        |
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐           │
│  │ MatMul │ │  Add   │ │Conv2d  │ │MaxPool │   ...     │
│  └────────┘ └────────┘ └────────┘ └────────┘           │
│                                                         │
│  • 计算图的基本单元                                      │
│  • 支持 VJP 自动微分                                     │
│  • NEAT 进化的最小粒度                                  │
│  • 统一 API：forward() + backward()                     │
└─────────────────────────────────────────────────────────┘
```

---

## Node 层（核心）

### 定义

Node 是计算图的**原子操作单元**，每个 Node：

- 执行单一数学运算
- 实现 `TraitNode` 接口
- 支持前向传播 (`calc_value_by_parents`)
- 支持反向传播 (`calc_grad_to_parent` - VJP 模式)

### 统一 API 设计

Node 层使用**统一的 VJP 模式 API**：

| API | 方法 | 说明 |
| --- | ---- | ---- |
| **前向传播** | `forward()` | 自动处理任意 batch 维度 |
| **反向传播** | `backward()` | VJP 模式，单样本是 batch_size=1 的特例 |

**设计理念**：单样本与批量使用相同的 API，底层统一使用 VJP 模式。

### 已实现的 Node

| 类型     | 节点                                          | 说明                 |
| -------- | --------------------------------------------- | -------------------- |
| **输入** | `Input`, `Parameter`                          | 数据入口、可训练参数 |
| **运算** | `Add`, `MatMul`, `Multiply`, `ScalarMultiply` | 基础数学运算         |
| **形状** | `Reshape`, `Flatten`                          | 张量形状变换         |
| **激活** | `Step`, `Tanh`, `Sigmoid`, `LeakyReLU`        | 非线性激活           |
| **损失** | `MSELoss`, `SoftmaxCrossEntropy`              | 损失计算             |
| **CNN**  | `Conv2d`, `MaxPool2d`, `AvgPool2d`            | 卷积与池化           |

### Node 特性

1. **独立性**：每个 Node 可独立使用
2. **可组合**：多个 Node 可自由连接
3. **可追踪**：自动微分完全支持
4. **NEAT 友好**：可在运行时添加/删除

---

## Layer 层（便捷 API）

### 定义

Layer 是**便捷函数**，内部创建并组合多个 Node，简化常见网络结构的构建。

**Layer 不是新的抽象层，只是语法糖！**

### 设计原则：Batch-First

**核心原则**：输入张量必须遵循 **Batch-First** 格式。

| 操作类型 | 维度格式 | 说明 |
| -------- | -------- | ---- |
| **FC/Linear** | `[batch, features]` | 必须 2D，batch 在第一维 |
| **CNN** | `[batch, C, H, W]` 或 `[C, H, W]` | 4D 时 batch 在第一维，3D 无 batch |

**关键特性**：
- **单样本是 batch_size=1 的特例**：FC 层单样本用 `[1, features]`，而非 `[features]`
- **Batch 维度始终在第一位**（如果存在）

| 理由           | 说明                                       |
| -------------- | ------------------------------------------ |
| **定位**       | Layer 是便捷 API，主要用于快速构建训练网络 |
| **统一 API**   | 单样本与批量使用相同的 forward/backward    |
| **符合主流**   | PyTorch `nn.Linear` 默认 Batch-First       |

```rust
// Layer 函数签名
pub fn linear(
    graph: &mut Graph,
    input: NodeId,        // [batch_size, in_features]
    in_features: usize,
    out_features: usize,
    batch_size: usize,
    name: Option<&str>,
) -> Result<LinearOutput, GraphError>

// 返回结构体暴露所有内部节点
pub struct LinearOutput {
    pub output: NodeId,   // 最终输出 [batch_size, out_features]
    pub weights: NodeId,  // 权重参数 [in_features, out_features]
    pub bias: NodeId,     // 偏置参数 [1, out_features]
    pub ones: NodeId,     // ones 矩阵 [batch_size, 1]（bias 广播用）
}
```

### Layer 与 Node 的对应关系

| 操作        | Node 级别              | Layer 级别                       |
| ----------- | ---------------------- | -------------------------------- |
| **全连接**  | `MatMul` + `Add`       | `linear()`                       |
| **卷积**    | `Conv2d`               | `conv2d()`                       |
| **池化**    | `MaxPool2d`/`AvgPool2d`| `max_pool2d()` / `avg_pool2d()`  |
| **激活**    | `ReLU`, `Sigmoid`...   | _(直接用 Node)_                  |
| **Reshape** | `Reshape`, `Flatten`   | _(直接用 Node)_                  |

**注意**：简单操作（激活、Reshape）不需要 Layer 包装，直接用 Node。

---

## PyTorch 风格 vs MatrixSlow 风格

### 关键区别

| 方面           | MatrixSlow 风格    | PyTorch 风格（我们采用） |
| -------------- | ------------------ | ------------------------ |
| **Conv2d**     | 每通道独立节点     | 单节点处理多通道         |
| **特征图合并** | 需要 `Concat` 节点 | 不需要                   |
| **CNN→FC**     | `Concat` → `FC`    | `Flatten` → `Linear`     |
| **复杂度**     | 节点多，图大       | 节点少，图紧凑           |

### 为什么选择 PyTorch 风格？

1. **易用性**：符合用户预期
2. **效率**：更少的节点数量
3. **扩展性**：便于后续优化
4. **项目目标**：「媲美 PyTorch 易用体验」

---

## 与 NEAT 的关系

### Node 层：NEAT 的基础

- **必需**：NEAT 需要在 Node 级别操作
- 变异操作：添加/删除节点、修改连接
- 权重进化：直接操作 Parameter 节点

### Layer 层：可选参与

```rust
// Layer 创建的节点仍可被 NEAT 访问
let fc = linear(&mut graph, input, 784, 128, Some("fc1"))?;

// NEAT 可以：
// 1. 访问内部参数
let weights = fc.weights;
let bias = fc.bias;

// 2. 在 Layer 输出后插入新节点
let new_node = graph.new_relu_node(fc.output)?;

// 3. 变异权重
graph.mutate_parameter(fc.weights, 0.1)?;
```

### NEAT 进化粒度

```
细粒度 ──────────────────────────────────────────────► 粗粒度
   │                    │                    │
   ▼                    ▼                    ▼
单个 Node          单个 Layer           网络模块
(MatMul)           (Linear)            (ResBlock)
   │                    │                    │
传统 NEAT          宏 NEAT              模块化 NEAT
```

---

## 实现路线

### 当前状态

- ✅ Node 层：核心框架完成，统一 VJP 模式
- ✅ Layer 层：`linear()`, `conv2d()`, `max_pool2d()`, `avg_pool2d()` 已实现

### 计划

1. **Phase 1** ✅：完成基础 Node（MatMul, Add, 激活, 损失）
2. **Phase 2** ✅：实现 Batch 机制（forward_batch, backward_batch）
3. **Phase 3** ✅：实现 Layer 便捷函数（linear）
4. **Phase 4** ✅：添加 CNN 节点（Conv2d, MaxPool2d, AvgPool2d）
5. **Phase 5** ✅：添加 CNN Layer（conv2d, max_pool2d, avg_pool2d）
6. **Phase 6**：MNIST CNN 端到端示例（LeNet 风格）
7. **Phase 7**：NEAT 集成

---

## 总结

| 层级      | 角色     | API 风格 | 说明                                 |
| --------- | -------- | -------- | ------------------------------------ |
| **Node**  | 原子操作 | 统一 VJP | forward() + backward()，自动处理 batch |
| **Layer** | 便捷 API | 统一 VJP | 封装 Node 组合，简化常见模式构建       |

**核心原则**：

- Node 是基础，保留两套 API 服务不同场景
- Layer 是便捷包装，只提供 Batch 版本避免冗余

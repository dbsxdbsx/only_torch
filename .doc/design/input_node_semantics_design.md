# Input 节点语义与数据共享可视化设计

> 定义 Input/TargetInput/Loss 在模型 scope 中的归属原则，以及同源数据节点的可视化标注方案。

---

## 1. 节点归属原则

在计算图可视化中，不同类型的节点有明确的归属规则：

| 节点类型 | 归属位置 | 理由 |
|---------|---------|------|
| **Input（数据输入）** | **模型内部** | 模型接口的一部分，定义了模型接受什么形状/类型的输入 |
| **Parameter（参数）** | 模型内部 | 模型学到的权重（已有行为） |
| **运算节点** | 模型内部 | 模型的计算逻辑（通过 Layer/Distribution group tag 或推断归入） |
| **TargetInput（标签）** | **模型外部** | 训练目标（真实标签），不属于模型的推理路径 |
| **Loss（损失函数）** | **模型外部** | 训练目标函数，连接模型输出与标签 |

**核心理念**：模型 scope 表示的是模型的**推理架构**（Input → 运算 → Output），训练相关的节点（TargetInput、Loss）在模型之外。

### 已修复问题

RNN/LSTM/GRU 场景下，Input 节点曾因推断算法的 bug 被错误地放在模型外部。

- **根因**：`rnn_hidden_ids` 中的折叠隐藏节点被纳入了 `node_children` 映射，但没有模型归属，导致 `all(|m| m.is_some())` 判定失败
- **修复**（commit `f5d6fd8`）：在 [`src/nn/var.rs`](../../src/nn/var.rs) 的推断算法中，构建 `node_children` 和迭代推断时排除 `rnn_hidden_ids` 中的隐藏节点
- **影响范围**：所有 RNN/LSTM/GRU 示例的 Input 现已正确归入模型 scope 内；SAC、GAN 等非循环示例行为不变

---

## 2. 跨模型数据流分类

多模型场景下，数据在模型之间的流动存在本质不同的类型：

### 2.1 图级跨模型流（已有实现）

**代表场景**：MNIST GAN

数据通过计算图中的 Var 边从一个模型流向另一个模型。图引擎知道这条连接的存在。

```
Generator 输出 (Var) ──→ Discriminator 输入 (Var)
                    图中有真实的边
```

**可视化**：虚拟 Input 节点（灰色虚线椭圆）+ 灰色虚线连接。

两种子类型：
- **梯度贯通**（GAN 的 G Loss 路径）：Generator → Discriminator，梯度从 G Loss 穿过 D 回到 G
- **梯度阻断**（GAN 的 D Fake 路径）：Generator → `detach()` → Discriminator，梯度在 detach 处被阻断

当前两种子类型使用相同的虚线样式，但 detach 节点本身（紫色虚线边框）以及 loss 路径颜色（绿色 vs 红色）已提供了足够的区分信息，无需额外改动。

### 2.2 Tensor 级数据共享（规划中 — Phase 1）

**代表场景**：CartPole SAC

同一份外部数据（如 `obs_batch`）分别通过 `input_named()` 进入多个模型，创建多个独立的 Input 节点。**图引擎不知道它们是同一份数据**。

```
obs_batch (Tensor)
  ├─ input_named("obs") → Critic1 的 Input 节点（独立）
  ├─ input_named("obs") → Critic2 的 Input 节点（独立）
  └─ input_named("obs") → Actor 的 Input 节点（独立）
```

**当前状态**：三个 Input 节点各自在模型内部，无视觉关联。用户看不出它们是同一份数据。

**规划方案**：链式虚线标注（见 §3）。

### 2.3 Input 与 TargetInput 跨类型共享（远期 — Phase 2）

**代表场景**：自编码器（Autoencoder）

同一份数据同时作为模型的输入和 loss 的目标：

```rust
let x_hat = autoencoder.forward(&graph.input_named(&x, "x")?)?;
let loss = x_hat.mse_loss(&x)?;  // x 同时是 Input 和 TargetInput
```

图中会出现两个节点（蓝色 Input + 橙色 TargetInput），底层数据相同但角色不同。

**当前状态**：无法检测。需要 Tensor 身份追踪机制（见 §4）。

---

## 3. 链式虚线标注方案（Phase 1）

### 3.1 方案概述

- 保持每个模型各自拥有独立的 Input 节点（在模型 scope 内部）
- 可视化层检测**同名 + 同 shape** 的 BasicInput 节点
- 用**链式虚线箭头**（n-1 条）在同名节点之间标注数据共享关系
- 触发条件：用户使用 `input_named()` 并给出相同名字

### 3.2 视觉效果

以 SAC 的三个模型共享 `obs` 输入为例：

```
┌── Actor ─────────────┐
│  [obs Input] → ...   │
└──────────────────────┘
         ▲
         ┊ 虚线（同源数据）
         ▼
┌── Critic1 ───────────┐
│  [obs Input] → ...   │
└──────────────────────┘
         ▲
         ┊ 虚线（同源数据）
         ▼
┌── Critic2 ───────────┐
│  [obs Input] → ...   │
└──────────────────────┘
```

采用**链式连接**（n-1 条线）而非全连接（n*(n-1)/2 条线），避免当共享节点数量较多时视觉过于混乱。

### 3.3 匹配规则

| 条件 | 必须 | 说明 |
|------|------|------|
| 节点类型 = BasicInput | 是 | 仅匹配数据输入节点，不含 TargetInput |
| 节点名称相同 | 是 | 通过 `input_named()` 的 name 参数 |
| 节点 shape 相同 | 是 | 防止同名但不同形状的误匹配 |
| 节点属于不同模型 | 是 | 同模型内的同名 Input 不标注 |

### 3.4 实现位置

在 [`src/nn/var.rs`](../../src/nn/var.rs) 的 `snapshot_to_dot` 方法的边生成阶段末尾，增加同名 Input 检测与链式虚线生成逻辑。

### 3.5 不影响的场景

| 场景 | 为什么不受影响 |
|------|--------------|
| MNIST GAN | 三个 Input 都是不同数据（real_images、z、z_g），名字不同，不匹配 |
| 单模型（XOR、MNIST 等） | 只有一个模型，不存在跨模型匹配 |
| RNN/LSTM/GRU | 只有一个模型，不存在跨模型匹配 |

---

## 4. 远期规划：Tensor 身份追踪（Phase 2）

### 需求场景

Phase 1 的名字匹配方案有两个局限：

1. **跨类型共享**：Input 与 TargetInput 使用同一份数据（自编码器），TargetInput 是自动命名的，无法通过名字匹配
2. **未命名 Input**：用户直接将 Tensor 传入 `forward()`（通过 `IntoVar` 自动转换），Input 获得自动名字，无法匹配

### 方案草案

给 Tensor 结构体增加 `source_id: Option<u64>` 字段：
- 由全局原子计数器分配
- `clone()` 保持相同 ID（同一份数据的副本）
- 运算产生的新 Tensor 获得新 ID（不同数据）
- 创建 Input/TargetInput 节点时记录 `source_id`
- 可视化层按 `source_id` 分组，绘制共享标注

### 实施条件

当出现真实的自编码器或类似需求时再实施。当前 Phase 1 的名字匹配方案已覆盖最常见的 SAC 式跨模型共享场景。

---

## 5. 设计决策记录

### 否决方案 1：合并为同一个图节点

**方案**：当同一份 Tensor 被多次用于创建 Input 时，复用同一个图节点。

**否决理由**：
- **与归属原则冲突**：共享节点的子节点分属多个模型，推断算法无法将其归入任何一个模型 → 被推到顶层 → 模型 scope 内没有 Input，架构图不完整
- **梯度语义模糊**：多个 backward 路径的梯度会在同一个节点累积（虽然对结果无影响，但增加了理解难度）

### 否决方案 2：统一 Input/TargetInput 为同一类型

**方案**：只保留一种 "Input" 节点类型，通过连接关系推断角色。

**否决理由**：
- **丢失角色语义**：蓝色（数据输入）vs 橙色（训练目标）的颜色区分是即时可读的视觉信号，合并后需要追踪连接才能分辨
- **推理图问题**：推理时不需要 TargetInput，如果类型统一，难以区分哪些节点应该出现在推理图中

### 采纳方案：独立节点 + 分层标注

**方案**：保持独立节点（维护模型完整性和角色区分），通过可视化标注层展示数据共享关系。

**优势**：
- 两个需求（模型完整性、数据共享可见性）通过**分层**同时满足
- 不改变图引擎的核心逻辑，纯可视化层增强
- 渐进式实施：Phase 1（名字匹配）→ Phase 2（Tensor ID）

---

## 6. 相关文档

- [计算图可视化指南](visualization_guide.md) — 节点样式、边类型、模型分组等可视化使用说明
- [梯度流控制机制](gradient_flow_control_design.md) — `detach`、`no_grad` 等机制
- [Graph 序列化与可视化设计](graph_serialization_design.md) — 可视化基础架构
- [未来功能规划](future_enhancements.md) — 项目整体功能路线图

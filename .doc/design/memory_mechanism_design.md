# Memory Mechanism Design（记忆/循环机制设计）

> 本文档阐述 only_torch 中记忆机制（循环结构）的设计决策，包括 NEAT 风格循环与传统 RNN 循环的关系、设计选择及实现路径。

---

## 📋 实现状态速览

| Phase | 状态 | 说明 |
|-------|------|------|
| Phase 1: 基础循环 | ✅ | `step()`/`reset()`/`connect_recurrent()` |
| Phase 2: BPTT | ✅ | 时间步快照 + 梯度累加 |
| Phase 2.5: State 节点 | ✅ | 修复跨时间梯度传递 |
| Phase 2 修复 A: 通用激活 | ✅ | 支持 tanh/sigmoid/任意组合 |
| Phase 2 修复 B: VJP 模式 | ✅ | 大 batch/hidden 高效训练 |
| Phase 3: 模板层 | ✅ | `rnn()`, `lstm()`, `gru()` Layer API |
| Phase 4: NEAT 集成 | ⏳ | 结构变异、物种形成 |

**验收指标**：
- 21/21 PyTorch 数值对照测试通过（RNN 7 + LSTM 7 + GRU 7）
- batch=64, hidden=256：241ms/5epochs，无 OOM
- IT-1 奇偶性检测：98% 准确率
- IT-3b RNN Layer：95.3% 准确率
- IT-3c LSTM Layer：93.8% 准确率
- IT-3d GRU Layer：90.6% 准确率

---

## 1. 核心概念辨析

### 1.1 两种"循环"是正交的

在机器学习中，"循环"（recurrent）一词有两种完全不同的含义：

| 概念 | NEAT 循环（拓扑循环） | 传统 RNN 循环（时间展开） |
|------|----------------------|--------------------------|
| **循环发生在哪** | 网络**拓扑结构**中（节点间形成环） | **时间维度**上（同一网络重复执行） |
| **权重共享** | ❌ 每个连接有独立权重 | ✅ 所有时间步共享同一套权重 |
| **状态/记忆** | 节点保留上一步输出值（双缓冲） | 隐藏状态 h_t 在时间步间传递 |
| **训练方法** | 进化算法（无梯度） | BPTT / TBPTT（梯度下降） |
| **结构** | 可进化（动态拓扑） | 固定（LSTM/GRU 门控结构） |

这两种机制可以**独立存在**，也可以**同时存在**：

```
                    │ 无时间展开           │ 有时间展开（RNN 风格）
 ───────────────────┼──────────────────────┼───────────────────────
 无拓扑循环（DAG）   │ 前馈网络（MLP, CNN） │ 传统 RNN/LSTM/GRU
 ───────────────────┼──────────────────────┼───────────────────────
 有拓扑循环         │ NEAT Recurrent       │ 两者结合（罕见）
```

### 1.2 NEAT 循环如何实现记忆

NEAT 的循环不是"死循环"，而是基于**离散时间步 + 双缓冲**：

```
每次 activate() 调用是一个时间步：
- 读取：上一步的节点输出值
- 写入：这一步的节点输出值
- 切换：双缓冲交替

节点 A 的自连接：
    t=0: A = f(input)           → A_old = 0, A_new = 0.76
    t=1: A = f(input + A_old)   → A_old = 0.76, A_new = 0.82
    t=2: A = f(input + A_old)   → A_old = 0.82, A_new = 0.85
    ...
```

### 1.3 NEAT 能否进化出 RNN/LSTM/Transformer？

| 目标 | 能否进化 | 说明 |
|------|---------|------|
| **记忆能力** | ✅ 可以 | 通过拓扑循环（自连接）实现 |
| **RNN 等价功能** | ✅ 功能等价 | 但结构不同（无权重共享） |
| **LSTM 门控结构** | ⚠️ 理论可能，实际极难 | 需要极大搜索空间，概率趋近于 0 |
| **Transformer/Attention** | ❌ 几乎不可能 | Q·K^T·V 模式太特殊 |

**关键洞察**：NEAT 循环是更底层、更通用的"记忆"概念，但它不会自然产生权重共享或门控机制。LSTM/Attention 等是人类设计的**强归纳偏置**，目的是让学习更稳定、更高效——而非增加表达能力。

> 💡 **实证案例**（来自 [NEAT 原始论文](../paper/NEAT_2002/summary.md)）：
>
> 在**双杆平衡无速度信息**（DPNV）任务中，网络必须从历史位置信息推断速度——这是一个典型的**非马尔可夫**任务，需要记忆机制。NEAT 演化出的解仅使用 **1 个隐藏节点 + 1 条自连接**，通过计算角度差的导数来估计速度，无需复杂的门控机制。这证明了：
> - 简单循环连接对于**短期记忆**任务足够
> - 不是所有需要记忆的任务都需要 LSTM/GRU
> - 但对于**长期依赖**，门控机制仍有优势（参见 [EXAMM 论文](../paper/EXAMM_2019/summary.md)的对比实验）

### 1.4 关键设计决策

- **单步永远是 DAG**：循环边的"环"是时间维度的，不是图结构的
- **State ≠ Input**：State 节点可接收梯度，Input 节点不可
- **BPTT = 时间调度**：不是"重复调用 backward"，而是执行器管理的跨时间梯度传递
- **NEAT 循环边 = Delay 边**：统一的抽象，`recurrent_depth` 只是 Delay 的步数

---

## 2. VJP 解释（Vector-Jacobian Product）

### 2.1 什么是 VJP

**VJP（向量-雅可比乘积）** 是反向传播的数学核心，相比显式 Jacobian 矩阵更高效：

```
传统 Jacobian 方式：
  y = f(x)     // x 是 [N] 向量，y 是 [M] 向量
  J = ∂y/∂x   // Jacobian 矩阵 [M × N]
  ∂L/∂x = (∂L/∂y)ᵀ · J   // 需要存储完整的 [M × N] 矩阵

VJP 方式：
  y = f(x)
  ∂L/∂x = vjp(f, x, ∂L/∂y)   // 直接计算结果向量 [N]，无需构造 [M × N] 矩阵
```

### 2.2 为什么 VJP 更高效

| 场景 | Jacobian 内存 | VJP 内存 |
|------|--------------|---------|
| batch=64, hidden=256 | O(N²) = 4MB/节点 | O(N) = 64KB/节点 |
| tanh 节点 | [16384 × 16384] 对角阵 | [16384] 向量 |

### 2.3 代码对应

> **历史记录**：`calc_jacobi_to_a_parent`、`get_node_jacobi` 等 Jacobian 模式 API 已废弃，当前统一为 VJP 模式（`grad` / `node.grad()`）。

```rust
// Jacobian 模式（已废弃，曾用于单样本调试）
// node.calc_jacobi_to_a_parent(...)  // 返回 [1, N] 或 [N, M] 矩阵

// VJP 模式（当前，用于训练）
node.calc_grad_to_parent(parent, upstream_grad, ...)  // 返回与 parent 同形的梯度
// 梯度获取：node.grad() 或 graph.get_node_grad(node_id)
```

---

## 3. 节点类型

| 类型 | 值来源 | 接收梯度 | 优化器参与 | 用途 |
|------|-------|---------|----------|------|
| `Input` | 用户设置 | ❌ | ❌ | 外部数据输入 |
| `Parameter` | 初始化+优化器 | ✅ | ✅ | 可学习权重 |
| `State` | 执行器注入 | ✅ | ❌ | RNN 隐藏状态 |

**State 节点的语义**：State 是"要记的东西"，不是"要学的东西"。

```rust
// state.rs 中的关键设计
impl TraitNode for State {
    // fn set_jacobi(...) { ... }  // 已废弃（Jacobian 模式）
    fn set_grad(...) { ... }    // VJP 模式支持（当前）
}
```

---

## 4. 分层架构

> 参考 [架构 V2 设计](./architecture_v2_design.md)

### 4.1 记忆机制在 3+1 层架构中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│ 第1层：高层 API（PyTorch 风格）                                   │
│   Graph 句柄 + Var（算子重载、链式调用）                         │
│   Module trait + 高层 Layer 封装（Linear, RNN, LSTM...）         │
│   用户无感知时间调度细节                                         │
│                                                                 │
│   🔸 记忆机制组件：rnn(), lstm(), gru() 便捷 API                 │
│      本质是原子节点的组合，不是新的核心概念                       │
├─────────────────────────────────────────────────────────────────┤
│ 第2层：演化 API（NEAT）                                          │
│   职责：结构变异、物种形成、recurrent_depth 配置                │
│   Genome → Graph 编译                                           │
│                                                                 │
│   🔸 记忆机制组件：NEAT 进化算法直接操作原子节点                  │
├─────────────────────────────────────────────────────────────────┤
│ 第3层：训练语义层                                                │
│   职责：forward/backward、BPTT 时间调度、Optimizer              │
│   梯度流控制：detach / no_grad（动态图架构下天然支持多次 backward，无需 retain_graph）│
│                                                                 │
│   🔸 记忆机制组件：BPTT 展开/梯度传递、TBPTT 截断                │
├─────────────────────────────────────────────────────────────────┤
│ 第4层：核心底座（GraphInner + Node + Tensor）  ← "+1"            │
│   职责：单步 DAG 的前向/局部求导                                 │
│   ⚠️ 不应知道"时间"的概念                                        │
│                                                                 │
│   🔸 记忆机制组件：State 节点、step()/reset() 语义               │
│      这是 NEAT 可直接变异的最小单元                              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 各层职责说明

| 架构层 | 记忆机制组件 | 职责 |
|--------|-------------|------|
| **第1层** 高层 API | `rnn()`, `lstm()`, `gru()` | 便捷封装，用户无需手动组装节点 |
| **第2层** 演化 API | NEAT 变异操作 | 直接操作第4层的原子节点进行结构搜索 |
| **第3层** 训练语义层 | BPTT / TBPTT | 管理时间维度的梯度传递和截断 |
| **第4层** 核心底座 | `State` 节点、`step()`/`reset()` | 原子级记忆能力，不感知"时间" |

**关键洞察**：
- BPTT 不是"把单步 backward 调 3 次"，而是**第3层训练语义层的时间调度问题**
- `rnn()`/`lstm()` 等 Layer API 是**第1层的便捷封装**，不是新的核心概念
- NEAT 直接操作**第4层的原子节点**（如 State），而非第1层的模板

### 4.3 训练方法选择

| 场景 | 推荐方法 |
|------|---------|
| 短序列 + 需要长程依赖 | Full BPTT（梯度下降） |
| 长序列 + 内存有限 | TBPTT（截断反向传播） |
| 结构搜索 + 小规模任务 | NEAT 进化（无梯度） |
| 混合策略 | NEAT 搜索结构 + 梯度微调权重 |

---

## 5. 实现路径

### 5.1 Phase 1: 基础循环支持 ✅

**目标**：实现记忆机制的核心基础设施

- [x] `recurrent_edges: HashMap<NodeId, NodeId>` 声明循环连接
- [x] `prev_values: HashMap<NodeId, Tensor>` 双缓冲
- [x] `step()` / `reset()` / `connect_recurrent()` API
- [x] `current_time_step()` / `has_recurrent_edges()` 查询

**双缓冲机制**：

```
┌─────────────────────────────────────────────────────────────┐
│  step() 流程：                                              │
│  1. 将 prev_values 传递给循环目标节点（State）               │
│  2. 执行正常前向传播                                         │
│  3. 将循环源节点的当前值存入 prev_values                     │
│  4. 保存快照用于 BPTT（训练模式下）                          │
│                                                             │
│  reset() 流程：                                             │
│  1. 清空 prev_values 和 step_history                        │
│  2. 将循环目标节点重置为零                                   │
│  3. 时间步归零                                               │
└─────────────────────────────────────────────────────────────┘
```

**验收**：8 个单元测试全部通过（见 `src/nn/tests/recurrent_basic.rs`）

### 5.2 Phase 2: BPTT 训练 ✅

**目标**：支持基于梯度的时间序列训练

#### 5.2.1 已实现

- [x] 时间步展开（隐式展开 + 快照机制）
- [x] TBPTT 截断选项
- [x] 前向传播结果与 PyTorch 完全匹配
- [x] BPTT 梯度与 PyTorch 完全匹配

#### 5.2.2 架构问题发现与解决（历史记录）

**问题**：最初将 `h_prev` 实现为 Input 节点，而 Input 节点不接收梯度，导致跨时间梯度链断裂。

**证据**（PyTorch 对照）：

| 参数 | PyTorch 梯度 | 修复前 | 修复后 |
|-----|-------------|-------|-------|
| `w_out` | -0.134053 | -0.134053 ✅ | -0.134053 ✅ |
| `w_scale` | -0.025092 | -0.020124 ❌ | -0.025092 ✅ |

**根本原因**：Input 节点在架构中是"梯度汇点"，但 RNN 的 hidden state 应该是**可导的中间量**。

**解决方案**：引入 State 节点类型（见 Phase 2.5）

#### 5.2.3 核心 API

```rust
// 完整 BPTT
graph.backward_through_time(&[params], loss)?;

// 截断 BPTT（只反向传播最近 k 步）
graph.backward_through_time_truncated(&[params], loss, Some(k))?;

// 查询
graph.history_len()      // 当前历史步数
graph.clear_history()    // 清除历史（保留循环状态）
```

### 5.3 Phase 2.5: State 节点类型 ✅

**目标**：修复跨时间梯度传递

**实现**：`src/nn/nodes/raw_node/state.rs`

```rust
pub struct State {
    value: Option<Tensor>,
    // jacobi: Option<Tensor>,  // 已废弃（Jacobian 模式）
    grad: Option<Tensor>,    // VJP 模式梯度（当前）
    shape: Vec<usize>,
}
```

**验收**：12 个单元测试全部通过（见 `src/nn/tests/node_state.rs`）

### 5.4 Phase 2 改进：通用化与效率优化 ✅

BPTT 最初实现有两个限制，现已修复：

1. **硬编码 tanh 导数** → 改为通用 "seeded backward"
   - 现在支持任意激活函数组合（tanh/sigmoid/混合）
   - 原则：BPTT 只负责时间调度，激活导数交给 Node

2. **Jacobian 模式内存爆炸** → 改为 VJP 模式
   - Jacobian：elementwise 节点产生 N×N 矩阵
   - VJP：直接计算 N 维梯度向量
   - 效果：batch=64, hidden=256 可正常训练

**核心 VJP 函数**：
- `backward_from_loss_vjp()` — loss → 参数
- `bptt_backward_from_node_vjp()` — 中间节点 → 参数
- `bptt_propagate_to_state_vjp()` — 传播到 State 节点

### 5.5 Phase 3: 模板层 ⏳

**目标**：提供便捷的循环层 API

**优先级**（按 EXAMM 论文推荐）：

| 单元 | 门数量 | 状态 | 说明 |
|------|--------|------|------|
| ∆-RNN | 1 | ⏳ 优先 | 性价比最高 |
| GRU | 2 | ⏳ | 稳定选择 |
| LSTM | 3 | ⏳ | 复杂但不一定更好 |

**∆-RNN 公式**：
```
z_t = σ(x_t · W_xz + h_{t-1} · W_hz)
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ tanh(x_t · W_xh)
```

### 5.6 Phase 4: NEAT 集成 ⏳

**目标**：支持网络结构进化

- [ ] 连接级变异（添加/删除边）
- [ ] 节点级变异（添加/删除节点）
- [ ] 物种形成（speciation）
- [ ] 可配置 `recurrent_depth`（1-N 时间步跳跃）
- [ ] Lamarckian 权重继承

---

## 6. 当前 Loss 模式：Many-to-One

### 6.1 当前实现

当前 BPTT 实现中，**只有最后一个时间步从 loss 反向传播**，中间时间步只传递来自未来的梯度：

```rust
// t=T（最后一步）：从 loss 反向传播
let state_grads = self.backward_from_loss_vjp(...)?;

// t<T（中间步）：只从 incoming_grad 传播
self.bptt_backward_from_node_vjp(from_node, incoming_grad, ...)?;
```

这对应 **many-to-one** 任务（如序列分类）：

```
输入序列: [x_1, x_2, x_3, ..., x_T]
输出: y_T（只在最后一步有 loss）
```

### 6.2 Many-to-Many 支持（未来扩展）

对于 **many-to-many** 任务（如语言模型，每步都有 loss）：

```
输入序列: [x_1, x_2, x_3, ..., x_T]
输出序列: [y_1, y_2, y_3, ..., y_T]（每步都有 loss 贡献）
```

需要在每个时间步都从 loss 反向传播，然后**合并**来自 loss 和来自未来的梯度：

```rust
// 每个时间步：
// grad = grad_from_loss + grad_from_future
```

这是 Phase 3 的可选扩展。

### 6.3 变长序列处理：Padding + Mask

**问题**：Batch 训练时，不同样本序列长度不同，如何对齐？

**方案**：采用经典的 **Padding + Mask** 机制：

1. **Padding**：将所有序列填充到 `max_len`，短序列用 0（或特殊 token）填充
2. **Mask**：生成 `[batch, 1]` 或 `[batch, hidden]` 的 mask 张量，标记有效时间步

**状态冻结公式**：

```
h_t = mask_t ⊙ h̃_t + (1 - mask_t) ⊙ h_{t-1}
```

其中：
- `h̃_t` = 正常 RNN 更新后的隐藏状态
- `mask_t` = 1 表示该样本在 t 时刻仍有效，0 表示已结束
- 结果：结束后的样本，其隐藏状态被"冻结"在最后有效时刻

**兼容性**：此机制与所有任务类型兼容：

| 任务类型 | mask 作用点 | 说明 |
|---------|-----------|------|
| many-to-one | 状态更新 | 冻结结束后的 hidden，最终只取一次 loss |
| many-to-many | 状态更新 + loss | 冻结 hidden + 只对有效步计 loss |
| one-to-many | loss | 只对有效输出步计 loss |
| one-to-one | 通常不需要 | 长度为 1 或同步映射 |

**实现方式（当前原子节点）**：

```rust
// 在图中添加 mask 输入（每时间步更新）
let mask = graph.new_input_node(&[batch, hidden], Some("mask"))?;

// 计算 h̃_t（正常 RNN 更新）
let h_tilde = graph.new_tanh_node(pre_hidden, None)?;

// 计算 delta = h̃_t - h_prev
let neg_h_prev = graph.new_scalar_multiply_node(neg_one, h_prev, None)?;
let delta = graph.new_add_node(&[h_tilde, neg_h_prev], None)?;

// h_t = h_prev + mask * delta
let masked_delta = graph.new_multiply_node(mask, delta, None)?;
let hidden = graph.new_add_node(&[h_prev, masked_delta], None)?;
```

**验收测试**：IT-3a（见第 8 节）

---

## 7. 未来考虑

### 7.1 可配置 recurrent_depth

当前实现只支持 `depth=1`（标准 RNN 循环）。EXAMM 支持 `depth=1-10`。

```rust
// 当前
recurrent_edges: HashMap<NodeId, NodeId>

// 未来可能
recurrent_edges: HashMap<NodeId, (NodeId, u32)>  // (from_node, depth)
```

`depth>1` 相当于时间维度的 skip connection，对某些任务可能有益。

### 7.2 可学习的初始隐藏状态

当前 `reset()` 将 State 重置为零。某些任务需要可学习的初始状态：

```rust
// 方案：添加可选的初始 State 参数
let h_0 = graph.parameter(&[batch, hidden], Some("h_0"))?;
graph.set_initial_state(h_prev, h_0)?;
```

---

## 8. 集成测试策略

采用**渐进式验证**：

| 阶段 | 测试类型 | Batch | 状态 |
|------|---------|-------|------|
| IT-1 | 奇偶性检测（固定长度） | ❌ 单序列 | ✅ 98% 准确率 |
| IT-2 | 奇偶性检测（固定长度） | ✅ Batch | ✅ 梯度正确 |
| IT-3a | 奇偶性检测（变长 + Padding/Mask） | ✅ Batch | ✅ 96.9% 准确率 |
| IT-3b | 奇偶性检测（变长 + RNN Layer） | ✅ Batch | ✅ 95.3% 准确率 |
| IT-3c | 奇偶性检测（变长 + LSTM Layer） | ✅ Batch | ✅ 93.8% 准确率 |
| IT-3d | 奇偶性检测（变长 + GRU Layer） | ✅ Batch | ✅ 90.6% 准确率 |

**集成测试说明**：
- **IT-3a**：用原子节点手工实现 padding + mask，验证变长语义的核心正确性
- **IT-3b/c/d**：分别用 `rnn()`, `lstm()`, `gru()` Layer API 实现同一任务，验收 Phase 3 封装易用性

**奇偶性检测任务**：

```
输入：0/1 序列（如 [1,0,1,1,0]）
输出：1 的个数是奇数(1)还是偶数(0)

为什么选这个任务：
✅ 必须有记忆才能解决
✅ 简单的二分类，100% 准确率可达
✅ 可自动生成无限训练数据
```

---

## 9. 开源项目记忆机制对比

### 9.1 核心对比表

| 维度 | neat-python | neat-rs | EXACT/EXAMM | neat-gru-rust |
|------|-------------|---------|-------------|---------------|
| **语言** | Python | Rust | C++ | Rust |
| **循环支持** | ✅ 任意拓扑 | ❌ 仅前馈 | ✅ 循环边 + 预制单元 | ✅ GRU 连接 |
| **双缓冲** | ✅ | N/A | ✅ | ✅ |
| **预制记忆单元** | ❌ | ❌ | ✅ 6 种 | ✅ GRU |
| **recurrent_depth** | 1 步固定 | N/A | 1-10 步 | 1 步固定 |
| **训练方法** | 纯进化 | 纯进化 | 进化 + BPTT | 纯进化 |
| **Lamarckian 继承** | ❌ | ❌ | ✅ | ❌ |

### 9.2 关键借鉴

| 借鉴来源 | 采纳内容 | 优先级 |
|---------|---------|--------|
| neat-python | 双缓冲机制 | 🔴 必须（已实现） |
| 所有项目 | 隐式 hidden state + reset() | 🔴 必须（已实现） |
| EXAMM | 可配置 recurrent_depth | 🟡 建议（Phase 4） |
| EXAMM | Lamarckian 权重继承 | 🟡 建议（Phase 4） |
| EXAMM | ∆-RNN 作为轻量级模板 | 🟢 可选（Phase 3） |
| neat-rs | ❌ **避免**禁止循环的设计 | — |

### 9.3 EXAMM 论文关键洞察

1. 没有"万能"记忆单元 — 不同任务最优单元不同
2. **简单神经元 + 复杂记忆单元混合效果最佳**
3. ∆-RNN 性价比最高（1 个门，表现接近 LSTM）
4. 演化出的网络非常紧凑（平均 16-29 隐藏节点，3-8 循环边）

---

## 10. 关键 API 示例

> **历史记录**：`new_parameter_node`、`get_node_jacobi` 已废弃，当前 API 为 `graph.parameter()`、`node.grad()`。

```rust
// 创建循环网络
let h_prev = graph.new_state_node(&[batch, hidden], Some("h_prev"))?;
graph.set_node_value(h_prev, Some(&Tensor::zeros(&[batch, hidden])))?;

let pre_hidden = graph.new_add_node(&[input_contrib, hidden_contrib], None)?;
let hidden = graph.new_tanh_node(pre_hidden, Some("hidden"))?;
graph.connect_recurrent(hidden, h_prev)?;

// 训练循环
for input in sequence {
    graph.set_node_value(x, Some(&input))?;
    graph.step(loss)?;
}
graph.backward_through_time(&params, loss)?;

// SGD 更新（当前使用 graph.get_node_grad() 或 node.grad()，旧 API 为 get_node_jacobi）
for &p in &params {
    if let Some(grad) = graph.get_node_grad(p)? {
        let val = graph.get_node_value(p)?.unwrap();
        let grad_reshaped = grad.reshape(val.shape());
        graph.set_node_value(p, Some(&(val - &grad_reshaped * lr)))?;
    }
}
graph.reset();  // 新序列前重置
```

---

## 11. 相关论文与资源

### NEAT 核心

- **NEAT 原始论文**：[Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) — [本地笔记](../paper/NEAT_2002/summary.md)
- **NEAT 后续综述（2021）**：[A Systematic Literature Review of the Successors of NeuroEvolution of Augmenting Topologies](https://cris.vub.be/ws/files/75376010/A_Systematic_Literature_Review_of_the.pdf)

### NEAT + 循环/记忆

- **EXAMM**：[Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution](https://arxiv.org/abs/1902.02390) — [本地笔记](../paper/EXAMM_2019/summary.md)
- **EXALT/EXAMM 框架**：https://github.com/travisdesell/exact

### RNN/BPTT

- **BPTT 原始论文**：[Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) - Rumelhart et al., 1986
- **LSTM**：[Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber, 1997
- **GRU**：[Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) - Cho et al., 2014

### 实现参考

- **neat-python**：https://github.com/CodeReclaimers/neat-python
- **PyTorch-NEAT**：https://github.com/uber-research/PyTorch-NEAT
- **radiate**：https://github.com/pkalivas/radiate
- **neat-gru-rust**：https://github.com/sakex/neat-gru-rust

---

## 12. 总结

| 问题 | 决策 | 依据 |
|------|------|------|
| 核心概念有多少？ | **极简**：原子节点 + State 节点 + Delay 边 | 架构一致性 |
| 需要硬编码 RNN/LSTM 吗？ | **不需要**：作为可选模板层提供 | Hybrid 方案 |
| hidden state 如何管理？ | **默认隐式**：可选显式接口 | 开源项目共识 |
| NEAT 和梯度训练兼容吗？ | **兼容**：NEAT 搜索结构 + 梯度微调权重 | EXAMM 验证 |
| 优先实现哪个记忆单元？ | **∆-RNN**（性价比最高） | EXAMM 论文 |
| 循环连接的正确抽象？ | **Delay(k) 边**：跨时间步的连接 | 架构文档 |
| BPTT 应在哪层实现？ | **执行引擎层** | 五层架构 |

---

*本文档记录了 only_torch 记忆机制的设计决策，综合了 NEAT/EXAMM 论文洞察及多个开源项目的实现经验。*

*最后更新：2024-12-29（Phase 2 完成；新增变长序列 Padding+Mask 方案文档）*

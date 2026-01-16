# Only-Torch 架构 V2 设计方案

> **状态**：待实现 (v2.3 - 准备开始 Phase 1)
> **作者**：架构评审
> **创建日期**：2025-12-30
> **最后更新**：2026-01-08
> **前置条件**：[自动微分统一设计](autodiff_unification_design.md) 已完成（Phase 1-5 全部 ✅）
> **背景**：基于对 Burn、Candle、Neuronika、tch-rs、neat-python、neat-rs 等框架的深度调研，以及对用户体验和梯度流控制兼容性的深入讨论，重新设计项目架构

---

## 设计摘要

本文档描述 only_torch 的 **Graph Handle + Smart Var** 架构设计，核心目标是提供 **PyTorch 级用户体验**，同时保持与 NEAT、LSTM/RNN、复杂梯度流控制的完全兼容。

> **注意**：本文档是架构设计的完整参考，无需参考其他设计文档。
>
> **黄金法则**：每个 Phase 完成后必须通过全面测试验证，确保无回归后才能进入下一阶段。

### 核心设计决策

| 决策 | 方案 | 理由 |
|------|------|------|
| Graph 结构 | `Rc<RefCell<GraphInner>>` 句柄 | 允许 Var 持有图引用 |
| Var 结构 | `NodeId + Rc<RefCell<GraphInner>>` | 支持算子重载和链式调用 |
| 算子重载 | `&a + &b`、`a + b` | PyTorch 风格数学表达式 |
| 链式调用 | `x.relu().sigmoid().matmul(&w)?` | 流畅的 API |
| forward 签名 | `fn forward(&self, x: Var)` | 无需 &Graph 参数 |
| RefCell 可见性 | 对用户完全隐藏 | 无需了解内部实现 |

### 关键概念澄清

| 问题 | 答案 |
|------|------|
| **第4层（核心底座）是否保留？** | ✅ **完全保留**。`GraphInner` 就是现有 `Graph` 的重命名，保留所有字段和方法。 |
| **Graph 是否变成 `pub(crate)`？** | ❌ **仍然是 `pub`**。新 `Graph` 和 `GraphInner` 都是 `pub`。现有粒度级操作通过 `graph.inner_mut()` 访问。 |
| **`randn` 是什么分布？** | **正态分布 N(0,1)**，与 PyTorch `torch.randn()` 语义一致。均匀分布使用 `rand()`。 |
| **`backward()` 为什么隐式先 `forward`？** | **用户体验优化**。`backward()` 采用 **ensure-forward**：若当前 pass 下 loss 尚未计算，则先触发一次 forward；若已计算（缓存命中），则不重复 forward。可选择显式模式。 |
| **新 Optimizer 与现有的是同一概念吗？** | ✅ **是同一概念**，但 API 层级不同。新版不需要传 `&mut graph`，Optimizer 内部持有图引用。 |
| **`Var` 在 forward 之前有值吗？** | ❌ **没有**。`Var` 只是节点引用（标签），`forward()` 或 `backward()`（ensure-forward：必要时先 forward）才触发实际计算。 |
| **为什么 `forward` 不是 Module trait 方法？** | 不同层的 `forward` 签名各异（MLP vs RNN vs Attention），无法统一。`parameters()` 签名一致，放入 trait。 |

### API 风格对比

```rust
// 旧设计（C 风格）
let h = graph.relu(fc1.forward(&mut graph, x));
let loss = graph.cross_entropy(h, y);
let loss_val = graph.backward(loss, &params)?;

// 新设计（PyTorch 风格）
let h = fc1.forward(x)?.relu();
let loss = h.cross_entropy(&y)?;
let loss_val = loss.backward()?;
```

---

## 1. 现状诊断

### 1.1 原五层架构 vs 代码现实

原 `high_level_architecture_design.md` 规划了五层架构，但实际代码与规划存在显著差距：

| 层级 | 文档规划 | 代码现实 | 状态 |
|------|---------|---------|------|
| 第1层 | Module/Optimizer/DataLoader | 仅有 Optimizer trait | ⚠️ 部分 |
| 第2层 | NEAT 演化 API | 未实现 | ❌ 缺失 |
| 第3层 | ExecutionEngine (Hybrid 模式) | **不存在** | ❌ 虚构 |
| 第4层 | ComputationGraph (IR) | Graph 部分承担 | ⚠️ 耦合 |
| 第5层 | Graph/Node/Tensor | 实际存在 | ✅ 完整 |

**关键发现**：
- `ExecutionEngine`、`ComputationGraph`、`Module` trait 在代码中**均不存在**
- 当前实际是 **Graph 一体化架构**：IR + 执行 + 状态管理耦合在 `Graph` 结构中
- 原"五层架构"更像是愿景文档，不是实现指南

### 1.2 当前代码的实际架构

```
┌─────────────────────────────────────────────────────────────────┐
│  辅助层                                                          │
│  ├── layer::* (linear, conv2d, rnn, lstm, gru)                  │
│  ├── optimizer (SGD, Adam, Optimizer trait)                     │
│  └── data (DataLoader, MnistDataset)                            │
├─────────────────────────────────────────────────────────────────┤
│  核心底座（Graph 一体化）                                         │
│  ├── Graph: 节点管理、边、循环边、pass_id、BPTT step_history     │
│  ├── Node/NodeHandle: 算子实现、前向计算、局部求导                │
│  └── Tensor: 数值计算、ndarray 封装                              │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 当前优势（可直接复用）

| 能力 | 状态 | 说明 |
|------|------|------|
| 动态拓扑 | ✅ | `on_topology_changed()` 支持 forward/backward 后加节点 |
| BPTT | ✅ | `step()`, `backward_through_time()`, `step_history` |
| 循环边 | ✅ | `connect_recurrent()`, `prev_values` 双缓冲 |
| Seed 管理 | ✅ | `Graph::new_with_seed()` 支持确定性训练 |
| 可视化 | ✅ | `GraphDescriptor`, Graphviz DOT 输出 |
| 序列化 | ✅ | JSON + bin 格式 |
| 梯度流控制 | ✅ | `detach`、`no_grad`、`retain_graph` 已设计 |

---

## 2. 参考框架调研

### 2.1 Burn（最接近 PyTorch）

**核心设计**：
```rust
#[derive(Module, Debug)]
pub struct MyNet<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> MyNet<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = activation::relu(x);
        self.fc2.forward(x)
    }
}
```

**关键特点**：
- `#[derive(Module)]` 宏自动生成参数收集
- `forward()` **不是** trait 方法 → 签名完全自由
- `Backend` 泛型支持多后端
- `Module` trait 核心方法：`visit()`, `map()`, `into_record()`, `load_record()`

### 2.2 Neuronika（动态图风格）

**核心设计**：
```rust
// 前向传播返回"可微分变量"
fn forward<I>(&self, input: I) -> VarDiff<impl Data + Gradient> {
    let x = self.lin1.forward(input);
    let x = neuronika::relu(x);
    self.lin2.forward(x)
}

// 训练
loss.forward();
loss.backward(1.0);
optimizer.step();
```

**关键特点**：
- Define-by-run 动态图
- 变量自带自动微分追踪
- 更接近 PyTorch 的 eager 模式

### 2.3 Candle（共享所有权设计参考）

**核心设计**：
```rust
pub struct Tensor_(
    id: TensorId,
    storage: Arc<RwLock<Storage>>,  // 共享存储，线程安全
    layout: Layout,
    op: BackpropOp,
    is_variable: bool,
    dtype: DType,
    device: Device,
);

pub struct Tensor(Arc<Tensor_>);  // 轻量句柄，Clone 只增加引用计数
```

**关键特点**：
- 使用 `Arc<RwLock<...>>` 实现共享所有权（线程安全）
- `Tensor` 本身是 `Clone` 的轻量句柄
- 对用户隐藏内部共享机制
- 支持算子重载：`let c = &a + &b;`

**对 only_torch 的启发**：
- 我们采用 `Rc<RefCell<...>>`（单线程版本），更轻量
- 同样对用户隐藏内部实现
- 同样支持算子重载

### 2.4 neat-python（NEAT 标准实现）

**核心设计**：
```python
class DefaultGenome:
    def __init__(self, key):
        self.nodes = {}        # node_id -> NodeGene
        self.connections = {}  # (in, out) -> ConnectionGene
        self.fitness = None

    def mutate(self, config): ...
    def crossover(self, other): ...
    def distance(self, other) -> float: ...
```

**关键特点**：
- **Innovation Number** 是 NEAT 核心 → 相同结构变异有相同编号
- **Species** 基于基因距离划分 → 保护创新
- Genome → Network 是编译过程

---

## 3. 设计决策分析

### 3.1 架构分层：为什么选择 3+1 层而非 5 层？

| 方案 | 优点 | 缺点 |
|------|------|------|
| **原5层** | 抽象清晰、理论完整 | 与现实脱节、过度设计、实现成本高 |
| **新3+1层** | 贴近现实、渐进式、风险低 | 抽象层次略低 |

**决策**：采用 **3+1 层**

**理由**：
1. 当前 `Graph` 已经是 IR + 执行的一体化实现，强行拆分收益低、成本高
2. 原"Hybrid 双模式"（Eager/Graph）在 CPU-only + NEAT 场景下价值有限
3. 渐进式架构升级风险更低

**新 3+1 层架构**：
```
第1层：高层 API（Var + Module + PyTorch 风格 Graph API）
第2层：演化 API（NEAT）
第3层：训练语义层（forward/backward/BPTT/Optimizer/梯度流控制）
第4层：核心底座（Graph + Node + Tensor）← 保持现有
```

### 3.2 高层 API 设计：Graph Handle + Smart Var 方案

经过深入讨论和多轮迭代，我们评估了多种方案：

| 方案 | 优点 | 缺点 |
|------|------|------|
| VarMap + VarBuilder | 参数命名清晰 | 用户仍需操作 NodeId，代码冗余 |
| ForwardContext + TrackedTensor | 算子重载、接近 PyTorch | "单次 forward-backward" 假设与复杂梯度流不兼容 |
| Graph + 简单 Var (NodeId 包装) | 简洁、零开销 | 无算子重载，API 冗长（C 风格） |
| EVar + GraphContext + 显式生命周期 | 支持算子重载 | 用户需要处理 `'g` 生命周期，定义模型时语法复杂 |
| **Graph Handle + Smart Var** | 支持算子重载、链式调用、无显式生命周期 | 轻微运行时开销（Rc clone） |

**最终决策**：采用 **Graph Handle + Smart Var** 方案

#### 3.2.1 核心设计理念

```
┌──────────────────────────────────────────────────────────────────────┐
│  用户视角（简洁、PyTorch 风格）                                        │
│                                                                       │
│    let graph = Graph::new();                                         │
│    let x = graph.input(&data)?;        // 返回 Var                   │
│    let h = x.relu();                   // 链式调用                    │
│    let y = h.matmul(&w)?;              // 算子重载                    │
│    let loss = (y - target).mse();      // 更多算子                    │
│    loss.backward()?;                   // 直接在 Var 上调用           │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│  内部实现（用户不可见）                                                │
│                                                                       │
│    Graph ──────► Rc<RefCell<GraphInner>>                             │
│                        ▲                                             │
│    Var ────────────────┘ + NodeId                                    │
│                                                                       │
│  GraphInner = 现有 Graph 的所有字段（nodes, edges, rng, ...）         │
└──────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 设计优势

1. **PyTorch 级用户体验**
   - 支持算子重载：`a + b`、`a * b`、`a - b`
   - 支持链式调用：`x.relu().matmul(&w).softmax()`
   - 支持方法式梯度控制：`fake.detach()`、`loss.backward()`

2. **对用户完全隐藏 RefCell 和生命周期**
   - 用户**永远不需要**写 `'a`、`'g` 等生命周期标注
   - 用户**永远不需要**知道 `RefCell` 的存在
   - 模型定义无任何泛型参数

3. **与 NEAT 和复杂梯度流 100% 兼容**
   - `GraphInner` 保留现有 Graph 的全部能力（动态拓扑、BPTT、循环边）
   - `detach`/`attach`/`retain_graph` 等梯度流控制完全支持
   - NEAT 可通过 `graph.inner_mut()` 访问底层进行拓扑变异

4. **性能开销极小**
   - `Rc::clone()` 只是引用计数 +1，纳秒级
   - `RefCell` 借用检查是 O(1) 操作
   - 相比 Python/PyTorch 的开销可忽略不计

#### 3.2.3 Trade-off 分析

| 方面 | 新方案 | 旧方案（简单 Var） |
|------|--------|-------------------|
| 用户体验 | ✅ PyTorch 级 | ⚠️ C 风格 |
| 算子重载 | ✅ `a + b` | ❌ `graph.add(a, b)` |
| 链式调用 | ✅ `x.relu().sigmoid()` | ❌ `graph.sigmoid(graph.relu(x))` |
| 梯度控制 | ✅ `fake.detach()` | ⚠️ `graph.detach(fake)` |
| Var 大小 | 16 字节 (NodeId + Rc) | 4 字节 (NodeId) |
| Copy trait | ❌ Clone only | ✅ Copy |
| 运行时检查 | ⚠️ RefCell borrow | ✅ 无 |

**结论**：用户体验的巨大提升远超轻微的运行时开销。

### 3.3 Module 设计：forward 是否应该是 trait 方法？

**决策**：`forward()` **不是** trait 方法

**理由**（参考 Burn）：
1. 不同层的 forward 签名差异大
2. trait 方法强制统一签名会导致大量 wrapper
3. Burn 实践证明普通方法足够好用

### 3.4 NEAT 与 Module 如何共存？

**决策**：**NEAT 独立**，通过 `Genome::compile() -> Graph` 桥接

**理由**：
1. NEAT 的核心是**结构可变**，而 Module 的核心是**结构固定**
2. 强行统一会导致 Module trait 过于复杂
3. `Genome → Graph` 编译是自然的桥接点

### 3.5 其他决策

| 决策 | 理由 |
|------|------|
| 不引入 Backend 泛型 | CPU-only 定位，避免复杂度 |
| 放弃 Hybrid 双模式 | CPU 场景收益有限 |
| NEAT 支持两级演化（Node + Layer） | Node 级是核心价值，Layer 级是实用需求 |

---

## 4. 新架构详细设计

### 4.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│  第1层：高层 API（PyTorch 风格）                                  │
│  ├── Graph（Rc<RefCell<GraphInner>> 句柄，用户友好接口）          │
│  ├── Var（NodeId + Graph 引用，支持算子重载和链式调用）           │
│  ├── Module trait（parameters 方法）                             │
│  ├── 高层 Layer 封装（Linear, Conv2d, RNN...）                   │
│  └── Init 初始化策略                                             │
├─────────────────────────────────────────────────────────────────┤
│  第2层：演化 API（NEAT）                                          │
│  ├── Genome（NodeGene + ConnectionGene）                         │
│  ├── InnovationTracker（历史标记管理）                            │
│  ├── Species + Population（物种与种群）                          │
│  └── Genome → Graph 编译器                                       │
├─────────────────────────────────────────────────────────────────┤
│  第3层：训练语义层（现有 + 增强）                                  │
│  ├── Optimizer（SGD, Adam）+ zero_grad / step / minimize         │
│  ├── DataLoader + Dataset                                        │
│  └── 梯度流控制：detach / no_grad / retain_graph                 │
├─────────────────────────────────────────────────────────────────┤
│  第4层：核心底座（GraphInner = 现有 Graph，仅重命名）              │
│  ├── GraphInner（原 Graph：节点管理、边、循环边、执行状态）        │
│  ├── Node + NodeHandle（算子实现）                                │
│  ├── Tensor（数值计算）                                           │
│  └── layer::* 底层辅助函数                                        │
└─────────────────────────────────────────────────────────────────┘
```

**重要说明**：第4层 `GraphInner` 就是 `src/nn/graph.rs` 中现有的 `Graph` 结构，只是换个名字。所有现有字段（`nodes`、`edges`、`recurrent_edges`、`rng`、`pass_id`、`step_history` 等）和方法（`forward_node`、`backward_nodes`、`step`、`backward_through_time` 等）**完全保留不变**。

### 4.1.1 现有测试的向后兼容性

现有测试（如 `test_adaline.rs`、`test_optimizer_example.rs`）有**两种迁移路径**：

**路径 A：继续使用底层 API（最小改动）**

```rust
// 通过 inner_mut() 访问 GraphInner
let graph = Graph::new();

// 通过 inner_mut() 访问 GraphInner，API 几乎不变
let mut g = graph.inner_mut();
let x = g.new_input_node(&[3, 1], Some("x"))?;
let w = g.new_parameter_node_seeded(&[1, 3], Some("w"), seed)?;
// ... 其他操作完全相同
g.forward_node(loss)?;
g.backward_nodes(&[w, b], loss)?;
drop(g);  // 释放借用

// 手动参数更新（与现有代码相同）
let mut g = graph.inner_mut();
let w_value = g.get_node_value(w)?.unwrap();
let w_grad = g.get_node_grad(w)?.unwrap();
g.set_node_value(w, Some(&(w_value - learning_rate * w_grad)))?;
```

**路径 B：迁移到新 API（推荐）**

```rust
let graph = Graph::new_with_seed(42);
let x = graph.input(&features)?;           // 返回 Var
let w = graph.parameter(&[1, 3], Init::Normal, "w")?;
let output = x.matmul(&w)? + &b;           // 算子重载
let loss = output.mse_loss(&target)?;
loss.backward()?;                           // 链式调用
optimizer.step()?;
```

**关键点**：底层 `GraphInner` API 完全保留，现有代码只需通过 `graph.inner_mut()` 访问即可继续工作。

**架构关系图**：

```
用户代码                          内部实现
─────────────────────────────────────────────────────────────────

  Graph::new()
       │
       ▼
┌─────────────┐     clone      ┌─────────────────────────────┐
│   Graph     │───────────────►│  Rc<RefCell<GraphInner>>    │
│  (handle)   │                │                             │
└─────────────┘                │  ┌───────────────────────┐  │
       │                       │  │     GraphInner        │  │
       │ .input()              │  │  ├── nodes: HashMap   │  │
       │ .parameter()          │  │  ├── edges: Vec       │  │
       ▼                       │  │  ├── recurrent_edges  │  │
┌─────────────┐     clone      │  │  ├── rng: StdRng      │  │
│    Var      │───────────────►│  │  ├── pass_id          │  │
│ (NodeId +   │                │  │  └── ...              │  │
│  Graph ref) │                │  └───────────────────────┘  │
└─────────────┘                └─────────────────────────────┘
       │
       │ .relu()
       │ .matmul(&other)
       │ .detach()
       │ .backward()
       ▼
   链式操作
```

### 4.2 核心接口定义

#### 4.2.1 Var（智能变量句柄）

```rust
use std::rc::Rc;
use std::cell::RefCell;

/// 智能变量句柄 - 携带图引用，支持算子重载和链式调用
///
/// # 设计原则
/// - 持有 `Rc<RefCell<GraphInner>>` 引用，实现算子重载
/// - 用户无需关心内部实现，像 PyTorch tensor 一样使用
/// - Clone 语义（非 Copy），但开销极低（Rc clone）
///
/// # 使用示例
/// ```rust
/// let graph = Graph::new();
/// let x = graph.input(&images)?;      // 返回 Var
/// let h = x.relu();                   // 链式调用
/// let y = h.matmul(&w)?;              // 方法调用
/// let z = &y + &b;                    // 算子重载
/// let loss = z.cross_entropy(&target)?;
/// loss.backward()?;                   // 直接在 Var 上调用
/// ```
#[derive(Clone, Debug)]
pub struct Var {
    /// 节点 ID
    id: NodeId,
    /// 图引用（用户不可见）
    graph: Rc<RefCell<GraphInner>>,
}

impl Var {
    /// 创建新的 Var（内部使用）
    pub(crate) fn new(id: NodeId, graph: Rc<RefCell<GraphInner>>) -> Self {
        Self { id, graph }
    }

    /// 获取节点 ID
    pub fn node_id(&self) -> NodeId {
        self.id
    }

    /// 检查两个 Var 是否来自同一个 Graph
    pub fn same_graph(&self, other: &Var) -> bool {
        Rc::ptr_eq(&self.graph, &other.graph)
    }

    /// 获取 Var 所属的 Graph handle
    ///
    /// 即使原始 Graph handle 已 drop，此方法仍返回有效的 Graph。
    /// 这是因为 Var 持有 GraphInner 的强引用（Rc）。
    pub fn get_graph(&self) -> Graph {
        Graph { inner: Rc::clone(&self.graph) }
    }

    // ==================== 链式激活函数 ====================
    //
    // 注意：以下方法在实现时可能会根据 4.2.1.3 节的 Trait 分层策略
    // 组织到不同的扩展 trait 中（如 VarActivationOps、VarLossOps 等）。
    // 这里展示的是用户最终使用的 API 形态。

    /// ReLU 激活
    pub fn relu(&self) -> Var {
        let id = self.graph.borrow_mut().new_relu_node(self.id, None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }

    /// Sigmoid 激活
    pub fn sigmoid(&self) -> Var {
        let id = self.graph.borrow_mut().new_sigmoid_node(self.id, None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }

    /// Tanh 激活
    pub fn tanh(&self) -> Var {
        let id = self.graph.borrow_mut().new_tanh_node(self.id, None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }

    /// Softmax
    pub fn softmax(&self) -> Var {
        let id = self.graph.borrow_mut().new_softmax_node(self.id, None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }

    // ==================== 链式矩阵运算 ====================

    /// 矩阵乘法
    pub fn matmul(&self, other: &Var) -> Result<Var, GraphError> {
        let id = self.graph.borrow_mut().new_mat_mul_node(&[self.id, other.id], None)?;
        Ok(Var::new(id, Rc::clone(&self.graph)))
    }

    // ==================== 损失函数 ====================

    /// Cross Entropy Loss
    pub fn cross_entropy(&self, target: &Var) -> Result<Var, GraphError> {
        let id = self.graph.borrow_mut()
            .new_softmax_cross_entropy_node(self.id, target.id, None)?;
        Ok(Var::new(id, Rc::clone(&self.graph)))
    }

    /// MSE Loss
    pub fn mse(&self) -> Var {
        // 假设 self 已经是 (pred - target)
        let id = self.graph.borrow_mut().new_mse_loss_from_diff_node(self.id, None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }

    /// MSE Loss (显式指定 target)
    pub fn mse_loss(&self, target: &Var) -> Result<Var, GraphError> {
        let id = self.graph.borrow_mut().new_mse_loss_node(self.id, target.id, None)?;
        Ok(Var::new(id, Rc::clone(&self.graph)))
    }

    // ==================== 梯度流控制 ====================

    /// 截断梯度流（返回自身以支持链式调用）
    pub fn detach(&self) -> Result<&Self, GraphError> {
        self.graph.borrow_mut().detach_node(self.id)?;
        Ok(self)
    }

    /// 恢复梯度流
    pub fn attach(&self) -> Result<&Self, GraphError> {
        self.graph.borrow_mut().attach_node(self.id)?;
        Ok(self)
    }

    // ==================== 执行 ====================

    /// 前向传播
    pub fn forward(&self) -> Result<(), GraphError> {
        self.graph.borrow_mut().forward_node(self.id)
    }

    /// 反向传播（ensure-forward）
    ///
    /// # 语义：ensure-forward
    /// - 若当前 pass 下该 loss 尚未计算，则先触发一次 `forward()`
    /// - 若该 loss 在当前 pass 已经 forward 过（缓存命中），则不重复 forward
    ///
    /// # 设计动机
    /// 在 define-and-run（静态图）语义下，Var 在执行前只是“节点引用”，
    /// backward 需要先确保 loss 的值已计算。调用形式保持 PyTorch 风格：`loss.backward()`。
    ///
    /// # PyTorch 风格
    /// 与 PyTorch 一致，`backward()` 计算**所有** requires_grad=True 的参数的梯度。
    /// 具体更新哪些参数由 `optimizer.step()` 控制（optimizer 绑定了特定参数）。
    ///
    /// ```rust
    /// optimizer.zero_grad()?;
    /// let loss = model.forward(x)?.cross_entropy(&y)?;
    /// loss.backward()?;      // 计算所有参数的梯度
    /// optimizer.step()?;     // 只更新 optimizer 绑定的参数
    /// ```
    pub fn backward(&self) -> Result<f32, GraphError> {
        let mut g = self.graph.borrow_mut();
        g.forward_node(self.id)?;  // ensure-forward：本 pass 未计算则计算，已计算则不重复
        let loss_val = g.get_node_value(self.id)?
            .ok_or_else(|| GraphError::ValueNotComputed(self.id))?
            .scalar();
        g.backward_from_loss(self.id)?;
        Ok(loss_val)
    }

    // ==================== 值访问与设置 ====================

    /// 获取标量值
    pub fn item(&self) -> Result<f32, GraphError> {
        let g = self.graph.borrow();
        let tensor = g.get_node_value(self.id)?
            .ok_or_else(|| GraphError::ValueNotComputed(self.id))?;
        Ok(tensor.scalar())
    }

    /// 获取 Tensor 值（克隆）
    pub fn value(&self) -> Result<Tensor, GraphError> {
        let g = self.graph.borrow();
        let tensor = g.get_node_value(self.id)?
            .ok_or_else(|| GraphError::ValueNotComputed(self.id))?;
        Ok(tensor.clone())
    }

    /// 设置节点的值（用于输入数据喂入）
    ///
    /// # 使用场景
    /// 在训练循环中更新输入节点的数据，而不是每次创建新节点。
    /// 这是避免图膨胀的关键 API。
    ///
    /// # 示例
    /// ```rust
    /// // ✅ 正确：创建一次，多次更新
    /// let x = graph.zeros(&[batch_size, 784])?;
    /// for batch in dataloader.iter() {
    ///     x.set_value(&batch)?;  // 只更新数据，不创建节点
    ///     // ...
    /// }
    /// ```
    pub fn set_value(&self, value: &Tensor) -> Result<(), GraphError> {
        self.graph.borrow_mut().set_node_value(self.id, Some(value))
    }

    /// 获取梯度（克隆）
    pub fn grad(&self) -> Result<Option<Tensor>, GraphError> {
        let g = self.graph.borrow();
        Ok(g.get_node_jacobi(self.id)?.cloned())
    }
}

// ==================== 算子重载 ====================
//
// 注意：所有算子重载在遇到跨图操作时会 panic。
// 这是设计决策，因为跨图操作是程序逻辑错误，应在开发阶段暴露。
// 如需安全的错误处理，请使用 try_add(), try_sub() 等方法。

impl std::ops::Add for &Var {
    type Output = Var;
    fn add(self, rhs: &Var) -> Var {
        assert!(self.same_graph(rhs), "Cannot operate on Vars from different Graphs");
        let id = self.graph.borrow_mut().new_add_node(&[self.id, rhs.id], None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }
}

impl std::ops::Sub for &Var {
    type Output = Var;
    fn sub(self, rhs: &Var) -> Var {
        assert!(self.same_graph(rhs), "Cannot operate on Vars from different Graphs");
        let id = self.graph.borrow_mut().new_sub_node(&[self.id, rhs.id], None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }
}

impl std::ops::Mul for &Var {
    type Output = Var;
    fn mul(self, rhs: &Var) -> Var {
        assert!(self.same_graph(rhs), "Cannot operate on Vars from different Graphs");
        let id = self.graph.borrow_mut().new_mul_node(&[self.id, rhs.id], None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }
}

impl std::ops::Div for &Var {
    type Output = Var;
    fn div(self, rhs: &Var) -> Var {
        assert!(self.same_graph(rhs), "Cannot operate on Vars from different Graphs");
        let id = self.graph.borrow_mut().new_div_node(&[self.id, rhs.id], None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }
}

impl std::ops::Neg for &Var {
    type Output = Var;
    fn neg(self) -> Var {
        let id = self.graph.borrow_mut().new_neg_node(self.id, None).unwrap();
        Var::new(id, Rc::clone(&self.graph))
    }
}

// 支持 Var（非引用）的算子重载
impl std::ops::Add for Var {
    type Output = Var;
    fn add(self, rhs: Var) -> Var { &self + &rhs }
}

impl std::ops::Sub for Var {
    type Output = Var;
    fn sub(self, rhs: Var) -> Var { &self - &rhs }
}

impl std::ops::Mul for Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Var { &self * &rhs }
}

impl std::ops::Div for Var {
    type Output = Var;
    fn div(self, rhs: Var) -> Var { &self / &rhs }
}

impl std::ops::Neg for Var {
    type Output = Var;
    fn neg(self) -> Var { -&self }
}
```

##### 4.2.1.1 Var 跨图安全性

Var 携带其所属 Graph 的引用。当两个 Var 参与运算时，会自动检查它们是否来自同一个 Graph：

- ✅ 同一 Graph 的 Var 可以自由组合
- ❌ 不同 Graph 的 Var 参与运算会 **panic**

```rust
// ✅ 正确：同一 Graph
let graph = Graph::new();
let a = graph.input(&data1)?;
let b = graph.input(&data2)?;
let c = &a + &b;  // OK

// ❌ 错误：不同 Graph
let graph1 = Graph::new();
let graph2 = Graph::new();
let a = graph1.input(&data1)?;
let b = graph2.input(&data2)?;
let c = &a + &b;  // panic: Cannot operate on Vars from different Graphs
```

**设计决策**：跨图操作是**程序逻辑错误**（类似数组越界），不是可恢复的运行时错误。因此使用 panic 而非 Result。

**常见问题澄清**：

| 场景 | 是否跨图？ | 说明 |
|------|-----------|------|
| GAN（Generator + Discriminator） | ❌ 不是 | 都在同一 Graph 中，用 detach 控制梯度 |
| Actor-Critic | ❌ 不是 | 都在同一 Graph 中 |
| Target Network | ❌ 不是 | 同一 Graph 中的两套参数 |
| 两个独立模型（如 A/B 测试） | ✅ 是 | 用 `var.value()` 提取 Tensor 传递 |

##### 4.2.1.2 Var 与 Graph 生命周期

**核心原则**：Var 持有 `Rc<RefCell<GraphInner>>` 强引用，因此：

1. **Graph handle 可以被 drop**：只要有任何 Var 存活，GraphInner 就不会被释放
2. **Var 可以恢复 Graph handle**：通过 `var.get_graph()` 获取新的 Graph handle

```rust
// 工厂模式：Graph handle 被 drop，但 Var 仍存活
fn build_model() -> (Var, Var, Vec<Var>) {
    let graph = Graph::new();
    let x = graph.input(&Tensor::zeros(&[1, 10]))?;
    let w = graph.parameter(&[10, 5], Init::Xavier, "w")?;
    let y = x.matmul(&w)?;
    (x, y, vec![w])
}  // graph handle dropped here

let (input, output, params) = build_model();
let graph = output.get_graph();  // ✅ 从 Var 恢复 Graph handle
graph.backward(&output)?;        // ✅ 继续使用
```

**设计理由**：这与 PyTorch 的 tensor 语义一致——tensor 可以独立于 model 存在。

##### 4.2.1.3 Var API 组织策略（Trait 分层设计）

**核心理念**：`Var` 是"节点句柄 + 建图语法糖"，不是节点本身。当你调用 `x.relu()` 时，实际上是在 GraphInner 中创建了一个 ReLU 节点，并返回指向它的新 `Var`。

**问题**：随着节点类型增加（激活函数、损失函数、CNN 算子、形状操作等），如果全部塞进 `impl Var`，会导致：
- 单文件膨胀（数千行）
- 难以维护和扩展
- 不同领域的算子混杂

**解决方案**：采用 **Trait 分层 + 核心集合** 的组织策略。

---

**层级 1：核心能力（直接在 `impl Var` 中）**

这些是 Var 的核心能力，用户最高频使用，必须直接可用（无需 import trait）：

```rust
impl Var {
    // ========== 身份与图访问 ==========
    pub fn node_id(&self) -> NodeId;
    pub fn same_graph(&self, other: &Var) -> bool;
    pub fn get_graph(&self) -> Graph;

    // ========== 执行控制 ==========
    pub fn forward(&self) -> Result<(), GraphError>;
    pub fn backward(&self) -> Result<f32, GraphError>;

    // ========== 值访问与设置 ==========
    pub fn value(&self) -> Result<Option<Tensor>, GraphError>;
    pub fn set_value(&self, value: &Tensor) -> Result<(), GraphError>;
    pub fn grad(&self) -> Result<Option<Tensor>, GraphError>;
    pub fn item(&self) -> Result<f32, GraphError>;

    // ========== 梯度流控制 ==========
    pub fn detach(&self) -> Result<Var, GraphError>;
    pub fn attach(&self) -> Result<Var, GraphError>;
}
```

**层级 2：算子重载（通过 `std::ops` trait）**

算术运算符通过标准库 trait 实现，用户自动可用：

```rust
// 自动可用，无需 import
let c = &a + &b;  // Add
let d = &a - &b;  // Sub
let e = &a * &b;  // Mul (逐元素)
let f = &a / &b;  // Div
let g = -&a;      // Neg
```

**层级 3：扩展 trait（按功能领域分组）**

长尾算子通过扩展 trait 组织，用户按需 import：

| Trait 名称 | 职责 | 包含的方法 |
|-----------|------|-----------|
| `VarActivationOps` | 激活函数 | `relu`, `sigmoid`, `tanh`, `softmax`, `leaky_relu`, `step` |
| `VarLossOps` | 损失函数 | `mse_loss`, `cross_entropy`, `perception_loss`, `bce_loss` |
| `VarMatrixOps` | 矩阵运算 | `matmul`, `transpose`, `reshape`, `flatten` |
| `VarVisionOps` | CNN/视觉 | `conv2d`, `max_pool2d`, `avg_pool2d`, `channel_bias_add` |
| `VarReductionOps` | 归约操作 | `sum`, `mean`, `max`, `min` |

**使用示例**：

```rust
use only_torch::nn::var::{Var, VarActivationOps, VarLossOps};

let h = x.relu();                    // VarActivationOps
let loss = output.mse_loss(&target)?; // VarLossOps
```

**层级 4：高层 Layer 封装（复杂组件）**

带配置参数、多个节点组合的复杂组件，**不应**暴露为 Var 方法，而应封装为 Layer：

```rust
// ❌ 不推荐：Var 上直接暴露复杂组件
let out = x.linear(784, 128, "fc1")?;  // 参数创建 + 乘法 + bias

// ✅ 推荐：使用 Layer 封装
let fc1 = Linear::new(&graph, 784, 128, true, "fc1")?;
let out = fc1.forward(x)?;
```

**理由**：Layer 封装可以：
- 管理自己的参数（weight, bias）
- 提供更丰富的配置（dropout, initialization 等）
- 实现 `Module` trait，支持参数收集和序列化

---

**扩展 Trait 设计模板**

当需要添加新节点到 Var API 时，遵循此模板：

```rust
/// 激活函数扩展 trait
pub trait VarActivationOps {
    /// ReLU 激活函数
    fn relu(&self) -> Var;
    /// Sigmoid 激活函数
    fn sigmoid(&self) -> Var;
    /// Tanh 激活函数
    fn tanh(&self) -> Var;
    // ... 更多激活函数
}

impl VarActivationOps for Var {
    fn relu(&self) -> Var {
        let id = self.graph.borrow_mut()
            .new_relu_node(self.id, None)
            .expect("Failed to create ReLU node");
        Var::new(id, Rc::clone(&self.graph))
    }
    // ... 其他实现
}
```

**宏生成减少样板**（可选优化）：

```rust
// 定义宏简化单输入节点的实现
macro_rules! impl_unary_var_op {
    ($trait_name:ident, $method:ident, $node_method:ident) => {
        fn $method(&self) -> Var {
            let id = self.graph.borrow_mut()
                .$node_method(self.id, None)
                .expect(concat!("Failed to create ", stringify!($method), " node"));
            Var::new(id, Rc::clone(&self.graph))
        }
    };
}
```

---

**文件组织建议**

```
src/nn/
├── var.rs              # Var 结构体 + 核心能力 + 算子重载
├── var_ops/
│   ├── mod.rs          # 导出所有扩展 trait
│   ├── activation.rs   # VarActivationOps
│   ├── loss.rs         # VarLossOps
│   ├── matrix.rs       # VarMatrixOps
│   ├── vision.rs       # VarVisionOps
│   └── reduction.rs    # VarReductionOps
└── prelude.rs          # 常用 trait 的便捷导出
```

**prelude 便捷导出**：

```rust
// src/nn/prelude.rs
pub use crate::nn::var::{Var, Init};
pub use crate::nn::var_ops::{VarActivationOps, VarLossOps, VarMatrixOps};
pub use crate::nn::graph::GraphHandle;
// ... 其他常用类型

// 用户代码
use only_torch::nn::prelude::*;
```

---

**新节点添加指南**

当需要支持新的节点类型时，按以下决策树选择暴露方式：

```
新节点需要暴露到 Var API 吗？
    │
    ├─ 是基础算子（单/双输入，无复杂配置）？
    │   ├─ 是 ──► 添加到对应的扩展 trait（如 VarActivationOps）
    │   └─ 否 ──► 考虑 Layer 封装
    │
    ├─ 是算术运算（+、-、*、/）？
    │   └─ 是 ──► 已通过 std::ops 实现，无需额外操作
    │
    └─ 是复杂组件（多节点组合、带参数、需配置）？
        └─ 是 ──► 封装为 Layer，实现 Module trait
```

**示例决策**：

| 节点类型 | 决策 | 理由 |
|---------|------|------|
| `ReLU` | `VarActivationOps` | 单输入、无配置 |
| `MSELoss` | `VarLossOps` | 双输入、无配置 |
| `Conv2d` | `VarVisionOps` 或 `Layer` | 有 stride/padding 参数，推荐 Layer |
| `Linear` | `Layer` | 需要创建 weight/bias 参数，必须 Layer |
| `BatchNorm` | `Layer` | 有 running_mean/var 状态，必须 Layer |

---

**Phase 1b 实现策略**

为保持 Phase 1b 的简洁性，当前阶段可以：

1. **暂时将所有方法放在 `impl Var` 中**（快速验证 API 可用性）
2. **Phase 2+ 重构**：当方法超过 ~20 个时，拆分为扩展 trait
3. **保持向后兼容**：通过 `prelude.rs` 统一导出，用户代码无感知

#### 4.2.2 Init（参数初始化策略）

```rust
/// 参数初始化策略
#[derive(Debug, Clone)]
pub enum Init {
    /// 常数初始化
    Constant(f32),
    /// 全零
    Zeros,
    /// 全一
    Ones,
    /// 均匀分布 [lo, hi]
    Uniform { lo: f32, hi: f32 },
    /// 正态分布
    Normal { mean: f32, std: f32 },
    /// Kaiming/He 初始化（适用于 ReLU）
    Kaiming,
    /// Xavier/Glorot 初始化（适用于 Sigmoid/Tanh）
    Xavier,
}

impl Init {
    /// 生成初始化后的 Tensor
    pub fn generate(&self, shape: &[usize], rng: &mut impl Rng) -> Tensor {
        match self {
            Init::Constant(v) => Tensor::fill(*v, shape),
            Init::Zeros => Tensor::zeros(shape),
            Init::Ones => Tensor::ones(shape),
            Init::Uniform { lo, hi } => Tensor::uniform(*lo, *hi, shape, rng),
            Init::Normal { mean, std } => Tensor::normal(*mean, *std, shape, rng),
            Init::Kaiming => {
                let fan_in = shape[0];
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::normal(0.0, std, shape, rng)
            }
            Init::Xavier => {
                let (fan_in, fan_out) = (shape[0], shape[1]);
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::normal(0.0, std, shape, rng)
            }
        }
    }
}
```

#### 4.2.3 Graph（图句柄）和 GraphInner

```rust
use std::rc::Rc;
use std::cell::RefCell;

/// GraphInner - 计算图的实际实现
///
/// 这是现有 Graph 结构的重命名，保留全部原有功能：
/// - 节点管理（nodes, node_count）
/// - 边管理（edges, recurrent_edges）
/// - 执行状态（pass_id, step_history）
/// - BPTT 支持
/// - 动态拓扑变更
///
/// 用户通常不直接操作 GraphInner，而是通过 Graph 句柄。
pub struct GraphInner {
    // 保持现有 Graph 的所有字段不变
    nodes: HashMap<NodeId, NodeHandle>,
    edges: Vec<(NodeId, NodeId)>,
    recurrent_edges: Vec<RecurrentEdge>,
    rng: StdRng,
    pass_id: u64,
    // ... 其他现有字段
}

/// Graph - 用户友好的图句柄
///
/// # 设计原则
/// - 是 `Rc<RefCell<GraphInner>>` 的薄封装
/// - Clone 语义：多个 Graph 引用同一个 GraphInner
/// - 创建的 Var 自动持有图引用
///
/// # 使用示例
/// ```rust
/// let graph = Graph::new();
/// let x = graph.input(&images)?;
/// let y = x.relu().matmul(&w)?;  // Var 上的链式调用
/// let loss = y.cross_entropy(&target)?;
/// loss.backward()?;
/// ```
#[derive(Clone)]
pub struct Graph {
    inner: Rc<RefCell<GraphInner>>,
}

impl Graph {
    // ==================== 创建 ====================

    /// 创建新图
    pub fn new() -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new())),
        }
    }

    /// 创建带种子的图（用于确定性训练）
    pub fn new_with_seed(seed: u64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(GraphInner::new_with_seed(seed))),
        }
    }

    // ==================== 创建变量（返回 Var，自动携带图引用）====================

    /// 创建输入节点并设置数据
    ///
    /// # 示例
    /// ```rust
    /// let x = graph.input(&images)?;  // 返回携带图引用的 Var
    /// let h = x.relu();               // 可直接链式调用
    /// ```
    pub fn input(&self, data: &Tensor) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_input_node(data.shape(), None)?;
        g.set_node_value(node_id, Some(data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建命名输入节点
    pub fn input_named(&self, data: &Tensor, name: &str) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_input_node(data.shape(), Some(name))?;
        g.set_node_value(node_id, Some(data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建参数节点（带初始化）
    ///
    /// # 示例
    /// ```rust
    /// let w = graph.parameter(&[784, 128], Init::Kaiming, "fc1.weight")?;
    /// ```
    pub fn parameter(&self, shape: &[usize], init: Init, name: &str) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_parameter_node(shape, Some(name))?;
        let init_data = init.generate(shape, g.rng_mut());
        g.set_node_value(node_id, Some(&init_data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建零张量
    pub fn zeros(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_input_node(shape, None)?;
        g.set_node_value(node_id, Some(&Tensor::zeros(shape)))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建全一张量
    pub fn ones(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_input_node(shape, None)?;
        g.set_node_value(node_id, Some(&Tensor::ones(shape)))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建随机张量（标准正态分布 N(0,1)）
    ///
    /// 与 PyTorch `torch.randn()` 语义一致。
    /// 如需均匀分布，请使用 `rand()`。
    pub fn randn(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_input_node(shape, None)?;
        let data = Tensor::normal(0.0, 1.0, shape, g.rng_mut());
        g.set_node_value(node_id, Some(&data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建随机张量（均匀分布 U(0,1)）
    ///
    /// 与 PyTorch `torch.rand()` 语义一致。
    /// 如需正态分布，请使用 `randn()`。
    pub fn rand(&self, shape: &[usize]) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_input_node(shape, None)?;
        let data = Tensor::uniform(0.0, 1.0, shape, g.rng_mut());
        g.set_node_value(node_id, Some(&data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    /// 创建常量节点
    pub fn constant(&self, data: &Tensor) -> Result<Var, GraphError> {
        let mut g = self.inner.borrow_mut();
        let node_id = g.new_constant_node(data.shape(), None)?;
        g.set_node_value(node_id, Some(data))?;
        Ok(Var::new(node_id, Rc::clone(&self.inner)))
    }

    // ==================== 执行（也可以在 Var 上调用）====================

    /// 前向传播
    pub fn forward(&self, output: &Var) -> Result<(), GraphError> {
        self.inner.borrow_mut().forward_node(output.node_id())
    }

    /// 反向传播（ensure-forward）
    ///
    /// 计算**所有** requires_grad=true 参数的梯度。
    /// 具体更新哪些参数由 optimizer.step() 控制。
    pub fn backward(&self, loss: &Var) -> Result<f32, GraphError> {
        loss.backward()
    }

    /// 反向传播（扩展版，支持 retain_graph）
    ///
    /// 用于多任务学习等需要多次 backward 的场景。
    pub fn backward_ex(
        &self,
        loss: &Var,
        retain_graph: bool,
    ) -> Result<f32, GraphError> {
        let mut g = self.inner.borrow_mut();
        g.forward_node(loss.node_id())?;
        let loss_val = g.get_node_value(loss.node_id())?
            .ok_or_else(|| GraphError::ValueNotComputed(loss.node_id()))?
            .scalar();
        g.backward_from_loss_ex(loss.node_id(), retain_graph)?;
        Ok(loss_val)
    }

    // ==================== 训练控制 ====================

    /// 清零所有参数的梯度
    pub fn zero_grad(&self, params: &[Var]) -> Result<(), GraphError> {
        let mut g = self.inner.borrow_mut();
        for param in params {
            g.clear_node_jacobi(param.node_id())?;
        }
        Ok(())
    }

    /// no_grad 作用域
    pub fn no_grad<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let was_grad_enabled = self.inner.borrow().is_grad_enabled();
        self.inner.borrow_mut().set_grad_enabled(false);
        let result = f();
        self.inner.borrow_mut().set_grad_enabled(was_grad_enabled);
        result
    }

    // ==================== 底层访问（NEAT 和高级用途）====================

    /// 获取底层 GraphInner 的不可变引用
    ///
    /// # 用途
    /// - 查询图状态
    /// - 可视化
    pub fn inner(&self) -> Ref<'_, GraphInner> {
        self.inner.borrow()
    }

    /// 获取底层 GraphInner 的可变引用
    ///
    /// # 用途
    /// - NEAT 拓扑变异
    /// - 直接操作底层节点
    /// - 序列化/反序列化
    pub fn inner_mut(&self) -> RefMut<'_, GraphInner> {
        self.inner.borrow_mut()
    }

    /// 获取底层 Rc（用于创建 Var）
    pub(crate) fn inner_rc(&self) -> Rc<RefCell<GraphInner>> {
        Rc::clone(&self.inner)
    }
}
```

##### 4.2.3.1 API 可见性设计

| API | 可见性 | 说明 |
|-----|--------|------|
| `Graph::new()`, `Graph::input()` 等 | `pub` | 用户主要接口 |
| `Graph::inner()` | `pub` | 返回 `Ref<'_, GraphInner>`，用于查询 |
| `Graph::inner_mut()` | `pub` | 返回 `RefMut<'_, GraphInner>`，NEAT/高级用户使用 |
| `Graph::inner_rc()` | `pub(crate)` | 内部使用，创建 Var |
| `Var::new()` | `pub(crate)` | 内部使用，只能通过 Graph 创建 |
| `Var::same_graph()`, `Var::get_graph()` | `pub` | 用户可用的辅助方法 |
| `GraphInner` 所有方法 | `pub` | 通过 `graph.inner()` / `graph.inner_mut()` 访问 |

**设计理由**：

- `graph.inner()` / `graph.inner_mut()` 是"逃生舱口"，允许 NEAT 等场景直接操作底层
- `Var::new()` 是 `pub(crate)` 确保 Var 只能通过 Graph 创建
- 普通用户永远不需要调用 `inner()`，但高级用户可以

**重要说明**：

1. **Graph 是句柄，GraphInner 是实现**
   - `Graph` 只是 `Rc<RefCell<GraphInner>>` 的薄封装
   - `GraphInner` 保留现有 `Graph` 的全部字段和方法
   - 迁移时只需将现有 `Graph` 重命名为 `GraphInner`

2. **Var 自动携带图引用**
   - 所有 `graph.input()`、`graph.parameter()` 等方法返回的 `Var` 都携带 `Rc<RefCell<GraphInner>>`
   - 这使得 `Var` 可以独立执行操作：`x.relu()`、`loss.backward()`

3. **算子重载通过 Var 实现**
   - 不在 `Graph` 上实现 `add(a, b)` 等方法
   - 而是让用户直接使用 `&a + &b` 或 `a + b`

#### 4.2.4 Module trait

```rust
/// 模块 trait
///
/// # 设计原则
/// - `forward()` **不是** trait 方法（签名各异）
/// - `new()` **不是** trait 方法（参数各异）
/// - `parameters()` 返回 `Vec<Var>`（签名一致，放入 trait）
/// - 由于 Var 携带图引用，`forward()` 不需要 `&Graph` 参数
///
/// # 为什么 forward() 和 new() 不是 trait 方法？
///
/// 不同层的签名差异太大，无法统一：
///
/// ```rust
/// // MLP
/// fn forward(&self, x: Var) -> Result<Var, GraphError>
///
/// // RNN（返回 hidden state）
/// fn forward(&self, x: Var, h: Var) -> Result<(Var, Var), GraphError>
///
/// // LSTM（返回 hidden + cell）
/// fn forward(&self, x: Var, h: Var, c: Var) -> Result<(Var, Var, Var), GraphError>
///
/// // Attention
/// fn forward(&self, q: Var, k: Var, v: Var, mask: Option<Var>) -> Result<Var, GraphError>
///
/// // new 同样各异
/// fn new(graph: &Graph, in: usize, out: usize) -> Self           // Linear
/// fn new(graph: &Graph, in: usize, hidden: usize, layers: usize) -> Self  // LSTM
/// ```
///
/// 强制放入 trait 需要 `dyn Any` 或复杂泛型，得不偿失。Burn 也采用相同设计。
///
/// # 使用示例
/// ```rust
/// struct MLP {
///     fc1: Linear,
///     fc2: Linear,
/// }
///
/// impl MLP {
///     // new 是普通方法，不是 trait 方法
///     fn new(graph: &Graph) -> Result<Self, GraphError> {
///         Ok(Self {
///             fc1: Linear::new(graph, 784, 128, true, "fc1")?,
///             fc2: Linear::new(graph, 128, 10, true, "fc2")?,
///         })
///     }
///
///     // forward 是普通方法，不是 trait 方法
///     // 不需要 graph 参数！Var 已携带图引用
///     fn forward(&self, x: Var) -> Result<Var, GraphError> {
///         let h = self.fc1.forward(x)?.relu();
///         self.fc2.forward(h)
///     }
/// }
///
/// // 只有 parameters() 需要实现 trait
/// impl Module for MLP {
///     fn parameters(&self) -> Vec<Var> {
///         [self.fc1.parameters(), self.fc2.parameters()].concat()
///     }
/// }
/// ```
pub trait Module {
    /// 获取所有可训练参数
    ///
    /// 这是 Module trait 的唯一必须实现的方法。
    /// 用于：
    /// - 优化器需要知道要更新哪些参数
    /// - 序列化/保存模型参数
    /// - 统计参数数量
    fn parameters(&self) -> Vec<Var>;

    /// 获取参数数量
    fn num_params(&self) -> usize {
        self.parameters().len()
    }

    /// 获取模块名称（用于调试）
    fn name(&self) -> &str {
        "Module"
    }

    /// 获取参数总元素数
    fn total_params(&self) -> usize {
        // 需要访问 GraphInner 获取形状，这里是估算
        self.parameters().len()
    }
}
```

**关键改进**：由于 `Var` 携带图引用，`forward()` 方法签名变得更简洁：

| 旧设计 | 新设计 |
|--------|--------|
| `fn forward(&self, graph: &mut Graph, x: Var) -> Var` | `fn forward(&self, x: Var) -> Result<Var, GraphError>` |

#### 4.2.5 高层 Layer 封装

```rust
/// 全连接层
///
/// # 设计改进
/// - `forward()` 不再需要 `&Graph` 参数
/// - 使用算子重载和链式调用
pub struct Linear {
    weight: Var,
    bias: Option<Var>,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    /// 创建线性层（参数自动注册到 Graph）
    ///
    /// # 示例
    /// ```rust
    /// let fc1 = Linear::new(&graph, 784, 128, true, "fc1")?;
    /// let output = fc1.forward(input)?;  // 不需要传 graph！
    /// ```
    pub fn new(
        graph: &Graph,
        in_features: usize,
        out_features: usize,
        bias: bool,
        name: &str,
    ) -> Result<Self, GraphError> {
        let weight = graph.parameter(
            &[in_features, out_features],
            Init::Kaiming,
            &format!("{}.weight", name),
        )?;

        let bias_var = if bias {
            Some(graph.parameter(
                &[1, out_features],
                Init::Zeros,
                &format!("{}.bias", name),
            )?)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_var,
            in_features,
            out_features,
        })
    }

    /// 前向传播
    ///
    /// # 关键改进
    /// - 不需要 `&Graph` 参数！
    /// - Var 已携带图引用，可直接调用方法
    ///
    /// # 示例
    /// ```rust
    /// let h = fc1.forward(x)?;           // 简洁！
    /// let h = fc1.forward(x)?.relu();    // 可链式调用
    /// ```
    pub fn forward(&self, x: Var) -> Result<Var, GraphError> {
        let out = x.matmul(&self.weight)?;
        match &self.bias {
            Some(b) => Ok(&out + b),  // ✅ 算子重载
            None => Ok(out),
        }
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Var> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }

    fn name(&self) -> &str {
        "Linear"
    }
}
```

**更多 Layer 示例**：

```rust
/// Conv2d 卷积层
pub struct Conv2d {
    weight: Var,
    bias: Option<Var>,
    // ... 其他参数
}

impl Conv2d {
    pub fn forward(&self, x: Var) -> Result<Var, GraphError> {
        let out = x.conv2d(&self.weight, self.stride, self.padding)?;
        match &self.bias {
            Some(b) => Ok(&out + b),
            None => Ok(out),
        }
    }
}

/// BatchNorm 层
pub struct BatchNorm {
    gamma: Var,
    beta: Var,
    running_mean: Var,
    running_var: Var,
}

impl BatchNorm {
    pub fn forward(&self, x: Var, training: bool) -> Result<Var, GraphError> {
        if training {
            // 训练模式：使用 batch 统计
            let (mean, var) = x.batch_stats()?;
            let normalized = (&x - &mean) / &(var.sqrt() + 1e-5);
            Ok(&(&self.gamma * &normalized) + &self.beta)
        } else {
            // 推理模式：使用 running 统计
            let normalized = (&x - &self.running_mean) / &(self.running_var.sqrt() + 1e-5);
            Ok(&(&self.gamma * &normalized) + &self.beta)
        }
    }
}

/// Dropout 层
pub struct Dropout {
    p: f32,
    graph: Rc<RefCell<GraphInner>>,  // 需要图引用来生成随机数
}

impl Dropout {
    pub fn forward(&self, x: Var, training: bool) -> Var {
        if training && self.p > 0.0 {
            x.dropout(self.p)
        } else {
            x
        }
    }
}
```

#### 4.2.6 Optimizer API

**PyTorch 风格设计**：

与 PyTorch 一致，`backward()` 计算**所有**参数的梯度，而 Optimizer 只更新**它绑定的**参数：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  loss.backward()         →  计算所有 requires_grad=true 参数的梯度      │
│  g_optimizer.step()      →  只更新 Generator 的参数                     │
│  d_optimizer.step()      →  只更新 Discriminator 的参数                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**与现有 Optimizer 的关系**：

| 现有 API（底层） | 新 API（高层） |
|-----------------|---------------|
| `SGD::new(&graph, lr)?` | `SGD::new(&graph, &params, lr)` |
| `optimizer.one_step(&mut graph, loss)?` | `loss.backward()? + optimizer.step()?` |
| `optimizer.update(&mut graph)?` | `optimizer.step()?` |
| 需要每次传 `&mut graph` | 不需要传 graph（内部持有） |

```rust
/// 优化器 trait（新设计）
///
/// # PyTorch 风格
/// - Optimizer 绑定特定参数
/// - `backward()` 计算所有梯度（由 Var 调用）
/// - `step()` 只更新 Optimizer 绑定的参数
///
/// # 设计决策：移除 backward_for / target_params
/// PyTorch 没有 `loss.backward(params)` 这种 API。
/// 正确的模式是：`loss.backward()` 计算所有梯度，`optimizer.step()` 更新特定参数。
/// GAN 等场景的梯度隔离通过 `detach()` 实现，而非 `target_params`。
/// 详见 [梯度流控制设计 - 附录 A](gradient_flow_control_design.md#附录-a设计决策为什么用-detach-而非-target_params)。
pub trait Optimizer {
    /// 清零所有参数的梯度
    fn zero_grad(&mut self) -> Result<(), GraphError>;

    /// 更新参数（只更新 Optimizer 绑定的参数）
    fn step(&mut self) -> Result<(), GraphError>;

    /// 一步完成：zero_grad → backward(ensure-forward) → step
    fn minimize(&mut self, loss: &Var) -> Result<f32, GraphError>;
}

/// Adam 优化器
pub struct Adam {
    params: Vec<Var>,
    graph: Rc<RefCell<GraphInner>>,  // 持有图引用
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    // 动量状态
    m: HashMap<NodeId, Tensor>,
    v: HashMap<NodeId, Tensor>,
    t: usize,
}

impl Adam {
    /// 创建 Adam 优化器
    ///
    /// # 示例
    /// ```rust
    /// let optimizer = Adam::new(&graph, &model.parameters(), 0.001);
    /// ```
    pub fn new(graph: &Graph, params: &[Var], lr: f32) -> Self {
        Self {
            params: params.to_vec(),
            graph: graph.inner_rc(),
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// 创建带完整配置的 Adam
    pub fn new_with_config(
        graph: &Graph,
        params: &[Var],
        lr: f32,
        betas: (f32, f32),
        eps: f32,
    ) -> Self {
        Self {
            params: params.to_vec(),
            graph: graph.inner_rc(),
            lr,
            betas,
            eps,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }
}

impl Optimizer for Adam {
    fn zero_grad(&mut self) -> Result<(), GraphError> {
        let mut g = self.graph.borrow_mut();
        for param in &self.params {
            g.clear_node_jacobi(param.node_id())?;
        }
        Ok(())
    }

    fn step(&mut self) -> Result<(), GraphError> {
        self.t += 1;
        let mut g = self.graph.borrow_mut();

        for param in &self.params {
            let node_id = param.node_id();

            // 获取梯度
            let grad = g.get_node_jacobi(node_id)?
                .ok_or_else(|| GraphError::GradientNotComputed(node_id))?
                .clone();

            // 获取当前值
            let value = g.get_node_value(node_id)?
                .ok_or_else(|| GraphError::ValueNotComputed(node_id))?
                .clone();

            // 初始化动量（如果需要）
            let m = self.m.entry(node_id).or_insert_with(|| Tensor::zeros(grad.shape()));
            let v = self.v.entry(node_id).or_insert_with(|| Tensor::zeros(grad.shape()));

            // Adam 更新
            let (beta1, beta2) = self.betas;
            *m = &(&*m * beta1) + &(&grad * (1.0 - beta1));
            *v = &(&*v * beta2) + &(&(&grad * &grad) * (1.0 - beta2));

            // 偏差修正
            let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
            let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));

            // 更新参数
            let update = &m_hat / &(&v_hat.sqrt() + self.eps);
            let new_value = &value - &(&update * self.lr);

            g.set_node_value(node_id, Some(&new_value))?;
        }

        Ok(())
    }

    fn minimize(&mut self, loss: &Var) -> Result<f32, GraphError> {
        self.zero_grad()?;

        // forward + backward（计算所有参数的梯度）
        let loss_val = loss.backward()?;

        // step（只更新这个 optimizer 绑定的参数）
        self.step()?;

        Ok(loss_val)
    }
}
```

**使用对比**：

| PyTorch | only_torch (新) | only_torch (旧) |
|---------|-----------------|-----------------|
| `optimizer.zero_grad()` | `optimizer.zero_grad()?` | `optimizer.zero_grad(&mut graph)?` |
| `optimizer.step()` | `optimizer.step()?` | `optimizer.step(&mut graph)?` |
| `optimizer.minimize(loss)` | `optimizer.minimize(&loss)?` | `optimizer.minimize(&mut graph, loss)?` |

#### 4.2.7 NEAT Genome（保持不变）

```rust
/// 节点类型
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NeatNodeType {
    Input,
    Hidden,
    Output,
}

/// 节点基因
#[derive(Debug, Clone)]
pub struct NodeGene {
    pub id: u32,
    pub node_type: NeatNodeType,
    pub activation: Activation,
    pub bias: f32,
}

/// 连接基因
#[derive(Debug, Clone)]
pub struct ConnectionGene {
    pub innovation: u32,
    pub from: u32,
    pub to: u32,
    pub weight: f32,
    pub enabled: bool,
}

/// 基因组
#[derive(Debug, Clone)]
pub struct Genome {
    pub id: u32,
    pub nodes: HashMap<u32, NodeGene>,
    pub connections: HashMap<u32, ConnectionGene>,
    pub fitness: Option<f32>,
}

impl Genome {
    pub fn minimal(inputs: usize, outputs: usize) -> Self;
    pub fn compile(&self) -> Result<Graph, GraphError>;
    pub fn mutate_add_connection(&mut self, tracker: &mut InnovationTracker);
    pub fn mutate_add_node(&mut self, tracker: &mut InnovationTracker);
    pub fn mutate_weights(&mut self, rate: f32, strength: f32);
    pub fn crossover(&self, other: &Genome) -> Genome;
    pub fn distance(&self, other: &Genome, config: &NeatConfig) -> f32;
}
```

#### 4.2.8 错误处理策略

##### 4.2.8.1 Panic vs Result 的设计决策

| 错误类型 | 处理方式 | 理由 |
|---------|---------|------|
| 跨图 Var 运算 | **panic** | 程序逻辑错误，类似数组越界 |
| 形状不匹配 | **Result** | 可能是用户输入错误，应优雅处理 |
| 节点不存在 | **Result** | 可能是 API 误用，可恢复 |
| 值/梯度未计算 | **Result** | 调用顺序问题，可通过提示修复 |
| RefCell 重入借用 | **panic** | 代码 bug，不应该发生 |

##### 4.2.8.2 算子重载的错误处理

由于 Rust 的 `std::ops` trait 要求返回具体类型（非 `Result`），算子重载使用 **panic**：

```rust
// ❌ 以下在编译时正确，但运行时 panic
let a = graph1.input(&data)?;
let b = graph2.input(&data)?;
let c = &a + &b;  // panic!

// ✅ 需要安全处理时，使用 try_ 方法
let c = a.try_add(&b)?;  // 返回 Result<Var, GraphError>
```

**try_* 方法列表**：

```rust
impl Var {
    pub fn try_add(&self, other: &Var) -> Result<Var, GraphError>;
    pub fn try_sub(&self, other: &Var) -> Result<Var, GraphError>;
    pub fn try_mul(&self, other: &Var) -> Result<Var, GraphError>;
    pub fn try_div(&self, other: &Var) -> Result<Var, GraphError>;
}
```

##### 4.2.8.3 GraphError 新增错误类型

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    // 现有错误类型...
    NodeNotFound(NodeId),
    InvalidShape { expected: Vec<usize>, got: Vec<usize> },

    // 新增错误类型
    /// 节点值尚未计算（需要先调用 forward）
    ValueNotComputed(NodeId),
    /// 节点梯度尚未计算（需要先调用 backward）
    GradientNotComputed(NodeId),
    /// 两个 Var 来自不同的 Graph（仅 try_* 方法使用）
    GraphMismatch { left: usize, right: usize },
    /// 形状不匹配（新增详细信息）
    ShapeMismatch { op: &'static str, left: Vec<usize>, right: Vec<usize> },
    /// 节点已被 detach，不能参与梯度计算
    NodeDetached(NodeId),
}
```

### 4.3 使用示例

#### 4.3.1 Define-and-Run 语义说明

**重要**：only_torch 采用 **define-and-run**（静态图）模式，与 PyTorch 的 define-by-run（动态图）不同：

```rust
let graph = Graph::new();
let x = graph.input(&images)?;      // x 是 Var（节点引用），此时 x 没有值
let h = x.relu();                   // h 也是 Var，此时 h 没有值
let y = h.matmul(&w)?;              // y 也是 Var，此时 y 没有值
let loss = y.cross_entropy(&t)?;    // loss 也是 Var，此时 loss 没有值

// 此时所有 Var 都只是"标签"（节点引用），没有实际计算

loss.backward()?;  // 这里会触发实际计算（ensure-forward：若尚未 forward 则先 forward；若已计算则不重复）
                   // 1. forward 计算所有节点的值
                   // 2. backward 计算所有参数的梯度
                   // 3. 返回 loss 的 f32 标量值
```

如果需要中间值：
```rust
h.forward()?;              // 显式触发 forward
let h_tensor = h.value()?; // 获取计算后的 Tensor
```

> **⚠️ 警告：图膨胀问题**
>
> 在 define-and-run 语义下，每次调用 `graph.input()`、`graph.zeros()` 或任何建图方法都会创建 **新节点**。
> 图不会自动删除节点——这是持久化计算图的固有特性。
>
> **错误示范**（图会无限膨胀）：
> ```rust
> for batch in dataloader.iter() {
>     let x = graph.input(&batch)?;  // ❌ 每次循环创建新节点！
>     let y = model.forward(x)?;     // ❌ 又创建一堆新节点！
>     // ... epoch 1: 100 个节点；epoch 2: 200 个节点...
> }
> ```
>
> **正确做法**（图规模恒定）：
> ```rust
> // ✅ 建图阶段：仅执行一次
> let x = graph.zeros(&[batch_size, 784])?;  // 创建输入节点
> let y = model.forward(x.clone())?;         // 创建计算子图
>
> for batch in dataloader.iter() {
>     x.set_value(&batch)?;   // ✅ 只更新数据，不创建节点
>     y.backward()?;          // ✅ 复用已有计算图
> }
> ```

#### 4.3.2 标准训练循环（新 PyTorch 风格 API）

```rust
use only_torch::nn::{Graph, Var, Linear, Module, Init};
use only_torch::nn::optimizer::{Adam, Optimizer};

// 定义模型 - 无需任何泛型或生命周期！
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new(graph: &Graph) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, 784, 128, true, "fc1")?,
            fc2: Linear::new(graph, 128, 10, true, "fc2")?,
        })
    }

    // ✅ forward 不需要 &mut graph 参数！Var 已携带图引用
    fn forward(&self, x: Var) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x)?.relu();    // 链式调用
        self.fc2.forward(h)
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

fn main() -> Result<(), GraphError> {
    // 初始化
    let graph = Graph::new_with_seed(42);
    let model = MLP::new(&graph)?;
    let params = model.parameters();
    let mut optimizer = Adam::new(&graph, &params, 0.001);

    // ✅ Plan A（更像 PyTorch 的写法）：建图一次，复用输入节点，用 set_value 喂新数据
    //
    // 重要：`graph.input(&tensor)` 每调用一次都会创建一个新的 Input 节点。
    // 在训练循环里反复调用会导致节点数量持续增长（图不会自动“删除节点”），最终影响内存与速度。
    //
    // 因此训练循环推荐：先创建固定形状的输入 Var（例如 zeros），再在循环里 set_value()。
    let x = graph.zeros(&[/* batch_size */, 784])?;
    let y = graph.zeros(&[/* batch_size */, 10])?;

    // 训练循环
    for epoch in 0..10 {
        for (images, labels) in dataloader.iter() {
            // ✅ PyTorch 风格：清零梯度
            optimizer.zero_grad()?;

            // ✅ 喂数据（复用输入节点）
            x.set_value(&images)?;
            y.set_value(&labels)?;

            // ✅ forward - 直接调用，无需传 graph
            let output = model.forward(x.clone())?;

            // ✅ 计算 loss - 链式调用
            let loss = output.cross_entropy(&y)?;

            // ✅ backward - 直接在 Var 上调用！
            let loss_val = loss.backward()?;

            // ✅ 更新参数
            optimizer.step()?;

            println!("Loss: {:.4}", loss_val);
        }
    }

    Ok(())
}
```

**对比旧 API**：

| 旧 API（C 风格） | 新 API（PyTorch 风格） |
|-----------------|----------------------|
| `graph.relu(x)` | `x.relu()` |
| `graph.add(a, b)` | `&a + &b` 或 `a + b` |
| `graph.matmul(a, b)` | `a.matmul(&b)?` |
| `graph.backward(loss, &params)?` | `loss.backward()?` |
| `graph.detach(fake)?` | `fake.detach()?` |
| `fn forward(&self, graph: &mut Graph, x: Var)` | `fn forward(&self, x: Var)` |
```

#### 4.3.3 简化版（使用 minimize）

```rust
fn main() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);
    let model = MLP::new(&graph)?;
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.001);

    // ✅ Plan A：建图一次 + set_value 喂数据
    let x = graph.zeros(&[/* batch_size */, 784])?;
    let y = graph.zeros(&[/* batch_size */, 10])?;

    for (images, labels) in dataloader.iter() {
        x.set_value(&images)?;
        y.set_value(&labels)?;
        let output = model.forward(x.clone())?;
        let loss = output.cross_entropy(&y)?;

        // ✅ 一行搞定：zero_grad → backward(ensure-forward) → step
        let loss_val = optimizer.minimize(&loss)?;

        println!("Loss: {:.4}", loss_val);
    }

    Ok(())
}
```

#### 4.3.4 算子重载示例

```rust
fn main() -> Result<(), GraphError> {
    let graph = Graph::new();

    // 创建变量
    let x = graph.input(&Tensor::new(&[1.0, 2.0, 3.0], &[3, 1]))?;
    let w = graph.parameter(&[3, 2], Init::Xavier, "w")?;
    let b = graph.parameter(&[1, 2], Init::Zeros, "b")?;

    // ✅ 使用算子重载 - 像写数学公式一样！
    let h = x.matmul(&w)?;
    let y = &h + &b;           // 算子重载: Add
    let y = y.relu();          // 链式调用

    // 计算 loss
    let target = graph.constant(&Tensor::new(&[0.5, 0.5], &[1, 2]))?;
    let diff = &y - &target;   // 算子重载: Sub
    let loss = (&diff * &diff).mean();  // 算子重载: Mul

    // 反向传播
    loss.backward()?;

    println!("Loss: {:.4}", loss.item()?);

    Ok(())
}
```

#### 4.3.5 GAN 训练（复杂梯度流）

```rust
/// Generator 模型
struct Generator {
    fc1: Linear,
    fc2: Linear,
}

impl Generator {
    fn new(graph: &Graph, latent_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, latent_dim, hidden_dim, true, "g_fc1")?,
            fc2: Linear::new(graph, hidden_dim, output_dim, true, "g_fc2")?,
        })
    }

    fn forward(&self, z: Var) -> Result<Var, GraphError> {
        let h = self.fc1.forward(z)?.relu();
        let out = self.fc2.forward(h)?.tanh();  // 输出范围 [-1, 1]
        Ok(out)
    }
}

/// Discriminator 模型
struct Discriminator {
    fc1: Linear,
    fc2: Linear,
}

impl Discriminator {
    fn new(graph: &Graph, input_dim: usize, hidden_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, input_dim, hidden_dim, true, "d_fc1")?,
            fc2: Linear::new(graph, hidden_dim, 1, true, "d_fc2")?,
        })
    }

    fn forward(&self, x: Var) -> Result<Var, GraphError> {
        let h = self.fc1.forward(x)?.relu();
        let out = self.fc2.forward(h)?.sigmoid();  // 输出概率
        Ok(out)
    }
}

fn train_gan(
    graph: &Graph,
    generator: &Generator,
    discriminator: &Discriminator,
    g_optimizer: &mut Adam,
    d_optimizer: &mut Adam,
    real_images: &Tensor,
    noise: &Tensor,
    batch_size: usize,
) -> Result<(f32, f32), GraphError> {
    // === 训练判别器 ===
    d_optimizer.zero_grad()?;

    // 真实样本
    let real = graph.input(real_images)?;
    let d_real = discriminator.forward(real)?;
    let real_labels = graph.ones(&[batch_size, 1])?;
    let d_loss_real = d_real.bce_loss(&real_labels)?;

    // 生成样本
    let z = graph.input(noise)?;
    let fake = generator.forward(z)?;

    // ✅ 关键：detach 阻止梯度流向 G（训练 D 时不更新 G）
    fake.detach()?;

    let d_fake = discriminator.forward(fake.clone())?;
    let fake_labels = graph.zeros(&[batch_size, 1])?;
    let d_loss_fake = d_fake.bce_loss(&fake_labels)?;

    // 总 D loss
    let d_loss = &d_loss_real + &d_loss_fake;  // ✅ 算子重载
    // ✅ backward() 计算所有梯度，但因为 fake 被 detach，G 的参数梯度为 0
    let d_loss_val = d_loss.backward()?;
    // ✅ step() 只更新 D 的参数（d_optimizer 只绑定了 D 的参数）
    d_optimizer.step()?;

    // === 训练生成器 ===
    g_optimizer.zero_grad()?;

    // ✅ 恢复 fake 的梯度流（训练 G 时需要梯度流过 D 到 G）
    fake.attach()?;

    let d_fake_for_g = discriminator.forward(fake)?;
    let g_labels = graph.ones(&[batch_size, 1])?;
    let g_loss = d_fake_for_g.bce_loss(&g_labels)?;
    // ✅ backward() 计算所有梯度
    let g_loss_val = g_loss.backward()?;
    // ✅ step() 只更新 G 的参数（g_optimizer 只绑定了 G 的参数）
    g_optimizer.step()?;

    Ok((d_loss_val, g_loss_val))
}

// 主训练循环
fn main() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 创建模型
    let generator = Generator::new(&graph, 100, 256, 784)?;
    let discriminator = Discriminator::new(&graph, 784, 256)?;

    // 创建优化器
    let mut g_optimizer = Adam::new(&graph, &generator.parameters(), 0.0002);
    let mut d_optimizer = Adam::new(&graph, &discriminator.parameters(), 0.0002);

    for epoch in 0..100 {
        for batch in dataloader.iter() {
            let noise = Tensor::randn(&[batch_size, 100]);
            let (d_loss, g_loss) = train_gan(
                &graph, &generator, &discriminator,
                &mut g_optimizer, &mut d_optimizer,
                &batch, &noise, batch_size,
            )?;
            println!("D Loss: {:.4}, G Loss: {:.4}", d_loss, g_loss);
        }
    }

    Ok(())
}
```

**GAN 梯度流说明**：

```
训练 D 时：                           训练 G 时：

  z ───► G ───► fake ──✂──► D        z ───► G ───► fake ───► D
              detach()               attach()
              阻止梯度                恢复梯度

  ✂ = fake.detach() 截断梯度流        fake.attach() 恢复梯度流
```

#### 4.3.6 多任务学习（retain_graph）

```rust
fn train_multitask(
    graph: &Graph,
    backbone: &Backbone,
    cls_head: &ClassificationHead,
    reg_head: &RegressionHead,
    optimizer: &mut Adam,
    images: &Tensor,
    cls_labels: &Tensor,
    reg_targets: &Tensor,
) -> Result<(f32, f32), GraphError> {
    optimizer.zero_grad()?;

    let x = graph.input(images)?;
    let features = backbone.forward(x)?;

    // 分类任务
    let cls_out = cls_head.forward(features.clone())?;  // clone Var（廉价）
    let cls_target = graph.input(cls_labels)?;
    let cls_loss = cls_out.cross_entropy(&cls_target)?;

    // 回归任务
    let reg_out = reg_head.forward(features)?;
    let reg_target = graph.input(reg_targets)?;
    let reg_loss = reg_out.mse_loss(&reg_target)?;

    // 两个 loss 都需要 backward
    // 第一个使用 retain_graph=true（保留计算图，供第二个 loss 使用）
    let cls_loss_val = graph.backward_ex(&cls_loss, true)?;   // 保留图
    let reg_loss_val = graph.backward_ex(&reg_loss, false)?;  // 释放图

    // step() 只更新 optimizer 绑定的参数
    optimizer.step()?;

    Ok((cls_loss_val, reg_loss_val))
}
```

#### 4.3.7 Actor-Critic 强化学习

```rust
/// Actor 网络（策略网络）
struct Actor {
    fc1: Linear,
    fc2: Linear,
}

impl Actor {
    fn new(graph: &Graph, state_dim: usize, hidden_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, state_dim, hidden_dim, true, "actor_fc1")?,
            fc2: Linear::new(graph, hidden_dim, action_dim, true, "actor_fc2")?,
        })
    }

    fn forward(&self, state: Var) -> Result<Var, GraphError> {
        let h = self.fc1.forward(state)?.relu();
        let logits = self.fc2.forward(h)?;
        Ok(logits.softmax())  // 输出动作概率分布
    }
}

impl Module for Actor {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

/// Critic 网络（价值网络）
struct Critic {
    fc1: Linear,
    fc2: Linear,
}

impl Critic {
    fn new(graph: &Graph, state_dim: usize, hidden_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            fc1: Linear::new(graph, state_dim, hidden_dim, true, "critic_fc1")?,
            fc2: Linear::new(graph, hidden_dim, 1, true, "critic_fc2")?,
        })
    }

    fn forward(&self, state: Var) -> Result<Var, GraphError> {
        let h = self.fc1.forward(state)?.relu();
        self.fc2.forward(h)  // 输出状态价值 V(s)
    }
}

impl Module for Critic {
    fn parameters(&self) -> Vec<Var> {
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}

/// Actor-Critic 训练步骤
fn train_actor_critic(
    graph: &Graph,
    actor: &Actor,
    critic: &Critic,
    actor_optimizer: &mut Adam,
    critic_optimizer: &mut Adam,
    state: &Tensor,
    action: usize,
    reward: f32,
    next_state: &Tensor,
    done: bool,
    gamma: f32,
) -> Result<(f32, f32), GraphError> {
    // ========== 计算 TD Target ==========
    let state_var = graph.input(state)?;
    let next_state_var = graph.input(next_state)?;

    // 计算当前状态价值
    let value = critic.forward(state_var.clone())?;

    // 计算下一状态价值（no_grad，不参与梯度计算）
    let next_value = graph.no_grad(|| {
        critic.forward(next_state_var)
    })?;

    // TD Target: r + γ * V(s')
    let gamma_tensor = graph.constant(&Tensor::scalar(gamma))?;
    let reward_tensor = graph.constant(&Tensor::scalar(reward))?;

    let td_target = if done {
        reward_tensor  // 终止状态，TD target = reward
    } else {
        &reward_tensor + &(&gamma_tensor * &next_value)  // ✅ 算子重载
    };

    // ========== 训练 Critic ==========
    critic_optimizer.zero_grad()?;

    // Critic Loss: MSE(V(s), TD_target)
    // 关键：detach td_target，不让梯度流向 next_value 的计算
    td_target.detach()?;
    let critic_loss = value.mse_loss(&td_target)?;
    // ✅ backward() 计算所有梯度
    let critic_loss_val = critic_loss.backward()?;
    // ✅ step() 只更新 Critic（critic_optimizer 只绑定了 Critic 的参数）
    critic_optimizer.step()?;

    // ========== 训练 Actor ==========
    actor_optimizer.zero_grad()?;

    // Advantage: TD_target - V(s)
    // 关键：detach value，Advantage 不应该更新 Critic
    value.detach()?;
    let advantage = &td_target - &value;  // ✅ 算子重载

    // 计算动作概率
    let action_probs = actor.forward(state_var)?;

    // 选择动作的 log 概率
    let log_prob = action_probs.log().select(action)?;  // 选择执行的动作

    // Actor Loss: -log(π(a|s)) * Advantage（策略梯度）
    let actor_loss = -(&log_prob * &advantage);  // ✅ 算子重载 + Neg
    // ✅ backward() 计算所有梯度
    let actor_loss_val = actor_loss.backward()?;
    // ✅ step() 只更新 Actor（actor_optimizer 只绑定了 Actor 的参数）
    actor_optimizer.step()?;

    Ok((actor_loss_val, critic_loss_val))
}

// 主训练循环
fn main() -> Result<(), GraphError> {
    let graph = Graph::new_with_seed(42);

    // 环境参数
    let state_dim = 4;   // CartPole 状态维度
    let action_dim = 2;  // CartPole 动作维度
    let hidden_dim = 64;

    // 创建网络
    let actor = Actor::new(&graph, state_dim, hidden_dim, action_dim)?;
    let critic = Critic::new(&graph, state_dim, hidden_dim)?;

    // 创建优化器
    let mut actor_optimizer = Adam::new(&graph, &actor.parameters(), 0.001);
    let mut critic_optimizer = Adam::new(&graph, &critic.parameters(), 0.001);

    // 训练循环
    for episode in 0..1000 {
        let mut state = env.reset();
        let mut total_reward = 0.0;

        loop {
            // 选择动作
            let state_tensor = Tensor::from_vec(&state, &[1, state_dim]);
            let state_var = graph.input(&state_tensor)?;
            let action_probs = actor.forward(state_var)?;
            action_probs.forward()?;  // 执行前向
            let action = sample_action(&action_probs.value()?);

            // 执行动作
            let (next_state, reward, done) = env.step(action);
            total_reward += reward;

            // 训练
            let next_state_tensor = Tensor::from_vec(&next_state, &[1, state_dim]);
            train_actor_critic(
                &graph, &actor, &critic,
                &mut actor_optimizer, &mut critic_optimizer,
                &state_tensor, action, reward, &next_state_tensor, done, 0.99,
            )?;

            if done {
                println!("Episode {}: Total Reward = {:.2}", episode, total_reward);
                break;
            }

            state = next_state;
        }
    }

    Ok(())
}
```

**Actor-Critic 梯度流说明**：

```
┌──────────────────────────────────────────────────────────────────┐
│                        Critic 训练                                │
│                                                                  │
│   state ───► Critic ───► V(s)                                    │
│                           │                                      │
│                           ▼                                      │
│   next_state ─► Critic ─► V(s') ──✂──► TD_target ◄── reward     │
│               (no_grad)   detach()                               │
│                                                                  │
│   Loss = MSE(V(s), TD_target)                                   │
│   梯度只更新 V(s) 的参数，不流向 V(s')                             │
├──────────────────────────────────────────────────────────────────┤
│                        Actor 训练                                 │
│                                                                  │
│   state ───► Actor ───► π(a|s) ───► log_prob                    │
│                                        │                         │
│         Advantage = TD_target - V(s) ◄─┘                        │
│                      ▲        ▲                                  │
│                      │        └── detach()                       │
│                      │            不更新 Critic                   │
│   Loss = -log_prob * Advantage                                   │
└──────────────────────────────────────────────────────────────────┘
```

##### 4.3.7.1 Target Network（DQN/TD3/SAC 风格）

Target Network 是强化学习中常用的稳定技术。**重要澄清**：Target Network 并非用 `detach()` 实现，而是使用**独立的参数集**。

```rust
/// DQN 风格的 Target Network 实现
struct DQN {
    // 主网络（训练用）
    q_fc1: Linear,
    q_fc2: Linear,
    // Target 网络（独立参数，不训练，定期复制主网络参数）
    target_fc1: Linear,
    target_fc2: Linear,
}

impl DQN {
    fn new(graph: &Graph, state_dim: usize, hidden_dim: usize, action_dim: usize) -> Result<Self, GraphError> {
        Ok(Self {
            // 主 Q 网络
            q_fc1: Linear::new(graph, state_dim, hidden_dim, true, "q_fc1")?,
            q_fc2: Linear::new(graph, hidden_dim, action_dim, true, "q_fc2")?,
            // Target Q 网络（独立参数）
            target_fc1: Linear::new(graph, state_dim, hidden_dim, true, "target_fc1")?,
            target_fc2: Linear::new(graph, hidden_dim, action_dim, true, "target_fc2")?,
        })
    }

    /// 主 Q 网络前向
    fn q_forward(&self, state: Var) -> Result<Var, GraphError> {
        let h = self.q_fc1.forward(state)?.relu();
        self.q_fc2.forward(h)
    }

    /// Target Q 网络前向
    fn target_forward(&self, state: Var) -> Result<Var, GraphError> {
        let h = self.target_fc1.forward(state)?.relu();
        self.target_fc2.forward(h)
    }

    /// 软更新 Target 网络参数
    fn soft_update(&self, graph: &Graph, tau: f32) -> Result<(), GraphError> {
        // target_params = tau * q_params + (1 - tau) * target_params
        for (target, main) in self.target_params().iter().zip(self.q_params().iter()) {
            let main_val = main.value()?;
            let target_val = target.value()?;
            let new_val = &(&main_val * tau) + &(&target_val * (1.0 - tau));
            graph.inner_mut().set_node_value(target.node_id(), Some(&new_val))?;
        }
        Ok(())
    }

    /// 主网络参数（用于训练）
    fn q_params(&self) -> Vec<Var> {
        [self.q_fc1.parameters(), self.q_fc2.parameters()].concat()
    }

    /// Target 网络参数（不训练）
    fn target_params(&self) -> Vec<Var> {
        [self.target_fc1.parameters(), self.target_fc2.parameters()].concat()
    }
}

fn train_dqn(
    graph: &Graph,
    dqn: &DQN,
    optimizer: &mut Adam,  // 只绑定 q_params，不绑定 target_params
    state: &Tensor,
    action: usize,
    reward: f32,
    next_state: &Tensor,
    done: bool,
    gamma: f32,
) -> Result<f32, GraphError> {
    optimizer.zero_grad()?;

    let state_var = graph.input(state)?;
    let next_state_var = graph.input(next_state)?;

    // 当前 Q 值（主网络）
    let q_values = dqn.q_forward(state_var)?;
    let q_value = q_values.select(action)?;

    // 下一状态的最大 Q 值（Target 网络）
    // no_grad 确保不计算 target 网络的梯度
    let next_q_max = graph.no_grad(|| {
        let next_q = dqn.target_forward(next_state_var)?;
        next_q.max_dim(1)
    })?;

    // TD Target
    let td_target = if done {
        graph.constant(&Tensor::scalar(reward))?
    } else {
        let gamma_t = graph.constant(&Tensor::scalar(gamma))?;
        let reward_t = graph.constant(&Tensor::scalar(reward))?;
        &reward_t + &(&gamma_t * &next_q_max)
    };

    // TD Target 不参与梯度传播
    td_target.detach()?;

    // Loss
    let loss = q_value.mse_loss(&td_target)?;
    let loss_val = loss.backward()?;  // ✅ 计算所有梯度
    optimizer.step()?;                 // ✅ 只更新主 Q 网络

    Ok(loss_val)
}

// 训练循环中定期调用
dqn.soft_update(&graph, 0.005)?;  // 软更新 Target 网络
```

**Target Network vs detach 的区别**：

| 机制 | 用途 | 特点 |
|------|------|------|
| `detach()` | 阻止梯度流向某个节点 | 节点仍参与 forward，只是不参与 backward |
| Target Network | 提供稳定的 TD target | 独立参数集，定期从主网络复制/软更新 |

**为什么不能用 detach 实现 Target Network**：Target Network 需要**固定的旧参数**来计算 TD target，而 `detach` 只是阻止梯度流，不能保持参数固定。

#### 4.3.8 NEAT 进化

```rust
use only_torch::neat::{Genome, Population, NeatConfig};

fn main() -> Result<(), GraphError> {
    let config = NeatConfig {
        population_size: 100,
        inputs: 2,
        outputs: 1,
        add_connection_rate: 0.05,
        add_node_rate: 0.03,
        weight_mutation_rate: 0.8,
        compatibility_threshold: 3.0,
        ..Default::default()
    };

    let mut population = Population::new(config);

    // XOR 数据
    let xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    // XOR 适应度函数
    let fitness_fn = |genome: &Genome| -> f32 {
        // 编译基因组为 Graph（新设计：返回 Graph 句柄）
        let graph = genome.compile().unwrap();
        let mut total_error = 0.0;

        for (inputs, expected) in &xor_data {
            // 创建输入
            let x = graph.input(&Tensor::new(inputs, &[1, 2])).unwrap();

            // 前向传播（通过编译后的网络结构）
            // Genome::compile 会设置好网络结构，x 经过网络后得到输出
            let output = genome.forward_compiled(&graph, x).unwrap();

            // 执行计算并获取结果
            output.forward().unwrap();
            let value = output.item().unwrap();

            total_error += (value - expected).powi(2);
        }

        // 适应度 = 4 - 总误差（最大适应度为 4）
        4.0 - total_error
    };

    // 进化循环
    for generation in 0..100 {
        population.evaluate(fitness_fn);

        let best = population.best_genome();
        let fitness = best.fitness.unwrap();
        println!("Gen {}: fitness = {:.4}", generation, fitness);

        if fitness > 3.9 {
            println!("Solution found!");

            // 可视化最佳网络
            let graph = best.compile()?;
            graph.inner().visualize("best_xor_network.dot")?;

            break;
        }

        population.evolve();
    }

    Ok(())
}
```

**NEAT 与新设计的集成**：

```rust
impl Genome {
    /// 编译基因组为可执行 Graph（新设计）
    pub fn compile(&self) -> Result<Graph, GraphError> {
        let graph = Graph::new();

        // 通过 graph.inner_mut() 访问 GraphInner 进行拓扑构建
        {
            let mut g = graph.inner_mut();

            // 1. 创建所有节点
            for (id, node_gene) in &self.nodes {
                match node_gene.node_type {
                    NeatNodeType::Input => {
                        g.add_neat_input_node(*id)?;
                    }
                    NeatNodeType::Hidden => {
                        g.add_neat_hidden_node(*id, node_gene.activation)?;
                    }
                    NeatNodeType::Output => {
                        g.add_neat_output_node(*id, node_gene.activation)?;
                    }
                }
            }

            // 2. 创建所有连接
            for (_, conn) in &self.connections {
                if conn.enabled {
                    g.add_neat_connection(conn.from, conn.to, conn.weight)?;
                }
            }

            // 3. 排序拓扑
            g.on_topology_changed()?;
        }

        Ok(graph)
    }

    /// 在编译后的图上执行前向传播
    pub fn forward_compiled(&self, graph: &Graph, input: Var) -> Result<Var, GraphError> {
        // 获取输入节点和输出节点
        let output_node_id = self.get_output_node_id();

        // 设置输入值
        {
            let mut g = graph.inner_mut();
            // 将 input 的值复制到 NEAT 输入节点
            let input_value = g.get_node_value(input.node_id())?.cloned();
            for input_id in self.get_input_node_ids() {
                g.set_node_value(input_id, input_value.as_ref())?;
            }
        }

        // 返回输出节点的 Var
        Ok(Var::new(output_node_id, graph.inner_rc()))
    }
}
```

---

## 5. 与梯度流控制和高级功能的兼容性

### 5.1 梯度流控制

本设计方案与 `gradient_flow_control_design.md` 中描述的所有高级场景完全兼容：

| 场景 | 新 API 支持方式 | 示例 |
|------|----------------|------|
| `no_grad` / eval 模式 | `graph.no_grad(\|\| { ... })` | 推理时禁用梯度 |
| `detach` 局部截断 | `var.detach()?` | GAN 训练、Actor-Critic |
| `attach` 恢复梯度 | `var.attach()?` | GAN Generator 训练 |
| `retain_graph` 多次 backward | `graph.backward_ex(&loss, &params, true)?` | 多任务学习 |
| 多任务学习 | 多次调用 `backward_ex` | 共享 backbone |
| GAN 训练 | `fake.detach()` + `fake.attach()` | 分离 G/D 训练 |
| Actor-Critic | `value.detach()` + `td_target.detach()` | 策略梯度 |

> **关于参数冻结 / `requires_grad`**
>
> 当前设计中，所有参数节点默认 `requires_grad=true`。如需"冻结"部分参数（如迁移学习），有两种方式：
>
> 1. **Optimizer 选择性绑定**（推荐）：`Adam::new(&graph, &trainable_params, lr)` 只绑定需要训练的参数
> 2. **`requires_grad` 机制**（Optional TODO）：详见 [梯度流控制设计 - 附录 B](gradient_flow_control_design.md#附录-brequires_grad--冻结机制可选功能)
>
> `detach` 与 `requires_grad=false` 的核心区别：`detach` 会**截断**梯度流，而 `requires_grad=false` 允许梯度**穿过**但不累积。99% 场景用 `detach` 即可。

### 5.2 与 LSTM/RNN 记忆机制的兼容性

新设计**完全兼容** LSTM、RNN、GRU 等带记忆机制的模型：

1. **GraphInner 保留所有现有能力**
   - `recurrent_edges` 循环边管理
   - `prev_values` 双缓冲机制
   - `step()` 和 `backward_through_time()` BPTT 支持
   - `step_history` 时间步记录

2. **Var 是 GraphInner 的薄封装**
   - `Var` 只是 `(NodeId, Rc<RefCell<GraphInner>>)` 的组合
   - 不改变任何底层语义
   - LSTM/RNN 的所有操作通过 `GraphInner` 执行

3. **示例：LSTM 前向传播**

```rust
struct LSTMCell {
    // ... 参数
}

impl LSTMCell {
    fn forward(&self, x: Var, h_prev: Var, c_prev: Var) -> Result<(Var, Var), GraphError> {
        // 所有操作都在同一个 GraphInner 上执行
        let gates = x.matmul(&self.w_ih)? + h_prev.matmul(&self.w_hh)? + &self.bias;

        // 分割 gates（通过 GraphInner 底层方法）
        let (i, f, g, o) = gates.split_4()?;

        let i = i.sigmoid();
        let f = f.sigmoid();
        let g = g.tanh();
        let o = o.sigmoid();

        // 新的 cell state 和 hidden state
        let c = &f * &c_prev + &i * &g;  // ✅ 算子重载
        let h = &o * &c.tanh();

        Ok((h, c))
    }
}

// BPTT 训练
fn train_lstm_bptt(
    graph: &Graph,
    lstm: &LSTMCell,
    sequence: &[Tensor],
    targets: &[Tensor],
) -> Result<f32, GraphError> {
    let mut h = graph.zeros(&[batch_size, hidden_size])?;
    let mut c = graph.zeros(&[batch_size, hidden_size])?;

    let mut total_loss = graph.zeros(&[1, 1])?;

    // 前向传播序列
    for (t, (x_t, y_t)) in sequence.iter().zip(targets).enumerate() {
        let x = graph.input(x_t)?;
        let y = graph.input(y_t)?;

        // LSTM 前向
        let (h_new, c_new) = lstm.forward(x, h.clone(), c.clone())?;

        // 计算 loss
        let loss_t = h_new.mse_loss(&y)?;
        total_loss = &total_loss + &loss_t;  // ✅ 算子重载

        // 通过 GraphInner 记录时间步（BPTT 用）
        graph.inner_mut().step()?;

        h = h_new;
        c = c_new;
    }

    // BPTT 反向传播（计算所有参数的梯度）
    let loss_val = total_loss.backward()?;
    // optimizer.step() 只更新 lstm.parameters()

    Ok(loss_val)
}
```

### 5.3 与 NEAT 动态拓扑的兼容性

新设计**100% 兼容** NEAT 的动态拓扑变更：

1. **GraphInner 保留动态拓扑能力**
   - `on_topology_changed()` 自动重排拓扑
   - 支持运行时添加/删除节点和边
   - NEAT 变异操作直接作用于 `GraphInner`

2. **通过 `graph.inner_mut()` 访问底层**

```rust
impl Genome {
    /// 编译基因组为可执行图
    pub fn compile(&self) -> Result<Graph, GraphError> {
        let graph = Graph::new();

        // 直接操作 GraphInner
        {
            let mut g = graph.inner_mut();

            // 创建节点
            for (id, node_gene) in &self.nodes {
                g.add_neat_node(*id, node_gene.node_type, node_gene.activation)?;
            }

            // 创建连接
            for (_, conn) in &self.connections {
                if conn.enabled {
                    g.add_neat_connection(conn.from, conn.to, conn.weight)?;
                }
            }
        }

        Ok(graph)
    }
}

// NEAT 变异直接操作 GraphInner
fn mutate_add_node(genome: &mut Genome, graph: &Graph, tracker: &mut InnovationTracker) {
    // 选择一条连接进行分裂
    let conn = genome.random_enabled_connection();

    // 创建新节点
    let new_node_id = tracker.next_node_id();
    genome.nodes.insert(new_node_id, NodeGene::new_hidden());

    // 操作 GraphInner 添加新节点
    graph.inner_mut().add_neat_node(
        new_node_id,
        NeatNodeType::Hidden,
        Activation::ReLU,
    ).unwrap();

    // 更新连接
    // ...
}
```

### 5.4 设计优势总结

| 优势 | 说明 |
|------|------|
| **PyTorch 级用户体验** | 算子重载 + 链式调用 + 无生命周期 |
| **完全兼容梯度流控制** | `detach`/`attach`/`retain_graph` 等 |
| **完全兼容 LSTM/RNN** | GraphInner 保留所有 BPTT 能力 |
| **完全兼容 NEAT** | 通过 `graph.inner_mut()` 访问底层进行拓扑变异 |
| **概念简单** | Graph 是句柄，Var 是带图引用的节点 ID |
| **运行时开销极小** | Rc clone + RefCell borrow，纳秒级 |

---

## 6. 未来改进方向

本节记录当前设计中已知的可改进之处，留待未来版本实现。

### 6.1 简化张量创建语法

**当前问题**：创建零张量、随机张量等需要通过 `graph` 调用：

```rust
// 当前设计
let fake_labels = graph.zeros(&[batch_size, 1])?;  // 需要 graph
let noise = graph.randn(&[batch_size, latent_dim])?;
```

**原因**：`Var` 必须关联到某个 `Graph`，因此需要知道从哪个图创建。

**未来改进方案**：

**方案 A：`zeros_like` / `randn_like`（推荐）**

```rust
impl Var {
    /// 创建与 self 形状相同的零张量
    pub fn zeros_like(&self) -> Result<Var, GraphError> {
        let shape = self.shape()?;
        let mut g = self.graph.borrow_mut();
        let node_id = g.new_input_node(&shape, None)?;
        g.set_node_value(node_id, Some(&Tensor::zeros(&shape)))?;
        Ok(Var::new(node_id, Rc::clone(&self.graph)))
    }

    /// 创建与 self 形状相同的随机张量
    pub fn randn_like(&self) -> Result<Var, GraphError> {
        let shape = self.shape()?;
        let mut g = self.graph.borrow_mut();
        let node_id = g.new_input_node(&shape, None)?;
        let data = Tensor::normal(0.0, 1.0, &shape, g.rng_mut());
        g.set_node_value(node_id, Some(&data))?;
        Ok(Var::new(node_id, Rc::clone(&self.graph)))
    }
}

// 使用：从已有 Var 推断图
let fake_labels = d_real.zeros_like()?;
let noise = latent.randn_like()?;
```

**方案 B：静态方法 + 显式图**

```rust
impl Var {
    pub fn zeros(graph: &Graph, shape: &[usize]) -> Result<Var, GraphError>;
    pub fn randn(graph: &Graph, shape: &[usize]) -> Result<Var, GraphError>;
}

// 使用
let fake_labels = Var::zeros(&graph, &[batch_size, 1])?;
```

**方案 C：保持现状**

`graph.zeros()` 语义清晰，且调用次数通常不多。当前设计已经足够好用，无需过度优化。

**结论**：在 Phase 2 或 Phase 3 中考虑实现方案 A（`zeros_like` / `randn_like`），作为便捷方法补充现有 API。

### 6.2 更丰富的算子重载

**当前问题**：只支持基本算术运算（`+`、`-`、`*`、`/`）。

**未来改进**：
- 标量运算：`var * 2.0`、`var + 1.0`
- 比较运算：`var > 0.5`（返回 mask）
- 赋值运算：`var += &other`（需要考虑语义）

### 6.3 更简洁的模型定义语法

**当前问题**：定义模型需要手动实现 `Module` trait。

**未来改进**：考虑引入类似 Burn 的 `#[derive(Module)]` 宏，自动生成 `parameters()` 方法。

```rust
// 未来可能的语法
#[derive(Module)]
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

// 自动生成：
// impl Module for MLP {
//     fn parameters(&self) -> Vec<Var> {
//         [self.fc1.parameters(), self.fc2.parameters()].concat()
//     }
// }
```

---

## 7. 实现路线图

> **黄金法则**：每个 Phase 完成后必须通过全面测试验证（包括现有测试无回归 + 新功能测试），确保稳定后才能进入下一阶段。

---

### 7.1 Phase 1a：GraphInner 重构（1 周）

**目标**：将现有 Graph 重命名为 GraphInner，确保无回归

- [x] 将 `Graph` 重命名为 `GraphInner` ✅ 通过 `type Graph = GraphInner` 兼容实现
- [x] 保留所有现有字段和方法 ✅
- [x] 更新所有内部引用 ✅

> **📝 实现策略**：采用类型别名 `pub type Graph = GraphInner` 保持向后兼容，
> 旧代码无需修改即可继续使用 `Graph`。未来命名清理阶段会移除此别名。

**🧪 Phase 1a 验收门禁**（必须全部通过才能进入 Phase 1b）：
- [x] `cargo test` 全部通过（733+ 单元测试）✅ 现已 822+
- [x] 关键集成测试验证：
  - [x] `cargo test test_mnist_batch` → 90%+ 准确率 ✅
  - [x] `cargo test test_california_housing_regression` → 70%+ R² ✅
  - [x] `cargo test test_mnist_gan` → 正常完成 ✅

---

### 7.2 Phase 1b：Graph Handle + Smart Var（2 周）

**目标**：实现新的 Graph 句柄和 Smart Var

- [x] **实现新 Graph（句柄）**
  - [x] `Graph` 包装 `Rc<RefCell<GraphInner>>`
  - [x] 实现 `new()`, `new_with_seed()`
  - [x] 实现 `input()`, `parameter()`, `parameter_seeded()`
  - [x] 实现 `zeros()`, `ones()`, `randn()`
  - [x] 实现 `constant()`
  - [x] 实现 `forward()`, `backward()`
  - [x] 实现 `zero_grad()`
  - [x] 实现 `no_grad_scope()`
  - [x] 实现 `inner()` 底层访问
- [x] **实现 Smart Var**
  - [x] `Var` 包含 `NodeId` + `Rc<RefCell<GraphInner>>`
  - [x] 实现核心方法（详见 §4.2.1.3 层级 1）：
    - [x] `node_id()`, `same_graph()`
    - [x] `get_graph()`
    - [x] `forward()`, `backward()`
    - [x] `value()`, `set_value()`, `grad()`, `item()`
    - [x] `detach()`, `attach()`
  - [x] 实现链式激活函数：`relu()`, `sigmoid()`, `tanh()`, `leaky_relu()`
  - [x] 实现链式运算：`matmul()`, `cross_entropy()`, `mse_loss()`
  - [x] 实现额外激活/损失：`step()`, `perception_loss()`
- [x] **实现算子重载**
  - [x] `Add`, `Sub`, `Mul` for `&Var`
  - [x] `Add`, `Sub`, `Mul` for `Var`
  - [x] `Neg` for `Var` and `&Var`
- [x] 实现 `Init` 枚举及初始化逻辑

> **📝 API 组织策略**：已按 §4.2.1.3 完成 Trait 分层拆分 ✅
> - `VarActivationOps`: `relu()`, `sigmoid()`, `tanh()`, `leaky_relu()`, `step()`
> - `VarLossOps`: `cross_entropy()`, `mse_loss()`, `perception_loss()`
> - `VarMatrixOps`: `matmul()`
> - 核心方法保留在 `impl Var` 中

**🧪 Phase 1b 验收门禁**（必须全部通过才能进入 Phase 2）：
- [x] 新增单元测试：`src/nn/tests/var_ops.rs`（测试算子重载、链式调用、扩展 trait）✅ 16 passed
- [x] 新增单元测试：`src/nn/tests/graph_handle.rs`（测试 Graph 句柄）✅ 21 passed
- [x] 用新 API 重写 XOR 测试：`tests/test_v2_api.rs::test_v2_xor_training` ✅ 通过
- [x] `cargo test` 全部通过 ✅（774+ 单元测试 + 12 集成测试）

---

### 7.3 Phase 2：Module + Optimizer 增强（2-3 周）

**目标**：实现 Module trait 和高层 Layer 封装

- [x] **新增底层节点**（需先实现才能支持 Var API）
  - [x] 实现 `Div` 节点（逐元素除法）✅ `src/nn/nodes/raw_node/ops/divide.rs`
  - [x] 实现独立 `Softmax` 节点 ✅ `src/nn/nodes/raw_node/ops/softmax.rs`
- [x] **完善 Var 算子重载**
  - [x] `Div` for `&Var` and `Var`（依赖底层 Div 节点）✅ `src/nn/var.rs`
  - [x] `Var::softmax()`（依赖底层 Softmax 节点）✅ `src/nn/var_ops/activation.rs`
- [x] 定义 `Module` trait（返回 `Vec<Var>`）✅ `src/nn/module.rs`
- [x] **重构 Optimizer**
  - [x] Optimizer 持有 `Rc<RefCell<GraphInner>>` 引用 ✅ `OptimizerV2`
  - [x] `zero_grad()` 不再需要 `&mut Graph` 参数 ✅
  - [x] `step()` 不再需要 `&mut Graph` 参数 ✅
  - [x] 实现 `minimize(&self, loss: &Var)` ✅
- [x] **实现高层 Layer**
  - [x] `Linear::new(graph, in, out, bias, name)` → 返回持有 Var 的 Linear ✅ `src/nn/layer/linear_v2.rs`
  - [x] `Linear::forward(x: Var)` → 不需要 graph 参数 ✅
  - [ ] 类似实现 `Conv2d`, `RNN`, `LSTM`, `GRU`（延后到 Phase 2.5）

**🧪 Phase 2 验收门禁**（必须全部通过才能进入 Phase 3）：
- [x] 新增单元测试：`src/nn/tests/module_trait.rs` ✅ 6 tests
- [x] 新增单元测试：`src/nn/tests/optimizer_v2.rs` ✅
- [x] 用新 API 重写 `test_mnist_linear.rs` 并通过（90%+ 准确率）✅ `tests/test_mnist_linear_v2.rs`
- [x] 用新 API 重写 `test_mnist_batch.rs` 并通过 ✅ `tests/test_mnist_batch_v2.rs`
- [x] `cargo test` 全部通过 ✅ 822 unit tests + V2 integration tests

---

### 7.4 Phase 3：NEAT MVP（4-6 周）

**目标**：实现最小可用的 NEAT 进化

- [ ] 实现 `NodeGene`, `ConnectionGene`, `Genome`
- [ ] 实现 `InnovationTracker`
- [ ] 实现 `Genome::compile() -> Graph`
- [ ] 实现基础变异（add_node, add_connection, mutate_weights）
- [ ] 实现 `Genome::crossover()` 和 `distance()`

**🧪 Phase 3 验收门禁**（必须全部通过才能进入 Phase 4）：
- [ ] 新增单元测试：`src/neat/tests/genome.rs`
- [ ] 新增单元测试：`src/neat/tests/mutation.rs`
- [ ] 新增集成测试：`tests/test_neat_xor.rs` → XOR 任务进化成功
- [ ] `cargo test` 全部通过

---

### 7.5 Phase 4：NEAT 完整（6-8 周）

**目标**：实现完整的 NEAT 进化系统

- [ ] 实现 `Species` 和 `Population`
- [ ] 实现物种划分算法
- [ ] 支持循环连接
- [ ] 实现进化可视化

**🧪 Phase 4 验收门禁**（必须全部通过才能进入 Phase 5）：
- [ ] 新增单元测试：`src/neat/tests/species.rs`
- [ ] 新增单元测试：`src/neat/tests/population.rs`
- [ ] 新增集成测试：`tests/test_neat_parity.rs` → Parity 任务进化成功
- [ ] `cargo test` 全部通过

---

### 7.6 Phase 5：Layer-Level NEAT（未来，8-12 周）

**目标**：实现 Layer 级别的网络架构演化

- [ ] 定义 `LayerGene` 枚举
- [ ] 实现 `Blueprint`
- [ ] 实现层级变异和交叉

**🧪 Phase 5 验收门禁**：
- [ ] 新增集成测试：`tests/test_neat_mnist_nas.rs` → MNIST 架构搜索
- [ ] `cargo test` 全部通过

---

### 7.7 测试迁移策略

**目标**：所有现有测试统一使用新的 Graph Handle + Var API。

#### 迁移前（现有测试风格）

```rust
#[test]
fn test_conv2d_forward() {
    let mut graph = Graph::new();
    let input_id = graph.new_input_node(&[1, 1, 5, 5], Some("input")).unwrap();
    let kernel_id = graph.new_parameter_node(&[1, 1, 3, 3], Some("kernel")).unwrap();
    let output_id = graph.new_conv2d_node(input_id, kernel_id, None).unwrap();

    graph.set_node_value(input_id, Some(&input_data)).unwrap();
    graph.set_node_value(kernel_id, Some(&kernel_data)).unwrap();
    graph.forward_node(output_id).unwrap();

    let result = graph.get_node_value(output_id).unwrap().unwrap();
    assert_tensor_eq!(&result, &expected, 1e-5);
}
```

#### 迁移后（新 API 风格）

```rust
#[test]
fn test_conv2d_forward() {
    let graph = Graph::new();
    let input = graph.input(&input_data).unwrap();
    let kernel = graph.parameter(&[1, 1, 3, 3], Init::Constant(1.0), "kernel").unwrap();
    let output = input.conv2d(&kernel).unwrap();

    output.forward().unwrap();
    let result = output.value().unwrap();
    assert_tensor_eq!(&result, &expected, 1e-5);
}
```

#### 迁移规则

| 旧 API | 新 API |
|--------|--------|
| `graph.new_input_node() + set_value()` | `graph.input(&data)` |
| `graph.new_parameter_node() + set_value()` | `graph.parameter(shape, init, name)` |
| `graph.new_xxx_node(parent_id, ...)` | `parent.xxx(...)` |
| `graph.forward_node(node_id)` | `var.forward()` 或 `loss.backward()` |
| `graph.get_node_value(node_id)` | `var.value()` |
| `graph.get_node_jacobi(node_id)` | `var.grad()` |

#### 特殊情况：需要底层操作

少数测试可能需要直接操作 `GraphInner`（如测试内部拓扑）：

```rust
#[test]
fn test_internal_topology() {
    let graph = Graph::new();
    let x = graph.input(&data).unwrap();
    let y = x.relu();

    // 需要访问底层时，使用 inner()（不可变）或 inner_mut()（可变）
    let inner = graph.inner();
    assert_eq!(inner.node_count(), 2);
    assert!(inner.get_edge(x.node_id(), y.node_id()).is_some());
}
```

---

## 8. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| RefCell 运行时 panic（重复借用） | 中 | 1. 单线程使用模式下风险低<br>2. 文档明确说明不可嵌套借用<br>3. 考虑 debug 模式下增加检测 |
| Var Clone 开销积累 | 低 | Rc::clone 只需几纳秒，实际可忽略 |
| Graph → GraphInner 重构工作量 | 中 | 1. 主要是重命名<br>2. 保持 GraphInner API 不变<br>3. 分阶段迁移 |
| 算子重载方法增多导致 Var impl 膨胀 | 中 | 1. 考虑 extension trait<br>2. 宏生成重复代码 |
| Module 封装与现有 layer 函数冲突 | 低 | 保留 layer 函数，Module 封装调用它们 |
| NEAT 与 BPTT 集成复杂 | 高 | 先做 DAG 网络进化，循环后加 |
| 用户误用 `graph.inner()` / `inner_mut()` 导致状态不一致 | 低 | 1. 文档明确这是高级 API<br>2. 正常使用无需访问 inner |

### 8.1 RefCell 安全使用指南

为避免 RefCell 运行时 panic，需遵守以下规则：

```rust
// ✅ 正确：短暂借用，立即释放
let x = graph.input(&data)?;
let y = x.relu();  // borrow_mut 在内部完成，立即释放

// ✅ 正确：顺序操作
let a = graph.input(&data1)?;
let b = graph.input(&data2)?;
let c = &a + &b;  // 每次操作都是独立的 borrow_mut

// ❌ 错误：同时持有多个可变引用（编译时无法检测，运行时 panic）
let inner1 = graph.inner_mut();
let inner2 = graph.inner_mut();  // panic!

// ✅ 正确：使用作用域限制借用
{
    let mut g = graph.inner_mut();
    g.some_operation();
}  // 借用在此释放
{
    let mut g = graph.inner_mut();  // 安全
    g.another_operation();
}
```

**设计保证**：正常使用 `Graph` 和 `Var` 的公开 API 时，不会触发 RefCell panic。只有直接操作 `graph.inner()` / `graph.inner_mut()` 时需要注意。

---

## 9. 决策记录

| 日期 | 决策 | 理由 |
|------|------|------|
| 2025-12-30 | 放弃五层架构，采用 3+1 层 | 与现实对齐，避免过度设计 |
| 2025-12-30 | forward() 不是 trait 方法 | Burn 实践验证，灵活性更重要 |
| 2025-12-30 | NEAT 独立于 Module | 结构可变 vs 结构固定的本质差异 |
| 2025-12-30 | 不引入 Backend 泛型 | CPU-only 定位，避免复杂度 |
| 2025-12-30 | 放弃 Hybrid 双模式 | CPU 场景收益有限 |
| 2025-12-31 | 放弃 VarMap/VarBuilder | 用户仍需操作 NodeId，不够简洁 |
| 2025-12-31 | 放弃 ForwardContext | "单次 forward-backward" 与复杂梯度流不兼容 |
| 2025-12-31 | 放弃简单 Var (NodeId only) | C 风格 API 用户体验差 |
| 2025-12-31 | 放弃 EVar + 显式生命周期 | 用户需要处理 `'g` 生命周期，模型定义复杂 |
| **2025-12-31** | **采用 Graph Handle + Smart Var** | **PyTorch 级用户体验，支持算子重载和链式调用** |
| **2025-12-31** | **Graph 变为 Rc<RefCell<GraphInner>>** | **允许 Var 持有图引用，实现算子重载** |
| **2025-12-31** | **Var 携带图引用** | **支持 `x.relu()`、`loss.backward()` 等链式调用** |
| **2025-12-31** | **内部使用 RefCell，对用户隐藏** | **用户无需知道 RefCell，无需写生命周期** |
| 2025-12-31 | Optimizer 增加 zero_grad/minimize | PyTorch 风格，用户体验提升 |
| 2025-12-31 | backward() 采用 ensure-forward 并返回 loss 值 | 若当前 pass 未计算则先 forward；若已计算（缓存命中）则不重复 forward；简化用户代码 |
| 2025-12-31 | forward() 和 new() 不是 Module trait 方法 | 不同层签名各异，无法统一（参考 Burn） |
| 2026-01-01 | randn() 使用正态分布 N(0,1) | 与 PyTorch `torch.randn()` 语义一致 |
| 2026-01-06 | `requires_grad` / 冻结机制列为 Optional TODO | `detach` + optimizer 选择性绑定已覆盖 99% 场景；详见 [梯度流控制设计 - 附录 B](gradient_flow_control_design.md#附录-brequires_grad--冻结机制可选功能) |
| 2026-01-08 | Phase 1 拆分为 1a（重构）和 1b（新增） | 降低风险，确保每步都有明确验收门禁 |
| 2026-01-08 | 每个 Phase 必须有验收门禁 | **黄金法则**：测试全过才能进入下一阶段 |
| 2026-01-09 | Var API 采用 Trait 分层设计 | 核心能力放 `impl Var`，长尾算子按领域分组到扩展 trait（如 `VarActivationOps`），复杂组件封装为 Layer；详见 §4.2.1.3 |
| 2026-01-09 | `set_value()` 列为 Var 核心 API | 训练循环中复用输入节点喂数据的关键 API，避免图膨胀 |
| 2026-01-09 | 图膨胀问题明确警告 | 在 §4.3.1 添加显式警告，强调不要在训练循环里反复 `graph.input()` |
| 2026-01-09 | 放弃 `placeholder()` 方法 | 与 TensorFlow 1.x 语义雷同易混淆，`zeros()` + `set_value()` 已满足需求 |
| 2026-01-09 | 完成 Var Trait 分层拆分 | 按 §4.2.1.3 策略拆分为 `VarActivationOps`、`VarLossOps`、`VarMatrixOps`，核心方法保留在 `impl Var` |
| 2026-01-09 | 在 GraphHandle 暴露 `no_grad_scope()` | 提供简洁的无梯度上下文 API，用于验证集评估等场景 |

### 9.1 关键设计决策详解

#### 为什么选择 `Rc<RefCell<GraphInner>>` 而非其他方案？

| 方案 | 问题 |
|------|------|
| `&'g RefCell<Graph>` + 生命周期 | 用户需要在模型定义中写 `'g`，复杂 |
| `Arc<RwLock<Graph>>` | 多线程场景可用，但单线程下 RwLock 开销大于 RefCell |
| `Var` 不持有 Graph 引用 | 无法实现算子重载和链式调用，C 风格 API |
| **`Rc<RefCell<GraphInner>>`** | ✅ 用户无需知道内部实现，API 最简洁 |

**为什么不用 `Arc<RwLock>`（像 Candle 那样）？**

| 对比 | `Rc<RefCell>` | `Arc<RwLock>` |
|------|---------------|---------------|
| 线程安全 | ❌ 单线程 | ✅ 多线程 |
| Clone 开销 | ~2 ns（非原子） | ~5-10 ns（原子操作） |
| 借用开销 | ~5 ns | ~10-20 ns |
| 复杂度 | 简单 | 需处理 poisoning |

**结论**：only_torch 定位是 CPU-only 学习框架，不追求多线程并行。单线程场景下 `Rc<RefCell>` 更简单、开销更低。未来如需多线程支持，可考虑迁移到 `Arc<RwLock>`。

#### RefCell 运行时开销分析

```
操作            | 开销      | 说明
----------------|----------|------------------
Rc::clone()     | ~2 ns    | 原子引用计数 +1
RefCell::borrow | ~5 ns    | 检查借用状态
RefCell::borrow_mut | ~5 ns | 检查借用状态

对比 PyTorch (Python):
- Python 函数调用 | ~100 ns
- Python 对象创建 | ~200 ns

结论：RefCell 开销相比 Python 可完全忽略
```

#### 为什么 Var 是 Clone 而非 Copy？

```rust
// Copy 版本（无法实现）
#[derive(Copy, Clone)]
struct Var {
    id: NodeId,
    graph: Rc<RefCell<GraphInner>>,  // ❌ Rc 不是 Copy
}

// Clone 版本（当前方案）
#[derive(Clone)]
struct Var {
    id: NodeId,
    graph: Rc<RefCell<GraphInner>>,  // ✅ Rc::clone() 很廉价
}
```

虽然 `Var` 不是 `Copy`，但 `Rc::clone()` 只需几纳秒，实际使用中无感知。

---

## 附录 A：API 对比

### PyTorch vs only_torch（新设计）

| PyTorch | only_torch（新设计） |
|---------|----------------------|
| `optimizer.zero_grad()` | `optimizer.zero_grad()?` |
| `output = model(x)` | `let output = model.forward(x)?` |
| `loss = criterion(output, y)` | `let loss = output.cross_entropy(&y)?` |
| `loss.backward()` | `loss.backward()?` |
| `optimizer.step()` | `optimizer.step()?` |
| `print(loss.item())` | `println!("{}", loss.item()?)` |
| `y = x.relu()` | `let y = x.relu()` |
| `z = a + b` | `let z = &a + &b` 或 `a + b` |
| `fake = fake.detach()` | `fake.detach()?` |
| `with torch.no_grad():` | `graph.no_grad(\|\| { ... })` |

### 与旧设计的对比

| 操作 | 旧设计（C 风格） | 新设计（PyTorch 风格） |
|------|-----------------|----------------------|
| 激活函数 | `graph.relu(x)` | `x.relu()` |
| 算术运算 | `graph.add(a, b)` | `&a + &b` |
| 矩阵乘法 | `graph.matmul(a, b)` | `a.matmul(&b)?` |
| 梯度截断 | `graph.detach(fake)?` | `fake.detach()?` |
| 反向传播 | `graph.backward(loss, &params)?` | `loss.backward()?` |
| 模型前向 | `model.forward(&mut graph, x)` | `model.forward(x)?` |

### 主要改进

1. **✅ 无需显式传 `&mut graph`**：Var 已携带图引用
2. **✅ 支持算子重载**：`&a + &b`、`&a * &b` 等
3. **✅ 支持链式调用**：`x.relu().sigmoid().matmul(&w)?`
4. **✅ 方法式梯度控制**：`fake.detach()`、`loss.backward()`
5. **✅ 用户无需了解 RefCell**：内部实现对用户完全透明

### 完整示例对比

**旧设计（C 风格）**：
```rust
fn train_step(graph: &mut Graph, model: &Model, x: &Tensor, y: &Tensor) -> Result<f32, GraphError> {
    let input = graph.input(x);
    let target = graph.input(y);

    let h = graph.relu(model.fc1.forward(graph, input));
    let output = model.fc2.forward(graph, h);
    let loss = graph.cross_entropy(output, target);

    let loss_val = graph.backward(loss, &model.parameters())?;
    Ok(loss_val)
}
```

**新设计（PyTorch 风格）**：
```rust
fn train_step(graph: &Graph, model: &Model, x: &Tensor, y: &Tensor) -> Result<f32, GraphError> {
    let input = graph.input(x)?;
    let target = graph.input(y)?;

    let output = model.forward(input)?;  // forward 不需要 graph 参数
    let loss = output.cross_entropy(&target)?;

    loss.backward()  // 直接在 Var 上调用
}
```

---

## 附录 B：与原架构文档的对照

| 原文档概念 | 新架构对应 | 状态 |
|-----------|-----------|------|
| Module trait | Module trait（返回 `Vec<Var>`，forward 无需 graph 参数） | ✅ 增强 |
| Parameter | `Var`（NodeId + 图引用，支持算子重载） | ✅ 增强 |
| Graph | `GraphInner`（核心实现）+ `Graph`（用户友好句柄） | ✅ 拆分 |
| VarMap / VarBuilder | **删除** | ❌ 放弃 |
| ForwardContext | **删除** | ❌ 放弃 |
| ExecutionEngine | 不需要（GraphInner 直接执行） | ❌ 删除 |
| ComputationGraph | 不需要（GraphInner 承担） | ❌ 删除 |
| Hybrid 模式 | 不需要 | ❌ 删除 |
| OTMF 格式 | 保留（JSON + bin） | ✅ 保留 |
| EvolvableModel | Genome（通过 `graph.inner_mut()` 操作底层） | ✅ 重设计 |

### 新旧架构结构对比

```
旧架构                              新架构
─────────────────────────────────────────────────────────────────

Graph                              Graph (handle)
├── nodes                          └── Rc<RefCell<GraphInner>>
├── edges                                      │
├── rng                            GraphInner ◄┘
├── ...                            ├── nodes
└── methods()                      ├── edges
                                   ├── rng
Var (简单)                         ├── ...
└── NodeId                         └── methods()

                                   Var (smart)
                                   ├── NodeId
                                   └── Rc<RefCell<GraphInner>>
                                       ├── relu(), sigmoid()...
                                       ├── detach(), attach()
                                       ├── backward()
                                       └── Add, Sub, Mul, Div
```

---

## 附录 C：参考资料

- **Burn**：https://github.com/tracel-ai/burn
- **Candle**：https://github.com/huggingface/candle
- **Neuronika**：https://github.com/neuronika/neuronika
- **neat-python**：https://github.com/CodeReclaimers/neat-python
- **neat-rs**：https://github.com/TLmaK0/neat-rs

---

## 附录 D：实现检查清单

### Phase 1 完成标准

- [ ] `GraphInner` 重命名完成，所有现有测试通过
- [ ] `Graph` 句柄实现，`new()`、`new_with_seed()` 可用
- [ ] `Var` 实现，携带图引用，包含 `same_graph()`、`get_graph()`
- [ ] 算子重载实现：`Add`、`Sub`、`Mul`、`Div`、`Neg`（含跨图检查）
- [ ] try_* 方法：`try_add()`、`try_sub()`、`try_mul()`、`try_div()`
- [ ] 链式激活函数：`relu()`、`sigmoid()`、`tanh()`、`softmax()`
- [ ] 梯度控制：`detach()`、`attach()`
- [ ] 执行方法：`forward()`、`backward()`
- [ ] 新错误类型：`ValueNotComputed`、`GradientNotComputed`、`GraphMismatch`、`ShapeMismatch`
- [ ] `test_xor.rs` 使用新 API 重写并通过

### Phase 2 完成标准

- [ ] `Module` trait 定义
- [ ] `Linear` 实现（forward 无需 graph 参数）
- [ ] `Optimizer` 重构（持有图引用，PyTorch 风格）
- [ ] `test_mnist_linear.rs` 使用新 API 重写并通过

---

## 附录 E：错误类型完整定义

```rust
/// only_torch 计算图错误类型
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    // ==================== 现有错误类型 ====================

    /// 节点不存在
    NodeNotFound(NodeId),

    /// 无效的形状（通用）
    InvalidShape { expected: Vec<usize>, got: Vec<usize> },

    /// 节点名称重复
    DuplicateName(String),

    /// 检测到循环依赖（非 BPTT 场景）
    CycleDetected,

    // ==================== 新增错误类型 ====================

    /// 节点值尚未计算（需要先调用 forward）
    ///
    /// 常见原因：在 define-and-run 模式下，创建 Var 后直接调用 value()
    /// 解决方法：先调用 `var.forward()` 或 `loss.backward()`
    ValueNotComputed(NodeId),

    /// 节点梯度尚未计算（需要先调用 backward）
    ///
    /// 常见原因：在 forward 后直接调用 grad()，没有先 backward
    /// 解决方法：先调用 `loss.backward()`
    GradientNotComputed(NodeId),

    /// 两个 Var 来自不同的 Graph
    ///
    /// 仅 try_* 方法返回此错误。算子重载（+、-、*、/）会直接 panic。
    /// 常见原因：意外使用了两个独立 Graph 的 Var 进行运算
    GraphMismatch {
        /// 左操作数所属 Graph 的标识（用于调试）
        left_graph_id: usize,
        /// 右操作数所属 Graph 的标识
        right_graph_id: usize,
    },

    /// 形状不匹配（详细版）
    ///
    /// 提供了具体的操作名和两边的形状，便于调试
    ShapeMismatch {
        /// 操作名称（如 "matmul"、"add"）
        op: &'static str,
        /// 左操作数形状
        left: Vec<usize>,
        /// 右操作数形状
        right: Vec<usize>,
    },

    /// 节点已被 detach，不能参与梯度计算
    ///
    /// 常见原因：在 detach 后的节点上调用了需要梯度的操作
    /// 解决方法：使用 `var.attach()` 恢复梯度流
    NodeDetached(NodeId),

    /// NEAT 相关：无效的创新号
    InvalidInnovation(u32),

    /// NEAT 相关：连接已存在
    ConnectionExists { from: u32, to: u32 },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::NodeNotFound(id) => write!(f, "Node {:?} not found", id),
            GraphError::ValueNotComputed(id) => {
                write!(f, "Node {:?} value not computed. Call forward() first.", id)
            }
            GraphError::GradientNotComputed(id) => {
                write!(f, "Node {:?} gradient not computed. Call backward() first.", id)
            }
            GraphError::GraphMismatch { left_graph_id, right_graph_id } => {
                write!(f, "Cannot operate on Vars from different Graphs (left: {}, right: {})",
                       left_graph_id, right_graph_id)
            }
            GraphError::ShapeMismatch { op, left, right } => {
                write!(f, "Shape mismatch in {}: left {:?}, right {:?}", op, left, right)
            }
            GraphError::NodeDetached(id) => {
                write!(f, "Node {:?} is detached and cannot participate in gradient computation", id)
            }
            // ... 其他错误类型
            _ => write!(f, "{:?}", self),
        }
    }
}

impl std::error::Error for GraphError {}
```

---

*本文档是 only_torch 架构 V2.3 (PyTorch 风格增强版) 的完整设计参考，经过多轮讨论和修订。核心创新是通过 `Rc<RefCell<GraphInner>>` 实现 PyTorch 级用户体验，同时完全兼容 NEAT 动态拓扑、LSTM/RNN 记忆机制、以及复杂梯度流控制（detach/attach/retain_graph）。V2.3 进一步移除了 backward_for，采用 PyTorch 风格的 Optimizer 设计。*


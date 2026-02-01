# 动态图生命周期设计（方案 C）

> **状态**：设计中  
> **作者**：架构评审  
> **创建日期**：2026-02-01  
> **目标**：实现 PyTorch 风格的节点自动生命周期管理，从根本上解决节点累积问题

---

## 目录

1. [问题背景](#1-问题背景)
2. [设计目标](#2-设计目标)
3. [核心设计](#3-核心设计)
4. [对现有组件的影响](#4-对现有组件的影响)
5. [实施路径](#5-实施路径)
6. [测试策略](#6-测试策略)
7. [与 NAS/NEAT 的兼容性](#7-与-nasneat-的兼容性)
8. [进阶选项：完全隐藏 Graph](#8-进阶选项完全隐藏-graph)

---

## 1. 问题背景

### 1.1 当前架构的节点累积问题

在强化学习（SAC）示例中发现，训练过程随时间显著变慢。根本原因是**节点只增不减**：

```
当前架构：
  Graph.nodes: HashMap<NodeId, NodeHandle>  ← 集中存储
  
  每次前向传播：
    - 闭包内节点：被 ModelState 缓存，复用 ✅
    - 闭包外节点：每次新建，永不删除 ❌
    
  训练 100 epochs × 1000 batches：
    - 闭包外产生 100,000+ 个冗余节点
    - HashMap 越来越大，查找/遍历开销上升
    - 训练速度指数级下降
```

### 1.2 问题的本质

| 当前设计 | 问题 |
|---------|------|
| `Var` 持有 `NodeId`（u64） | 只是 ID，无法追踪节点引用状态 |
| `release_intermediate_results()` | 只清值，不删节点 |
| `ModelState` 缓存机制 | 只覆盖闭包内，闭包外不覆盖 |

### 1.3 为什么 GAN/MNIST 正常，SAC 不行？

**关键差异在于损失函数的形式**：

| 场景 | 损失函数形式 | 被 ModelState 缓存？ |
|------|-------------|---------------------|
| GAN | `MseLoss.forward()` | ✅ 是（MseLoss 内部有缓存） |
| MNIST | `CrossEntropyLoss.forward()` | ✅ 是 |
| SAC | `log_probs * α - Q` 手工组合 | ❌ 否（闭包外运算） |

强化学习的损失函数涉及多个模型输出的组合，这些组合运算**必然发生在闭包外**。

### 1.4 临时解决方案的局限

当前已实现的 `checkpoint()` + `prune_nodes_after()` API：

```rust
let checkpoint = graph.checkpoint();
for epoch in 0..epochs {
    // ... 训练 ...
    graph.prune_nodes_after(checkpoint)?;  // 手动清理
}
```

**问题**：
- 需要用户手动调用，容易遗漏
- 是"治标"方案，根本问题未解决
- 不符合 PyTorch 的用户体验

---

## 2. 设计目标

### 2.1 核心目标

| 目标 | 说明 |
|------|------|
| **自动生命周期管理** | 节点用完自动释放，无需手动清理 |
| **参数持久化** | weight/bias 跨 batch 保留，累积学习 |
| **运算节点临时化** | 每次前向传播重建，用完即弃 |
| **API 向后兼容** | 用户代码最小改动或零改动 |

### 2.2 设计哲学

采用 **PyTorch 的成功范式**：

```rust
// only_torch 的工作方式（方案 C）
let model = Linear::new(10, 5)?;  // 参数持久

for batch in dataloader {
    let output = model.forward(&batch)?;   // 运算节点每次重建
    let loss = output.cross_entropy(&y)?;  // Var 方法，无 Criterion 类
    loss.backward()?;                       // backward 后运算节点可释放
    optimizer.step()?;                      // 参数更新
}
// ← output, loss 离开作用域，运算节点自动释放（Rc 引用计数归零）
```

### 2.3 为什么参数保留、运算重建？

| 节点类型 | 代表什么 | 生命周期需求 |
|---------|---------|-------------|
| **参数节点** | 模型的可学习知识 | 必须持久，否则学到的东西丢失 |
| **运算节点** | 当前 batch 的中间计算 | 只对当前 batch 有意义，用完即弃 |

**重建开销可忽略**：PyTorch 已用大规模生产验证，创建节点的开销（内存分配+初始化）远小于实际矩阵运算。

---

## 3. 核心设计

### 3.1 当前架构 vs 新架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        当前架构                                  │
├─────────────────────────────────────────────────────────────────┤
│  Graph: HashMap<NodeId, Node>    ← 集中存储，只增不减            │
│  Var: NodeId + Rc<GraphInner>    ← 只是 ID，不控制生命周期       │
│  ModelState: 缓存闭包内的子图    ← 闭包外不覆盖                  │
│  Criterion: 按 NodeId 缓存       ← 依赖 ModelState 的稳定 ID     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        新架构（方案 C）                          │
├─────────────────────────────────────────────────────────────────┤
│  Graph: 参数节点注册表           ← 只存持久节点（可选隐藏）      │
│  Var: Rc<NodeInner>              ← 直接持有，引用计数控制生命周期│
│  ModelState: ❌ 删除             ← 可视化改用 Var.visualize()    │
│  Criterion: ❌ 删除              ← 统一用 Var 方法如 mse_loss()  │
│                                                                  │
│  运算节点：创建 → 计算 → Var 离开作用域自动释放                  │
│  参数节点：持久存在，直到模型销毁                                │
│  可视化：从任意 Var 沿 parents 链遍历整个上游图                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Var 结构变更

### 当前设计（NodeId 是什么）

```rust
// 当前的 NodeId：只是一个 u64 数字
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

// 当前的 Var：持有 ID，通过 ID 在 HashMap 中查找节点
pub struct Var {
    id: NodeId,                        // 只是一个数字 ID
    graph: Rc<RefCell<GraphInner>>,    // 图的引用（用于查表）
}

// 当前的 GraphInner：集中存储所有节点
pub struct GraphInner {
    nodes: HashMap<NodeId, NodeHandle>,  // 节点集中存储
    // ...
}

// 当前的 NodeHandle：节点的完整数据
pub struct NodeHandle {
    raw_node: NodeType,              // 节点类型（Add/MatMul/ReLU 等）
    last_forward_pass_id: u64,
    last_backward_pass_id: u64,
    is_detached: bool,
}
```

**问题**：`Var` 只是一个 ID，离开作用域时节点仍在 HashMap 中，不会释放。

### 新设计（方案 C）

```rust
// NodeId 保留，但主要用于可视化和调试
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u64);

// 新的 Var：直接持有节点的引用计数指针
pub struct Var {
    node: Rc<NodeInner>,                    // 直接持有节点（引用计数）
    graph: Weak<RefCell<GraphInner>>,       // 弱引用，用于：
                                            //   - 获取新 NodeId
                                            //   - 访问 pass_id 计数器
                                            //   - 检查 is_eval_mode
}

// 新的 NodeInner：原 NodeHandle 的内容 + 父节点引用
pub struct NodeInner {
    // === 节点标识（用于可视化/调试）===
    id: NodeId,
    name: Option<String>,
    
    // === 原 NodeHandle 的核心内容（使用 RefCell，因为 value/grad 是非 Copy 类型）===
    raw_node: RefCell<NodeType>,            // 包含 value: Option<Tensor> 和 grad: Option<Tensor>
    
    // === 内部可变字段（使用 Cell，适用于 Copy 类型）===
    last_forward_pass_id: Cell<u64>,
    last_backward_pass_id: Cell<u64>,
    is_detached: Cell<bool>,
    
    // === 父节点引用（强引用，保证反向传播时存活）===
    parents: Vec<Rc<NodeInner>>,
}

// Var 离开作用域时，Rc 引用计数自动管理
// 无需显式 Drop 实现，Rust 默认行为就是引用计数 -1
```

**设计说明**：
- **单线程设计**：使用 `Rc`/`Weak` 而非 `Arc`（性能更好，足够当前需求）
- **内部可变性**：
  - `RefCell<T>` 用于 `raw_node`（`value`/`grad` 是非 Copy 类型，需要运行时借用检查）
  - `Cell<T>` 用于 `last_forward_pass_id` 等（Copy 类型，无需借用检查）
- **value/grad 位置**：存储在 `raw_node: NodeType` 内部，每个节点类型自己管理
- **延迟计算**：`raw_node.borrow().value` 为 `Option<Tensor>`，`None` 表示尚未计算

### 3.3 节点分类与生命周期

| 节点类型 | 存储位置 | 生命周期 | 说明 |
|---------|---------|---------|------|
| **Parameter** | Graph 参数注册表 + Var | 持久（直到模型销毁） | weight, bias |
| **Input** | 仅 Var | 随 Var 作用域 | 每次 forward 的输入 |
| **运算节点** | 仅 Var | 随 Var 作用域 | matmul, relu, add 等结果 |

### 3.4 边管理策略（为什么用 Rc 强引用而非 Weak）

#### 为什么 parents 必须用 Rc 强引用

考虑这个场景（如果错误地使用 Weak）：

```rust
let x = graph.input(&data)?;  // var_x 持有 Rc<x_inner>
let y = x.relu()?;            // var_y 持有 Rc<y_inner>
let z = y.softmax()?;         // var_z 持有 Rc<z_inner>

// ❌ 错误设计：parents 用 Weak
drop(var_x);  // x_inner 引用计数 → 0 → 释放！
drop(var_y);  // y_inner 引用计数 → 0 → 释放！

z.backward()?;  // 💥 崩溃！z.parents 中的 Weak 已失效！
```

正确设计：

```rust
// ✅ 正确设计：parents 用 Rc
struct NodeInner {
    parents: Vec<Rc<NodeInner>>,   // 强引用
}

drop(var_x);  // x_inner 引用计数 -1，但 y.parents 还持有 Rc<x> → 不释放
drop(var_y);  // y_inner 引用计数 -1，但 z.parents 还持有 Rc<y> → 不释放

z.backward()?;  // ✅ 所有上游节点都活着

drop(var_z);   // z 释放 → y 释放 → x 释放（级联释放）
```

#### 为什么不会造成循环引用？

**核心论点：计算图的构建本质上是单向的（DAG）**

```
节点创建顺序:  x  →  y  →  z  →  loss
                    ↓      ↓      ↓
parents 引用:      [x]   [y]    [z]

引用方向:  loss.parents → z.parents → y.parents → x
           （单向链，没有回路）
```

**不可能出现循环引用**，因为：
1. 创建节点 B 时，B 可以引用已存在的 A（B.parents 包含 Rc\<A\>）
2. 但 A **不可能**引用 B（B 在创建 A 时还不存在！）
3. 这是由计算图的**构建时序**决定的，不是软件设计的约束

#### RNN 的"循环"不是图循环

RNN 在每个时间步展开后：

```
h_0 ← f(x_0, h_init)
  ↓
h_1 ← f(x_1, h_0)      h_1.parents = [Rc<h_0>, Rc<x_1>]
  ↓
h_2 ← f(x_2, h_1)      h_2.parents = [Rc<h_1>, Rc<x_2>]
  ↓
output

引用方向: output → h_2 → h_1 → h_0 （单向链，DAG）
```

h_2 依赖 h_1，h_1 依赖 h_0，但 **h_0 不依赖 h_2**。这是单向依赖链，不是循环。

**设计决策**：
- `parents: Vec<Rc<NodeInner>>` 使用**强引用**
- 只要 loss 节点还活着，整个上游计算图都不会被释放
- 当 loss 离开作用域时，整个图级联释放

### 3.5 循环引用的防御机制

#### 为什么用户无法构造循环引用？

1. **API 设计保证**：所有操作（add, matmul, relu 等）只接收已存在的 Var，创建新的 Var
2. **parents 不可变**：节点创建后，`parents` 字段不暴露修改接口
3. **Rust 所有权保证**：不能让已存在的对象引用尚未创建的对象

```rust
// 不可能的情况（编译器/API 会阻止）：
let a = some_op()?;
let b = other_op(&a)?;
a.add_parent(&b)?;  // ❌ 这个 API 不存在，且 a 已经是不可变的
```

#### 防御性运行时检测（可选）

作为双重保险，可在 `backward()` 中添加循环检测：

```rust
impl Var {
    pub fn backward(&self) -> Result<f32, GraphError> {
        let mut visited = HashSet::new();
        self.backward_with_cycle_check(&mut visited)
    }
    
    fn backward_with_cycle_check(
        &self, 
        visited: &mut HashSet<NodeId>
    ) -> Result<f32, GraphError> {
        // 检测循环
        if !visited.insert(self.node.id) {
            return Err(GraphError::CycleDetected(format!(
                "检测到循环依赖：节点 {} 被重复访问",
                self.node.id
            )));
        }
        
        // 递归处理 parents
        for parent in &self.node.parents {
            // ... 计算梯度
        }
        
        visited.remove(&self.node.id);  // 回溯时移除（允许 DAG 中的菱形结构）
        Ok(loss_value)
    }
}
```

**注意**：正常使用下这个检测永远不会触发，但它能捕获潜在的内部 bug。

### 3.5 Graph 的新角色

```rust
pub struct Graph {
    inner: Rc<RefCell<GraphInner>>,
}

pub struct GraphInner {
    name: String,
    
    // 全局计数器
    next_node_id: u64,              // 分配新 NodeId
    forward_pass_id: u64,           // 前向传播批次标记
    backward_pass_id: u64,          // 反向传播批次标记
    
    // 参数注册表（弱引用，不控制参数生命周期）
    parameters: HashMap<String, Weak<NodeInner>>,
    
    // 状态相关
    is_eval_mode: bool,
    rng: Option<StdRng>,
    
    // 循环/BPTT 相关（保留）
    recurrent_edges: HashMap<NodeId, NodeId>,
    prev_values: HashMap<NodeId, Tensor>,
    time_step: u64,
    step_history: Vec<HashMap<NodeId, StepSnapshot>>,
}
```

**变化**：
- 移除 `nodes: HashMap<NodeId, NodeHandle>`（运算节点不再集中存储）
- `parameters` 改用 **`Weak` 弱引用**（见下方说明）
- 保留循环/BPTT 相关字段

#### 参数的持有关系

```
┌─────────────────────────────────────────────────────────────┐
│  参数的生命周期由 Layer 控制，Graph 只是"观察者"            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Linear {                                                   │
│       weight: Var  ←─── Rc<NodeInner>  （主持有者，强引用）  │
│       bias: Var    ←─── Rc<NodeInner>  （主持有者，强引用）  │
│   }                                                          │
│        │                                                     │
│        │ 注册                                                │
│        ▼                                                     │
│   Graph.parameters: HashMap<String, Weak<NodeInner>>         │
│                               └─── 弱引用，不阻止释放        │
│                                                              │
│   当 Linear 销毁时：                                         │
│   - weight/bias 的 Rc 引用计数归零 → 参数释放                │
│   - Graph.parameters 中的 Weak 自动失效                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**optimizer.step() 的实现**：
```rust
impl Optimizer {
    pub fn step(&mut self) -> Result<(), GraphError> {
        for weak_param in self.graph.parameters.values() {
            if let Some(param) = weak_param.upgrade() {
                // 参数还活着，更新它
                self.update_param(&param)?;
            }
            // 如果 upgrade() 返回 None，说明参数已被销毁，跳过
        }
        Ok(())
    }
}
```

### 3.6 前向传播流程

```rust
// 用户代码（与现在相同）
let x = graph.input(&batch)?;           // 创建 Input Var
let h = self.fc1.forward(&x).relu();    // 创建运算 Var
let out = self.fc2.forward(&h);         // 创建运算 Var
let loss = out.cross_entropy(&target)?; // 创建运算 Var

loss.backward()?;
optimizer.step()?;

// ← 代码块结束，x, h, out, loss 离开作用域
// ← 运算节点自动释放（Rc 引用计数归零）
// ← 参数节点保留（在 Graph.parameters 中）
```

### 3.7 反向传播流程

```rust
impl Var {
    pub fn backward(&self) -> Result<f32, GraphError> {
        // 1. 从 self 向上遍历到所有叶子节点（通过 parents）
        // 2. 计算梯度（与现在相同的 VJP 机制）
        // 3. 累积到参数节点的 grad 字段
        // 4. 返回 loss 值
        
        // 注意：不需要清理节点，Var 离开作用域时自动释放
    }
}
```

---

## 4. 对现有组件的影响

### 4.1 ModelState：完全移除

#### 为什么可以彻底移除？

在方案 C 中，**每个 Var 通过 parents 链持有整个上游计算图**：

```
loss.visualize() 能自动发现：

loss
  └─ parents ─→ cross_entropy
                  └─ parents ─→ fc2_output
                                  └─ parents ─→ relu
                                                  └─ parents ─→ fc1_output
                                                                  └─ parents ─→ [input, W1, b1]
```

**不需要任何闭包或包装器**，只需从 `loss` 开始遍历 `parents` 链。

#### 新的可视化 API

```rust
// ==================== 方式 A：从 Var 调用 ====================

let loss = model.forward(&x)?.cross_entropy(&target)?;
loss.visualize("model.png")?;

// 多出口点
Var::visualize_all(&[&g_loss, &d_loss], "gan.png")?;

// ==================== 方式 B：从 Graph 调用（保留习惯）====================

// 与旧 API 风格一致
graph.visualize("model.png", &loss)?;
graph.visualize_all("gan.png", &[&g_loss, &d_loss])?;
```

#### 关键特性：无需前向传播即可可视化

**节点创建 ≠ 前向传播**，图结构在调用操作时就已建立：

```rust
// ==================== 仅构建图，不执行计算 ====================

// 使用形状占位符，无需实际数据
let x = graph.input_shape(&[32, 784])?;
let target = graph.input_shape(&[32, 10])?;

// 这些调用只是创建节点和 parents 链，不执行矩阵运算
let h = self.fc1.forward(&x)?.relu();
let out = self.fc2.forward(&h)?;
let loss = out.cross_entropy(&target)?;

// 此时：
// - 没有任何计算发生
// - 但图结构已经完整！
// - parents 链已经建立！

loss.visualize("architecture.png")?;  // ✅ 画出完整架构图

// ==================== 使用场景 ====================

// 1. 训练前验证模型架构
// 2. 生成文档中的架构图
// 3. 调试复杂模型结构
// 4. 无需真实数据就能检查模型
```

#### 多模型场景示例

```rust
// ==================== GAN ====================
let fake = generator.forward(&noise)?;
let d_real = discriminator.forward(&real)?;
let d_fake = discriminator.forward(&fake)?;
let g_loss = d_fake.bce_loss(&ones)?;
let d_loss = d_real.bce_loss(&ones)?.add(&d_fake.bce_loss(&zeros)?)?;

// 完整捕获 Generator + Discriminator + 数据流
graph.visualize_all("gan_full.png", &[&g_loss, &d_loss])?;

// ==================== SAC 强化学习 ====================
let q1 = critic1.forward(&state, &action)?;
let q2 = critic2.forward(&state, &action)?;
let pi = actor.forward(&state)?;
let alpha_loss = compute_alpha_loss(&pi, &log_alpha)?;

// 完整捕获 Actor + Critic1 + Critic2 + Alpha + 数据流
graph.visualize_all("sac_full.png", &[&q1, &q2, &pi, &alpha_loss])?;
```

#### 实现原理

```rust
impl Var {
    /// 从当前节点向上游遍历，可视化整个计算图
    pub fn visualize(&self, path: &str) -> Result<(), GraphError> {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        self.collect_graph(&mut visited, &mut nodes, &mut edges);
        
        let dot = generate_dot(&nodes, &edges);
        std::fs::write(path, dot)?;
        Ok(())
    }
    
    fn collect_graph(
        &self,
        visited: &mut HashSet<NodeId>,
        nodes: &mut Vec<NodeInfo>,
        edges: &mut Vec<EdgeInfo>,
    ) {
        // 避免重复访问（处理 DAG 中的菱形结构）
        if !visited.insert(self.node.id) {
            return;
        }
        
        // 收集当前节点
        nodes.push(NodeInfo {
            id: self.node.id,
            name: self.node.name.clone(),
            node_type: self.node.raw_node.describe(),
        });
        
        // 递归遍历所有父节点
        for parent in &self.node.parents {
            edges.push(EdgeInfo {
                from: parent.id,
                to: self.node.id,
            });
            
            // 递归
            self.wrap_parent(parent).collect_graph(visited, nodes, edges);
        }
    }
    
    /// 合并多个出口点的计算图
    pub fn visualize_all(vars: &[&Var], path: &str) -> Result<(), GraphError> {
        let mut visited = HashSet::new();
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        
        for var in vars {
            var.collect_graph(&mut visited, &mut nodes, &mut edges);
        }
        
        let dot = generate_dot(&nodes, &edges);
        std::fs::write(path, dot)?;
        Ok(())
    }
}
```

#### 命名/分组如何处理？

不需要 ModelState 的闭包，通过**节点名称前缀**自动分组：

```rust
// Layer 创建节点时自动设置名称
impl Linear {
    pub fn forward(&self, x: &Var) -> Var {
        // 节点名称自动带上 layer 名称前缀
        let h = x.matmul(&self.weight);  // 名称: "fc1/matmul"
        h.add(&self.bias)                 // 名称: "fc1/add"
    }
}

// 可视化时自动按前缀分组
// fc1/matmul, fc1/add     → 分组 "fc1"
// fc2/matmul, fc2/add     → 分组 "fc2"
// actor/fc1/..., actor/fc2/...  → 分组 "actor"
// critic/fc1/..., critic/fc2/... → 分组 "critic"
```

#### 与 ModelState 对比

| 特性 | 旧方案（ModelState 闭包） | 新方案（parents 遍历） |
|------|--------------------------|----------------------|
| 可视化范围 | 仅闭包内的操作 | **整个上游计算图** |
| 多模型场景 | 需要嵌套闭包 | `visualize_all()` 一行搞定 |
| RL 复杂流程 | 很难覆盖所有操作 | **自动捕获所有相关节点** |
| 代码侵入性 | 需要修改 forward 结构 | **零侵入** |
| 命名/分组 | 依赖闭包嵌套 | 通过节点名称前缀自动推断 |

**结论**：**彻底移除 ModelState**，删除 `src/nn/model_state.rs`

### 4.2 RNN/LSTM/GRU 层

当前的循环层内部有 `unroll_cache` 按 `(batch_size, seq_len)` 缓存展开结构的 NodeId。

**新架构下的问题**：
- 运算节点每次 ID 不同
- 缓存永远不命中

**处理方案**：移除层内部的缓存，每次重新展开

```rust
// 当前设计（有缓存）
pub struct Rnn {
    // 参数...
    unroll_cache: RefCell<HashMap<(usize, usize), NodeId>>,
}

// 新设计（无缓存）
pub struct Rnn {
    w_ih: Var,
    w_hh: Var,
    b_h: Var,
    // 移除 unroll_cache
}

impl Rnn {
    pub fn forward(&self, x: &Var) -> Result<Var, GraphError> {
        // 每次都重新展开（开销可忽略，因为只是创建节点）
        // 实际的矩阵运算无论缓存与否都要执行
        self.unroll(x, seq_len)
    }
}
```

**为什么可行**：
- 展开只是创建节点结构，开销很小
- 真正的计算（矩阵运算）无论是否缓存都要执行
- 方案 C 保证运算节点用完自动释放，不会累积

**图打印**：forward 后、Var 离开作用域前打印，能看到完整的时间步展开。

### 4.3 Criteria（损失函数）

当前的 `CrossEntropyLoss`、`MseLoss` 等使用 `LossState` 按 `output.node_id()` 缓存。

**新架构下**：
- output 节点每次都是新的（ID 不同）
- 缓存永远不命中
- **完全移除 Criterion 类**，统一使用 Var 方法

```rust
// ==================== 移除 Criterion 类 ====================
// 不再需要：
// let criterion = MseLoss::new();
// let loss = criterion.forward(&output, &target)?;

// ==================== 统一使用 Var 方法 ====================

// 无参数版本（使用默认值）
let loss = output.mse_loss(&target)?;
let loss = output.cross_entropy(&target)?;
let loss = output.huber_loss(&target)?;         // 默认 delta=1.0

// 有参数版本（显式指定）
let loss = output.huber_loss_with(&target, 0.5)?;  // delta=0.5
```

**设计要点**：
1. **参数固化在节点内部**：每次调用创建的节点存储当时的参数值
2. **每次调用可传不同参数**：更灵活，无需预先创建 Criterion 实例
3. **图打印时显示参数**：如 `Huber(δ=0.5)` 而非简单的 `Huber`

**需要删除的文件**：`src/nn/criterion.rs`（整个文件）

### 4.4 可视化

**当前**：依赖 ModelState 闭包缓存图结构

**新架构**：从 Var 沿 parents 链遍历，无需缓存

```rust
// 推荐的可视化工作流
fn visualize_model(model: &MyModel) -> Result<(), GraphError> {
    // 用形状占位符构建图（不执行实际计算）
    let x = Var::input_shape(&[1, INPUT_DIM])?;
    let output = model.forward(&x)?;
    
    // 从 output 向上游遍历整个图
    output.visualize("model.png")?;
    
    Ok(())
    // ← output 离开作用域，运算节点自动清理
}

// GAN 等多模型可视化
let fake = generator.forward(&noise)?;
let d_output = discriminator.forward(&fake)?;
let loss = d_output.bce_loss(&labels)?;  // 直接调用 Var 方法

// 打印完整的 G → D 数据流
Var::visualize_all(&[&loss], "gan.png")?;

loss.backward()?;
```

### 4.5 序列化

参数序列化从 `Graph.parameters` 注册表读取，不受影响。

### 4.6 BPTT 机制

BPTT 相关字段（`step_history` 等）保留在 GraphInner 中。

**注意**：如果使用传统的 `step()` + `connect_recurrent()` 机制（而非展开式），`recurrent_edges` 中存储的 `from_node` NodeId 在方案 C 下可能不稳定。建议：
- 优先使用展开式设计（当前 RNN/LSTM/GRU 层已采用）
- 如需传统机制，改用语义标识或值传递

---

## 5. 实施路径

### Phase 0：当前状态（已完成）

- ✅ `checkpoint()` + `prune_nodes_after()` API 作为临时方案
- ✅ 让 SAC 示例先跑起来验证算法正确性

### Phase 1：架构设计（本文档）

- [x] 确定 Var 新结构（`Rc<NodeInner>` + `Weak<RefCell<GraphInner>>`）
- [x] 设计边管理策略（parents 用 `Rc` **强引用**，保证反向传播时存活）
- [x] 设计 Graph 新角色（仅保留参数注册表和全局配置）
- [x] 确定 API 兼容性策略（渐进式迁移）
- [x] 评审通过后进入 Phase 2

### Phase 2：核心实现

> **实施策略**：渐进式迁移，保持 API 兼容，每步可独立测试

#### Step 2.1：NodeInner 工厂方法 ✅ 已完成
- [x] 实现 `NodeInner` 结构（`src/nn/nodes/node_inner.rs`）
- [x] 基础创建方法（`new`, `new_leaf`）
- [x] 访问器和修改器方法
- [x] 单元测试（9 个，覆盖创建、引用计数、级联释放）

#### Step 2.2：Var 结构过渡改造 ✅ 已完成
- [x] 添加 `node: Option<Rc<NodeInner>>` 字段（过渡期）
- [x] 修改 `node_id()` 优先从 `self.node` 获取
- [x] 修改 `value()`, `grad()` 等优先从 `self.node` 访问
- [x] 保持旧路径 `graph.borrow().xxx(self.id)` 作为后备
- [x] 单元测试：验证新旧路径等价（8 个测试）

#### Step 2.3：GraphInner 参数注册表
- [x] 添加 `parameters: HashMap<String, Weak<NodeInner>>`
- [x] `next_node_id` 计数器：复用现有 `next_id: u64`，无需额外改动
- [x] 实现参数注册/查询 API（7 个方法）
- [x] 暂时保留 `nodes: HashMap` 用于过渡
- [x] 单元测试：参数注册和弱引用行为（7 个测试）

#### Step 2.4：节点创建流程改造
- [ ] 新增 `NodeInner::new_xxx()` 系列工厂方法（各节点类型）
- [ ] `GraphInner::new_xxx_node()` 返回 `Rc<NodeInner>` 并设置 parents
- [ ] Var 算子重载使用新创建流程
- [ ] 单元测试：节点创建和 parents 链正确性

#### Step 2.5：前向传播改造
- [ ] 实现 `NodeInner::forward_recursive()` 基于 parents 遍历
- [ ] 从 loss 节点向上遍历，拓扑排序执行
- [ ] 使用 `last_forward_pass_id` 避免重复计算
- [ ] Var.forward() 调用新实现
- [ ] 单元测试：前向传播正确性、pass_id 去重

#### Step 2.6：反向传播改造
- [ ] 实现 `NodeInner::backward_recursive()` 基于 parents 遍历
- [ ] 从 loss 向上拓扑逆序计算梯度
- [ ] 使用 `last_backward_pass_id` 和 `is_detached` 控制流
- [ ] Var.backward() 调用新实现
- [ ] 适配 `zero_grad()`（清除参数节点的梯度，遍历 `parameters` 注册表）
- [ ] 单元测试：反向传播正确性、梯度累积、detach 行为

#### Step 2.7：移除过渡代码
- [ ] Var.graph 从 `Rc` 改为 `Weak<RefCell<GraphInner>>`
- [ ] 移除 Var.node 的 `Option` 包装
- [ ] 移除 GraphInner 的 `nodes: HashMap<NodeId, NodeHandle>`
- [ ] 移除 GraphInner 的 `forward_edges`, `backward_edges`
- [ ] 更新所有依赖旧结构的代码
- [ ] 回归测试：所有现有测试通过

#### Step 2.8：完整性验证
- [ ] 运行所有单元测试
- [ ] 运行所有集成测试
- [ ] 验证 xor 示例收敛
- [ ] 验证 mnist 示例准确率
- [ ] 验证 cartpole_sac 节点不累积（**核心目标**）

**补充测试场景**（评审建议）：
- [ ] 深度网络（100+ 层）：验证递归遍历不栈溢出
- [ ] 高频创建销毁（10000+ 次/秒）：验证 Rc alloc/dealloc 性能
- [ ] 菱形依赖（DAG）：验证多路径汇合时**梯度**正确累积
- [ ] detach 后子图隔离：验证梯度不穿透 detach 边界
- [ ] Graph 销毁后 Var 操作：验证 `Weak` 失效时的健壮性（panic 或优雅降级）

### Phase 3：功能适配

- [ ] **删除** `src/nn/model_state.rs`（完全移除）
- [ ] **删除** `src/nn/criterion.rs`（完全移除，统一用 Var 方法）
- [ ] 可视化模块适配（改用 Var.visualize() 遍历 parents）
- [ ] 序列化模块适配
- [ ] BPTT/循环机制适配（移除 RNN 层内缓存）

### Phase 4：验证与文档

- [ ] 现有示例全部通过（xor, mnist, gan, sac...）
- [ ] 性能对比测试
- [ ] 更新用户文档
- [ ] 更新 architecture_roadmap.md

---

## 6. 测试策略

### 6.1 回归测试

所有现有示例必须通过：

| 示例 | 验证点 |
|------|--------|
| `xor` | 基础 MLP，收敛性 |
| `mnist` | 图像分类，准确率 |
| `mnist_gan` | GAN 训练，detach 行为 |
| `california_housing` | 回归任务 |
| `parity_*` | RNN/LSTM/GRU 变长序列 |
| `cartpole_sac` | **重点**：强化学习，节点不累积 |

### 6.2 性能测试

| 指标 | 验收标准 |
|------|---------|
| 节点数量 | 训练过程中保持恒定（不累积） |
| 内存使用 | 不随 epoch 增长 |
| 单次迭代耗时 | 与当前架构相当或更快 |

### 6.3 新增测试

- [ ] 节点生命周期测试（创建、释放时机）
- [ ] 多模型数据流测试（GAN、SAC）
- [ ] 边界情况（空图、单节点、深度网络）

---

## 7. 与 NAS/NEAT 的兼容性

方案 C 的动态图特性**非常适合 NAS/NEAT**：

| NAS 需求 | 方案 C 支持 |
|---------|------------|
| 运行时添加连接 | ✅ 下次 forward 自动体现 |
| 运行时删除连接 | ✅ 不连接即自动释放 |
| 动态改变网络深度 | ✅ 只需改变 forward 代码路径 |
| 参数共享 | ✅ 多个 Var 可引用同一参数 |
| 多网络并行进化 | ✅ 每个 Graph 独立 |

```rust
// NEAT 变异示例
fn mutate_add_node(&mut self, graph: &Graph) {
    // 在两个节点之间插入新节点
    let new_weight = graph.parameter(&[in_dim, out_dim], Init::Xavier, "new_w")?;
    // 下次 forward 时自动使用新拓扑
}

fn mutate_remove_connection(&mut self) {
    // 只需不在 forward 中使用该连接
    // 对应的运算节点自然不会创建
}
```

---

## 附录 A：关键代码变更清单

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `src/nn/var.rs` | **重构** | Var 结构变更为 Rc<NodeInner> + Weak<RefCell<GraphInner>> |
| `src/nn/graph/inner/mod.rs` | **重构** | 移除 nodes HashMap |
| `src/nn/graph/inner/core.rs` | **重构** | 修改前向/反向传播逻辑 |
| `src/nn/graph/handle.rs` | 适配 | Graph API 适配 |
| `src/nn/model_state.rs` | **删除** | 完全移除，可视化改用 Var 遍历 |
| `src/nn/criterion.rs` | **删除** | 完全移除，统一使用 Var 方法 |
| `src/nn/nodes/node_handle.rs` | **重构** | 拆分为 NodeInner |
| `src/nn/graph/inner/visualization.rs` | 适配 | 适配新的节点遍历方式 |

---

## 8. 进阶选项：完全隐藏 Graph

> 本节讨论一个更激进的架构选项：**彻底移除用户可见的 Graph 对象**，实现完全的 PyTorch 风格。

### 8.1 当前 vs PyTorch 风格对比

```rust
// ==================== 当前风格（显式 Graph）====================
let graph = Graph::new("my_model");
let x = graph.input(&data)?;
let y = model.forward(&x)?;
loss.backward()?;
graph.visualize("model.png", &loss)?;

// ==================== PyTorch 风格（无显式 Graph）====================
let x = Tensor::from(&data).requires_grad(true);  // 直接标记需要梯度
let y = model.forward(&x)?;
loss.backward()?;
loss.visualize("model.png")?;  // 直接从 Var 可视化
```

### 8.2 Graph 当前职责的替代方案

| 当前职责 | 替代方案 |
|---------|---------|
| 创建输入节点 | `Tensor::requires_grad(true)` 或 `Var::input(&tensor)` |
| 参数注册表 | 存储在 Module/Layer 中，`model.parameters()` |
| 训练/评估模式 | `model.train()` / `model.eval()` |
| 随机数种子 | `only_torch::manual_seed(42)` 全局设置 |
| 循环/BPTT 状态 | 存储在 RNN 层内部 |
| 可视化 | `var.visualize()` 从 Var 遍历 |

### 8.3 新的用户代码风格

```rust
// ==================== 完全 PyTorch 风格 ====================

// 全局设置
only_torch::manual_seed(42);

// 定义模型
struct MyModel {
    fc1: Linear,
    fc2: Linear,
}

impl MyModel {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 256),
            fc2: Linear::new(256, 10),
        }
    }
    
    fn forward(&self, x: &Var) -> Var {
        self.fc2.forward(&self.fc1.forward(x).relu())
    }
}

// 训练
let model = MyModel::new();
let optimizer = Adam::new(model.parameters(), 0.001);

for (data, target) in dataloader {
    // 输入：直接从 Tensor 创建 Var
    let x = Var::input(&data);
    let t = Var::input(&target);
    
    // 前向
    let output = model.forward(&x);
    let loss = output.cross_entropy(&t)?;
    
    // 反向
    loss.backward()?;
    
    // 优化
    optimizer.step()?;
    optimizer.zero_grad()?;
}

// 可视化（任意时刻，从 Var 开始）
loss.visualize("model.png")?;
```

### 8.4 实现要点

#### 8.4.1 输入节点的创建

```rust
impl Var {
    /// 从 Tensor 创建输入 Var（替代 graph.input()）
    pub fn input(tensor: &Tensor) -> Self {
        let node = Rc::new(NodeInner {  // 如需多线程支持，可改为 Arc
            id: NodeId::next(),  // 全局 ID 生成器
            raw_node: NodeType::Input(tensor.shape().to_vec()),
            parents: vec![],
            // ...
        });
        
        // 立即执行 forward（设置值）
        node.set_value(tensor);
        
        Self { node, graph: Weak::new() }  // graph 为空
    }
}
```

#### 8.4.2 参数管理

```rust
trait Module {
    /// 收集所有参数
    fn parameters(&self) -> Vec<Var>;
    
    /// 设置训练/评估模式
    fn train(&mut self);
    fn eval(&mut self);
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Var> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}
```

#### 8.4.3 全局配置

```rust
// src/lib.rs
thread_local! {
    static CONFIG: RefCell<GlobalConfig> = RefCell::new(GlobalConfig::default());
}

pub fn manual_seed(seed: u64) {
    CONFIG.with(|c| c.borrow_mut().seed = Some(seed));
}

pub fn is_grad_enabled() -> bool {
    CONFIG.with(|c| c.borrow().grad_enabled)
}
```

### 8.5 优缺点分析

| 方面 | 隐藏 Graph | 保留 Graph |
|------|-----------|-----------|
| 用户体验 | ⭐⭐⭐ PyTorch 风格，更简洁 | ⭐⭐ 需要额外的 Graph 对象 |
| 学习曲线 | ⭐⭐⭐ PyTorch 用户无缝迁移 | ⭐⭐ 需要理解 Graph 概念 |
| 代码复杂度 | ⭐⭐ 需要全局状态管理 | ⭐⭐⭐ 状态集中在 Graph |
| 调试能力 | ⭐⭐ 可能更难追踪 | ⭐⭐⭐ Graph 提供清晰边界 |
| 多图支持 | ⭐ 需要额外机制 | ⭐⭐⭐ 天然支持多个独立图 |

### 8.6 建议

**推荐分两步走**：

1. **Phase 1（方案 C 核心）**：保留 Graph，但大幅简化其职责
   - 移除 ModelState、Criterion
   - 实现 Rc/parents 生命周期管理
   - 可视化从 Var 遍历

2. **Phase 2（可选进阶）**：隐藏 Graph
   - 提供 `Var::input()` 替代 `graph.input()`
   - 参数管理移至 Module trait
   - Graph 变为可选的"高级功能"

这样可以渐进式地向 PyTorch 风格靠拢，同时保持向后兼容。

---

## 附录 B：参考资料

- [PyTorch Autograd 机制](https://pytorch.org/docs/stable/notes/autograd.html)
- [Rust Arc 和 Weak 文档](https://doc.rust-lang.org/std/sync/struct.Arc.html)
- 项目内参考：`MatrixSlow/` Python 实现

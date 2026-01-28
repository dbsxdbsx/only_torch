# 神经架构演化设计（Neural Architecture Evolution）

> 本文档描述 only_torch 的神经架构演化机制设计——一种结合 **NEAT 风格拓扑变异** 与 **梯度训练** 的混合策略。
>
> **定位**：这是项目愿景的核心功能，区别于传统固定架构的深度学习框架。

---

## 1. 核心理念

### 1.1 与传统 NEAT 的区别

| 维度 | 传统 NEAT | only_torch 方案 |
|------|----------|----------------|
| **权重优化** | 纯进化（慢） | 梯度下降（快） |
| **权重继承** | Darwinian（重新初始化） | **Lamarckian**（继承训练后的权重） |
| **物种形成** | 必须（保护新结构） | ❌ 不需要（梯度训练快速验证结构价值） |
| **创新号** | 必须（交叉对齐） | ❌ 不需要（不做交叉） |
| **交叉操作** | 核心机制 | ❌ 不需要（只用变异） |
| **结构变异** | 核心机制 | ✅ 核心机制 |
| **循环/记忆** | ✅ 支持（拓扑循环） | ✅ 支持（复用现有记忆机制） |

### 1.2 两种边类型

演化过程中存在两种边，它们的语义和约束完全不同：

| 边类型 | 英文 | 值传递时机 | 拓扑约束 | 用途 |
|--------|------|-----------|---------|------|
| **普通边** | Forward Edge | 同一时间步内立即传递 | **必须保持 DAG** | 标准计算流 |
| **循环边** | Recurrent Edge | 延迟一个时间步传递 | **允许形成图论环** | 记忆/循环机制 |

**关键设计原则**：

```
单步计算永远是 DAG
    ↓
循环边的"环"是时间维度的，不是计算维度的
    ↓
不会产生死循环
```

**循环边如何避免死循环？**

循环边使用**双缓冲机制**——每个时间步读取的是上一步缓存的值，而不是当前正在计算的值：

```
时间步 t=0:
    读取: A_old = 0（初始值）
    计算: A_new = f(input_0 + w * A_old)
    更新: A_old ← A_new

时间步 t=1:
    读取: A_old = 上一步的值
    计算: A_new = f(input_1 + w * A_old)
    更新: A_old ← A_new
```

> **与 RNN/LSTM 的关系**：循环边是节点层级的记忆原语，而 RNN/LSTM 是这些原语的组合封装。详见 [记忆机制设计](./memory_mechanism_design.md)。

### 1.3 设计哲学

```
┌─────────────────────────────────────────────────────────────────┐
│  NEAT 负责"探索"：什么样的网络结构可能更好？                       │
│  梯度负责"利用"：给定结构，找到最优权重                            │
│  两者交替：结构变异 → 梯度训练 → 评估 → 结构变异 → ...             │
└─────────────────────────────────────────────────────────────────┘
```

这种混合策略来自 [EXAMM 论文](../paper/EXAMM_2019/summary.md)，其核心洞察是：

> **梯度下降比进化快得多**（权重优化）
> **但梯度下降无法搜索结构**（需要进化）
> → 各取所长：进化搜索结构 + 梯度优化权重

### 1.4 模块位置

Evolution 作为独立模块，与 `nn` 平级：

```
only_torch/
├── nn/           # 定义网络
│   ├── graph/
│   ├── node/
│   └── layer/
├── evolution/    # 演化网络（核心特色）
│   ├── mod.rs
│   ├── evolution.rs      # Evolution 结构体
│   ├── mutation.rs       # 变异操作
│   └── tasks/            # EvolutionTask trait + 预定义任务
│       ├── mod.rs        # 定义 EvolutionTask trait + re-export
│       ├── supervised.rs
│       ├── rl.rs
│       └── gan.rs
├── optim/        # 优化器
└── data/         # 数据加载
```

**设计理由**：
- **突出核心特色**：Evolution 是项目愿景的核心，值得独立模块
- **概念清晰**：`nn` 定义网络，`evolution` 演化网络，`optim` 优化参数
- **符合惯例**：类似 PyTorch 的 `torch.nn` vs `torch.optim` 平级设计
- **易于扩展**：未来可能有多种演化策略

### 1.5 聚合节点（AggregateNode）

**问题**：当前架构中，操作节点（如 Tanh, Sigmoid）有固定的父节点数量。当 Add Edge 变异向一个已有输入的节点添加新边时，节点的计算逻辑无法处理多余的输入。

**参考**：在原始 NEAT 论文和 neat-python 中，每个节点天然支持任意数量的输入，通过**聚合函数**将它们合并：

```python
# neat-python 的节点计算
node_inputs = [values[i] * w for i, w in links]
s = aggregation(node_inputs)  # 先聚合（sum/mean/max/...）
output = activation(bias + response * s)  # 再激活
```

**解决方案**：引入 `AggregateNode`——一个内部节点类型，仅在演化过程中自动创建。

```
演化添加边的场景：

Before:  A ──w1──> B ──> C
         想添加 X ──> B（但 B 已有输入 A）

After:   A ──w1──┐
                 ├──> Agg ──w3──> B ──> C
         X ──w2──┘
                  ↑
             AggregateNode（sum）
```

**设计原则**：

| 方面 | 决策 |
|------|------|
| **可见性** | `pub(crate)` — 仅内部使用，不暴露给用户 |
| **位置** | `src/nn/graph/inner/evolution/aggregate.rs` |
| **默认聚合** | `sum`（与 NEAT 论文一致） |
| **可扩展** | 未来可支持 `mean`, `max`, `product` 等 |

```rust
// 内部定义，不导出
#[derive(Clone)]
pub(crate) enum AggregationType {
    Sum,      // 默认：加权求和
    // Mean,  // 未来可扩展
    // Max,
    // Product,
}

/// 聚合节点：将多个输入汇合为一个输出
/// 仅在演化过程中自动创建，用户无法直接使用
pub(crate) struct AggregateNode {
    aggregation: AggregationType,
}
```

**与现有节点的关系**：

| 节点类型 | 父节点数量 | 示例 |
|----------|-----------|------|
| 激活函数节点 | **1** | Tanh, Sigmoid, ReLU |
| 二元运算节点 | **2** | MatMul, Sub, Div |
| 多元运算节点 | **N** | Add（现有）, Concat |
| 聚合节点（新增） | **任意** | AggregateNode（内部） |

> **Note**：现有的 `AddNode` 虽然也支持多输入，但它是用户可见的公开节点。`AggregateNode` 是演化专用的内部实现细节。

### 1.6 演化颗粒度

| 颗粒度 | 变异操作 | 搜索空间 | 灵活性 |
|--------|----------|----------|--------|
| **Node**（当前） | Add/Remove Node/Edge | 大 | ✅ 最高 |
| **Layer**（后续可选） | Add/Remove Dense/Conv | 小 | 受限但更快 |

**MVP 阶段**：采用 **Node 层次**演化

- 最细粒度，最灵活
- 能发现任意拓扑（NEAT 核心优势）
- 验证核心机制

**后续扩展**（可选）：支持 Layer 层次作为"宏变异"

```rust
pub enum MutationType {
    // Node 层次（细粒度）
    AddNode,
    AddEdge,
    RemoveNode,
    RemoveEdge,

    // Layer 层次（粗粒度，后续可选）
    // AddDenseLayer { units: usize },
    // AddConvLayer { filters: usize, kernel: usize },
    // RemoveLayer,
}
```

> **注**：从 Node 视角看，Layer 只是一种"模式"（Dense = 全连接节点组）。
> Node 层次演化理论上可以演化出类似 Layer 的结构，只是需要更多时间。

---

## 2. 主循环设计

### 2.1 两层判定标准

| 层级 | 指标 | 作用 |
|------|------|------|
| **任务层** | Accuracy / F1 / Episode 奖励等 | 决定是否**彻底结束** |
| **优化层** | Loss 变化率 / 梯度范数 | 决定是否**收敛**（该换拓扑了） |

**关键区分**：
- **收敛** ≠ **达标**
- 收敛只是说明"当前拓扑已榨干"，需要变异换个结构继续尝试
- 达标才是真正完成任务

### 2.2 EvolutionTask trait

不同学习范式的"训练"和"评估"方式不同，通过 trait 抽象：

```rust
/// 演化任务 trait：抽象不同学习范式
pub trait EvolutionTask {
    /// 训练直到收敛，返回 loss（用于比较全局最优）
    fn train_until_converged(&self, graph: &mut Graph, config: &ConvergenceConfig) -> f32;

    /// 评估任务指标，返回分数（越高越好）
    fn evaluate(&self, graph: &Graph) -> f32;
}
```

| 学习范式 | train_until_converged | evaluate |
|----------|----------------------|----------|
| **监督学习** | 用训练集做梯度下降 | 在测试集上算 Accuracy |
| **强化学习** | 与环境交互 + 策略梯度 | 跑 Episode 取奖励 |
| **GAN** | G/D 交替训练 | FID / IS 等指标 |
| **无监督学习** | 重构/对比学习 | 重构误差等 |

### 2.3 预定义任务实现

为用户提供开箱即用的 Task。

**优化器所有权**：优化器是 Task 的内部实现细节，由 Task 持有和管理。

```rust
// ========== 监督学习 ==========
pub struct SupervisedTask {
    train_data: Dataset,
    test_data: Dataset,
    metric: MetricType,
    optimizer: Optimizer,  // Task 内部持有优化器
}

impl SupervisedTask {
    pub fn new(train_data: Dataset, test_data: Dataset, metric: MetricType) -> Self {
        Self {
            train_data,
            test_data,
            metric,
            optimizer: Adam::default(),  // 默认优化器
        }
    }

    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }
}

impl EvolutionTask for SupervisedTask {
    fn train_until_converged(&self, graph: &mut Graph, config: &ConvergenceConfig) -> f32 {
        let mut detector = ConvergenceDetector::new(config);
        loop {
            let loss = graph.forward(&self.train_data);
            let grad_norm = graph.backward();
            self.optimizer.step(graph);  // 使用 Task 内部的优化器
            if detector.is_converged(loss, grad_norm) {
                return loss;
            }
        }
    }

    fn evaluate(&self, graph: &Graph) -> f32 {
        // 评估时禁用梯度（纯推理）
        graph.with_no_grad(|| {
            graph.compute_metric(&self.test_data, &self.metric)
        })
    }
}

// ========== 强化学习 ==========
pub struct RLTask<E: Environment> {
    env: E,
    optimizer: Optimizer,  // Task 内部持有优化器
}

impl<E: Environment> RLTask<E> {
    pub fn new(env: E) -> Self {
        Self {
            env,
            optimizer: Adam::default(),
        }
    }

    pub fn with_optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = optimizer;
        self
    }
}

impl<E: Environment> EvolutionTask for RLTask<E> {
    fn train_until_converged(&self, graph: &mut Graph, config: &ConvergenceConfig) -> f32 {
        // 与环境交互，收集经验，做策略梯度
        // ...
    }

    fn evaluate(&self, graph: &Graph) -> f32 {
        // 跑一个 episode，返回总奖励
        self.env.run_episode(graph)
    }
}
```

### 2.4 Evolution 结构体（Only Touch 设计）

**核心理念**：用户只需 touch 一个结构 `Evolution`，调用一个方法 `.run()`。

```rust
/// 一次神经架构演化的完整定义
pub struct Evolution {
    // ========== 核心输入（所有权）==========
    /// 待演化的网络
    pub graph: Graph,
    /// 任务定义（封装训练+评估）
    pub task: Box<dyn EvolutionTask>,

    // ========== 达标判定 ==========
    pub target_metric: f32,   // 默认 0.95
    pub eval_runs: usize,     // 默认 3

    // ========== 收敛判定 ==========
    pub loss_tolerance: f32,  // 默认 1e-4
    pub grad_tolerance: f32,  // 默认 1e-5
    pub patience: usize,      // 默认 5

    // ========== 变异概率 ==========
    pub prob_add_node: f32,              // 默认 0.15
    pub prob_add_edge: f32,              // 默认 0.30
    pub prob_add_recurrent_edge: f32,    // 默认 0.15
    pub prob_remove_edge: f32,           // 默认 0.20
    pub prob_remove_recurrent_edge: f32, // 默认 0.10
    pub prob_remove_node: f32,           // 默认 0.10
}

impl Evolution {
    /// 便捷构造：只需 Graph 和 Task，其他用默认值
    pub fn new(graph: Graph, task: impl EvolutionTask + 'static) -> Self {
        Self {
            graph,
            task: Box::new(task),
            // 默认值
            target_metric: 0.95,
            eval_runs: 3,
            loss_tolerance: 1e-4,
            grad_tolerance: 1e-5,
            patience: 5,
            prob_add_node: 0.15,
            prob_add_edge: 0.30,
            prob_add_recurrent_edge: 0.15,
            prob_remove_edge: 0.20,
            prob_remove_recurrent_edge: 0.10,
            prob_remove_node: 0.10,
        }
    }

    /// 执行演化（消耗 self，返回演化后的 Graph）
    pub fn run(self) -> Graph {
        let mut graph = self.graph;  // 取出所有权
        let mut best_snapshot: Option<Snapshot> = None;
        let mut best_loss = f32::INFINITY;
        let mut rng = rand::thread_rng();  // 随机数生成器

        loop {
            // Step 1: 训练到收敛（由 task 决定如何训练）
            let loss = self.task.train_until_converged(&mut graph, &self.convergence_config());

            // Step 2: 评估（N 次取最低值，evaluate 内部已禁用梯度）
            let min_score = (0..self.eval_runs)
                .map(|_| self.task.evaluate(&graph))
                .fold(f32::INFINITY, f32::min);

            if min_score >= self.target_metric {
                return graph;  // 成功！返回所有权
            }

            // Step 3: 更新全局最优 / 回滚
            if loss < best_loss {
                best_loss = loss;
                best_snapshot = Some(graph.snapshot());
            } else if let Some(ref snapshot) = best_snapshot {
                graph.restore(snapshot);
            }

            // Step 4: 随机变异（先检查可行性，保证成功）
            self.random_mutation(&mut graph, &mut rng);

            // 回到 Step 1
            // 关键：回滚后不是"卡住"，而是"换个方向再试"
        }
    }

    fn convergence_config(&self) -> ConvergenceConfig {
        ConvergenceConfig {
            loss_tolerance: self.loss_tolerance,
            grad_tolerance: self.grad_tolerance,
            patience: self.patience,
        }
    }
}
```

### 2.5 任务达标判定

采用**多次评估取最低值**，避免随机性误判：

| 聚合策略 | 风险 |
|----------|------|
| 取**平均值** | 偶然低分被平均掉 |
| 取**最高值** | 偶然高分，不代表稳定 |
| 取**最低值** | ✅ 确保最差情况也达标 |

### 2.6 整体流程图

```
┌─────────────────────────────────────────────────────────────────┐
│  Evolution::run()  —— Only Touch                                 │
│                                                                  │
│    输入：self（包含 graph 所有权 + task + 配置）                   │
│    输出：Graph（演化后的网络所有权）                               │
│                                                                  │
│    loop {                                                        │
│        ┌─────────────────────────────────────────────────────┐   │
│        │  task.train_until_converged()  —— 优化层             │   │
│        │    由 EvolutionTask 实现决定如何训练                  │   │
│        │    return loss                                       │   │
│        └─────────────────────────────────────────────────────┘   │
│                                                                  │
│        ┌─────────────────────────────────────────────────────┐   │
│        │  task.evaluate() × N  —— 任务层                      │   │
│        │    N 次评估取最低值 >= 目标？                          │   │
│        │    └─ Yes → return graph（成功！）                    │   │
│        └─────────────────────────────────────────────────────┘   │
│                                                                  │
│        更新全局最优 / 回滚                                        │
│        随机变异                                                   │
│    }                                                             │
│                                                                  │
│  终止条件：任务指标达标                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.7 调用示例（Only Touch）

```rust
// ========== 最简用法：两行代码 ==========
let task = SupervisedTask::new(train_data, test_data, Metric::Accuracy);
let trained = Evolution::new(graph, task).run();

// ========== 自定义参数 ==========
let trained = Evolution {
    graph,
    task: Box::new(SupervisedTask::new(train_data, test_data, Metric::Accuracy)),
    target_metric: 0.99,
    patience: 3,
    ..Default::default()
}.run();

// ========== 强化学习 ==========
let trained = Evolution::new(graph, RLTask::new(CartPoleEnv::new()))
    .run();

// ========== GAN ==========
let trained = Evolution::new(generator, GANTask::new(real_data, discriminator))
    .run();
```

### 2.8 Only Touch 理念体现

| 维度 | Touch 的东西 |
|------|-------------|
| **结构** | `Evolution` — 一个结构包含一切 |
| **方法** | `.run()` — 一个方法完成一切 |
| **输入** | `Graph` — 一个网络（所有权转移） |
| **输出** | `Graph` — 一个网络（所有权返回） |

---

## 3. 收敛判定

### 3.1 判定条件

采用**双条件**判定，满足任一即认为收敛：

| 条件 | 公式 | 说明 |
|------|------|------|
| **Loss 稳定** | `max(recent) - min(recent) < ε × min(recent)` | 最近 k 个 epoch 的 loss 相对波动 < ε |
| **梯度消失** | `‖∇L‖₂ < δ` | 梯度 L2 范数 < δ |

### 3.2 推荐默认值

| 参数 | 默认值 | 含义 | 来源 |
|------|--------|------|------|
| `loss_tolerance` | **1e-4** | Loss 相对变化 < 0.01% 视为稳定 | PyTorch/TensorFlow early stopping 常用值 |
| `grad_tolerance` | **1e-5** | 梯度范数 < 此值视为"平坦" | 32-bit float 精度下的典型阈值 |
| `patience` | **5** | 连续 5 个 epoch 满足条件 | 2024 论文实证验证最优 |

**调整建议**：

| 场景 | loss_tolerance | grad_tolerance | patience |
|------|----------------|----------------|----------|
| 简单任务（XOR） | 1e-5 | 1e-6 | 3 |
| 中等任务 | 1e-4 | 1e-5 | 5 |
| 复杂任务 | 1e-3 | 1e-4 | 10 |

> **注**：这些参数对梯度下降和梯度上升都适用。因为收敛的本质是"梯度消失"——无论优化方向如何，当 ∇J → 0 时都表示到达极值点。

### 3.3 实现

```rust
pub struct ConvergenceDetector {
    loss_history: VecDeque<f32>,
    patience: usize,
    loss_tolerance: f32,
    grad_tolerance: f32,
}

impl ConvergenceDetector {
    pub fn is_converged(&mut self, loss: f32, grad_norm: f32) -> bool {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > self.patience {
            self.loss_history.pop_front();
        }

        // 条件1：Loss 稳定
        if self.loss_history.len() >= self.patience {
            let max = self.loss_history.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let min = self.loss_history.iter().cloned().fold(f32::INFINITY, f32::min);
            let relative_change = (max - min) / (min.abs() + 1e-8);

            if relative_change < self.loss_tolerance {
                return true;
            }
        }

        // 条件2：梯度消失
        if grad_norm < self.grad_tolerance {
            return true;
        }

        false
    }

    pub fn reset(&mut self) {
        self.loss_history.clear();
    }
}
```

---

## 4. 变异操作

### 4.1 变异类型

| 操作 | 名称 | 效果 | 默认概率 |
|------|------|------|:--------:|
| **Add Node** | 添加节点 | 在已有边上插入新节点 | 15% |
| **Add Edge** | 添加普通边 | 在两个节点间添加前向连接（保持 DAG） | 30% |
| **Add Recurrent Edge** | 添加循环边 | 添加跨时间步的循环连接（允许自环） | 15% |
| **Remove Edge** | 删除边 | 删除一条普通边 | 20% |
| **Remove Recurrent Edge** | 删除循环边 | 删除一条循环边 | 10% |
| **Remove Node** | 删除节点 | 删除隐藏节点及其连接 | 10% |

> **概率说明**：默认概率经过调整以平衡结构探索与记忆能力演化。循环边变异概率较低是因为不是所有任务都需要记忆机制。

### 4.2 Add Node（添加节点）

在一条已有边上插入新节点，原边被拆分为两条新边：

```
Before:  A ──w──> B

After:   A ──rand──> C ──rand──> B
              ↑           ↑
           随机初始化   随机初始化
                  │
            激活函数（如 tanh）

关键：
  - 原边被删除，产生两条全新的边
  - 两条新边的权重都随机初始化（增加探索性）
  - 新节点激活函数 = tanh（或可配置）
  - 后续梯度训练会优化这些权重
```

> **命名说明**：此操作在 NEAT 论文中称为 "Add Node Mutation"，实际操作是"分裂一条已有边并插入新节点"。
>
> **与 NEAT 的区别**：NEAT 原论文建议新入边权重=1、新出边权重=继承，以保持信息流。我们选择全部随机初始化，因为后续会进行充分的梯度训练，初始值不影响最终收敛结果，且能增加探索多样性。

**节点类型选择**：只要符合输入输出规范，任何节点类型都可以插入（激活函数、运算算子等）。

```rust
fn add_node(graph: &mut Graph, edge_id: EdgeId, rng: &mut impl Rng) -> Result<NodeId, GraphError> {
    let (src, dst, _old_weight) = graph.get_edge_info(edge_id)?;

    // 1. 随机选择一个符合输入输出规范的节点类型
    let node_type = random_valid_node_type(rng);  // Tanh / ReLU / Sigmoid / Add / ...
    let new_node = graph.new_hidden_node(node_type)?;

    // 2. 创建新边：src → new_node（权重随机）
    let weight1 = rng.gen_range(-1.0..1.0);
    graph.add_edge(src, new_node, weight1)?;

    // 3. 创建新边：new_node → dst（权重随机）
    let weight2 = rng.gen_range(-1.0..1.0);
    graph.add_edge(new_node, dst, weight2)?;

    // 4. 删除原边
    graph.remove_edge(edge_id)?;

    Ok(new_node)
}

/// 随机选择一个符合语法的节点类型
fn random_valid_node_type(rng: &mut impl Rng) -> NodeType {
    let types = [
        NodeType::Tanh,
        NodeType::ReLU,
        NodeType::Sigmoid,
        NodeType::LeakyReLU { alpha: 0.01 },
        // 未来可扩展更多类型
    ];
    *types.choose(rng).unwrap()
}
```

### 4.3 Add Edge（添加普通边）

在两个没有直接连接的节点间添加**普通边**（前向边）。

**关键约束**：必须保持 DAG，即 `src` 必须在拓扑序上在 `dst` 之前。

**多输入处理**：如果 `dst` 节点已有输入且是固定输入数量的节点（如 Tanh），需要自动插入 `AggregateNode` 来汇合多个输入。

```rust
fn add_edge(graph: &mut Graph, rng: &mut impl Rng) -> Result<EdgeId, GraphError> {
    // 1. 获取所有可能的 (src, dst) 对
    //    - src 必须在拓扑序上在 dst 之前（保持 DAG）← 关键约束！
    //    - src 和 dst 之间没有直接边
    let candidates = graph.get_possible_new_edges();

    // 2. 随机选择一对（调用前已检查非空）
    let (src, dst) = candidates.choose(rng).unwrap();

    // 3. 检查 dst 是否需要聚合节点
    let dst_node = graph.get_node(*dst)?;
    let needs_aggregate = dst_node.has_fixed_inputs() && graph.node_has_parents(*dst);
    
    if needs_aggregate {
        // 3a. dst 是固定输入节点且已有父节点 → 插入聚合节点
        //
        //  Before:  existing_parent ──> dst
        //           src（想连接到 dst）
        //
        //  After:   existing_parent ──┐
        //                             ├──> Agg ──> dst
        //           src ──────────────┘
        //
        let existing_parents = graph.get_node_parents(*dst);
        
        // 创建聚合节点
        let agg = graph.new_aggregate_node(AggregationType::Sum)?;  // 内部 API
        
        // 将原有父节点重定向到聚合节点
        for parent in existing_parents {
            graph.remove_evolution_edge(parent, *dst)?;
            graph.add_evolution_edge(parent, agg)?;
        }
        
        // 新边连接到聚合节点
        graph.add_evolution_edge(*src, agg)?;
        
        // 聚合节点连接到 dst
        graph.add_evolution_edge(agg, *dst)?;
    } else {
        // 3b. dst 支持多输入 或 尚无父节点 → 直接添加边
        graph.add_evolution_edge(*src, *dst)?;
    }

    Ok(())
}

/// 获取所有可添加普通边的候选对（保证不破坏 DAG）
fn get_possible_new_edges(&self) -> Vec<(NodeId, NodeId)> {
    let topo_order = self.topological_sort();
    let mut candidates = Vec::new();

    for (i, &src) in topo_order.iter().enumerate() {
        for &dst in &topo_order[i+1..] {  // dst 必须在 src 之后
            if !self.has_edge(src, dst) {
                candidates.push((src, dst));
            }
        }
    }
    candidates
}
```

> **为什么需要 DAG 约束？** 普通边在同一时间步内立即传递值。如果形成环（A → B → C → A），则 A 的计算需要 C，C 需要 B，B 需要 A——死循环。
>
> **为什么需要聚合节点？** 现有操作节点（如 Tanh）有固定的父节点数量。通过插入 `AggregateNode`，我们可以将多个输入汇合为一个，而不破坏现有节点的计算逻辑。

### 4.4 Add Recurrent Edge（添加循环边）

添加一条**循环边**（跨时间步的连接），用于赋予网络记忆能力。

**与普通边的区别**：
- **无 DAG 约束**：允许任意方向的连接，包括自环（A → A）
- **延迟传递**：值在下一个时间步才可用（通过双缓冲机制）
- **需要 State 节点**：循环边的目标是 State 节点，用于存储上一步的值

```
自环示例（节点 A 指向自己）：

    ┌──────────┐
    │          │
    ▼          │ recurrent
    A ─────────┘
    
实际执行：
    t=0: A = f(input_0 + 0)           → A_old = 0
    t=1: A = f(input_1 + w * A_old)   → A_old = A_{t=0}
    t=2: A = f(input_2 + w * A_old)   → A_old = A_{t=1}
```

```rust
fn add_recurrent_edge(graph: &mut Graph, rng: &mut impl Rng) -> Result<(), GraphError> {
    // 1. 获取所有可添加循环边的候选对
    //    - 源节点：任何非 Input 节点（需要有输出值）
    //    - 目标节点：State 节点（或自动创建）
    //    - 该循环边尚不存在
    let candidates = graph.get_possible_recurrent_edges();

    if candidates.is_empty() {
        return Err(GraphError::NoValidMutation);
    }

    // 2. 随机选择一对
    let (src, dst_state) = candidates.choose(rng).unwrap();

    // 3. 添加循环边，权重随机初始化
    let weight = rng.gen_range(-1.0..1.0);
    graph.connect_recurrent_weighted(*src, *dst_state, weight)?;

    Ok(())
}
```

> **复用现有基础设施**：循环边变异直接复用 `connect_recurrent()` API，该 API 在记忆机制设计中已实现（详见 [memory_mechanism_design.md](./memory_mechanism_design.md)）。

### 4.5 Remove Edge（删除普通边）

```rust
fn remove_edge(graph: &mut Graph, rng: &mut impl Rng) -> Result<(), GraphError> {
    // 1. 获取所有可删除的普通边（调用前已检查非空）
    let removable = graph.get_removable_edges();

    // 2. 随机选择一条边删除
    let edge_id = removable.choose(rng).unwrap();
    graph.remove_edge(*edge_id)?;

    // 3. 清理孤立节点
    graph.remove_orphan_nodes()
}
```

### 4.6 Remove Recurrent Edge（删除循环边）

```rust
fn remove_recurrent_edge(graph: &mut Graph, rng: &mut impl Rng) -> Result<(), GraphError> {
    // 1. 获取所有循环边
    let recurrent_edges = graph.get_recurrent_edges();

    if recurrent_edges.is_empty() {
        return Err(GraphError::NoValidMutation);
    }

    // 2. 随机选择一条循环边删除
    let (src, dst) = recurrent_edges.choose(rng).unwrap();
    graph.disconnect_recurrent(*src, *dst)?;

    // 3. 如果目标 State 节点不再被引用，可选择删除
    graph.remove_orphan_state_nodes()
}
```

### 4.7 Remove Node（删除节点）

```rust
fn remove_node(graph: &mut Graph, rng: &mut impl Rng) -> Result<(), GraphError> {
    // 1. 获取所有隐藏节点（调用前已检查非空）
    let hidden_nodes = graph.get_hidden_nodes();

    // 2. 随机选择一个节点删除
    let node_id = hidden_nodes.choose(rng).unwrap();

    // 3. 删除节点及其所有连接（包括普通边和循环边）
    graph.remove_node(*node_id)
}
```

### 4.8 random_mutation()（完整实现）

**设计原则**：先检查可行性，再选择变异类型，避免失败。

```rust
impl Evolution {
    fn random_mutation(&self, graph: &mut Graph, rng: &mut impl Rng) {
        // 1. 收集所有可行的变异类型
        let mut candidates: Vec<(MutationType, f32)> = Vec::new();

        // 普通边/节点变异
        if graph.has_edges() {
            candidates.push((MutationType::AddNode, self.prob_add_node));
        }
        if graph.has_possible_new_edges() {
            candidates.push((MutationType::AddEdge, self.prob_add_edge));
        }
        if graph.has_removable_edges() {
            candidates.push((MutationType::RemoveEdge, self.prob_remove_edge));
        }
        if graph.has_hidden_nodes() {
            candidates.push((MutationType::RemoveNode, self.prob_remove_node));
        }

        // 循环边变异
        if graph.has_possible_recurrent_edges() {
            candidates.push((MutationType::AddRecurrentEdge, self.prob_add_recurrent_edge));
        }
        if graph.has_recurrent_edges() {
            candidates.push((MutationType::RemoveRecurrentEdge, self.prob_remove_recurrent_edge));
        }

        // 2. 如果没有可行变异，跳过（极端情况）
        if candidates.is_empty() {
            eprintln!("警告：当前网络没有可行的变异操作");
            return;
        }

        // 3. 归一化概率并按权重随机选择
        let total_weight: f32 = candidates.iter().map(|(_, w)| w).sum();
        let threshold = rng.gen::<f32>() * total_weight;
        let mut cumulative = 0.0;
        let mut selected = &candidates[0].0;

        for (mutation_type, weight) in &candidates {
            cumulative += weight;
            if cumulative >= threshold {
                selected = mutation_type;
                break;
            }
        }

        // 4. 执行变异（此时保证成功）
        match selected {
            MutationType::AddNode => {
                let edges = graph.get_edges();
                let edge_id = edges.choose(rng).unwrap();
                let _ = add_node(graph, *edge_id, rng);
            }
            MutationType::AddEdge => {
                let _ = add_edge(graph, rng);
            }
            MutationType::AddRecurrentEdge => {
                let _ = add_recurrent_edge(graph, rng);
            }
            MutationType::RemoveEdge => {
                let _ = remove_edge(graph, rng);
            }
            MutationType::RemoveRecurrentEdge => {
                let _ = remove_recurrent_edge(graph, rng);
            }
            MutationType::RemoveNode => {
                let _ = remove_node(graph, rng);
            }
        }
    }
}
```

---

## 5. Lamarckian 权重继承

### 5.1 什么是 Lamarckian 继承

| 类型 | 子代权重 | 说明 |
|------|---------|------|
| **Darwinian** | 随机初始化 | 不继承父代训练后的权重 |
| **Lamarckian** | 继承父代训练后的权重 | "后天习得"可以遗传 |

### 5.2 在 only_torch 中的实现

**Lamarckian 继承是自然发生的**——未受影响的边保留训练后的权重：

| 边类型 | 权重来源 |
|--------|---------|
| **未受影响的已有边** | 保留训练后的值（Lamarckian 继承） |
| **新创建的边**（包括 Add Node 分裂产生的两条） | 随机初始化 |

```rust
// 示例：Add Node 变异
// 变异前：
//   Input ──0.8──> H1 ──0.6──> Output
//                       ↑
//                 已训练的权重
//
// 变异后（在 H1→Output 上插入 H2）：
//   Input ──0.8──> H1 ──rand──> H2 ──rand──> Output
//           ↑          ↑            ↑
//        保留      新边随机      新边随机
//     (Lamarckian)
```

**设计理由**：
- 未受影响的边保留权重 → 保留已学到的知识
- 新边随机初始化 → 增加探索多样性
- 后续梯度训练会优化所有权重 → 初始值不影响最终结果

---

## 6. 接受策略与回滚机制

### 6.1 决策规则

采用**严格比较**策略：只有当新结构的 loss **严格优于**全局最优时才接受。

```rust
fn should_accept(new_loss: f32, best_loss: f32) -> bool {
    new_loss < best_loss  // 严格小于
}
```

### 6.2 回滚后的行为

**关键设计**：回滚后不是"卡住"，而是"换个方向再试"。

```
全局最优 A（loss=0.30）
    ↓ 随机变异1（比如 Add Edge）
结构 B（loss=0.35）→ 更差，回滚到 A
    ↓ 随机变异2（比如 Add Node，或者 Add Edge 但选了不同的节点对）
结构 C（loss=0.28）→ 更好！更新全局最优 = C
    ...
```

因为变异是**随机**的，每次从同一起点出发可能产生完全不同的结构变化。

### 6.3 为什么不会陷入局部最优？

| 机制 | 作用 |
|------|------|
| **变异类型随机** | 可能是 Add Node / Add Edge / Remove Edge / Remove Node |
| **变异位置随机** | 选择哪条边/哪个节点是随机的 |
| **新边权重随机** | 即使拓扑相同，权重不同可能收敛到不同局部最优 |

这三重随机性保证了探索的多样性。

---

## 7. 关于重复结构

### 7.1 问题描述

理论上，变异可能产生与之前相同的拓扑结构：

```
Step 0:  Input → Output           (结构 A)
Step 1:  Input → Hidden → Output  (结构 B)
Step 2:  Input → Output           (结构 A')  ← 拓扑与 A 相同
```

### 7.2 为什么这不是问题？

| 因素 | 说明 |
|------|------|
| **新边权重随机** | 即使拓扑相同，新边权重不同 → 训练后可能收敛到不同局部最优 |
| **重复概率低** | 变异空间大，连续两次完全相同的变异概率很低 |
| **Lamarckian 继承** | 未受影响的边保留权重 → 即使重复也能快速收敛 |

### 7.3 MVP 策略：不记录历史

对于 MVP，采用最简单的策略——不检测重复：

- **实现简单**
- **随机性提供多样性**
- **即使重复也有探索价值**（权重不同）

### 7.4 后续优化（可选）

如果未来发现大量重复尝试影响效率，可以添加拓扑哈希：

```rust
// 计算拓扑结构的规范化哈希
fn topology_hash(graph: &Graph) -> u64 {
    // 1. 提取邻接表
    // 2. 规范化排序
    // 3. 计算哈希
}

// 记录尝试过的结构
let mut tried_topologies: HashSet<u64> = HashSet::new();

// 变异时跳过已尝试的结构
loop {
    apply_mutation(graph, mutation)?;
    let hash = topology_hash(graph);
    if !tried_topologies.contains(&hash) {
        tried_topologies.insert(hash);
        break;
    } else {
        graph.restore(snapshot)?;  // 回滚，尝试其他变异
    }
}
```

---

## 8. 与现有模块的集成

### 8.1 依赖的现有功能

| 模块 | 功能 | 状态 |
|------|------|:----:|
| `Graph` | 节点/边管理、前向/反向传播 | ✅ 已有 |
| `Graph::on_topology_changed()` | 拓扑变化后清理缓存 | ✅ 已有 |
| `Optimizer` | SGD/Adam 参数更新 | ✅ 已有 |
| `Graph::snapshot()` / `restore()` | 状态保存/恢复 | ⏳ 需要实现 |

### 8.2 需要新增的 Graph API

```rust
impl Graph {
    // ========== 普通边/节点查询 ==========
    fn get_hidden_nodes(&self) -> Vec<NodeId>;                    // ✅ 已实现
    fn get_edges(&self) -> Vec<(NodeId, NodeId)>;                 // ✅ 已实现
    fn get_removable_edges(&self) -> Vec<(NodeId, NodeId)>;       // ✅ 已实现
    fn get_possible_new_edges(&self) -> Vec<(NodeId, NodeId)>;    // ✅ 已实现
    fn has_edges(&self) -> bool;                                   // ✅ 已实现
    fn has_removable_edges(&self) -> bool;                         // ✅ 已实现

    // ========== 循环边查询 ==========
    fn get_recurrent_edges(&self) -> Vec<(NodeId, NodeId)>;
    fn get_possible_recurrent_edges(&self) -> Vec<(NodeId, NodeId)>;  // 无 DAG 约束
    fn has_recurrent_edges(&self) -> bool;
    fn has_possible_recurrent_edges(&self) -> bool;

    // ========== 普通边/节点修改 ==========
    fn add_evolution_edge(&mut self, src: NodeId, dst: NodeId) -> Result<(), GraphError>;   // ✅ 已实现
    fn remove_evolution_edge(&mut self, src: NodeId, dst: NodeId) -> Result<(), GraphError>; // ✅ 已实现
    fn remove_evolution_node(&mut self, node_id: NodeId) -> Result<(), GraphError>;          // ✅ 已实现
    fn remove_orphan_nodes(&mut self) -> Result<usize, GraphError>;                          // ✅ 已实现

    // ========== 聚合节点（内部 API，不暴露给用户）==========
    // 在 add_evolution_edge 检测到需要时自动调用
    pub(crate) fn new_aggregate_node(&mut self, agg_type: AggregationType) -> Result<NodeId, GraphError>;
    pub(crate) fn node_has_parents(&self, node_id: NodeId) -> bool;
    pub(crate) fn node_has_fixed_inputs(&self, node_id: NodeId) -> bool;

    // ========== 循环边修改（复用现有记忆机制 API）==========
    fn connect_recurrent_weighted(&mut self, src: NodeId, dst: NodeId, weight: f32) -> Result<(), GraphError>;
    fn disconnect_recurrent(&mut self, src: NodeId, dst: NodeId) -> Result<(), GraphError>;
    fn remove_orphan_state_nodes(&mut self) -> Result<(), GraphError>;

    // ========== 状态管理 ==========
    fn snapshot(&self) -> Result<GraphSnapshot, GraphError>;
    fn restore(&mut self, snapshot: GraphSnapshot) -> Result<(), GraphError>;
}

/// 快照需要包含循环边状态
pub struct GraphSnapshot {
    node_values: HashMap<NodeId, Tensor>,
    node_grads: HashMap<NodeId, Tensor>,
    recurrent_edges: HashMap<NodeId, NodeId>,  // 循环边拓扑
    prev_values: HashMap<NodeId, Tensor>,       // 循环状态（双缓冲）
}
```

---

## 9. MVP 验收标准

### 9.1 XOR 进化测试

| 指标 | 目标 |
|------|------|
| **任务** | XOR 二分类 |
| **初始结构** | 2 输入 → 1 输出（无隐藏层） |
| **目标 Accuracy** | ≥ 100%（4 个样本全部正确） |
| **评估次数** | 3 次取最低值 |
| **终止条件** | 仅以任务指标达标为准 |
| **成功率** | > 90%（10 次独立运行） |

### 9.2 测试用例

```rust
#[test]
fn test_xor_evolution() {
    // 最小初始结构
    let mut graph = Graph::new_with_seed(42);
    let input = graph.new_input_node(&[4, 2], Some("input"))?;
    let output = graph.new_parameter_node(&[2, 1], Some("output"))?;
    let pred = graph.new_matmul_node(input, output, None)?;

    // XOR 数据
    let (train_data, test_data) = xor_dataset();

    // 创建任务
    let task = SupervisedTask::new(train_data, test_data, Metric::Accuracy);

    // 运行演化（Only Touch：两行代码）
    let trained = Evolution::new(graph, task)
        .run();

    // 验证达标
    let task = SupervisedTask::new(train_data, test_data, Metric::Accuracy);
    assert!(task.evaluate(&trained) >= 1.0);
}

#[test]
fn test_xor_evolution_custom_config() {
    let graph = Graph::minimal_structure(2, 1);
    let (train_data, test_data) = xor_dataset();

    // 自定义配置
    let trained = Evolution {
        graph,
        task: Box::new(SupervisedTask::new(train_data, test_data, Metric::Accuracy)),
        target_metric: 1.0,
        patience: 3,  // XOR 简单任务用更小的 patience
        ..Default::default()
    }.run();

    assert!(trained.is_valid());
}
```

---

## 10. 适用范围

### 10.1 MVP 支持

| 场景 | 支持程度 | 说明 |
|------|---------|------|
| **Supervised Learning** | ✅ 完全支持 | 单网络演化，标准训练流程 |
| **RL（单网络 Policy）** | ✅ 完全支持 | 只演化 Policy 网络 |
| **RL（Actor-Critic）** | ⚠️ 部分支持 | 只演化 Actor，Critic 固定或手动管理 |
| **GAN** | ⚠️ 部分支持 | 只演化 Generator，Discriminator 固定或手动管理 |

### 10.2 复杂场景的处理

对于需要**自定义训练逻辑**的场景（如自定义 PPO/SAC、GAN 训练技巧），用户可实现 `EvolutionTask` trait：

```rust
// 示例：用户自定义的 PPO 训练
struct MyPPO {
    env: CartPole,
    // ... 自定义 PPO 参数
}

impl EvolutionTask for MyPPO {
    fn train_until_converged(&self, graph: &mut Graph, config: &ConvergenceConfig) -> f32 {
        // 用户完全控制训练逻辑
        // 可以是 PPO、SAC、DDPG、任何算法
        let mut detector = ConvergenceDetector::new(config);
        loop {
            let loss = self.ppo_update(graph);  // 用户的 PPO 逻辑
            if detector.is_converged(loss, /* grad_norm */) {
                return loss;
            }
        }
    }

    fn evaluate(&self, graph: &Graph) -> f32 {
        // 用户定义评估：跑 N 个 episode 取平均
        (0..10).map(|_| self.env.run_episode(graph)).sum::<f32>() / 10.0
    }
}

// 使用
let trained = Evolution::new(graph, MyPPO::new(env)).run();
```

---

## 11. 实现路径

| Phase | 任务 | 验收 | 状态 |
|:-----:|------|------|:----:|
| **1** | Graph 拓扑查询/修改 API（普通边） | 单元测试：查询/添加/删除节点边 | ✅ 完成 |
| **1b** | 聚合节点自动插入（复用 Add 节点） | 单元测试：固定输入检测/自动插入 | ✅ 完成 |
| **2** | Graph 循环边 API（复用记忆机制） | 单元测试：添加/删除循环边 | ⏳ 待实现 |
| **3** | Graph 状态快照/恢复 | 单元测试：snapshot/restore（含循环状态） | ⏳ 待实现 |
| **4** | 变异操作实现 | 单元测试：6 种变异 | ⏳ 待实现 |
| **5** | `EvolutionTask` trait + `SupervisedTask` | 单元测试：训练+评估 | ⏳ 待实现 |
| **6** | `Evolution` 结构体 + `.run()` | 集成测试：XOR 进化 | ⏳ 待实现 |
| **7** | 记忆任务验证（可选） | 集成测试：奇偶性检测（需要循环边） | ⏳ 待实现 |

**Phase 1 已完成的 API**：
- `get_hidden_nodes()` — 获取可删除的操作节点
- `has_edges()` / `get_edges()` — 边查询
- `get_possible_new_edges()` — 获取可安全添加的新边候选（保持 DAG）
- `get_removable_edges()` / `has_removable_edges()` — 可删除边查询
- `add_evolution_edge()` — 添加边（带 DAG 约束和环检测）
- `remove_evolution_edge()` — 删除边
- `remove_evolution_node()` — 删除节点及其连接
- `remove_orphan_nodes()` — 清理孤立节点（链式删除）

---

## 12. 未来扩展（Optional）

> 以下功能为未来可选扩展，MVP 阶段不实现。

### 12.1 多网络演化

支持同时演化多个网络（Actor-Critic、GAN）：

```rust
// 方案 A：MultiEvolution 结构体
pub struct MultiEvolution {
    graphs: Vec<Graph>,  // 多个网络
    task: Box<dyn MultiEvolutionTask>,
    // ...
}

pub trait MultiEvolutionTask {
    fn train_until_converged(&self, graphs: &mut [Graph], config: &ConvergenceConfig) -> f32;
    fn evaluate(&self, graphs: &[Graph]) -> f32;
}

// 使用示例
let (trained_actor, trained_critic) = MultiEvolution::new(
    vec![actor, critic],
    ActorCriticTask::new(env),
).run();
```

```rust
// 方案 B：Builder 模式 + 闭包
let (trained_g, trained_d) = Evolution::new_multi([generator, discriminator])
    .with_train_fn(|graphs| {
        // 用户的 GAN 训练逻辑
        let [g, d] = graphs;
        for _ in 0..d_steps { train_discriminator(d, g); }
        for _ in 0..g_steps { train_generator(g, d); }
        compute_loss(g, d)
    })
    .with_eval_fn(|graphs| {
        // FID / IS 等评估
        compute_fid(graphs[0])
    })
    .run();
```

### 12.2 Layer 层次演化

支持更粗粒度的"宏变异"：

```rust
pub enum MutationType {
    // Node 层次（细粒度，当前支持）
    AddNode,
    AddEdge,
    RemoveNode,
    RemoveEdge,

    // Layer 层次（粗粒度，未来可选）
    AddDenseLayer { units: usize },
    AddConvLayer { filters: usize, kernel_size: usize },
    AddLSTMLayer { hidden_size: usize },
    RemoveLayer { layer_id: LayerId },
}
```

**优势**：搜索空间更小，收敛更快
**劣势**：限制了结构多样性

### 12.3 并行演化

支持多个演化实例并行运行：

```rust
// 岛屿模型：多个种群并行演化，定期交换最优个体
let results = Evolution::parallel(
    num_islands: 4,
    graph_factory: || Graph::minimal_structure(2, 1),
    task: SupervisedTask::new(...),
    migration_interval: 10,  // 每 10 次变异交换一次
).run();
```

### 12.4 协同演化

支持多个网络协同演化（如 GAN 的 G 和 D）：

```rust
pub struct CoEvolution {
    populations: Vec<Vec<Graph>>,  // 多个种群
    fitness_fn: Box<dyn Fn(&[Graph]) -> Vec<f32>>,  // 协同适应度
}

// 使用
let (best_g, best_d) = CoEvolution::new()
    .add_population(generators)
    .add_population(discriminators)
    .with_fitness(|[g, d]| {
        // G 和 D 的适应度相互依赖
        (g_fitness, d_fitness)
    })
    .run();
```

---

## 13. 参考资料

### 论文

- **NEAT**：[Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) — [本地笔记](../paper/NEAT_2002/summary.md)
- **EXAMM**：[Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution](https://arxiv.org/abs/1902.02390) — [本地笔记](../paper/EXAMM_2019/summary.md)

### 开源实现

- **radiate**（Rust）：https://github.com/pkalivas/radiate
- **neat-python**：https://github.com/CodeReclaimers/neat-python
- **EXACT/EXAMM**（C++）：https://github.com/travisdesell/exact

---

## 14. 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| Evolution | Evolution | 一次完整的神经架构演化（包含 Graph + Task + 配置） |
| EvolutionTask | EvolutionTask | 抽象不同学习范式的训练+评估方式 |
| 拓扑变异 | Topology Mutation | 改变网络结构（添加/删除节点边） |
| Lamarckian 继承 | Lamarckian Inheritance | 未受影响的边保留训练后的权重 |
| 收敛 | Convergence | Loss 不再显著下降的状态（优化层判定） |
| 达标 | Task Complete | 任务指标满足用户要求（任务层判定） |
| 全局最优 | Global Best | 历史上 loss 最低的结构（含参数快照） |
| 添加节点 | Add Node | 在已有边上插入新节点（NEAT 标准术语） |
| 添加边 | Add Edge | 在两个未直连的节点间添加新普通边（保持 DAG） |
| 普通边 | Forward Edge | 同一时间步内立即传递值的边，必须保持 DAG |
| 循环边 | Recurrent Edge | 跨时间步传递值的边，允许形成图论环（记忆机制） |
| 自环 | Self-Loop | 节点指向自己的循环边，最简单的记忆单元 |
| 双缓冲 | Double Buffering | 循环边避免死循环的机制：读上一步值，写当前值 |
| 聚合节点 | AggregateNode | 内部节点类型，将多个输入汇合为一个输出（仅演化过程使用） |
| 聚合函数 | Aggregation Function | 将多个输入合并的函数（sum/mean/max/...），源自 NEAT 设计 |

---

*本文档描述了 only_torch 的神经架构演化机制，采用 NEAT 风格拓扑变异 + 梯度训练的混合策略，支持循环边（记忆机制）的演化。*

*最后更新：2026-01-27*

---

## 附录 A：聚合节点实现

### A.1 设计背景

在 NEAT 原始论文中，每个隐藏节点的计算方式是：

```
output = activation(Σ(w_i × input_i) + bias)
```

即**先聚合（加权求和）**，**再激活**。这意味着节点天然支持任意数量的输入。

而在我们的架构中，操作节点（如 `TanhNode`）有固定的父节点数量：

```rust
// TanhNode 的计算
fn calc_value_by_parents(&mut self, parents: &[NodeHandle]) {
    let input = parents[0].value();  // 只取第一个！
    self.value = input.tanh();
}
```

当 `add_evolution_edge()` 想向一个已有输入的 `TanhNode` 添加新边时，就会出现问题。

### A.2 实现方案：复用 Add 节点

经过评估，我们选择**复用现有的 Add 节点**作为聚合节点，而不是创建新的节点类型。

**理由**：
- Add 节点已经支持多输入和求和逻辑
- 避免增加新的节点类型，保持架构简洁
- 演化创建的聚合节点使用特殊命名前缀（`_evo_agg`）以便识别

### A.3 使用场景

```
场景：向已有输入的 Tanh 节点添加新边

Before:
    A ──────> Tanh ──> Output
              ↑
          已有 1 个父节点

想添加：B ──> Tanh

After:
    A ──────┐
            ├──> Add ──> Tanh ──> Output
    B ──────┘
              ↑
         Add 节点（名称: _evo_agg）
```

### A.4 核心实现

```rust
impl GraphInner {
    /// 检查节点是否有固定的父节点数量
    pub fn expected_parent_count(&self, node_id: NodeId) -> Option<usize> {
        match node.node_type() {
            // 单输入节点：1 个父节点
            NodeType::Tanh(_) | NodeType::Sigmoid(_) | ... => Some(1),
            // 双输入节点：2 个父节点
            NodeType::MatMul(_) | NodeType::Divide(_) | ... => Some(2),
            // 可变输入节点：None
            NodeType::Add(_) => None,
            ...
        }
    }

    pub fn add_evolution_edge(&mut self, src: NodeId, dst: NodeId) -> Result<(), GraphError> {
        // ...验证...
        
        // 检查是否需要插入聚合节点
        let needs_aggregate = self.has_fixed_parent_count(dst) 
                           && self.evolution_node_has_parents(dst);

        if needs_aggregate {
            // 收集所有父节点（包括新的 src）
            let mut all_parents = self.evolution_get_parents(dst);
            all_parents.push(src);

            // 创建 Add 节点
            let agg_node = self.new_add_node(&all_parents, Some("_evo_agg"))?;

            // 重新连接
            // ...
        } else {
            // 直接添加边
            self.add_evolution_edge_internal(src, dst)?;
        }
        Ok(())
    }
}
```

### A.5 测试覆盖

| 测试 | 验证内容 |
|------|----------|
| `test_aggregate_node_auto_insert` | 向固定输入节点添加边时自动插入 Add |
| `test_aggregate_node_not_inserted_for_add_node` | Add 节点不触发聚合插入 |
| `test_aggregate_node_not_inserted_for_first_edge` | 无父节点时不触发聚合插入 |
| `test_expected_parent_count` | 各节点类型的预期父节点数量 |

### A.6 未来扩展

如果需要支持不同的聚合策略（mean, max, product），可以：

1. 添加新的节点类型（`MeanNode`, `MaxNode` 等）
2. 或在演化配置中添加聚合策略选项

当前 MVP 使用 sum（Add 节点），与 NEAT 论文默认行为一致。

---

## 附录 B：解耦性分析

### B.1 架构层次

```
┌─────────────────────────────────────────────────────────────────┐
│  用户层（PyTorch 风格）                                          │
│    Module (Linear, Rnn, Conv2d...)                               │
│    ModelState, Var, ForwardInput                                 │
│    examples/ 中的所有模型                                         │
├─────────────────────────────────────────────────────────────────┤
│  句柄层                                                          │
│    Graph（Rc<RefCell<GraphInner>> 的薄封装）                      │
├─────────────────────────────────────────────────────────────────┤
│  底层（NEAT 演化操作于此层）                                      │
│    GraphInner                                                    │
│    NodeHandle, forward_edges, backward_edges                     │
│    ← AggregateNode 在这里                                        │
└─────────────────────────────────────────────────────────────────┘
```

### B.2 解耦性验证

| 问题 | 影响 | 说明 |
|------|:----:|------|
| **MIMO 结构** | ❌ 不影响 | AggregateNode 在底层，用户通过 `Module` 和 `Var` 操作，两者完全解耦 |
| **高层级进化** | ❌ 不影响 | Node 层次和 Layer 层次是独立的抽象，互不干扰 |
| **PyTorch 风格** | ❌ 不影响 | 用户通过 Module/Var 操作，不接触底层节点 |

### B.3 设计原则

1. **AggregateNode 是 NEAT 演化的内部实现细节**
   - `pub(crate)` 可见性，用户无法直接创建
   - 只在 `add_evolution_edge()` 检测到需要时自动插入

2. **用户 API 保持不变**
   - `Graph`, `Module`, `Var` 等高层抽象不受影响
   - 现有示例代码无需修改

3. **未来扩展安全**
   - Layer 层次演化操作 Module，不涉及底层节点
   - 并行演化、协同演化等高级功能不受影响

### B.4 与 PyTorch 风格 Module 联合使用（Optional）

> ⚠️ 此功能为**可选扩展**，MVP 阶段不实现。

**场景**：用户希望用 PyTorch 风格定义初始网络，然后运行 NEAT 演化。

**问题**：Module 的 `forward()` 是静态代码，不会使用演化新增的节点。

**可选方案**：

```rust
// 方案 A：提供 DynamicModule（动态前向传播）
pub struct DynamicModule {
    graph: Graph,
    input_nodes: Vec<NodeId>,
    output_nodes: Vec<NodeId>,
}

impl DynamicModule {
    /// 从演化后的 Graph 创建动态模块
    pub fn from_evolved(graph: Graph) -> Self { ... }
    
    /// 动态前向传播：根据图结构自动计算
    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Var> { ... }
}

// 使用
let evolved_graph = Evolution::new(graph, task).run();
let model = DynamicModule::from_evolved(evolved_graph);
let output = model.forward(&[input]);
```

```rust
// 方案 B：从 GraphInner 重建 Module 描述
pub fn describe_evolved_network(graph: &Graph) -> NetworkDescription {
    // 返回演化后网络的拓扑描述
    // 用户可据此手动重建 Module
}
```

**MVP 建议**：NEAT 演化直接操作底层 `GraphInner`，不与 Module 混用。

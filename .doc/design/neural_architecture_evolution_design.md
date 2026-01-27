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

### 1.2 设计哲学

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

### 1.3 模块位置

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

### 1.4 演化颗粒度

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
    pub prob_add_node: f32,       // 默认 0.20
    pub prob_add_edge: f32,       // 默认 0.40
    pub prob_remove_edge: f32,    // 默认 0.30
    pub prob_remove_node: f32,    // 默认 0.10
}

impl Evolution {
    /// 便捷构造：只需 Graph 和 Task，其他用默认值
    pub fn new(graph: Graph, task: impl EvolutionTask + 'static) -> Self {
        Self {
            graph,
            task: Box::new(task),
            // 默认值（2024 论文验证的工程最优）
            target_metric: 0.95,
            eval_runs: 3,
            loss_tolerance: 1e-4,
            grad_tolerance: 1e-5,
            patience: 5,
            prob_add_node: 0.20,
            prob_add_edge: 0.40,
            prob_remove_edge: 0.30,
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
| **Add Node** | 添加节点 | 在已有边上插入新节点 | 20% |
| **Add Edge** | 添加边 | 在两个节点间添加新连接 | 40% |
| **Remove Edge** | 删除边 | 删除一条边 | 30% |
| **Remove Node** | 删除节点 | 删除隐藏节点及其连接 | 10% |

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

### 4.3 Add Edge（添加边）

在两个没有直接连接的节点间添加新边：

```rust
fn add_edge(graph: &mut Graph, rng: &mut impl Rng) -> EdgeId {
    // 1. 获取所有可能的 (src, dst) 对
    //    - src 必须在拓扑序上在 dst 之前（保持 DAG）
    //    - src 和 dst 之间没有直接边
    let candidates = graph.get_possible_new_edges();

    // 2. 随机选择一对（调用前已检查非空）
    let (src, dst) = candidates.choose(rng).unwrap();

    // 3. 添加边，权重随机初始化
    let weight = rng.gen_range(-1.0..1.0);
    graph.add_edge(*src, *dst, weight)
}
```

### 4.4 Remove Edge（删除边）

```rust
fn remove_edge(graph: &mut Graph, rng: &mut impl Rng) {
    // 1. 获取所有可删除的边（调用前已检查非空）
    let removable = graph.get_removable_edges();

    // 2. 随机选择一条边删除
    let edge_id = removable.choose(rng).unwrap();
    graph.remove_edge(*edge_id);

    // 3. 清理孤立节点
    graph.remove_orphan_nodes();
}
```

### 4.5 Remove Node（删除节点）

```rust
fn remove_node(graph: &mut Graph, rng: &mut impl Rng) {
    // 1. 获取所有隐藏节点（调用前已检查非空）
    let hidden_nodes = graph.get_hidden_nodes();

    // 2. 随机选择一个节点删除
    let node_id = hidden_nodes.choose(rng).unwrap();

    // 3. 删除节点及其所有连接
    graph.remove_node(*node_id);
}
```

### 4.6 random_mutation()（完整实现）

**设计原则**：先检查可行性，再选择变异类型，避免失败。

```rust
impl Evolution {
    fn random_mutation(&self, graph: &mut Graph, rng: &mut impl Rng) {
        // 1. 收集所有可行的变异类型
        let mut candidates: Vec<(MutationType, f32)> = Vec::new();

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
                add_node(graph, *edge_id, rng);
            }
            MutationType::AddEdge => {
                add_edge(graph, rng);
            }
            MutationType::RemoveEdge => {
                remove_edge(graph, rng);
            }
            MutationType::RemoveNode => {
                remove_node(graph, rng);
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
    // 拓扑查询
    fn get_hidden_nodes(&self) -> Vec<NodeId>;
    fn get_removable_edges(&self) -> Vec<EdgeId>;
    fn get_possible_new_edges(&self) -> Vec<(NodeId, NodeId)>;

    // 拓扑修改
    fn add_edge(&mut self, src: NodeId, dst: NodeId, weight: f32) -> Result<EdgeId, GraphError>;
    fn remove_edge(&mut self, edge_id: EdgeId) -> Result<(), GraphError>;
    fn remove_node(&mut self, node_id: NodeId) -> Result<(), GraphError>;
    fn remove_orphan_nodes(&mut self) -> Result<(), GraphError>;

    // 状态管理
    fn snapshot(&self) -> Result<GraphSnapshot, GraphError>;
    fn restore(&mut self, snapshot: GraphSnapshot) -> Result<(), GraphError>;
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

| Phase | 任务 | 验收 |
|:-----:|------|------|
| **1** | Graph 拓扑修改 API | 单元测试：添加/删除节点边 |
| **2** | Graph 状态快照/恢复 | 单元测试：snapshot/restore |
| **3** | 变异操作实现 | 单元测试：4 种变异 |
| **4** | `EvolutionTask` trait + `SupervisedTask` | 单元测试：训练+评估 |
| **5** | `Evolution` 结构体 + `.run()` | 集成测试：XOR 进化 |

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
| 添加边 | Add Edge | 在两个未直连的节点间添加新边 |

---

*本文档描述了 only_torch 的神经架构演化机制，采用 NEAT 风格拓扑变异 + 梯度训练的混合策略。*

*最后更新：2026-01-25*

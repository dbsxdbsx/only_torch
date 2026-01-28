# 神经架构演化设计（层级进化）

> **设计理念**：单网络迭代演化，结合 **层级拓扑变异** 与 **梯度训练** 的混合策略。
>
> **定位**：项目愿景的核心功能，区别于传统固定架构的深度学习框架。

---

## 1. 核心理念

### 1.1 设计哲学

```
┌─────────────────────────────────────────────────────────────────┐
│  进化负责"探索"：什么样的网络结构可能更好？                        │
│  梯度负责"利用"：给定结构，找到最优权重                            │
│  两者交替：结构变异 → 梯度训练 → 评估 → 结构变异 → ...             │
└─────────────────────────────────────────────────────────────────┘
```

这种混合策略来自 [EXAMM 论文](../paper/EXAMM_2019/summary.md)：

> **梯度下降比进化快得多**（权重优化）
> **但梯度下降无法搜索结构**（需要进化）
> → 各取所长：进化搜索结构 + 梯度优化权重

### 1.2 为什么选择层级进化（而非节点级）

| 方面 | 节点级进化 | 层级进化 |
|------|-----------|---------|
| **粒度** | 单个操作节点 | 完整层（Linear, RNN, LSTM...） |
| **参数管理** | 需手动处理 Parameter | 层自动管理权重 |
| **搜索空间** | 巨大，收敛慢 | 适中，更高效 |
| **实现复杂度** | 高 | 低 |
| **与现有 API 兼容** | 需要新设计 | 直接复用 Linear/RNN 等 |
| **运行效率** | 逐节点计算 | 矩阵批量计算（更快） |

### 1.3 层级进化的表达能力

通过层级进化 + 超参数变异，可以覆盖节点级 NEAT 的 **95%+** 功能：

| 节点级能力 | 层级对应方式 | 能否实现 |
|-----------|-------------|:--------:|
| 添加激活节点 | `InsertLayer(Activation)` | ✅ |
| 添加带权重连接 | `InsertLayer(Linear)` | ✅ |
| 删除连接/节点 | `RemoveLayer` | ✅ |
| 改变网络深度 | Insert/Remove 层 | ✅ |
| 改变网络宽度 | `MutateHiddenSize` | ✅ |
| 添加记忆/循环 | `InsertLayer(RNN/LSTM)` | ✅ |
| 跳跃连接 | `InsertLayer(SkipConnection)` | ✅ |

---

## 2. 主循环设计

### 2.1 单网络迭代演化

```
┌─────────────────────────────────────────────────────────────────┐
│  Evolution::run()  —— Only Touch                                │
│                                                                 │
│    输入：self（包含 graph 所有权 + task + 配置）                  │
│    输出：Graph（演化后的网络所有权）                              │
│                                                                 │
│    loop {                                                       │
│        ┌─────────────────────────────────────────────────────┐  │
│        │  task.train_until_converged()  —— 优化层            │  │
│        │    梯度下降直到收敛                                  │  │
│        │    return loss                                      │  │
│        └─────────────────────────────────────────────────────┘  │
│                                                                 │
│        ┌─────────────────────────────────────────────────────┐  │
│        │  task.evaluate() × N  —— 任务层                     │  │
│        │    N 次评估取最低值 >= 目标？                        │  │
│        │    └─ Yes → return graph（成功！）                  │  │
│        └─────────────────────────────────────────────────────┘  │
│                                                                 │
│        更新全局最优 / 回滚                                       │
│        随机变异                                                  │
│    }                                                            │
│                                                                 │
│  终止条件：任务指标达标                                          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 两层判定标准

| 层级 | 指标 | 作用 |
|------|------|------|
| **任务层** | Accuracy / F1 / Episode 奖励等 | 决定是否**彻底结束** |
| **优化层** | Loss 变化率 / 梯度范数 | 决定是否**收敛**（该换拓扑了） |

**关键区分**：
- **收敛** ≠ **达标**
- 收敛只是说明"当前拓扑已榨干"，需要变异换个结构继续尝试
- 达标才是真正完成任务

### 2.3 EvolutionTask trait

不同学习范式的"训练"和"评估"方式不同，通过 trait 抽象：

```rust
/// 演化任务 trait：抽象不同学习范式
pub trait EvolutionTask {
    /// 训练直到收敛，返回 loss
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

---

## 3. 可演化的层类型

### 3.1 MVP 必须实现

| 层类型 | 描述 | 可变异参数 |
|--------|------|-----------|
| **Linear** | 全连接层 | `out_features` |
| **Activation** | 激活函数 | `type` (ReLU, Tanh, Sigmoid, GELU...) |
| **RNN** | 简单循环层 | `hidden_size` |
| **LSTM** | 长短期记忆 | `hidden_size` |
| **GRU** | 门控循环单元 | `hidden_size` |

### 3.2 可选扩展

| 层类型 | 描述 | 可变异参数 |
|--------|------|-----------|
| **Conv2d** | 卷积层 | `out_channels`, `kernel_size` |
| **Dropout** | 正则化 | `p` (dropout 概率) |

### 3.3 跳跃边与 AggregateNode

#### 什么是跳跃边？

**跳跃边**是一条绕过中间层、直接连接两个非相邻节点的**无参数边**（identity connection）。

```
普通连接：  Input → Layer1 → Layer2 → Layer3 → Output
                      ↓        ↓        ↓
                   数据依次流过每一层

跳跃边：    Input ─────────────────────────┐
              │                            │  ← 跳跃边（无参数，直接传数据）
              └─> Layer1 → Layer2 → Layer3 ─┴─> Output
```

**关键特性**：
- **无参数**：跳跃边本身不带权重，只传递原始数据（与 ResNet/DenseNet 一致）
- **任意跳跃**：可以从任意节点跳到任意后续节点，自由度高于 ResNet（固定间隔）和 DenseNet（全连接）

#### 什么是 AggregateNode？

当多条边汇入同一节点时，需要一种方式来**聚合**这些输入。`AggregateNode` 就是专门做这件事的节点。

```
问题：Layer3 现在有两个输入（Layer2 的输出 + Input 的跳跃），怎么合并？

解决：插入 AggregateNode

Input ─────────────────────────┐
  │                            │
  └─> Layer1 → Layer2 → Layer3 ─┴─> AggregateNode → Output
                                         ↑
                                    聚合多个输入
```

#### 聚合策略

| 策略 | 公式 | 典型应用 | 维度要求 |
|------|------|----------|----------|
| **Add** | `y = x₁ + x₂` | ResNet 残差 | 必须相同 |
| **Concat** | `y = [x₁, x₂]` | DenseNet 密集 | 可以不同 |
| **Mean** | `y = (x₁ + x₂) / 2` | 模型集成 | 必须相同 |
| **Max** | `y = max(x₁, x₂)` | 特征选择 | 必须相同 |

```rust
pub enum AggregateStrategy {
    Add,                    // 默认，ResNet 风格
    Concat { dim: i32 },    // DenseNet 风格
    Mean,
    Max,
}
```

#### 演化如何产生跳跃边？

通过 `AddSkipEdge` 变异操作，系统会**自动插入 AggregateNode**：

```
变异前：Input → Linear → ReLU → Linear → Output

执行 AddSkipEdge(Input, Output前的位置)：

变异后：Input ────────────────────────────┐
          │                               │
          └─> Linear → ReLU → Linear ─────┴─> Aggregate(Add) → Output
                                                     ↑
                                              自动插入的聚合节点
```

#### 能演化出什么？

| 结构 | 跳跃模式 | 我们能否演化出 |
|------|----------|:--------------:|
| **ResNet** | 每隔 2-3 层跳跃 + Add | ✅ |
| **DenseNet** | 每层连到后续所有层 + Concat | ✅ |
| **任意拓扑** | 选择性跳跃（如只跳到第 3、4 层） | ✅ |

**这就是"层级 + 边级"混合进化的威力**：不预设固定模式，让演化自己探索最优拓扑。

#### 相关变异操作

| 操作 | 作用 |
|------|------|
| `AddSkipEdge` | 添加跳跃边（自动插入 AggregateNode） |
| `RemoveSkipEdge` | 移除跳跃边 |
| `MutateAggregateStrategy` | 改变聚合策略（Add↔Concat...） |

---

## 4. 变异操作

### 4.1 拓扑变异

| 操作 | 描述 | 示例 |
|------|------|------|
| **InsertLayer** | 在两层之间插入新层 | 插入 `Linear(4)` |
| **RemoveLayer** | 移除某一层 | 移除中间的 Activation |
| **ReplaceLayerType** | 替换层类型 | `ReLU` → `GELU` |

### 4.2 跳跃连接变异

| 操作 | 描述 | 示例 |
|------|------|------|
| **AddSkipEdge** | 添加跳跃边（自动插入 AggregateNode） | 从 Layer1 跳到 Layer3 |
| **RemoveSkipEdge** | 移除跳跃边 | 移除残差连接 |
| **MutateAggregateStrategy** | 改变聚合策略 | `Add` → `Concat` |

```
AddSkipEdge 示例：

Before:  Input → Linear → ReLU → Linear → Output

After:   Input → Linear → ReLU → Linear ──┐
           │                              ├──> Aggregate(Add) → Output
           └──────────────────────────────┘
```

### 4.3 超参数变异

| 操作 | 描述 | 示例 |
|------|------|------|
| **GrowHiddenSize** | 增大隐藏层 | `Linear(32)` → `Linear(64)` |
| **ShrinkHiddenSize** | 缩小隐藏层 | `Linear(64)` → `Linear(32)` |
| **MutateActivationType** | 改变激活函数类型 | `ReLU` → `Tanh` |

### 4.4 变异概率（默认值）

| 变异类型 | 概率 | 说明 |
|----------|------|------|
| InsertLayer | 0.12 | 增加网络深度 |
| RemoveLayer | 0.12 | 减少网络深度 |
| ReplaceLayerType | 0.08 | 替换层类型 |
| AddSkipEdge | 0.08 | 添加跳跃连接 |
| RemoveSkipEdge | 0.05 | 移除跳跃连接 |
| MutateAggregateStrategy | 0.05 | 改变聚合策略 |
| GrowHiddenSize | 0.20 | 增大隐藏层 |
| ShrinkHiddenSize | 0.25 | 缩小隐藏层（鼓励简化） |
| OtherHyperparameter | 0.05 | 其他超参数变异 |

> **设计理念**：`ShrinkHiddenSize` 概率略高于 `GrowHiddenSize`，鼓励演化出更紧凑的网络。跳跃连接相关变异概率较低，避免网络过早变得复杂。

---

## 5. 网络规模控制

### 5.1 从最小结构开始

遵循 NEAT "从最小开始"原则：

```
初始网络：Input → Linear(1) → Output
           ↑
      最简结构

逐步演化：
  → Input → Linear(1) → ReLU → Output
  → Input → Linear(4) → ReLU → Output
  → Input → Linear(4) → ReLU → Linear(2) → Output
  → ...
```

### 5.2 规模上限约束

硬性限制网络最大规模：

```rust
pub struct SizeConstraints {
    /// 最大层数（默认 10）
    pub max_layers: usize,
    /// 最大隐藏层大小（默认 64）
    pub max_hidden_size: usize,
    /// 最大总参数量（默认 10000）
    pub max_total_params: usize,
}
```

超出限制的变异会被拒绝，网络不会失控增长。

### 5.3 缩减操作

支持网络简化：

| 操作 | 描述 |
|------|------|
| `RemoveLayer` | 删除一个非必要层 |
| `ShrinkHiddenSize` | 缩小隐藏层（减半、减 1） |

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
    ↓ 随机变异1（比如 InsertLayer）
结构 B（loss=0.35）→ 更差，回滚到 A
    ↓ 随机变异2（比如 GrowHiddenSize）
结构 C（loss=0.28）→ 更好！更新全局最优 = C
    ...
```

因为变异是**随机**的，每次从同一起点出发可能产生完全不同的结构变化。

### 6.3 为什么不会陷入局部最优？

| 机制 | 作用 |
|------|------|
| **变异类型随机** | 可能是 InsertLayer / RemoveLayer / GrowHiddenSize / ... |
| **变异位置随机** | 选择哪一层是随机的 |
| **新层参数随机** | 即使拓扑相同，参数不同可能收敛到不同局部最优 |

---

## 7. Lamarckian 权重继承

### 7.1 什么是 Lamarckian 继承

| 类型 | 子代权重 | 说明 |
|------|---------|------|
| **Darwinian** | 随机初始化 | 不继承父代训练后的权重 |
| **Lamarckian** | 继承父代训练后的权重 | "后天习得"可以遗传 |

### 7.2 在层级进化中的实现

**Lamarckian 继承是自然发生的**——未受影响的层保留训练后的权重：

| 层类型 | 权重来源 |
|--------|---------|
| **未受影响的已有层** | 保留训练后的值（Lamarckian 继承） |
| **新插入的层** | 随机初始化 |

```rust
// 示例：InsertLayer 变异
// 变异前：
//   Input → Linear(8) [已训练] → Output
//
// 变异后（插入 ReLU）：
//   Input → Linear(8) [保留] → ReLU [新增] → Output
//              ↑                   ↑
//           保留权重            无参数
//        (Lamarckian)
```

---

## 8. 数据结构

### 8.1 LayerGene

```rust
/// 层基因：描述一个层的配置
#[derive(Clone, Debug)]
pub struct LayerGene {
    /// 全局创新号（用于交叉对齐，可选）
    pub innovation_number: u64,
    /// 层类型和配置
    pub layer_config: LayerConfig,
    /// 是否启用
    pub enabled: bool,
}

/// 层配置枚举
#[derive(Clone, Debug)]
pub enum LayerConfig {
    Linear {
        in_features: usize,
        out_features: usize,
    },
    Activation {
        activation_type: ActivationType,
    },
    Rnn {
        input_size: usize,
        hidden_size: usize,
    },
    Lstm {
        input_size: usize,
        hidden_size: usize,
    },
    Gru {
        input_size: usize,
        hidden_size: usize,
    },
    /// 聚合节点：汇合多个输入
    Aggregate {
        strategy: AggregateStrategy,
    },
    // ... 其他层类型
}
```

### 8.2 NetworkGenome

```rust
/// 网络基因组：完整的网络拓扑描述
pub struct NetworkGenome {
    /// 层基因列表（有序）
    pub layers: Vec<LayerGene>,
    /// 生成方式
    pub generated_by: String,
}
```

---

## 9. API 设计

### 9.1 Evolution 结构体

```rust
/// 一次神经架构演化的完整定义
pub struct Evolution {
    // ========== 核心输入（所有权）==========
    /// 待演化的网络
    pub graph: Graph,
    /// 任务定义（封装训练+评估）
    pub task: Box<dyn EvolutionTask>,

    // ========== 达标判定 ==========
    /// 目标指标（达标就停止）
    pub target_metric: f32,   // 默认 0.95
    /// 评估次数（取最低值）
    pub eval_runs: usize,     // 默认 3

    // ========== 收敛判定 ==========
    pub loss_tolerance: f32,  // 默认 1e-4
    pub grad_tolerance: f32,  // 默认 1e-5
    pub patience: usize,      // 默认 5

    // ========== 变异概率 ==========
    pub prob_insert_layer: f32,
    pub prob_remove_layer: f32,
    pub prob_replace_layer_type: f32,
    pub prob_add_skip_edge: f32,
    pub prob_remove_skip_edge: f32,
    pub prob_mutate_aggregate: f32,
    pub prob_grow_hidden: f32,
    pub prob_shrink_hidden: f32,

    // ========== 规模约束 ==========
    pub initial_hidden_size: usize,  // 默认 1
    pub max_layers: usize,           // 默认 10
    pub max_hidden_size: usize,      // 默认 64
    pub max_total_params: usize,     // 默认 10000

    // ========== 可用层类型 ==========
    pub available_layer_types: Vec<LayerType>,
}

impl Default for Evolution {
    fn default() -> Self {
        Self {
            // ... 使用合理默认值
            target_metric: 0.95,
            eval_runs: 3,
            loss_tolerance: 1e-4,
            grad_tolerance: 1e-5,
            patience: 5,
            // 变异概率（总和 = 1.0）
            prob_insert_layer: 0.12,
            prob_remove_layer: 0.12,
            prob_replace_layer_type: 0.08,
            prob_add_skip_edge: 0.08,
            prob_remove_skip_edge: 0.05,
            prob_mutate_aggregate: 0.05,
            prob_grow_hidden: 0.20,
            prob_shrink_hidden: 0.25,
            // prob_other: 0.05
            initial_hidden_size: 1,
            max_layers: 10,
            max_hidden_size: 64,
            max_total_params: 10000,
            available_layer_types: vec![LayerType::Linear, LayerType::Activation],
        }
    }
}
```

### 9.2 Evolution::run() 实现

```rust
impl Evolution {
    /// 便捷构造：只需 Graph 和 Task，其他用默认值
    pub fn new(graph: Graph, task: impl EvolutionTask + 'static) -> Self {
        Self {
            graph,
            task: Box::new(task),
            ..Default::default()
        }
    }

    /// 执行演化（消耗 self，返回演化后的 Graph）
    pub fn run(self) -> Graph {
        let mut graph = self.graph;
        let mut best_snapshot: Option<Snapshot> = None;
        let mut best_loss = f32::INFINITY;
        let mut rng = rand::thread_rng();

        loop {
            // Step 1: 训练到收敛
            let loss = self.task.train_until_converged(&mut graph, &self.convergence_config());

            // Step 2: 评估（N 次取最低值）
            let min_score = (0..self.eval_runs)
                .map(|_| self.task.evaluate(&graph))
                .fold(f32::INFINITY, f32::min);

            if min_score >= self.target_metric {
                return graph;  // 成功！返回
            }

            // Step 3: 更新全局最优 / 回滚
            if loss < best_loss {
                best_loss = loss;
                best_snapshot = Some(graph.snapshot());
            } else if let Some(ref snapshot) = best_snapshot {
                graph.restore(snapshot);
            }

            // Step 4: 随机变异
            self.random_mutation(&mut graph, &mut rng);
        }
    }
}
```

### 9.3 使用示例（Only Touch）

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
    max_layers: 5,
    ..Default::default()
}.run();

// ========== 强化学习 ==========
let trained = Evolution::new(graph, RLTask::new(CartPoleEnv::new()))
    .run();
```

### 9.4 Only Touch 理念体现

| 维度 | Touch 的东西 |
|------|-------------|
| **结构** | `Evolution` — 一个结构包含一切 |
| **方法** | `.run()` — 一个方法完成一切 |
| **输入** | `Graph` — 一个网络（所有权转移） |
| **输出** | `Graph` — 一个网络（所有权返回） |

---

## 10. 收敛判定

### 10.1 判定条件

采用**双条件**判定，满足任一即认为收敛：

| 条件 | 公式 | 说明 |
|------|------|------|
| **Loss 稳定** | `max(recent) - min(recent) < ε × min(recent)` | 最近 k 个 epoch 的 loss 相对波动 < ε |
| **梯度消失** | `‖∇L‖₂ < δ` | 梯度 L2 范数 < δ |

### 10.2 推荐默认值

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `loss_tolerance` | **1e-4** | Loss 相对变化 < 0.01% 视为稳定 |
| `grad_tolerance` | **1e-5** | 梯度范数 < 此值视为"平坦" |
| `patience` | **5** | 连续 5 个 epoch 满足条件 |

### 10.3 实现

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
}
```

---

## 11. 实现路径

| Phase | 任务 | 验收标准 | 状态 |
|:-----:|------|----------|:----:|
| **1** | LayerGene / NetworkGenome 数据结构 | 单元测试 | ⏳ |
| **2** | 变异操作实现（InsertLayer, RemoveLayer, MutateHiddenSize...） | 单元测试 | ⏳ |
| **3** | Genome → Graph 转换 | 能构建并 forward | ⏳ |
| **4** | ConvergenceDetector | 单元测试 | ⏳ |
| **5** | EvolutionTask trait + SupervisedTask | 单元测试 | ⏳ |
| **6** | Evolution 结构体 + .run() | 集成测试：XOR 进化 | ⏳ |
| **7** | 记忆任务验证（RNN/LSTM） | 集成测试：Parity 任务 | ⏳ |

---

## 12. MVP 验收标准

### 12.1 XOR 进化测试

| 指标 | 目标 |
|------|------|
| **任务** | XOR 二分类 |
| **初始结构** | Input → Linear(1) → Output |
| **目标 Accuracy** | ≥ 100%（4 个样本全部正确） |
| **评估次数** | 3 次取最低值 |
| **成功率** | > 90%（10 次独立运行） |

### 12.2 测试用例

```rust
#[test]
fn test_xor_evolution() {
    // 最小初始结构
    let graph = Graph::minimal_structure(2, 1);  // Input → Linear(1) → Output

    // XOR 数据
    let (train_data, test_data) = xor_dataset();

    // 创建任务
    let task = SupervisedTask::new(train_data, test_data, Metric::Accuracy);

    // 运行演化（Only Touch：两行代码）
    let trained = Evolution::new(graph, task).run();

    // 验证达标
    assert!(task.evaluate(&trained) >= 1.0);
}
```

---

## 13. 扩展性设计（预留）

> **说明**：本节描述未来如何扩展到更细粒度的节点级演化。MVP 阶段不实现，但代码设计时应预留扩展点。

### 13.1 当前设计的扩展能力

| 扩展需求 | 当前支持 | 扩展难度 |
|----------|:--------:|:--------:|
| 添加新层类型 | ✅ | 低 |
| 添加新变异操作 | ✅ | 低 |
| AggregateNode 新策略 | ✅ | 低 |
| 任意拓扑结构 | ⚠️ | 中 |
| 节点级增删 | ❌ | 高 |
| 带权重的跳跃边 | ❌ | 中 |
| MetaNode 模块 | ❌ | 高 |

### 13.2 未来架构演进

#### Phase 1：显式图结构（v1.1）

将 `NetworkGenome` 从层列表改为显式图结构：

```rust
// 当前（MVP）
pub struct NetworkGenome {
    pub layers: Vec<LayerGene>,  // 隐含线性序列
}

// 未来（v1.1）
pub struct NetworkGenome {
    /// 基因单元（节点）
    pub units: HashMap<UnitId, GeneUnit>,
    /// 边信息（显式表达拓扑）
    pub edges: Vec<EdgeGene>,
    /// 输入/输出节点
    pub input_ids: Vec<UnitId>,
    pub output_ids: Vec<UnitId>,
}
```

#### Phase 2：可插拔基因单元（v1.2）

引入 `GeneUnit` 枚举，支持多种粒度：

```rust
/// 基因单元（可扩展）
pub enum GeneUnit {
    /// 层级单元（当前）
    Layer(LayerGene),
    /// 聚合节点
    Aggregate(AggregateGene),
    // ========== 未来扩展 ==========
    /// 细粒度节点（如带权重的 Add）
    Node(NodeGene),
    /// 元节点（组合多个节点的模块）
    MetaNode(MetaGene),
}

/// 边基因（支持带权重的边）
pub struct EdgeGene {
    pub src: UnitId,
    pub dst: UnitId,
    /// None = identity（无权重），Some = 带权重变换
    pub transform: Option<TransformGene>,
}
```

#### Phase 3：变异操作 trait 化（v2.0）

将变异操作抽象为 trait，实现可插拔：

```rust
/// 变异操作 trait
pub trait Mutation: Send + Sync {
    /// 变异名称
    fn name(&self) -> &str;
    /// 应用变异
    fn apply(&self, genome: &mut NetworkGenome, rng: &mut impl Rng) -> Result<(), MutationError>;
    /// 是否适用于当前基因组
    fn is_applicable(&self, genome: &NetworkGenome) -> bool;
}

// 注册变异操作
pub struct MutationRegistry {
    mutations: Vec<(f32, Box<dyn Mutation>)>,  // (概率, 操作)
}

impl MutationRegistry {
    pub fn register(&mut self, prob: f32, mutation: impl Mutation + 'static) {
        self.mutations.push((prob, Box::new(mutation)));
    }
}
```

### 13.3 节点级扩展示例

#### 带权重的跳跃边

```rust
// 当前：跳跃边是 identity（无权重）
Input ────────────────────────┐
  └─> Linear → ReLU → Linear ─┴─> Aggregate → Output

// 未来：跳跃边可带权重变换
Input ─> Linear(proj) ────────┐  ← 投影层（可学习）
  └─> Linear → ReLU → Linear ─┴─> Aggregate → Output
```

实现方式：`EdgeGene.transform = Some(LinearTransform { ... })`

#### MetaNode 模块

```rust
/// 元节点：封装常用模块（如 ResBlock、Attention）
pub struct MetaGene {
    pub template: MetaTemplate,
    pub params: MetaParams,
}

pub enum MetaTemplate {
    ResBlock,      // 残差块
    Attention,     // 注意力模块
    InceptionCell, // Inception 模块
    Custom(String), // 用户自定义
}
```

### 13.4 MVP 阶段的预留措施

为便于未来扩展，MVP 实现时应注意：

| 措施 | 说明 |
|------|------|
| 使用 `NodeId` 而非索引 | 便于未来支持非线性拓扑 |
| 变异逻辑独立封装 | 每个变异操作一个函数/结构体 |
| `LayerConfig` 保持开放 | 枚举可随时添加新变体 |
| 拓扑信息与权重分离 | `NetworkGenome` 只存拓扑，权重在 `Graph` 中 |

---

## 14. 参考文献

1. **NEAT 原始论文**（2002）：Evolving Neural Networks through Augmenting Topologies
2. **EXAMM 论文**（2019）：Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution
3. **LayerNAS**（2023）：Neural Architecture Search in Polynomial Complexity
4. **NFNets**（2021）：High-Performance Large-Scale Image Recognition Without Normalization

---

## 附录 A：激活函数类型

```rust
#[derive(Clone, Copy, Debug)]
pub enum ActivationType {
    ReLU,
    LeakyReLU { alpha: f32 },
    Tanh,
    Sigmoid,
    GELU,
    SiLU,
    Softmax,
}
```

---

## 附录 B：超参数变异策略

### B.1 增大隐藏层（GrowHiddenSize）

```rust
fn grow_hidden_size(current: usize, max: usize) -> usize {
    let options = [
        current + 1,           // 加 1
        current * 2,           // 翻倍
    ];
    
    options
        .into_iter()
        .filter(|&s| s <= max)
        .choose(&mut rng)
        .unwrap_or(current)
}
```

### B.2 缩小隐藏层（ShrinkHiddenSize）

```rust
fn shrink_hidden_size(current: usize, min: usize) -> usize {
    let options = [
        current - 1,           // 减 1
        current / 2,           // 减半
    ];
    
    options
        .into_iter()
        .filter(|&s| s >= min && s < current)
        .choose(&mut rng)
        .unwrap_or(current)
}
```

---

## 附录 C：典型演化轨迹

### C.1 基础演化（无跳跃连接）

```
Generation  0: Input → Linear(1) → Output                    [params: 3]
    ↓ InsertLayer(ReLU)
Generation  1: Input → Linear(1) → ReLU → Output             [params: 3]
    ↓ GrowHiddenSize
Generation  5: Input → Linear(4) → ReLU → Output             [params: 12]
    ↓ InsertLayer(Linear)
Generation 12: Input → Linear(4) → ReLU → Linear(2) → Output [params: 18]
    ↓ ShrinkHiddenSize（发现 2 足够了）
Generation 25: Input → Linear(3) → ReLU → Linear(2) → Output [params: 14]
    ↓ 达标！
```

### C.2 含跳跃连接的演化

```
Generation  0: Input → Linear(4) → ReLU → Output             [params: 12]
    ↓ AddSkipEdge（Input 跳到 Output）
Generation  8: Input → Linear(4) → ReLU ──┐
                │                         ├──> Agg(Add) → Output
                └─────────────────────────┘
    ↓ MutateAggregateStrategy（Add → Concat）
Generation 15: Input → Linear(4) → ReLU ──┐
                │                         ├──> Agg(Concat) → Output
                └─────────────────────────┘
                [注：Concat 后维度变为 4+1=5]
    ↓ 达标！
```

---

## 附录 D：术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| Evolution | Evolution | 一次完整的神经架构演化 |
| EvolutionTask | EvolutionTask | 抽象不同学习范式的训练+评估方式 |
| 层级进化 | Layer-level Evolution | 以完整层为单位进行结构变异 |
| Lamarckian 继承 | Lamarckian Inheritance | 未受影响的层保留训练后的权重 |
| 收敛 | Convergence | Loss 不再显著下降的状态 |
| 达标 | Task Complete | 任务指标满足用户要求 |
| 全局最优 | Global Best | 历史上 loss 最低的结构（含参数快照） |
| AggregateNode | Aggregate Node | 聚合多个输入的特殊节点 |
| 聚合策略 | Aggregate Strategy | 多输入聚合方式（Add/Concat/Mean/Max） |
| 跳跃连接 | Skip Connection | 跨层的边（非相邻层间的连接） |

---

*本文档描述了 only_torch 的神经架构演化机制，采用层级拓扑变异 + 梯度训练的混合策略。*

*最后更新：2026-01-27*

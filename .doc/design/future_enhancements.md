# 未来功能规划

> 本文档整理了经过架构分析后确认值得实现的未来功能，按优先级和依赖关系排序。
>
> **来源**：整合自 `architecture_v2_design.md` 的 Phase 3-5 及 §6 未来改进项。

---

## 1. NEAT 神经进化算法支持

**优先级**：🔴 高（项目愿景核心）

**背景**：根据项目规则，NEAT 融合是项目愿景的重要组成部分，允许训练时动态调整演化网络结构。

### 设计要点

| 方面 | 说明 |
|------|------|
| **个体独立性** | 每个进化个体拥有独立的 `ModelState` |
| **动态拓扑** | compute 闭包每次调用可动态构建网络 |
| **节点增删** | 需要实现从图中移除节点的逻辑 |
| **与现有设计兼容** | 当前 `ModelState` + 节点复用机制不会干扰 NEAT |

### 实现阶段

#### Phase 3：NEAT MVP（4-6 周）

**目标**：实现最小可用的 NEAT 进化

- [ ] 实现 `NodeGene`, `ConnectionGene`, `Genome`
- [ ] 实现 `InnovationTracker`（创新号追踪器）
- [ ] 实现 `Genome::compile() -> Graph`
- [ ] 实现基础变异：`add_node`, `add_connection`, `mutate_weights`
- [ ] 实现 `Genome::crossover()` 和 `distance()`

**验收标准**：
- [ ] 单元测试：`src/neat/tests/genome.rs`
- [ ] 单元测试：`src/neat/tests/mutation.rs`
- [ ] 集成测试：`tests/test_neat_xor.rs` → XOR 任务进化成功
- [ ] `cargo test` 全部通过

#### Phase 4：NEAT 完整（6-8 周）

**目标**：实现完整的 NEAT 进化系统

- [ ] 实现 `Species` 和 `Population`
- [ ] 实现物种划分算法
- [ ] 支持循环连接
- [ ] 实现进化可视化

**验收标准**：
- [ ] 单元测试：`src/neat/tests/species.rs`
- [ ] 单元测试：`src/neat/tests/population.rs`
- [ ] 集成测试：`tests/test_neat_parity.rs` → Parity 任务进化成功
- [ ] `cargo test` 全部通过

#### Phase 5：Layer-Level NEAT（8-12 周）

**目标**：实现 Layer 级别的网络架构演化

- [ ] 定义 `LayerGene` 枚举
- [ ] 实现 `Blueprint`
- [ ] 实现层级变异和交叉

**验收标准**：
- [ ] 集成测试：`tests/test_neat_mnist_nas.rs` → MNIST 架构搜索
- [ ] `cargo test` 全部通过

### NEAT 相关错误类型

实现 NEAT 时需要添加以下错误类型：

```rust
pub enum GraphError {
    // ... 现有错误 ...

    /// NEAT 相关：无效的创新号
    InvalidInnovation(u32),

    /// NEAT 相关：连接已存在
    ConnectionExists { from: u32, to: u32 },
}
```

### 预期使用示例

```rust
// 概念示例
struct NeatIndividual {
    genome: Genome,           // 网络拓扑基因
    state: ModelState,        // 独立的计算图状态
}

impl NeatIndividual {
    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            // 根据 genome 动态构建网络
            self.genome.build_network(input)
        })
    }
}
```

---

## 2. 多输入模型扩展

**优先级**：🟡 中

**背景**：强化学习等场景需要多输入支持，如 Critic 模型需要同时接收 state 和 action。

### 方案 A：多个 ForwardInput 参数

```rust
// 扩展 ModelState
pub fn forward2<X1, X2, F>(&self, x1: X1, x2: X2, compute: F) -> Result<Var, GraphError>
where
    X1: ForwardInput,
    X2: ForwardInput,
    F: FnOnce(&Var, &Var) -> Result<Var, GraphError>;
```

### 方案 B：元组作为输入

```rust
// 为元组实现 ForwardInput trait
impl<A: ForwardInput, B: ForwardInput> ForwardInput for (A, B) {
    type Output = (Var, Var);
    // ...
}

// 使用
let output = model.forward((state, action), |(s, a)| {
    let combined = s.concat(a)?;
    self.critic.forward(&combined)
})?;
```

### 缓存键处理

```rust
// 多输入时缓存键为形状元组
cache_key = (state.feature_shape(), action.feature_shape())
// 例如: ([4], [2])
```

### 应用场景

- **Critic 网络**：Q(s, a) 需要 state 和 action 两个输入
- **Siamese 网络**：两个输入共享编码器
- **条件生成**：输入 + 条件向量

---

## 3. 多输出模型扩展

**优先级**：🟡 中

**背景**：部分模型需要多个输出，如 Actor-Critic 共享特征层但有不同输出头。

### 方案 A：返回元组

```rust
pub fn forward(&self, x: &Tensor) -> Result<(Var, Var), GraphError> {
    self.state.forward(x, |input| {
        let features = self.shared.forward(input);
        let actor_out = self.actor.forward(&features);
        let critic_out = self.critic.forward(&features);
        Ok((actor_out, critic_out))
    })
}
```

### 方案 B：暴露多个输出方法

```rust
impl ActorCritic {
    pub fn forward_actor(&self, x: &Tensor) -> Result<Var, GraphError> { ... }
    pub fn forward_critic(&self, x: &Tensor) -> Result<Var, GraphError> { ... }
    pub fn forward_both(&self, x: &Tensor) -> Result<(Var, Var), GraphError> { ... }
}
```

### 应用场景

| 场景 | 输出 | 说明 |
|------|------|------|
| **Multi-head** | 多个分类头 | 多任务学习 |
| **Actor-Critic** | (action_probs, state_value) | 强化学习 |
| **VAE** | (reconstruction, latent) | 变分自编码器 |

---

## 4. 过程宏简化模型定义

**优先级**：🟢 低（优化体验，非必需）

**背景**：当前模型定义需要手动添加 `state` 字段和实现 `forward`/`parameters` 方法，可通过过程宏自动生成。

### 当前写法（手动）

```rust
pub struct XorMLP {
    fc1: Linear,
    fc2: Linear,
    state: ModelState,  // 手动添加
}

impl XorMLP {
    pub fn new(graph: &Graph) -> Self {
        Self {
            fc1: Linear::new(graph, 2, 8, true),
            fc2: Linear::new(graph, 8, 1, true),
            state: ModelState::new(graph),  // 手动添加
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {
            let h = self.fc1.forward(input).sigmoid();
            Ok(self.fc2.forward(&h))
        })
    }

    pub fn parameters(&self) -> Vec<Var> {  // 手动实现
        [self.fc1.parameters(), self.fc2.parameters()].concat()
    }
}
```

### 目标写法（过程宏）

```rust
#[derive(Model)]
pub struct XorMLP {
    fc1: Linear,
    fc2: Linear,
}

impl XorMLP {
    #[forward]
    pub fn forward(&self, input: &Var) -> Result<Var, GraphError> {
        let h = self.fc1.forward(input).sigmoid();
        Ok(self.fc2.forward(&h))
    }
}
```

### 宏自动生成

1. 添加 `state: ModelState` 字段
2. 包装 `forward` 方法调用 `state.forward()`
3. 遍历所有实现 `HasParameters` trait 的字段生成 `parameters()` 方法

### 实现步骤

1. 创建 `only_torch_macros` crate
2. 实现 `#[derive(Model)]` 派生宏
3. 实现 `#[forward]` 属性宏
4. 在 `only_torch` crate 中 re-export

---

## 5. API 便捷方法扩展

**优先级**：🟢 低（便捷性优化）

### 5.1 `zeros_like` / `randn_like` 方法

**问题**：创建零张量、随机张量需要通过 `graph` 调用

```rust
// 当前
let fake_labels = graph.zeros(&[batch_size, 1])?;
let noise = graph.randn(&[batch_size, latent_dim])?;
```

**改进**：从已有 Var 推断图

```rust
impl Var {
    pub fn zeros_like(&self) -> Result<Var, GraphError>;
    pub fn randn_like(&self) -> Result<Var, GraphError>;
}

// 使用
let fake_labels = d_real.zeros_like()?;
let noise = latent.randn_like()?;
```

### 5.2 标量运算支持

**当前问题**：只支持 Var 之间的运算

**改进**：支持 Var 与标量运算

```rust
// 目标
let scaled = var * 2.0;
let shifted = var + 1.0;
let mask = var > 0.5;  // 返回 mask Var
```

### 5.3 `Var::attach()` 方法

**当前**：`graph.attach_node(node_id)`

**改进**：与 `detach()` 对称的 API

```rust
impl Var {
    pub fn attach(&self) -> Result<(), GraphError> {
        self.graph.borrow_mut().attach_node(self.id)
    }
}
```

---

## 6. 错误类型精细化

**优先级**：🟢 低（可选优化）

**当前状态**：使用 `InvalidOperation(String)` 覆盖多种错误

**改进**：更精确的错误类型，便于用户处理

```rust
pub enum GraphError {
    // ... 现有错误 ...

    /// 节点值尚未计算（需要先调用 forward）
    ValueNotComputed(NodeId),

    /// 节点梯度尚未计算（需要先调用 backward）
    GradientNotComputed(NodeId),

    /// 两个 Var 来自不同的 Graph
    GraphMismatch { left_graph_id: usize, right_graph_id: usize },

    /// 节点已被 detach，不能参与梯度计算
    NodeDetached(NodeId),
}
```

**好处**：
- 错误信息更明确
- 用户可以 match 特定错误类型进行处理

---

## 依赖关系图

```
┌─────────────────┐
│  NEAT 支持       │ ← 项目愿景核心（Phase 3-5）
└────────┬────────┘
         │ 可能需要
         ▼
┌─────────────────┐     ┌─────────────────┐
│  多输入扩展      │────▶│  多输出扩展      │
└─────────────────┘     └─────────────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
         ┌─────────────────┐
         │  过程宏简化      │ ← 优化体验
         └─────────────────┘
                    │
                    ▼
    ┌───────────────────────────┐
    │  API 便捷方法 / 错误精细化  │ ← 可选优化
    └───────────────────────────┘
```

---

## 实施建议

| 优先级 | 功能 | 触发条件 |
|--------|------|---------|
| 🔴 高 | **NEAT** | 项目愿景核心，基础功能稳定后实现 |
| 🟡 中 | **多输入/多输出** | 遇到 RL 等具体需求时实现 |
| 🟢 低 | **过程宏** | API 稳定后，作为用户体验优化 |
| 🟢 低 | **API 便捷方法** | 按需添加，不影响核心功能 |
| 🟢 低 | **错误类型精细化** | 可选优化，当前 `InvalidOperation` 已可用 |

---

## 参考资料

- [NEAT 论文](./../paper/NEAT_2002/summary.md)
- [EXAMM 论文](./../paper/EXAMM_2019/summary.md)
- [项目路线图](../architecture_roadmap.md)
- [架构 V2 设计（已归档）](../_archive/architecture_v2_design.md) — Phase 1-2 已完成，本文档整合其 Phase 3-5 及未来改进项

# 未来功能规划

> 本文档整理了经过架构分析后确认值得实现的未来功能，按优先级和依赖关系排序。
>
> **来源**：整合自 `architecture_v2_design.md` 的 Phase 3-5 及 §6 未来改进项。

---

## 1. 神经架构演化（Neural Architecture Evolution）

**优先级**：🔴 高（项目愿景核心）

**详细设计**：📄 [neural_architecture_evolution_design.md](./neural_architecture_evolution_design.md)

### 概述

采用 **NEAT 风格拓扑变异 + 梯度训练** 的混合策略，区别于传统 NEAT 的纯进化方式：

| 维度 | 传统 NEAT | only_torch 方案 |
|------|----------|----------------|
| **权重优化** | 纯进化（慢） | 梯度下降（快） |
| **权重继承** | Darwinian | **Lamarckian**（继承训练后的权重） |
| **物种形成** | 必须 | ❌ 不需要 |
| **创新号/交叉** | 必须 | ❌ 不需要 |
| **结构变异** | 核心机制 | ✅ 核心机制 |

### 核心流程

```
初始化：最小结构（Input → Output）
    │
    ▼
┌─────────────────────────────────┐
│  梯度训练直到收敛               │◀─────┐
└─────────────────────────────────┘      │
    │                                     │
    ▼                                     │
  达到目标？ ──Yes──▶ 成功退出            │
    │No                                   │
    ▼                                     │
┌─────────────────────────────────┐      │
│  随机变异                        │      │
│  (Add Node / Add Edge /          │      │
│   Remove Edge / Remove Node)     │      │
└─────────────────────────────────┘      │
    │                                     │
    ▼                                     │
  贪婪决策：如果更差则回滚 ───────────────┘
```

### 实现路径

| Phase | 任务 | 验收 |
|:-----:|------|------|
| **1** | Graph 拓扑修改 API | 单元测试：添加/删除节点边 |
| **2** | Graph 状态快照/恢复 | 单元测试：snapshot/restore |
| **3** | 变异操作实现 | 单元测试：4 种变异 |
| **4** | 收敛判定器 | 单元测试：收敛检测 |
| **5** | 主循环实现 | 集成测试：XOR 进化 |

### MVP 验收标准

| 指标 | 目标 |
|------|------|
| **任务** | XOR 二分类 |
| **初始结构** | 2 输入 → 1 输出（无隐藏层） |
| **目标 loss** | < 0.01 |
| **成功率** | > 90%（10 次运行） |

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

- [神经架构演化设计](./neural_architecture_evolution_design.md) — **核心设计文档**，详细描述混合策略
- [NEAT 论文](./../paper/NEAT_2002/summary.md)
- [EXAMM 论文](./../paper/EXAMM_2019/summary.md)
- [记忆机制设计](./memory_mechanism_design.md) — 包含 NEAT 循环与 RNN 的关系
- [项目路线图](../architecture_roadmap.md)
- [架构 V2 设计（已归档）](../_archive/architecture_v2_design.md) — Phase 1-2 已完成，本文档整合其 Phase 3-5 及未来改进项

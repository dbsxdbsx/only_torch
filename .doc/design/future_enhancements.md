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

## 2. 概率分布模块

**优先级**：🔴 高（SAC-Continuous / Hybrid SAC 的核心依赖）

**背景**：SAC-Continuous 的策略网络需要对连续动作进行重参数化采样（reparameterization trick），并计算 log_prob 用于熵正则化和 Actor loss。这需要一个完整的概率分布模块，是当前项目**最大的系统性缺口**。

> **注**：SAC-Discrete（当前 sac/cartpole）不需要此模块——softmax + log_softmax 已足够。
> 此模块仅在扩展到连续/混合动作空间时才需要。

### 需要实现的分布

| 分布 | 核心方法 | 用途 |
|------|---------|------|
| **Normal** | `rsample()` `log_prob()` `entropy()` | 连续动作基础采样 |
| **TanhNormal** | `sample()` + tanh squash + log_prob 修正 | SAC-Continuous 标准策略 |
| **Categorical** | `sample()` `log_prob()` `entropy()` | Hybrid SAC 离散部分（计算图内版本） |

### 设计方案

建议作为独立模块 `src/nn/distributions/`，组合已有计算图节点：

```rust
// src/nn/distributions/mod.rs
pub mod normal;
pub mod tanh_normal;
pub mod categorical;

pub use normal::Normal;
pub use tanh_normal::TanhNormal;
pub use categorical::Categorical;
```

```rust
// Normal 分布核心 API
pub struct Normal {
    mean: Var,   // 均值（来自网络输出）
    std: Var,    // 标准差（来自网络输出，通常经 exp/softplus 变换）
}

impl Normal {
    /// 重参数化采样：mean + std * ε, ε ~ N(0,1)
    pub fn rsample(&self) -> Var { ... }
    /// 对数概率密度
    pub fn log_prob(&self, value: &Var) -> Var { ... }
    /// 熵
    pub fn entropy(&self) -> Var { ... }
}
```

```rust
// TanhNormal 分布核心 API（Squashed Gaussian）
pub struct TanhNormal {
    base_dist: Normal,
}

impl TanhNormal {
    /// 采样 + tanh squashing
    pub fn rsample(&self) -> (Var, Var) { /* (squashed_action, raw_action) */ }
    /// log_prob 带 tanh 修正：log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u) + ε)
    pub fn log_prob(&self, raw_action: &Var) -> Var { ... }
}
```

### 依赖

- **Exp 节点**（Var 层）：`log_std.exp()` → std
- **Clamp 节点**（Var 层）：`log_std` 数值裁剪
- **Tensor::normal()**：已有，用于生成 ε ~ N(0,1)

### 参考

- PyTorch: `torch.distributions.Normal`, `torch.distributions.TransformedDistribution`
- rustRL: 使用外部 crate `tch-distr`（`tch_distr::Normal`）
- 论文: Haarnoja et al. 2018 Appendix C（Enforcing Action Bounds）

---

## 3. 数据共享可视化增强

**优先级**：🟡 中

**详细设计**：📄 [input_node_semantics_design.md](./input_node_semantics_design.md)

### 背景

多模型场景（如 SAC）中，同一份数据分别通过 `input_named()` 进入多个模型，创建多个独立的 Input 节点。当前可视化中这些节点互不关联，用户无法看出它们是同一份数据。

### Phase 1：链式虚线标注（名字匹配）

- 可视化层检测**同名 + 同 shape** 的 BasicInput 节点
- 用链式虚线箭头（n-1 条）标注数据共享关系
- 不改变图引擎核心逻辑，纯可视化层增强

### Phase 2（远期）：Tensor 身份追踪

- 给 Tensor 增加 `source_id` 字段，自动追踪数据来源
- 支持 Input 与 TargetInput 跨类型共享检测（自编码器场景）

---

## ~~4. 多输入模型扩展~~ （已由动态图架构解决）

> **状态**：✅ 已解决
>
> 动态图架构移除了 `ModelState` 和 `ForwardInput`，多输入现在**天然支持**：
> 直接多次调用 `graph.input()` 创建多个输入 Var，在 forward 中自由组合即可。
>
> **已有示例**：`dual_input_add`、`siamese_similarity`、`multi_io_fusion`

<details>
<summary>原方案（仅供参考，已不适用）</summary>

**原优先级**：🟡 中

**原背景**：强化学习等场景需要多输入支持，如 Critic 模型需要同时接收 state 和 action。

方案 A / B 均基于已删除的 `ModelState` 和 `ForwardInput` trait，不再适用。

</details>

---

## ~~5. 多输出模型扩展~~ （已由动态图架构解决）

> **状态**：✅ 已解决
>
> 动态图架构下，forward 直接返回多个 Var（元组/结构体）即可，无需特殊支持。
>
> **已有示例**：`dual_output_classify`、`multi_label_point`、`multi_io_fusion`

<details>
<summary>原方案（仅供参考，已不适用）</summary>

**原优先级**：🟡 中

**原背景**：部分模型需要多个输出，如 Actor-Critic 共享特征层但有不同输出头。

方案 A / B 均基于已删除的 `ModelState`，不再适用。

</details>

---

## 6. 过程宏简化模型定义

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

## 7. API 便捷方法扩展

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

## 8. 错误类型精细化

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
┌─────────────────────┐
│  NEAT 支持           │ ← 项目愿景核心
└────────┬────────────┘
         │ 可能需要
         ▼
┌─────────────────────┐
│  概率分布模块         │ ← SAC-Continuous / Hybrid SAC 核心依赖
│  (Normal/TanhNormal) │
└────────┬────────────┘
         │ 依赖
         ▼
┌─────────────────────┐
│  Exp / Clamp 节点    │ ← 分布模块的底层依赖
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐     ┌───────────────────────────┐
│  过程宏简化           │     │  API 便捷方法 / 错误精细化  │
└─────────────────────┘     └───────────────────────────┘

注：多输入/多输出扩展已由动态图架构解决（§3/§4），不再是待办项。
```

---

## 实施建议

| 优先级 | 功能 | 触发条件 |
|--------|------|---------|
| 🔴 高 | **NEAT** | 项目愿景核心，基础功能稳定后实现 |
| ✅ 完成 | ~~概率分布模块~~ | 已实现 Categorical / Normal / TanhNormal，详见 [分布模块设计](./distributions_design.md) |
| ✅ 完成 | ~~Exp / Clip 节点~~ | 已实现，为分布模块提供底层支持 |
| 🟡 中 | **数据共享可视化** | 多模型共享 Input 的链式虚线标注，详见 [Input 语义设计](./input_node_semantics_design.md) |
| ✅ 完成 | ~~多输入/多输出~~ | 已由动态图架构解决，示例已有演示 |
| 🟢 低 | **过程宏** | API 稳定后，作为用户体验优化 |
| 🟢 低 | **API 便捷方法** | 按需添加，不影响核心功能 |
| 🟢 低 | **错误类型精细化** | 可选优化，当前 `InvalidOperation` 已可用 |

---

## 参考资料

- [神经架构演化设计](./neural_architecture_evolution_design.md) — **核心设计文档**，详细描述混合策略
- [待扩展节点类型规划](./future_node_types.md) — Exp/Clamp/概率分布等节点的详细规划
- [动态图生命周期设计](../_archive/dynamic_graph_lifecycle_design.md) — 已归档，已解决多输入/多输出和节点累积问题
- [Hybrid SAC 论文](./../paper/RL/SAC复合actions.pdf) — Delalleau et al. 2019，离散+连续+混合动作框架
- [NEAT 论文](./../paper/NEAT_2002/summary.md)
- [EXAMM 论文](./../paper/EXAMM_2019/summary.md)
- [记忆机制设计](./memory_mechanism_design.md) — 包含 NEAT 循环与 RNN 的关系
- [项目路线图](../architecture_roadmap.md)
- [架构 V2 设计（已归档）](../_archive/architecture_v2_design.md) — Phase 1-2 已完成，本文档整合其 Phase 3-5 及未来改进项

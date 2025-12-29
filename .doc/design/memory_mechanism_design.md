# Memory Mechanism Design（记忆/循环机制设计）

> 本文档阐述 only_torch 中记忆机制（循环结构）的设计决策，包括 NEAT 风格循环与传统 RNN 循环的关系、设计选择及实现路径。

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

## 2. 设计决策：Hybrid 方案

### 2.1 设计目标

- **概念极简**：核心只引入最少的记忆/循环概念
- **工程可用**：允许使用成熟的模板（RNN/LSTM 等）作为搜索偏置
- **NEAT 兼容**：支持细粒度（连接级）进化

### 2.2 分层架构

```
┌─────────────────────────────────────────────────────────────┐
│  Level 0: 原子节点（核心，必须）                              │
│  ├── MatMul, Add, Sigmoid, Tanh, Softmax, Concat, ...       │
│  ├── 带延迟的循环连接（recurrent edge）                       │
│  └── step() / reset() 语义                                   │
│                                                             │
│  这是 NEAT 可以操作的最小单元                                 │
├─────────────────────────────────────────────────────────────┤
│  Level 1: 模板层（可选，便捷 API）                            │
│  ├── RNNCell, LSTMCell, GRUCell                             │
│  ├── AttentionHead                                          │
│  └── 本质是原子节点的组合，不是新的核心概念                    │
│                                                             │
│  用户/NEAT 可以选择使用或不使用                               │
├─────────────────────────────────────────────────────────────┤
│  Level 2: 展开/训练层                                        │
│  ├── 时间序列展开（BPTT/TBPTT）                              │
│  ├── 进化算法（NEAT）                                        │
│  └── 混合训练策略                                            │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 核心设计要点

1. **允许图中有环**：当启用 NEAT 模式时，Graph 不再强制是 DAG
2. **双缓冲机制**：每个节点保存上一步输出值
3. **step() 语义**：每次 forward 是一个时间步
4. **reset() 语义**：重置所有节点状态（新序列开始时）
5. **模板是语法糖**：RNNCell 等只是原子节点的便捷组合
6. **隐式状态管理**（所有开源项目共识）：
   - 默认：用户无需显式传入 hidden state
   - API 示例：`output = net.forward(input)` 而非 `output, h = net.forward(input, h)`
   - 可选：高级接口允许显式状态访问（用于多实例并发、手动 checkpoint 等场景）

### 2.4 训练方法选择

| 场景 | 推荐方法 |
|------|---------|
| 短序列 + 需要长程依赖 | Full BPTT（梯度下降） |
| 长序列 + 内存有限 | TBPTT（截断反向传播） |
| 结构搜索 + 小规模任务 | NEAT 进化（无梯度） |
| 混合策略 | NEAT 搜索结构 + 梯度微调权重 |

> 💡 **实证验证**：EXAMM 论文（[本地笔记](../paper/EXAMM_2019/summary.md)）通过 484 万个 RNN 的大规模实验，验证了混合策略的有效性。每个候选网络仅需训练 10 epochs（得益于 Lamarckian 权重继承），20 核并行约 30 分钟即可完成一次完整的结构搜索。

## 3. 与 NAS（神经架构搜索）的关系

| 方法 | 搜索粒度 | 灵活性 | 效率 |
|------|---------|--------|------|
| **层级 NAS** | 层/模块 | 低 | 高 |
| **NEAT 细粒度** | 连接/节点 | 高 | 低 |
| **Hybrid（本方案）** | 可选（连接级或模块级） | 中-高 | 中 |

only_torch 的 Hybrid 方案允许在两个极端之间灵活选择。

## 4. 实现路径

### Phase 1: 基础循环支持
- [ ] Graph 支持非 DAG 模式（允许环）
- [ ] 节点双缓冲机制
- [ ] step() / reset() API
- [ ] 简单 RNN 示例（类似 MatrixSlow）

### Phase 2: BPTT 训练
- [ ] 时间步展开（Unroll）
- [ ] TBPTT 截断选项
- [ ] 与现有 backward() 兼容

### Phase 3: 模板层（按 EXAMM 论文推荐优先级）
- [ ] ∆-RNN（性价比最高，仅 1 个门）
- [ ] GRU / MGU（稳定选择）
- [ ] LSTM（复杂但不一定更好，优先级最低）
- [ ] 作为原子节点组合的便捷 API

### Phase 4: NEAT 集成
- [ ] 连接级变异（添加/删除边）
- [ ] 节点级变异（添加/删除节点）
- [ ] 物种形成（speciation）
- [ ] 可配置 recurrent_depth（1-N 时间步跳跃，参考 EXAMM）
- [ ] Lamarckian 权重继承（子代继承父代权重，大幅提高效率）
- [ ] 可选的模块级变异

## 5. 开源项目记忆机制对比

基于对多个 NEAT 相关开源项目的深入分析，以下是它们在记忆机制实现上的对比：

### 5.1 核心机制对比表

| 维度 | neat-python | neat-rs | EXACT/EXAMM | neat-gru-rust |
|------|-------------|---------|-------------|---------------|
| **语言** | Python | Rust | C++ | Rust |
| **循环支持** | ✅ 任意拓扑环 | ❌ 仅前馈 | ✅ 循环边 + 预制单元 | ✅ GRU 连接 |
| **双缓冲** | ✅ `values = [{}, {}]` | N/A | ✅（时间步展开） | ✅（连接级状态） |
| **预制记忆单元** | ❌ | ❌ | ✅ 6 种 | ✅ GRU |
| **recurrent_depth** | 1 步固定 | N/A | 1-10 步可配置 | 1 步固定 |
| **训练方法** | 纯进化 | 纯进化 | 进化 + BPTT | 纯进化 |
| **Lamarckian 继承** | ❌ | ❌ | ✅ | ❌ |
| **step()/reset()** | ✅ | N/A | ✅ | ✅ |
| **隐式 hidden state** | ✅ | N/A | ✅ | ✅ |
| **并行/分布式** | ❌ | ❌ | ✅ MPI | ❌ |

### 5.2 各项目详细分析

#### neat-python — 最简循环实现

```python
# 核心双缓冲机制
self.values = [{}, {}]  # 两个缓冲区
self.active = 0

def activate(self, inputs):
    ivalues = self.values[self.active]      # 读取上一步
    ovalues = self.values[1 - self.active]  # 写入当前步
    self.active = 1 - self.active           # 切换
```

**优点**：实现简单优雅，支持任意拓扑循环
**缺点**：无梯度训练，无预制记忆单元

#### neat-rs — 仅前馈（警示案例）

```rust
fn can_connect_to(&self, to: &Self) -> bool {
    self.value() < to.value()  // 强制拓扑顺序，禁止循环
}
```

**教训**：only_torch 应**避免**这种设计限制，必须允许任意拓扑循环。

#### EXACT/EXAMM — 最完整实现

```cpp
// 循环边支持可配置的时间跳跃
int32_t recurrent_depth;  // 1-10 步

void propagate_forward(int32_t time) {
    outputs[time + recurrent_depth] = input_node->output_values[time] * weight;
}
```

**预制记忆单元库**（按 EXAMM 论文推荐优先级排序）：

| 单元 | 门数量 | 参数量 | EXAMM 论文评价 |
|------|--------|--------|---------------|
| ∆-RNN | 1 | 最少 | ⭐ 性价比最高 |
| UGRNN | 1 | 少 | 轻量级选择 |
| MGU | 1 | 少 | 单独差，混合最好 |
| GRU | 2 | 中 | 稳定选择 |
| LSTM | 3 | 最多 | 复杂但不一定更好 |

**关键特性**：
- Lamarckian 权重继承（子代继承父代权重，减少训练时间）
- 混合训练：进化搜索结构 + BPTT 微调权重

#### neat-gru-rust — GRU 内置于连接

```rust
pub struct ConnectionGru<T> {
    memory: T,           // 持久状态
    prev_input: T,       // 上一步输入
    // ... 6 个权重参数
}

fn reset_state(&mut self) {
    self.memory = T::zero();
    self.prev_input = T::zero();
}
```

**特点**：GRU 是连接类型（不是节点类型），每个连接维护自己的状态。

### 5.3 共识与最佳实践

**所有项目的共识**：
1. **隐式状态管理**：用户无需显式传入 hidden state
2. **reset() 语义**：新序列开始时清空状态
3. **创新号机制**：用于交叉时的基因对齐

**EXAMM 论文的关键洞察**：
1. 没有"万能"记忆单元 — 不同任务最优单元不同
2. **简单神经元 + 复杂记忆单元混合效果最佳**
3. ∆-RNN 性价比最高（1 个门，表现接近 LSTM）
4. 演化出的网络非常紧凑（平均 16-29 隐藏节点，3-8 循环边）

### 5.4 only_torch 借鉴要点

| 借鉴来源 | 采纳内容 | 优先级 |
|---------|---------|--------|
| neat-python | 双缓冲机制 | 🔴 必须 |
| 所有项目 | 隐式 hidden state + reset() | 🔴 必须 |
| EXAMM | 可配置 recurrent_depth | 🟡 建议 |
| EXAMM | Lamarckian 权重继承 | 🟡 建议 |
| EXAMM | ∆-RNN 作为轻量级模板 | 🟢 可选 |
| τ-NEAT | 连接级时间延迟（FIFO 缓冲区） | 🟢 可选 |
| NEAT-LSTM-IM | 两阶段训练（无监督+RL） | 🟢 可选 |
| neat-rs | ❌ 避免禁止循环的设计 | — |

## 6. 相关论文与资源

### 6.1 NEAT 核心论文

- **⭐ NEAT 原始论文**：[Evolving Neural Networks through Augmenting Topologies](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) — **[本地笔记](../paper/NEAT_2002/summary.md)**
  - 三大核心创新：历史标记（创新号）、物种形成、从最小结构增量生长
  - 在双杆平衡任务上比 Cellular Encoding 快 25 倍
  - 消融实验证明每个组件都不可或缺
- **⭐ NEAT 后续综述（2021）**：[A Systematic Literature Review of the Successors of NeuroEvolution of Augmenting Topologies](https://cris.vub.be/ws/files/75376010/A_Systematic_Literature_Review_of_the.pdf)
  - 系统梳理 NEAT 发明后 18 年间的 61 种后继方法
  - 识别出 6 种具有记忆能力的 x-NEAT 方法：NEAT-CTRNN、NEAR、NEAT-LSTM、NEAT-LSTM-IM、τ-NEAT、τ-HyperNEAT
  - **τ-NEAT**：在连接级引入 FIFO 缓冲区实现时间延迟，与 EXAMM 的 recurrent_depth 互补
  - **NEAT-LSTM-IM**：两阶段训练策略（无监督预训练 + RL 微调），解决欺骗性优化问题
  - 验证了"直接编码 + 可选模板层"是主流设计选择（~2/3 方法采用直接编码）

### 6.2 NEAT + 循环/记忆结构

- **⭐ 用神经进化研究 RNN 记忆结构（EXAMM）**：[Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution](https://arxiv.org/abs/1902.02390) — **[本地笔记](../paper/EXAMM_2019/summary.md)**
  - 验证了"NEAT 搜索结构 + 梯度微调权重"混合策略的有效性
  - 比较了 ∆-RNN、GRU、LSTM、MGU、UGRNN 等记忆单元
  - 关键发现：简单神经元 + 复杂记忆单元混合效果最佳
- **NEAT + GRU/LSTM**：[EXALT/EXAMM 框架](https://github.com/travisdesell/exact) - C++ 实现的 NEAT + LSTM/GRU/CNN
- **权重继承对神经进化的影响**：[An Experimental Study of Weight Initialization and Weight Inheritance Effects on Neuroevolution](https://arxiv.org/abs/2009.09644)
- **激活函数变异**：[Evolving Parsimonious Networks by Mixing Activation Functions](https://arxiv.org/abs/1703.07122)

### 6.3 RNN/BPTT 相关

- **BPTT 原始论文**：[Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) - Rumelhart et al., 1986
- **LSTM**：[Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) - Hochreiter & Schmidhuber, 1997
- **GRU**：[Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) - Cho et al., 2014

### 6.4 Rust/Python 实现参考

- **neat-python**：https://github.com/CodeReclwordsaimers/neat-python - 纯 Python NEAT 实现
- **PyTorch-NEAT**：https://github.com/uber-research/PyTorch-NEAT - Uber 的 PyTorch NEAT
- **radiate**：https://github.com/pkalivas/radiate - Rust 遗传编程引擎
- **neat-gru-rust**：https://github.com/sakex/neat-gru-rust - Rust NEAT + GRU

## 7. 总结

| 问题 | 决策 | 依据 |
|------|------|------|
| 核心概念有多少？ | **极简**：原子节点 + 带延迟的循环连接 | 所有项目共识 |
| 需要硬编码 RNN/LSTM 吗？ | **不需要**：作为可选模板层提供 | Hybrid 方案 |
| hidden state 如何管理？ | **默认隐式**：可选显式接口 | 所有项目共识 |
| NEAT 和梯度训练兼容吗？ | **兼容**：NEAT 搜索结构 + 梯度微调权重 | EXAMM 验证 |
| 优先实现哪个记忆单元？ | **∆-RNN**（性价比最高） | EXAMM 论文 |
| 能进化出任意记忆机制吗？ | **理论可以**：实际效率取决于搜索空间 | NEAT 论文 |

---

*本文档记录了 only_torch 记忆机制的设计决策，综合了 NEAT/EXAMM 论文洞察、NEAT 后续综述（2021）及多个开源项目的实现经验。后续实现时可作为参考准绳。*


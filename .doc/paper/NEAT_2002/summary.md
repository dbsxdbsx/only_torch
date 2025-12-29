# NEAT 论文笔记：通过拓扑增强演化神经网络

> **论文**：Evolving Neural Networks through Augmenting Topologies
> **作者**：Kenneth O. Stanley, Risto Miikkulainen
> **年份**：2002
> **期刊**：Evolutionary Computation 10(2): 99-127
> **链接**：[UT Austin](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
> **本地 PDF**：[paper.pdf](./paper.pdf)

---

## 1. 核心贡献

提出 **NEAT**（NeuroEvolution of Augmenting Topologies），一种能够**同时演化神经网络拓扑结构和权重**的方法，通过三个关键创新解决了此前 TWEANN（Topology and Weight Evolving Artificial Neural Networks）方法面临的主要难题。

---

## 2. 解决的三大问题

### 2.1 竞争规约问题（Competing Conventions Problem）

**问题**：相同功能的网络可能有多种不同的拓扑表示，导致交叉操作产生损坏的后代。

**NEAT 解决方案**：**历史标记（Historical Markings）**
- 每个新基因被分配一个全局递增的**创新号（Innovation Number）**
- 创新号记录基因的历史起源，永不改变
- 交叉时，通过创新号对齐匹配基因（matching genes）
- 不匹配的基因分为：
  - **Disjoint genes**：在另一方创新号范围内但不匹配
  - **Excess genes**：超出另一方创新号范围

```
Parent1: [1, 2, 3, 4, 5,    8   ]
Parent2: [1, 2, 3,    5, 6, 7   ]
                ↓
Matching:  1, 2, 3, 5
Disjoint:  4, 6, 7
Excess:    8
```

### 2.2 创新保护问题

**问题**：新增结构（节点/边）通常会**暂时降低**网络适应度，在优化完成前就被淘汰。

**NEAT 解决方案**：**物种形成（Speciation）**
- 使用**显式适应度共享（Explicit Fitness Sharing）**
- 相似拓扑的网络被分入同一物种，主要在物种内竞争
- 兼容性距离公式：

  ```
  δ = c₁·E/N + c₂·D/N + c₃·W̄
  ```

  其中 E=excess 基因数，D=disjoint 基因数，W̄=匹配基因的平均权重差

- 每个物种内的调整适应度：`f'ᵢ = fᵢ / (物种内个体数)`
- 这样创新结构在自己的"生态位"中有时间优化

### 2.3 搜索空间维度问题

**问题**：随机初始拓扑包含大量无用结构，浪费搜索资源。

**NEAT 解决方案**：**从最小结构开始增量生长**
- 初始种群**无隐藏节点**（只有输入直连输出）
- 通过变异逐步添加结构：
  - **Add Connection**：在两个未连接的节点间添加边
  - **Add Node**：分裂现有边，插入新节点
- 只有**证明有用的结构才会保留**
- 始终在最低维度的权重空间中搜索

---

## 3. 基因编码

### 3.1 基因组结构

```
Genome = [Node Genes] + [Connection Genes]

Connection Gene:
  - In Node
  - Out Node
  - Weight
  - Enabled/Disabled
  - Innovation Number
```

### 3.2 变异操作

| 变异类型 | 说明 |
|----------|------|
| **权重变异** | 80% 概率触发，每个权重 90% 概率扰动、10% 概率重置 |
| **添加连接** | 在两个未连接节点间添加边 |
| **添加节点** | 分裂现有边，新入边权重=1，新出边权重=原权重 |

### 3.3 交叉操作

- 匹配基因：随机从两个父代中选择
- Disjoint/Excess 基因：从**更优秀的父代**继承
- 禁用基因有 75% 概率在后代中也被禁用

---

## 4. 实验结果

### 4.1 XOR 问题

| 指标 | 结果 |
|------|------|
| 平均代数 | 32 代 |
| 平均评估次数 | 4,755 个网络 |
| 解网络隐藏节点 | 2.35 个（最优解只需 1 个） |
| 成功率 | 100%（100 次运行） |

### 4.2 双杆平衡（有速度信息）

| 方法 | 评估次数 | 网络数 |
|------|----------|--------|
| Evolutionary Programming | 307,200 | 2,048 |
| Conventional NE | 80,000 | 100 |
| SANE | 12,600 | 200 |
| ESP | 3,800 | 200 |
| **NEAT** | **3,600** | 150 |

### 4.3 双杆平衡（无速度信息）- 最难任务

| 方法 | 评估次数 | 加速比 |
|------|----------|--------|
| Cellular Encoding (CE) | 840,000 | 1× |
| ESP | 169,466 | 5× |
| **NEAT** | **33,184** | **25×** |

**关键发现**：
- NEAT 比 CE 快 **25 倍**
- NEAT 比 ESP 快 **5 倍**
- NEAT 从不需要重启（ESP 平均重启 4.06 次）

---

## 5. 消融实验

验证每个组件都是必要的：

| 消融版本 | 评估次数 | 失败率 | 相比完整版 |
|----------|----------|--------|-----------|
| 无生长（固定拓扑） | 30,239 | 80% | 8.5× 慢 |
| 无物种形成 | 25,600 | 25% | 7× 慢 |
| 随机初始拓扑 | 23,033 | 5% | 7× 慢 |
| 无交叉 | 5,557 | 0% | 1.5× 慢 |
| **完整 NEAT** | **3,600** | **0%** | 基准 |

**结论**：所有组件相互依赖，缺一不可。

---

## 6. 关键洞察

### 6.1 为什么从最小结构开始？

> "The goal is not to minimize only the final product, but all intermediate networks along the way as well."

- 搜索始终在**最低维度**空间进行
- 只添加**证明有用**的结构
- 避免浪费时间删除随机初始化的无用结构

### 6.2 为什么物种形成有效？

- 保护创新结构免于过早淘汰
- 允许同时搜索**多个不同维度**的空间
- 防止种群过早收敛到单一拓扑

### 6.3 NEAT 的独特性

> "NEAT is unique because structures become increasingly more complex as they become more optimal, strengthening the analogy between GAs and natural evolution."

NEAT 同时实现：
- **优化（Optimization）**：找到更好的权重
- **复杂化（Complexification）**：找到更好的结构

---

## 7. NEAT 的记忆机制

> ⚠️ **重要洞察**：NEAT 不使用预制的记忆单元（如 LSTM/GRU），而是通过**演化出的循环连接**实现记忆能力。

### 7.1 非马尔可夫任务与记忆需求

双杆平衡任务分两种：

| 版本 | 马尔可夫性 | 记忆需求 |
|------|-----------|---------|
| **有速度信息（DPV）** | ✅ 马尔可夫 | 不需要记忆（当前状态足够决策） |
| **无速度信息（DPNV）** | ❌ 非马尔可夫 | **需要记忆**（必须从历史推断速度） |

### 7.2 NEAT 如何实现记忆

NEAT 通过**循环边（recurrent connections）**实现记忆：

```
简单神经元 + 自连接 = 记忆能力

节点 A 的自连接工作原理：
  t=0: A = tanh(input)                    → 无历史
  t=1: A = tanh(input + w·A_{t-1})        → 记住 t=0 的状态
  t=2: A = tanh(input + w·A_{t-1})        → 记住 t=1 的状态（含 t=0 的信息）
```

### 7.3 DPNV 的典型解结构

论文 Figure 8 展示了一个优雅的解：

```
输入（杆角度）──→ [隐藏节点] ──→ 输出（力）
                    ↻ (自连接)
```

> "Using the recurrent connection to itself, the single hidden node determines whether the poles are **falling away or towards each other**."

**工作原理**：
- 自连接计算**角度差的导数**（相当于估计速度）
- 只需 **1 个隐藏节点 + 1 条循环边**
- 无需显式计算每个杆的速度

### 7.4 NEAT 记忆 vs 门控记忆单元

| 特性 | NEAT 循环连接 | LSTM/GRU 门控单元 |
|------|--------------|------------------|
| **结构来源** | 演化产生 | 人工设计 |
| **门控机制** | ❌ 无 | ✅ 有（forget/input/output gate） |
| **参数量** | 极少（1 个权重/边） | 较多（每个门都有权重矩阵） |
| **长期依赖** | 仅适合短期记忆 | 专门设计解决长期依赖 |
| **可解释性** | 结构简单，易理解 | 结构复杂，较难解释 |

### 7.5 关键结论

> **"简单循环连接对于某些任务足够，不一定需要复杂的门控机制。"**

这解释了为什么 EXAMM 论文发现"简单神经元 + 复杂记忆单元"混合效果最好——两者**互补**：
- 简单循环连接：轻量、适合短期依赖
- 门控单元（LSTM/GRU）：重量、适合长期依赖

---

## 8. 典型参数设置

| 参数 | 值 |
|------|-----|
| 种群大小 | 150（困难任务 1000） |
| c₁, c₂（拓扑兼容系数） | 1.0 |
| c₃（权重兼容系数） | 0.4（困难任务 3.0） |
| 兼容性阈值 δₜ | 3.0（困难任务 4.0） |
| 添加节点概率 | 0.03 |
| 添加连接概率 | 0.05（大种群 0.3） |
| 权重变异概率 | 80% |
| 物种停滞代数限制 | 15 代 |

---

## 9. 对 only_torch 的启示

### 9.1 核心设计原则

| NEAT 原则 | only_torch 实现建议 |
|-----------|---------------------|
| 历史标记 | 为每个节点/边分配全局创新号 |
| 物种形成 | 基于拓扑相似度分组，实现适应度共享 |
| 最小起始 | 初始网络只有输入→输出直连 |
| 增量生长 | Add Node / Add Connection 变异 |

### 9.2 与 EXAMM 的对比

| 方面 | NEAT | EXAMM |
|------|------|-------|
| 权重训练 | 纯进化 | 反向传播 + SGD |
| 节点类型 | 简单神经元 | 预制记忆单元（LSTM 等） |
| 记忆机制 | 演化出的循环边 | 预制门控单元 |
| 效率 | 适合小规模任务 | 适合时间序列预测 |

### 9.3 实现优先级

1. **Phase 1**：实现基因编码（创新号、节点/连接基因）
2. **Phase 2**：实现交叉操作（基于创新号对齐）
3. **Phase 3**：实现物种形成和适应度共享
4. **Phase 4**：实现从最小结构开始的增量生长

---

## 10. 引用格式

```bibtex
@article{stanley2002evolving,
  title={Evolving neural networks through augmenting topologies},
  author={Stanley, Kenneth O and Miikkulainen, Risto},
  journal={Evolutionary computation},
  volume={10},
  number={2},
  pages={99--127},
  year={2002},
  publisher={MIT Press}
}
```


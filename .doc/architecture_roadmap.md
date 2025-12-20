# Only Torch 架构路线图

> 最后更新: 2025-12-20
> 战略定位: **简化版 PyTorch in Rust**，为 NEAT 预留扩展性
> MVP 目标: **XOR with Optimizer**

## 文档索引

```
.doc/
├── architecture_roadmap.md              # ← 你在这里（主入口）
├── design/                              # 当前有效的设计文档
│   ├── gradient_clear_and_accumulation_design.md   # 梯度机制
│   └── optimizer_architecture_design.md            # 优化器架构
├── reference/                           # 参考资料
│   └── python_MatrixSlow_pid.md         # MatrixSlow 项目分析
└── _archive/                            # 暂缓/远期愿景
    ├── high_level_architecture_design.md   # 5层架构愿景（远期）
    └── graph_execution_refactor.md         # 底层重构方案（暂缓）
```

---

## 当前状态概览

```
模块               完成度    状态
─────────────────────────────────
tensor/            ~80%     ✅ 基本完成
nn/graph           ~90%     ✅ 核心完成
nn/nodes           ~30%     🔄 基础节点OK，缺激活/损失函数
nn/optimizer       ~70%     ✅ SGD/Adam可用，缺Momentum等
vision/            ~70%     ✅ 基本完成
logic/             0%       ❌ 预留
neat/              0%       ❌ 远期特色
```

## 已实现节点

| 类型 | 节点             | 状态 |
| :--- | :--------------- | :--: |
| 输入 | Input, Parameter |  ✅  |
| 运算 | Add, MatMul      |  ✅  |
| 激活 | Step             |  ✅  |
| 损失 | PerceptionLoss   |  ✅  |

## 缺失的关键节点

- **激活函数**: Tanh, Softplus, ReLU, Sigmoid, Softmax
- **损失函数**: CrossEntropyLoss, MSELoss
- **运算节点**: Sub, Neg, Mul(逐元素), Div, Reshape

---

## 优先级路线图

### MVP: XOR with Optimizer (2-3 周)

|  #  | 任务                 | 说明                          | 验收                  | NEAT 友好性 | 状态 |
| :-: | :------------------- | :---------------------------- | :-------------------- | :---------- | :--: |
| M1  | Optimizer 基础功能   | SGD/Adam 参数更新             | 参数能正常更新        | ✅ 无影响   |  ✅  |
| M2  | 实现 Tanh 节点       | XOR 必需的非线性激活          | forward/backward 正确 | ✅ 新节点   |  ❌  |
| M3  | XOR 监督学习示例     | 用 Optimizer 端到端训练       | 收敛>99%              | ✅ 验证     |  ❌  |
| M4  | 验证图的动态扩展能力 | 确保 Graph 支持运行时添加节点 | 单元测试通过          | ⭐ 关键     |  ❌  |

### 阶段二：MNIST 基础 (4-6 周)

|  #  | 任务                 | 说明                 | NEAT 友好性       |
| :-: | :------------------- | :------------------- | :---------------- |
| P1  | Softmax+CrossEntropy | 分类必需             | ✅ 新节点         |
| P2  | ReLU/Sigmoid 节点    | 通用激活             | ✅ 新节点         |
| P3  | Reshape/Flatten 节点 | CNN 数据流转换       | ✅ 结构操作       |
| P4  | Conv2d 节点          | 参考 MatrixSlow 实现 | ⚠️ 需设计可进化性 |
| P5  | Pooling 节点         | MaxPool/AvgPool      | ⚠️ 需设计可进化性 |
| P6  | MNIST 端到端示例     | LeNet 风格           | ✅ 验证           |

### 阶段三：NEAT 神经进化 (8-12 周)

| 任务                    | 说明                    | 依赖           |
| :---------------------- | :---------------------- | :------------- |
| NodeGene/ConnectionGene | NEAT 基因表示           | Graph 动态扩展 |
| 拓扑变异操作            | 添加节点/连接           | 基础节点类型   |
| 权重变异                | 利用现有 Parameter 机制 | Optimizer 可选 |
| 适应度评估              | 利用现有 forward 机制   | Graph 正确性   |
| 物种分化                | 基因相似度计算          | NodeGene 完成  |
| XOR 进化实验            | 从零进化解决 XOR        | 以上全部       |

---

## 目标架构

```
only_torch/
├── tensor/          # 张量核心 ✅
├── nn/
│   ├── graph        # 计算图 ✅
│   ├── nodes/       # 节点层
│   │   ├── 输入: Input, Parameter, Constant
│   │   ├── 激活: ReLU, Tanh, Sigmoid, Softmax, Step
│   │   ├── 运算: Add, Sub, Mul, Div, MatMul, Reshape
│   │   └── 损失: MSE, CrossEntropy, PerceptionLoss
│   ├── optimizer/   # 优化器
│   │   └── SGD, Momentum, Adam, LRScheduler
│   └── context/     # 运行上下文
│       └── no_grad, train/eval模式
├── vision/          # 视觉处理 ✅
├── data/            # 数据加载 (待实现)
│   └── DataLoader, Dataset, 内置数据集
├── neat/            # 神经进化 (远期)
└── rl/              # 强化学习 (远期)
```

---

## 下一步行动计划

### 当前优先：M2 实现 Tanh 节点

XOR 问题需要非线性激活函数。Tanh 是最简单的选择：

```rust
// Tanh节点实现要点
// forward: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
// backward: d(tanh)/dx = 1 - tanh²(x)
```

### M4 关键：验证 NEAT 友好性

在 MVP 完成后，必须验证 Graph 的动态扩展能力：

```rust
// 测试：在已有图中动态添加节点
let mut graph = Graph::new();
let a = graph.new_parameter_node(&[1, 1], Some("a"))?;
let b = graph.new_parameter_node(&[1, 1], Some("b"))?;
let add1 = graph.new_add_node(&[a, b], None)?;

// 执行一次训练...
graph.forward_node(add1)?;

// 动态添加新节点（NEAT变异时的典型操作）
let c = graph.new_parameter_node(&[1, 1], Some("c"))?;
let add2 = graph.new_add_node(&[add1, c], None)?;  // 新增节点

// 新图仍然能正常工作
graph.forward_node(add2)?;
```

如果这个测试失败，需要在进入阶段三之前修复。

---

## 架构约束（为 NEAT 预留）

设计新节点时，牢记以下约束：

1. **节点必须可克隆** - NEAT 需要复制基因
2. **节点必须可序列化** - 保存/加载进化历史
3. **Graph 必须支持动态修改** - 运行时添加/删除节点
4. **避免全局状态** - 多个 Graph 实例可能并行进化

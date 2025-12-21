# Only Torch 架构路线图

> 最后更新: 2025-12-22
> 战略定位: **简化版 PyTorch in Rust**，为 NEAT 预留扩展性
> MVP 目标: **XOR with Optimizer**

## 文档索引

```
.doc/
├── architecture_roadmap.md              # ← 你在这里（主入口）
├── design/                              # 当前有效的设计文档
│   ├── api_layering_and_seed_design.md             # API分层与种子管理
│   ├── batch_mechanism_design.md                   # Batch Forward/Backward 机制（重要）
│   ├── broadcast_mechanism_design.md               # 广播机制设计
│   ├── gradient_clear_and_accumulation_design.md   # 梯度机制
│   ├── node_vs_layer_design.md                     # Node vs Layer 架构设计 ⭐ NEW
│   ├── optimization_strategy.md                    # 性能优化策略
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
nn/nodes           ~75%     ✅ Conv2d/MaxPool2d/AvgPool2d 已完成
nn/optimizer       ~70%     ✅ SGD/Adam可用，缺Momentum等
data/              ~60%     🔄 MNIST 已完成，DataLoader 基础可用
vision/            ~70%     ✅ 基本完成
logic/             0%       ❌ 预留
neat/              0%       ❌ 远期特色
```

## 已实现节点

| 类型 | 节点                                    | 状态 |
| :--- | :-------------------------------------- | :--: |
| 输入 | Input, Parameter                        |  ✅  |
| 运算 | Add, MatMul, Reshape, Flatten           |  ✅  |
| 激活 | Step, Tanh, Sigmoid, LeakyReLU/ReLU     |  ✅  |
| CNN  | Conv2d, MaxPool2d, AvgPool2d            |  ✅  |
| 损失 | PerceptionLoss, SoftmaxCrossEntropyLoss |  ✅  |

## 缺失的关键节点

- **激活函数**: Softplus, Softmax (独立版)
- **损失函数**: MSELoss
- **运算节点**: Sub, Neg, Mul(逐元素), Div

## 集成测试进度

> 对应 MatrixSlow Python 示例的 Rust 实现验证

| Rust 测试                     | 对应 MatrixSlow 示例          | 状态 | 说明                                |
| ----------------------------- | ----------------------------- | :--: | ----------------------------------- |
| `test_adaline.rs`             | `ch02/adaline.py`             |  ✅  | 最基础的计算图+自动微分             |
| `test_adaline_batch.rs`       | `ch03/adaline_batch.py`       |  ✅  | 批量处理                            |
| `test_optimizer_example.rs`   | `ch03/optimizer_example.py`   |  ✅  | SGD/Adam 优化器验证                 |
| `test_xor.rs`                 | -                             |  ✅  | **MVP 展示：非线性分类问题**        |
| `test_logistic_regression.rs` | `ch04/logistic_regression.py` |  ❌  | 需要 Sigmoid 节点 (已有) + 测试代码 |
| `test_nn_iris.rs`             | `ch05/nn_iris.py`             |  ❌  | 需要多层网络+Softmax                |
| `test_mnist.rs`               | `ch05/nn_mnist.py`            |  ✅  | **MVP：MLP + SoftmaxCrossEntropy**  |

---

## 优先级路线图

### MVP: XOR with Optimizer (2-3 周)

|  #  | 任务                 | 说明                                  | 验收                                                        | NEAT 友好性 | 状态 |
| :-: | :------------------- | :------------------------------------ | :---------------------------------------------------------- | :---------- | :--: |
| M1  | Optimizer 基础功能   | SGD/Adam 参数更新                     | 参数能正常更新                                              | ✅ 无影响   |  ✅  |
| M1b | Granular 种子 API    | `_seeded` 方法确保测试可重复          | 集成测试确定性                                              | ✅ 无影响   |  ✅  |
| M2  | 实现 Tanh 节点       | XOR 必需的非线性激活                  | forward/backward 正确                                       | ✅ 新节点   |  ✅  |
| M3  | XOR 监督学习示例     | 用 Optimizer 端到端训练               | 收敛 100%                                                   | ✅ 验证     |  ✅  |
| M4  | 验证图的动态扩展能力 | 确保 Graph 支持运行时添加节点         | 单元测试通过                                                | ⭐ 关键     |  ✅  |
| M4b | Graph 级别种子 API   | `Graph::new_with_seed()` 简化用户代码 | 详见 [API 分层设计](design/api_layering_and_seed_design.md) | ⭐ 关键     |  ✅  |

### 阶段二：MNIST 基础 (4-6 周)

|  #  | 任务                 | 说明                                   | NEAT 友好性       | 状态 |
| :-: | :------------------- | :------------------------------------- | :---------------- | :--: |
| P1  | Softmax+CrossEntropy | 分类必需                               | ✅ 新节点         |  ✅  |
| P1b | Sigmoid 节点         | 通用激活                               | ✅ 新节点         |  ✅  |
| P1c | DataLoader + MNIST   | 数据加载                               | ✅ 基础设施       |  ✅  |
| P2  | LeakyReLU/ReLU 节点  | 底层 LeakyReLU + 便捷 ReLU (slope=0.0) | ✅ 新节点         |  ✅  |
| P3  | Reshape/Flatten 节点 | CNN 数据流转换（PyTorch 风格）         | ✅ 结构操作       |  ✅  |
| P4  | Conv2d 节点          | PyTorch 风格（多通道内部处理）         | ✅ Jacobi+Batch   |  ✅  |
| P5  | Pooling 节点         | MaxPool2d/AvgPool2d                   | ✅ Jacobi+Batch   |  ✅  |
| P6  | MNIST CNN 端到端     | LeNet 风格                             | ✅ 验证           |  🔄  |

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
│   │   ├── 激活: LeakyReLU/ReLU, Tanh, Sigmoid, Softmax, Step
│   │   ├── 运算: Add, Sub, Mul, Div, MatMul, Reshape
│   │   └── 损失: MSE, CrossEntropy, PerceptionLoss
│   ├── optimizer/   # 优化器
│   │   └── SGD, Momentum, Adam, LRScheduler
│   └── context/     # 运行上下文
│       └── no_grad, train/eval模式
├── vision/          # 视觉处理 ✅
├── data/            # 数据加载 ✅ (基础完成)
│   └── MnistDataset, transforms, DataLoader (MVP)
├── neat/            # 神经进化 (远期)
└── rl/              # 强化学习 (远期)
```

---

## 下一步行动计划

### ✅ 已完成：M2 Tanh 节点 & M3 XOR 示例

XOR 问题已成功解决！网络结构：`Input(2) → Hidden(4, Tanh) → Output(1)`，约 30 个 epoch 收敛到 100%准确率。

### ✅ 已完成：M4 验证 NEAT 友好性

Graph 的动态扩展能力已验证通过！关键实现：

1. **新增 `on_topology_changed()` 方法**：在拓扑变化后调用，清除所有 Jacobi 但保留 value
2. **12 个综合测试**覆盖各种场景：
   - 基本动态添加（forward/backward 后添加节点）
   - 多次连续拓扑变化
   - 链式添加、分支添加
   - NEAT 变异模拟（添加节点、添加连接）

```rust
// 使用示例
graph.forward_node(loss)?;
graph.backward_nodes(&[w], loss)?;

// NEAT 变异：添加新节点
let new_node = graph.new_parameter_node(&[1, 1], Some("new"))?;
let new_add = graph.new_add_node(&[old_node, new_node], None)?;

// 通知拓扑变化（清除 Jacobi，保留 value）
graph.on_topology_changed();

// 继续训练
graph.forward_node(new_loss)?;
graph.backward_nodes(&[w, new_node], new_loss)?;
```

### ✅ 已完成：M4b Graph 级别种子 API

Graph 级别的种子管理已实现：

```rust
// 创建带种子的图（确定性）
let graph = Graph::new_with_seed(42);

// 或动态设置种子
let mut graph = Graph::new();
graph.set_seed(42);

// 参数创建自动使用 Graph 的 RNG
let w = graph.new_parameter_node(&[3, 2], Some("w"))?;

// Granular API 仍可覆盖
let b = graph.new_parameter_node_seeded(&[1, 1], Some("b"), 999)?;
```

**8 个新测试**验证了：确定性、NEAT 多图并行、种子覆盖等场景。

### 🎉 阶段二核心完成！

**已完成：**

- ✅ Sigmoid 激活节点 + `jacobi_diag()` 重构
- ✅ SoftmaxCrossEntropyLoss 融合节点（数值稳定）
- ✅ DataLoader 模块 + MNIST 数据集（自动下载/缓存）
- ✅ MNIST MLP MVP 集成测试（验证 loss 下降趋势）

**下一步：**

1. ~~实现 ReLU 激活节点~~ ✅ 已完成（LeakyReLU + ReLU）
2. ~~实现 Conv2d / Pooling 节点（CNN 基础）~~ ✅ 已完成
   - Conv2d: 支持 stride/padding，Jacobi+Batch 双模式
   - MaxPool2d: 稀疏梯度反传（记录最大值索引）
   - AvgPool2d: 均匀梯度分配
3. 实现 CNN Layer 便捷函数（conv_layer, pool_layer）
4. MNIST CNN 端到端示例（LeNet 风格）
5. 完善 MNIST MLP 示例（提升准确率，添加评估指标）

---

## 架构约束（为 NEAT 预留）

设计新节点时，牢记以下约束：

1. **节点必须可克隆** - NEAT 需要复制基因
2. **节点必须可序列化** - 保存/加载进化历史
3. **Graph 必须支持动态修改** - 运行时添加/删除节点
4. **避免全局状态** - 多个 Graph 实例可能并行进化

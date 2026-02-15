# Only Torch 架构路线图

> 最后更新: 2026-02-15
> 战略定位: **简化版 PyTorch in Rust**，为 NEAT 预留扩展性
> 当前阶段: **动态图架构已稳定，PyTorch 风格 API 已实现，性能调优阶段完成（含可选 BLAS 加速）**

## 文档索引

```
.doc/
├── architecture_roadmap.md              # ← 你在这里（主入口）
├── design/                              # 当前有效的设计文档
│   ├── neural_architecture_evolution_design.md      # ⭐ NEAT 神经架构演化（核心愿景）
│   ├── input_node_semantics_design.md              # 数据共享可视化设计
│   ├── api_layering_and_seed_design.md             # API 分层与种子管理
│   ├── batch_mechanism_design.md                   # Batch Forward/Backward 机制
│   ├── broadcast_mechanism_design.md               # 广播机制设计
│   ├── gradient_clear_and_accumulation_design.md   # 梯度清零与累积机制
│   ├── gradient_flow_control_design.md             # ⭐ 梯度流控制（detach/no_grad）
│   ├── data_loader_design.md                       # DataLoader 设计
│   ├── distributions_design.md                     # 概率分布模块设计
│   ├── graph_serialization_design.md               # 序列化与可视化
│   ├── memory_mechanism_design.md                  # 记忆/循环机制设计
│   ├── node_vs_layer_design.md                     # Node vs Layer 架构设计
│   ├── optimization_strategy.md                    # 性能优化策略
│   ├── optimizer_architecture_design.md            # 优化器架构
│   ├── rl_roadmap.md                               # RL 模块路线图
│   └── visualization_guide.md                      # 可视化使用指南
├── reference/                           # 参考资料
│   └── python_MatrixSlow_pid.md         # MatrixSlow 项目分析
└── _archive/                            # 已完成/历史文档
    ├── architecture_v2_design.md
    ├── autodiff_unification_design.md
    ├── dynamic_graph_lifecycle_design.md
    ├── future_enhancements.md           # 原未来功能规划（已全部完成/归档）
    ├── future_node_types.md             # 原节点扩展规划（已全部完成/归档）
    ├── graph_execution_refactor.md
    ├── graph_refactoring_design.md
    ├── input_node_unification_design.md
    └── multi_input_forward_design.md
```

---

## 当前状态概览

```
模块               完成度    状态
─────────────────────────────────
tensor/            ~90%     ✅ 基本完成（含 variance/pow/pad/repeat/square/reciprocal/log10/log2/relu6/hard_tanh/one_hot）
nn/graph           ~95%     ✅ 动态图架构 + PyTorch 风格 API + GraphError 精细化
nn/nodes           ~95%     ✅ 73 个节点类型（含完整激活函数族 + 归一化族 + 选择/排序）
nn/layer           ~90%     ✅ Linear/Conv2d/Pool/RNN/LSTM/GRU + BatchNorm/LayerNorm/RMSNorm/GroupNorm/InstanceNorm + Embedding/MultiHeadAttention
nn/debug           100%     ✅ 节点类型枚举 + 调试工具（strum 自动获取）
nn/optimizer       ~70%     ✅ SGD/Adam 可用（set_value_owned 零拷贝 + 可选 BLAS 加速）
data/              ~85%     ✅ MNIST + California Housing + DataLoader + Transform 管线（8 种数据增强）
vision/            ~70%     ✅ 基本完成
rl/                ~45%     ✅ GymEnv + SAC-Discrete（CartPole）+ SAC-Continuous（Pendulum）
logic/             0%       ❌ 预留
neat/              0%       ❌ 远期特色
```

## PyTorch 风格 API

### 核心组件

| 组件 | 说明 | 状态 |
|------|------|------|
| **Graph** | 计算图（参数注册表、种子管理） | ✅ |
| **Var** | 节点变量（Rc 引用计数管理生命周期） | ✅ |
| **Module** trait | 模型定义（parameters + forward） | ✅ |
| **Layer** | Linear/Conv2d/Pool/RNN/LSTM/GRU/BatchNorm/LayerNorm/RMSNorm/GroupNorm/InstanceNorm/Embedding/MultiHeadAttention | ✅ |
| **Optimizer** | SGD/Adam + with_params 选择性优化 | ✅ |
| **DetachedVar** | 轻量 detach 包装（GAN/RL 梯度隔离） | ✅ |
| **Loss** 方法 | `var.mse_loss()` / `var.cross_entropy()` / `var.bce_loss()` 等 | ✅ |

### 示例 API 风格

```rust
// 创建模型（PyTorch 风格）
let graph = Graph::new_with_seed(42);
let model = MyModel::new(&graph)?;
let mut optimizer = Adam::new(&graph, model.parameters(), 0.001);

// 训练循环
for (x, target) in &dataloader {
    optimizer.zero_grad()?;
    let output = model.forward(&x)?;
    let loss_val = output.mse_loss(&target)?.backward()?;
    optimizer.step()?;
}
```

### 示例覆盖

| 示例 | 功能验证 | 状态 |
|------|---------|------|
| `xor` | 基础 MLP | ✅ |
| `sine_regression` | 回归任务 | ✅ |
| `iris` | 多分类 | ✅ |
| `mnist` | CNN 图像分类 | ✅ |
| `mnist_gan` | GAN 训练 + detach | ✅ |
| `california_housing` | 房价回归 + DataLoader | ✅ |
| `parity_rnn_fixed_len` | RNN 展开式 | ✅ |
| `parity_rnn_var_len` | RNN 变长 + BucketedDataLoader | ✅ |
| `parity_lstm_var_len` | LSTM 变长 | ✅ |
| `parity_gru_var_len` | GRU 变长 | ✅ |
| `dual_input_add` | 多输入 | ✅ |
| `siamese_similarity` | 共享编码器 | ✅ |
| `dual_output_classify` | 多输出 + 多 Loss | ✅ |
| `multi_io_fusion` | 多输入 + 多输出 | ✅ |
| `multi_label_point` | BceLoss 多标签 | ✅ |
| `sac/cartpole` | SAC-Discrete 强化学习 | ✅ |
| `sac/pendulum` | SAC-Continuous 强化学习 | ✅ |

## 已实现节点

> 使用 `nn::debug::print_registered_node_types()` 可查看完整列表（自动从 `NodeType` 枚举获取）

| 类型 | 节点                                                        | 数量 |
| :--- | :---------------------------------------------------------- | :--: |
| 输入 | Input (Data/Target), Parameter, State                       |  3   |
| 算术 | Add, Subtract, Multiply, Divide, Negate, Pow, Square, Reciprocal |  8   |
| 矩阵/卷积 | MatMul, Conv2d, MaxPool2d, AvgPool2d                   |  4   |
| 形状 | Reshape, Flatten, Select, Gather, Narrow, Permute, Stack, Concat, Pad, Repeat |  10  |
| 选择 | TopK, SortNode                                              |  2   |
| 归约 | Maximum, Minimum, Amax, Amin, Sum, Mean                     |  6   |
| 激活 | Sigmoid, Tanh, ReLU, LeakyReLU, ReLU6, HardTanh, Softmax, LogSoftmax, SoftPlus, GELU, Swish, ELU, SELU, Mish, HardSwish, HardSigmoid, Step, Sign, Abs, Ln, Log10, Log2, Exp, Sqrt | 24 |
| 裁剪 | Clip                                                        |  1   |
| 条件 | WhereCond                                                   |  1   |
| 损失 | MSE, MAE, BCE, Huber, SoftmaxCrossEntropy                   |  5   |
| 归一化 | BatchNormOp, LayerNormOp, RMSNormOp                       |  3   |
| 辅助 | Identity, Dropout, ZerosLike, Detach                        |  4   |
| **合计** |                                                         | **71 + 2 复合** |

> 注：Input 和 Parameter 各含子类型，实际枚举变体总数 73。

> 运行 `nn::debug::print_registered_node_types()` 可查看完整列表（自动从 `NodeType` 枚举获取）。
>
> 原规划节点已全部完成，未来可按需添加新节点类型。

## 集成测试进度

> 对应 MatrixSlow Python 示例的 Rust 实现验证

| Rust 测试                          | 对应 MatrixSlow 示例          | 状态 | 说明                                |
| ---------------------------------- | ----------------------------- | :--: | ----------------------------------- |
| `test_adaline.rs`                  | `ch02/adaline.py`             |  ✅  | 最基础的计算图+自动微分             |
| `test_adaline_batch.rs`            | `ch03/adaline_batch.py`       |  ✅  | 批量处理                            |
| `test_optimizer_example.rs`        | `ch03/optimizer_example.py`   |  ✅  | SGD/Adam 优化器验证                 |
| `test_xor.rs`                      | -                             |  ✅  | **MVP 展示：非线性分类问题**        |
| `test_logistic_regression.rs`      | `ch04/logistic_regression.py` |  ❌  | 需要 Sigmoid 节点 (已有) + 测试代码 |
| `test_nn_iris.rs`                  | `ch05/nn_iris.py`             |  ❌  | 需要多层网络+Softmax                |
| `test_mnist.rs`                    | `ch05/nn_mnist.py`            |  ✅  | **MVP：MLP + SoftmaxCrossEntropy**  |
| `test_simple_regression_full_batch.rs` | -                         |  ✅  | **MSE 回归验证：y=2x+1（全批量）** |
| `test_california_housing_price.rs` | -                             |  ✅  | **California Housing 房价回归**     |

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

|  #  | 任务                 | 说明                                   | NEAT 友好性     | 状态 |
| :-: | :------------------- | :------------------------------------- | :-------------- | :--: |
| P1  | Softmax+CrossEntropy | 分类必需                               | ✅ 新节点       |  ✅  |
| P1b | Sigmoid 节点         | 通用激活                               | ✅ 新节点       |  ✅  |
| P1c | DataLoader + MNIST   | 数据加载                               | ✅ 基础设施     |  ✅  |
| P2  | LeakyReLU/ReLU 节点  | 底层 LeakyReLU + 便捷 ReLU (slope=0.0) | ✅ 新节点       |  ✅  |
| P3  | Reshape/Flatten 节点 | CNN 数据流转换（PyTorch 风格）         | ✅ 结构操作     |  ✅  |
| P4  | Conv2d 节点          | PyTorch 风格（多通道内部处理）         | ✅ VJP 模式     |  ✅  |
| P5  | Pooling 节点         | MaxPool2d/AvgPool2d                    | ✅ VJP 模式     |  ✅  |
| P6  | MNIST CNN 端到端     | LeNet 风格                             | ✅ 验证         |  🔄  |

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
│   ├── graph/       # 计算图（动态图 + Rc 生命周期）✅
│   ├── nodes/       # 73 个节点类型 ✅
│   ├── layer/       # Linear/Conv2d/Pool/RNN/LSTM/GRU/Norm/Embedding/Attention ✅
│   ├── optimizer/   # SGD/Adam ✅
│   └── module       # Module trait ✅
├── vision/          # 视觉处理 ✅
├── data/            # DataLoader + MNIST + California Housing + Transform 管线 ✅
├── rl/              # GymEnv + SAC-Discrete + SAC-Continuous ✅
├── neat/            # 神经进化（远期核心）
└── logic/           # 逻辑推理（预留）
```

---

## 未来方向

| 优先级 | 功能 | 状态 | 详细设计 |
|:------:|------|------|----------|
| 🔴 高 | **NEAT 神经架构演化** | 待实现 | [neural_architecture_evolution_design.md](design/neural_architecture_evolution_design.md) |
| 🟡 中 | **数据共享可视化** | 待实现 | [input_node_semantics_design.md](design/input_node_semantics_design.md) |
| 💤 暂缓 | **过程宏** `#[derive(Model)]` | 等待 RFC 3698 | 样板代码仅 `parameters()` 约 5 行，投入产出比低 |
| 💤 暂缓 | **强化学习改良** | 基础完成 | [rl_roadmap.md](design/rl_roadmap.md) |

---

## 新增节点检查清单

新增节点时，确保完成以下步骤：

- [ ] `src/tensor/ops/` — Tensor 级方法 + 测试
- [ ] `src/nn/nodes/raw_node/ops/xxx.rs` — 节点实现（`TraitNode` trait）
- [ ] `src/nn/nodes/raw_node/ops/mod.rs` — 模块导出
- [ ] `src/nn/nodes/raw_node/mod.rs` — `define_node_types!` 添加变体（含元数据）
- [ ] `src/nn/graph/inner/node_builders.rs` — `create_xxx_node()` 构建方法
- [ ] `src/nn/var/ops/` — Var 扩展方法（如需要）
- [ ] `src/nn/tests/node_xxx.rs` — 节点测试 + `mod.rs` 注册
- [ ] Python 对照值（复杂计算验证）
- [ ] `cargo test` 全部通过

---

## 架构约束（为 NEAT 预留）

设计新节点时，牢记以下约束：

1. **节点必须可克隆** - NEAT 需要复制基因
2. **节点必须可序列化** - 保存/加载进化历史
3. **Graph 必须支持动态修改** - 运行时添加/删除节点
4. **避免全局状态** - 多个 Graph 实例可能并行进化

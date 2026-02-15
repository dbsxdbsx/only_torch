# 待扩展节点类型规划

> **状态**：✅ 大部分已完成
> **创建日期**：2026-02-01
> **最后更新**：2026-02-15
> **目标**：记录主流深度学习框架中常见但当前项目尚未实现的节点类型，供后期按需扩展

---

## 相关文档

- [architecture_roadmap.md](../architecture_roadmap.md) - 项目主入口，整体架构路线图
- [dynamic_graph_lifecycle_design.md](../_archive/dynamic_graph_lifecycle_design.md) - 动态图生命周期设计（已归档）
- [node_vs_layer_design.md](./node_vs_layer_design.md) - Node vs Layer 架构设计

---

## 目录

1. [当前已实现概览](#1-当前已实现概览)
2. [待扩展节点列表](#2-待扩展节点列表)
3. [优先级说明](#3-优先级说明)
4. [实现参考](#4-实现参考)
5. [调试工具](#5-调试工具)

---

## 1. 当前已实现概览

截至 2026-02-15，项目已实现 **73 个节点类型**（详见 `NodeType` 枚举）+ 多种 Var 级操作：

| 类别 | 数量 | 节点 |
|------|------|------|
| 输入/参数/状态 | 3 | Input、Parameter、State |
| 算术运算 | 7 | Add、Subtract、Multiply、Divide、Negate、Pow、Square、Reciprocal |
| 矩阵/卷积 | 4 | MatMul、Conv2d、MaxPool2d、AvgPool2d |
| 形状变换 | 10 | Reshape、Flatten、Select、Gather、Stack、Concat、Narrow、Permute、Pad、Repeat |
| 选择/排序 | 2 | TopK、SortNode |
| 比较/归约 | 6 | Maximum、Minimum、Amax、Amin、Sum、Mean |
| 激活函数 | 26 | Sigmoid、Tanh、ReLU、LeakyReLU、Softmax、LogSoftmax、SoftPlus、Step、Sign、Abs、Ln、Log10、Log2、Exp、Sqrt、GELU、Swish、ELU、SELU、Mish、HardSwish、HardSigmoid、ReLU6、HardTanh |
| 裁剪 | 1 | Clip |
| 条件 | 1 | WhereCond |
| 损失函数 | 5 | MSE、MAE、BCE、Huber、SoftmaxCrossEntropy |
| 归一化 | 3 | BatchNormOp、LayerNormOp、RMSNormOp |
| 辅助节点 | 4 | Identity、Detach、Dropout、ZerosLike |

> 此外还有 Var 级操作（无独立 NodeType）：Squeeze、Unsqueeze、Split、Chunk
>
> Layer 层（组合节点构建）：Linear、Conv2d、MaxPool2d、AvgPool2d、RNN、LSTM、GRU、
> BatchNorm、LayerNorm、RMSNorm、GroupNorm、InstanceNorm、Embedding、MultiHeadAttention

---

## 2. 待扩展节点列表

### ~~2.1 数学运算节点~~ （已全部完成）

| 节点 | 公式 | 用途 | 状态 |
|------|------|------|--------|
| ~~**Exp**~~ | `y = e^x` | SAC：`log_std.exp()` → std | ✅ 已实现 |
| ~~**Sqrt**~~ | `y = √x` | Adam 优化器：`sqrt(v + ε)` | ✅ 已实现 |
| ~~**Pow**~~ | `y = x^n` | 通用幂运算 | ✅ 已实现 |
| ~~**Square**~~ | `y = x²` | 独立节点（非 pow 组合），梯度 2x | ✅ 已实现 |
| ~~**Reciprocal**~~ | `y = 1/x` | 独立节点（非 pow 组合），梯度 -1/x² | ✅ 已实现 |
| ~~**Log10/Log2**~~ | `y = log₁₀(x)` / `y = log₂(x)` | 特定场景的对数计算 | ✅ 已实现 |

### ~~2.2 裁剪/限制节点~~ （已全部完成）

| 节点 | 公式 | 用途 | 状态 |
|------|------|------|--------|
| ~~**Clamp/Clip**~~ | `y = clamp(x, min, max)` | SAC：`log_std` 裁剪；PPO：ratio clipping | ✅ 已实现 |
| ~~**ReLU6**~~ | `y = min(max(0, x), 6)` | MobileNet 等轻量网络，独立节点 | ✅ 已实现 |
| ~~**HardTanh**~~ | `y = clamp(x, min, max)` | 硬 Tanh 激活，独立节点 | ✅ 已实现 |

### ~~2.3 形状/维度操作节点~~ （已全部完成）

| 节点 | 功能 | 用途 | 状态 |
|------|------|------|--------|
| ~~**Transpose/Permute**~~ | 维度转置 | 多头注意力：`[B,H,S,D] ↔ [B,S,H,D]` | ✅ 已实现 |
| ~~**Squeeze**~~ | 移除大小为 1 的维度 | `[1,3,1] → [3]` | ✅ 已实现（Var 方法） |
| ~~**Unsqueeze**~~ | 插入大小为 1 的维度 | `[3] → [1,3,1]` | ✅ 已实现（Var 方法） |
| ~~**Split**~~ | 张量分割 | 多输出网络、GRU 门分离 | ✅ 已实现（Var 方法） |
| ~~**Chunk**~~ | 等分分割 | 类似 Split，但按数量等分 | ✅ 已实现（Var 方法） |
| ~~**Repeat/Tile**~~ | 张量重复 | 广播扩展 | ✅ 已实现（独立 Node） |
| ~~**Pad**~~ | 常量值填充 | CNN same-padding / 序列对齐 | ✅ 已实现（独立 Node） |

### ~~2.4 激活函数节点~~ （已全部完成）

> **状态**：✅ 全部实现，位于 `src/nn/nodes/raw_node/ops/`。

### ~~2.5 归一化层节点~~ （已全部完成）

| 节点 | 功能 | 用途 | 状态 |
|------|------|------|--------|
| ~~**LayerNorm**~~ | 层归一化 | Transformer 标准组件 | ✅ 已实现（Node + Layer） |
| ~~**BatchNorm**~~ | 批归一化 | CNN 标准组件 | ✅ 已实现（Node + Layer） |
| ~~**InstanceNorm**~~ | 实例归一化 | 风格迁移、GAN | ✅ 已实现（Layer） |
| ~~**GroupNorm**~~ | 组归一化 | 小 batch 场景 | ✅ 已实现（Layer） |
| ~~**RMSNorm**~~ | RMS 归一化 | LLaMA 等现代 LLM | ✅ 已实现（Node + Layer） |

### ~~2.6 序列/NLP 相关节点~~ （已全部完成）

| 节点 | 功能 | 用途 | 状态 |
|------|------|------|--------|
| ~~**Embedding**~~ | 词嵌入查表 | NLP 任务输入层 | ✅ 已实现（Layer） |
| ~~**Attention**~~ | 多头注意力 | Transformer 核心组件 | ✅ 已实现（Layer: MultiHeadAttention） |
| ~~**Pad**~~ | 填充 | 序列对齐、卷积边界处理 | ✅ 已实现（独立 Node） |

### ~~2.7 概率分布节点（模块级）~~ （已完成）

> **状态**：✅ 已全部实现，位于 `src/nn/distributions/`。

### ~~2.8 其他节点~~ （已全部完成）

| 节点 | 功能 | 用途 | 状态 |
|------|------|------|--------|
| ~~**Where/Cond**~~ | 条件选择 | `y = cond ? a : b` | ✅ 已实现 |
| ~~**OneHot**~~ | 独热编码 | 分类标签转换 | ✅ 已实现（Tensor 方法，非计算图节点） |
| ~~**TopK**~~ | 取前 K 个最大值 | Beam Search、采样 | ✅ 已实现 |
| ~~**Sort/ArgSort**~~ | 排序 | 排序相关任务 | ✅ 已实现 |

---

## 3. 优先级说明

> **全部原规划节点已实现完毕。** 以下记录仅为历史参考。

### ✅ 已全部完成

原高/中/低优先级的所有节点均已实现。详见上方各节的 ✅ 标记。

---

## 4. 实现参考

### 4.1 实现模式

新节点应遵循现有实现模式：

```rust
// 1. 在 src/tensor/ops/ 添加 Tensor 级方法
// 2. 在 src/nn/nodes/raw_node/ops/ 创建节点文件
// 3. 实现 TraitNode trait
// 4. 在 define_node_types! 宏中添加变体
// 5. 在 GraphInner 中添加 create_xxx_node() 方法
// 6. 在 var/ops 中添加 Var 扩展方法
// 7. 添加 Tensor 测试 + Node 测试
```

### 4.2 外部参考

- PyTorch 文档：https://pytorch.org/docs/stable/nn.functional.html
- JAX 文档：https://jax.readthedocs.io/en/latest/jax.numpy.html
- 项目内参考：`MatrixSlow/matrixslow/ops/ops.py`

---

## 5. 调试工具

项目提供了 `nn::debug` 模块用于查看已注册的节点类型。

**使用 `strum` 自动获取**：节点列表直接从 `NodeType` 枚举自动提取，无需手动维护。

```rust
use only_torch::nn::debug::{
    describe_registered_node_types,  // 获取结构化列表
    print_registered_node_types,      // 直接打印
    get_node_type_summary,            // 获取分类统计
    node_type_count,                  // 获取节点数量（编译时常量）
    check_missing_metadata,           // 检查缺失的元数据
};
```

**添加新节点时**：
1. 在 `define_node_types!` 宏中添加变体 → 自动被 `strum` 识别
2. 元数据由宏内联声明，无需额外维护 `debug.rs`
3. `test_all_nodes_have_metadata` 测试会检查一致性

---

## 附录：节点实现检查清单

新增节点时，确保完成以下步骤：

- [ ] `src/tensor/ops/` — Tensor 级方法 + 测试
- [ ] `src/nn/nodes/raw_node/ops/xxx.rs` — 节点实现
- [ ] `src/nn/nodes/raw_node/ops/mod.rs` — 模块导出
- [ ] `src/nn/nodes/raw_node/mod.rs` — `define_node_types!` 添加变体（含元数据）
- [ ] `src/nn/graph/inner/node_builders.rs` — GraphInner 构建方法
- [ ] `src/nn/var/ops/xxx.rs` — Var 扩展方法（如需要）
- [ ] `src/nn/tests/node_xxx.rs` — 节点测试文件
- [ ] `src/nn/tests/mod.rs` — 测试模块注册
- [ ] Python 对照测试（复杂计算验证）
- [ ] `cargo test` 全部通过

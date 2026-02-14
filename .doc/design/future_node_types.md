# 待扩展节点类型规划

> **状态**：规划中  
> **创建日期**：2026-02-01  
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

截至 2026-02-12，项目已实现 **41 个节点类型**（详见 `NodeType` 枚举）：

| 类别 | 数量 | 节点 |
|------|------|------|
| 输入/参数/状态 | 3 | Input、Parameter、State |
| 算术运算 | 4 | Add、Subtract、Multiply、Divide |
| 矩阵/卷积 | 4 | MatMul、Conv2d、MaxPool2d、AvgPool2d |
| 形状变换 | 5 | Reshape、Flatten、Select、Gather、Stack |
| 比较/归约 | 6 | Maximum、Minimum、Amax、Amin、Sum、Mean |
| 激活函数 | 10 | Sigmoid、Tanh、LeakyReLU、Softmax、LogSoftmax、SoftPlus、Step、Sign、Abs、Ln |
| 损失函数 | 5 | MSE、MAE、BCE、Huber、SoftmaxCrossEntropy |
| 辅助节点 | 4 | Identity、Detach、Dropout、ZerosLike |

---

## 2. 待扩展节点列表

### 2.1 数学运算节点

| 节点 | 公式 | 用途 | 优先级 |
|------|------|------|--------|
| ~~**Exp**~~ | `y = e^x` | SAC：`log_std.exp()` → std | ✅ 已实现 |
| ~~**Sqrt**~~ | `y = √x` | Adam 优化器：`sqrt(v + ε)` | ✅ 已实现 |
| **Pow** | `y = x^n` | 通用幂运算 | 🟡 中 |
| **Square** | `y = x²` | 可用 `x * x` 替代，但语义更清晰 | 🟢 低 |
| **Reciprocal** | `y = 1/x` | 可用 `1.0 / x` 替代 | 🟢 低 |
| **Log10/Log2** | `y = log₁₀(x)` | 特定场景的对数计算 | 🟢 低 |

### 2.2 裁剪/限制节点

| 节点 | 公式 | 用途 | 优先级 |
|------|------|------|--------|
| ~~**Clamp/Clip**~~ | `y = clamp(x, min, max)` | SAC：`log_std` 裁剪；PPO：ratio clipping | ✅ 已实现 |
| **ReLU6** | `y = min(max(0, x), 6)` | MobileNet 等轻量网络 | 🟢 低 |
| **HardTanh** | `y = clamp(x, -1, 1)` | 硬 Tanh 激活 | 🟢 低 |

### 2.3 形状/维度操作节点

| 节点 | 功能 | 用途 | 优先级 |
|------|------|------|--------|
| **Transpose/Permute** | 维度转置 | 多头注意力：`[B,H,S,D] ↔ [B,S,H,D]` | 🟡 中 |
| **Squeeze** | 移除大小为 1 的维度 | `[1,3,1] → [3]` | 🟡 中 |
| **Unsqueeze** | 插入大小为 1 的维度 | `[3] → [1,3,1]` | 🟡 中 |
| **Split** | 张量分割 | 多输出网络、GRU 门分离 | 🟡 中 |
| **Chunk** | 等分分割 | 类似 Split，但按数量等分 | 🟢 低 |
| **Repeat/Tile** | 张量重复 | 广播扩展 | 🟢 低 |

### 2.4 激活函数节点

| 节点 | 公式 | 用途 | 优先级 |
|------|------|------|--------|
| **GELU** | `y = x · Φ(x)` | Transformer 标准激活（BERT、GPT） | 🟡 中 |
| **ELU** | `y = x if x>0 else α(e^x-1)` | 平滑 ReLU 变体 | 🟢 低 |
| **SELU** | `y = λ · ELU(x)` | 自归一化网络 | 🟢 低 |
| **Swish/SiLU** | `y = x · sigmoid(x)` | EfficientNet、Transformer 变体 | 🟡 中 |
| **Mish** | `y = x · tanh(softplus(x))` | YOLO v4+ | 🟢 低 |
| **HardSwish** | `y = x · ReLU6(x+3)/6` | MobileNet v3 | 🟢 低 |
| **HardSigmoid** | `y = ReLU6(x+3)/6` | 轻量 Sigmoid 近似 | 🟢 低 |

### 2.5 归一化层节点

| 节点 | 功能 | 用途 | 优先级 |
|------|------|------|--------|
| **LayerNorm** | 层归一化 | Transformer 标准组件 | 🟡 中 |
| **BatchNorm** | 批归一化 | CNN 标准组件 | 🟡 中 |
| **InstanceNorm** | 实例归一化 | 风格迁移、GAN | 🟢 低 |
| **GroupNorm** | 组归一化 | 小 batch 场景 | 🟢 低 |
| **RMSNorm** | RMS 归一化 | LLaMA 等现代 LLM | 🟢 低 |

### 2.6 序列/NLP 相关节点

| 节点 | 功能 | 用途 | 优先级 |
|------|------|------|--------|
| **Embedding** | 词嵌入查表 | NLP 任务输入层 | 🟢 低 |
| **Attention** | 注意力计算 | Transformer 核心组件 | 🟢 低 |
| **Pad** | 填充 | 序列对齐、卷积边界处理 | 🟢 低 |

### ~~2.7 概率分布节点（模块级）~~ （已完成）

> **状态**：✅ 已全部实现，位于 `src/nn/distributions/`。
> 详见 [概率分布模块设计](./distributions_design.md)、[RL 路线图](./rl_roadmap.md)。
>
> SAC-Discrete、SAC-Continuous、Hybrid SAC 三个示例均已使用。

| 分布 | 状态 |
|------|------|
| ~~**Normal**~~ | ✅ 已实现（rsample / log_prob / entropy） |
| ~~**TanhNormal**~~ | ✅ 已实现（Squashed Gaussian + Jacobian 修正） |
| ~~**Categorical**~~ | ✅ 已实现（probs / log_probs / entropy / sample） |

### 2.8 其他可能需要的节点

| 节点 | 功能 | 用途 | 优先级 |
|------|------|------|--------|
| **Where/Cond** | 条件选择 | `y = cond ? a : b` | 🟡 中 |
| **OneHot** | 独热编码 | 分类标签转换 | 🟢 低 |
| **TopK** | 取前 K 个最大值 | Beam Search、采样 | 🟢 低 |
| **Sort/ArgSort** | 排序 | 排序相关任务 | 🟢 低 |

---

## 3. 优先级说明

### ~~🔴 原高优先级（强化学习必需）~~ — 已全部完成

以下节点和模块最初作为 SAC-Continuous / Hybrid SAC 的核心依赖而列为高优先级，现已全部实现：

- ~~Exp~~ ✅ — `src/nn/nodes/raw_node/ops/exp.rs`
- ~~Clamp/Clip~~ ✅ — `src/nn/nodes/raw_node/ops/clip.rs`
- ~~Sqrt~~ ✅ — `src/nn/nodes/raw_node/ops/sqrt.rs`
- ~~Normal 分布~~ ✅ — `src/nn/distributions/normal.rs`
- ~~TanhNormal 分布~~ ✅ — `src/nn/distributions/tanh_normal.rs`
- ~~Categorical 分布~~ ✅ — `src/nn/distributions/categorical.rs`

> SAC-Discrete、SAC-Continuous、Hybrid SAC 三个示例均已完成并可运行。
> RL 后续改良方向详见 [RL 路线图](./rl_roadmap.md)。

### 🟡 中优先级（功能扩展）

这些用于**扩展模型能力**，支持更复杂的网络架构：

- ~~**Categorical 分布**~~ ✅ 已实现
- **Transpose/Permute** - 多头注意力、复杂张量操作
- **LayerNorm/BatchNorm** - Transformer、CNN 架构
- **GELU/Swish** - 现代激活函数
- **Split/Squeeze/Unsqueeze** - 灵活的张量操作
- **Where/Cond** - 条件分支

### 🟢 低优先级（锦上添花）

这些用于**特定场景**或有简单替代方案：

- **Square** - 可用 `x * x` 替代
- **ELU/SELU/Mish** - 特定网络架构
- **Embedding** - NLP 任务
- **TopK/Sort** - 特定算法

---

## 4. 实现参考

### 4.1 实现模式

新节点应遵循现有实现模式：

```rust
// 1. 在 src/nn/nodes/raw_node/ops/ 创建节点文件
// 2. 实现 new_from_shapes() 工厂方法
// 3. 实现 TraitNode trait
// 4. 在 NodeType 枚举中添加变体
// 5. 在 GraphInner 中添加 new_xxx_node() 方法
// 6. 在 var/ops 中添加 Var 扩展方法
// 7. 添加测试文件 src/nn/tests/node_xxx.rs
```

### 4.2 参考实现

| 节点类型 | 参考现有节点 |
|---------|-------------|
| Exp | Ln（单输入数学函数） |
| Sqrt | Abs（单输入数学函数） |
| Clamp | 需新实现（三参数：input, min, max） |
| Transpose | Reshape（形状变换） |
| LayerNorm | 需新实现（含可学习参数） |
| GELU | Sigmoid（激活函数） |

### 4.3 外部参考

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

// 方式 1：直接打印（调试用）
print_registered_node_types();

// 方式 2：获取结构化数据（程序化处理）
let nodes = describe_registered_node_types();
for node in &nodes {
    println!("{} [{}]: {}", node.name, node.category, node.description);
}

// 方式 3：获取统计信息
let summary = get_node_type_summary();
// [("输入", 1), ("参数", 1), ("激活", 10), ...]
```

**添加新节点时**：
1. 在 `NodeType` 枚举中添加变体 → 自动被 `strum` 识别
2. 在 `debug.rs` 的 `get_node_metadata()` 中添加描述 → 提供类别和 Var 方法信息
3. 如果忘记添加描述，`test_all_nodes_have_metadata` 测试会失败并提示

---

## 附录：节点实现检查清单

新增节点时，确保完成以下步骤：

- [ ] `src/nn/nodes/raw_node/ops/xxx.rs` - 节点实现（含 `new_from_shapes()`）
- [ ] `src/nn/nodes/raw_node/ops/mod.rs` - 模块导出
- [ ] `src/nn/nodes/raw_node/mod.rs` - NodeType 枚举添加变体（strum 自动识别）
- [ ] `src/nn/debug.rs` - `get_node_metadata()` 添加描述（否则测试失败提醒）
- [ ] `src/nn/graph/inner/node_builders.rs` - GraphInner 构建方法
- [ ] `src/nn/var/ops/xxx.rs` - Var 扩展方法（如需要）
- [ ] `src/nn/tests/node_xxx.rs` - 测试文件
- [ ] `src/nn/tests/mod.rs` - 测试模块注册
- [ ] Python 对照测试（复杂计算验证）
- [ ] 运行 `cargo test tests::debug::` 确认元数据已添加

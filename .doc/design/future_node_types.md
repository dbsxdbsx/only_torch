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

截至 2026-02-01，项目已实现 **40 个节点类型**（详见 `NodeType` 枚举）：

| 类别 | 数量 | 节点 |
|------|------|------|
| 输入/参数/状态 | 3 | Input、Parameter、State |
| 算术运算 | 4 | Add、Subtract、Multiply、Divide |
| 矩阵/卷积 | 4 | MatMul、Conv2d、MaxPool2d、AvgPool2d |
| 形状变换 | 5 | Reshape、Flatten、Select、Gather、Stack |
| 比较/归约 | 6 | Maximum、Minimum、Amax、Amin、Sum、Mean |
| 激活函数 | 10 | Sigmoid、Tanh、LeakyReLU、Softmax、LogSoftmax、SoftPlus、Step、Sign、Abs、Ln |
| 损失函数 | 5 | MSE、MAE、BCE、Huber、SoftmaxCrossEntropy |
| 辅助节点 | 3 | Identity、Dropout、ZerosLike |

---

## 2. 待扩展节点列表

### 2.1 数学运算节点

| 节点 | 公式 | 用途 | 优先级 |
|------|------|------|--------|
| **Exp** | `y = e^x` | SAC：`log_std.exp()` → std | 🔴 高 |
| **Sqrt** | `y = √x` | Adam 优化器：`sqrt(v + ε)` | 🔴 高 |
| **Pow** | `y = x^n` | 通用幂运算 | 🟡 中 |
| **Square** | `y = x²` | 可用 `x * x` 替代，但语义更清晰 | 🟢 低 |
| **Reciprocal** | `y = 1/x` | 可用 `1.0 / x` 替代 | 🟢 低 |
| **Log10/Log2** | `y = log₁₀(x)` | 特定场景的对数计算 | 🟢 低 |

### 2.2 裁剪/限制节点

| 节点 | 公式 | 用途 | 优先级 |
|------|------|------|--------|
| **Clamp/Clip** | `y = clamp(x, min, max)` | SAC：`log_std` 裁剪 (`-20` ~ `2`)；PPO：ratio clipping | 🔴 高 |
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

### 2.7 概率分布节点（模块级）

> **说明**：以下不是单个计算图节点，而是需要作为**独立模块**（`src/nn/distributions/`）实现的概率分布。
> 它们组合已有节点（Exp、Ln、Tanh 等）提供高层 API，是 SAC-Continuous / Hybrid SAC 的**最大缺口**。
>
> **参考实现**：PyTorch `torch.distributions`、rustRL 使用的 `tch-distr` crate。

| 分布 | 核心方法 | 用途 | 优先级 |
|------|---------|------|--------|
| **Normal** | `rsample()`（重参数化采样）、`log_prob()`、`entropy()` | SAC-Continuous Actor：连续动作采样 | 🔴 高 |
| **TanhNormal** | `sample()` + tanh squash + log_prob 修正 | SAC-Continuous：Squashed Gaussian 策略（Haarnoja 2018 Appendix C） | 🔴 高 |
| **Categorical** | `sample()`、`log_prob()`、`entropy()` | SAC-Discrete / Hybrid：离散动作采样（当前 cartpole_sac 用纯 Tensor 手工实现，Hybrid 需计算图内版本） | 🟡 中 |

**TanhNormal log_prob 修正公式**（Enforcing Action Bounds）：

```
log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u) + ε)
```

其中 `u` 是 squash 前的原始采样值，`a = tanh(u)`。

**依赖关系**：Normal / TanhNormal 依赖 §2.1 的 **Exp** 节点和 §2.2 的 **Clamp** 节点。

### 2.8 其他可能需要的节点

| 节点 | 功能 | 用途 | 优先级 |
|------|------|------|--------|
| **Where/Cond** | 条件选择 | `y = cond ? a : b` | 🟡 中 |
| **OneHot** | 独热编码 | 分类标签转换 | 🟢 低 |
| **TopK** | 取前 K 个最大值 | Beam Search、采样 | 🟢 低 |
| **Sort/ArgSort** | 排序 | 排序相关任务 | 🟢 低 |

---

## 3. 优先级说明

### 🔴 高优先级（强化学习必需）

这些是 SAC-Continuous / Hybrid SAC / PPO 等强化学习算法的**核心依赖**：

**计算图节点**（缺失会直接阻塞算法实现）：

1. **Exp** - SAC Actor 的 `log_std.exp()` → std 转换
2. **Clamp** - SAC 的 `log_std` 裁剪（`[-20, 2]`），PPO 的 ratio clipping（`[1-ε, 1+ε]`）
3. **Sqrt** - Adam 优化器的 `sqrt(v + ε)`（如果内置优化器需要）

**概率分布模块**（比单个节点更关键的系统性缺口）：

4. **Normal 分布** - 连续动作 SAC 的重参数化采样 + log_prob
5. **TanhNormal 分布** - Squashed Gaussian 策略（标准 SAC-Continuous 必需）

> **注**：SAC-Discrete（当前 cartpole_sac 示例）不需要以上任何内容，已有 softmax / log_softmax 足够。
> 以上仅在扩展到 SAC-Continuous / Hybrid SAC 时才需要。

**临时替代方案**：
- Exp：可用 `softplus(x) ≈ ln(1 + e^x)` 近似，但不精确
- Clamp：可用 `maximum(minimum(x, max), min)` 组合实现
- Sqrt：Adam 可在 Tensor 级别实现，无需计算图节点
- Normal/TanhNormal：无替代方案，必须实现

### 🟡 中优先级（功能扩展）

这些用于**扩展模型能力**，支持更复杂的网络架构：

- **Categorical 分布** - Hybrid SAC 中离散部分需要在计算图内采样
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
// 6. 在 var_ops 中添加 Var 扩展方法
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
- [ ] `src/nn/var_ops/xxx.rs` - Var 扩展方法（如需要）
- [ ] `src/nn/tests/node_xxx.rs` - 测试文件
- [ ] `src/nn/tests/mod.rs` - 测试模块注册
- [ ] Python 对照测试（复杂计算验证）
- [ ] 运行 `cargo test tests::debug::` 确认元数据已添加

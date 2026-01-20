# 广播机制设计

> 最后更新: 2026-01-19
> 状态: **已完成**（Tensor 层 + 工具函数 + Node 层 + Layer 层简化）
> 影响范围: Tensor 层、Node 层、Layer 层、NEAT 演化

---

## 核心决策

**Only Torch 采用完整的 NumPy 风格广播机制。**

- **Tensor 层**：直接使用 ndarray 原生广播（已验证可用）
- **Node 层**：Forward 支持广播，Backward 对广播维度求和
- **NEAT 兼容**：广播是"执行语义"，不影响"结构语义"

---

## 设计背景

### 为什么改变原有设计？

原设计采用"显式节点广播"（如 `ones @ b`），存在以下问题：

| 问题 | 说明 |
|---|---|
| **图膨胀** | 每次 forward 创建新的 ones 节点 |
| **复杂性** | 用户需要手动处理广播逻辑 |
| **不必要** | ndarray 原生已支持完整广播 |

### 关键发现

经过验证，ndarray 原生支持完整 NumPy 广播：

```rust
// ndarray 原生广播测试结果
[3, 2] + [1, 2]       ✅ batch 广播
[3, 2] + [2]          ✅ 维度数不同
[3, 2] * [1, 2]       ✅ 乘法广播
[2,3,2,2] + [1,3,1,1] ✅ Conv bias 广播
```

**✅ 已解决**：原 Tensor 层的 `is_same_shape` 限制已移除，现已启用完整广播支持。

---

## NumPy 广播规则

### 规则定义

1. **从右向左对齐维度**
2. **维度兼容条件**：相等，或其中一个为 1
3. **维度数不同时**：较短的形状前面补 1

### 示例

| 形状 A | 形状 B | 对齐后 B | 结果形状 | 兼容？ |
|---|---|---|---|---|
| `[32, 128]` | `[128]` | `[1, 128]` | `[32, 128]` | ✅ |
| `[32, 128]` | `[1, 128]` | - | `[32, 128]` | ✅ |
| `[32, 128]` | `[32, 1]` | - | `[32, 128]` | ✅ |
| `[2, 3, 4]` | `[3, 4]` | `[1, 3, 4]` | `[2, 3, 4]` | ✅ |
| `[2, 3, 4]` | `[3, 1]` | `[1, 3, 1]` | `[2, 3, 4]` | ✅ |
| `[32, 128]` | `[32, 64]` | - | - | ❌ |

### 广播满足交换律

```
A: [32, 128]
B: [1, 128]

A + B 的结果 == B + A 的结果  ✅
```

---

## 分层设计

### 第 1 层：Tensor 层 ✅ 已实现

**策略**：移除形状检查，使用 ndarray 原生广播 + 自定义错误信息。

```rust
// 修改后（当前实现）
fn add_within_tensors(a: &Tensor, b: &Tensor) -> Tensor {
    // 检查广播兼容性，使用自定义错误信息
    assert!(
        a.can_broadcast_with(b),
        "{}",
        TensorError::IncompatibleShape  // "张量形状不兼容"
    );
    // 使用 ndarray 原生广播
    Tensor { data: &a.data + &b.data }
}
```

**新增的 Tensor 方法**（`tensor/property.rs`）：

```rust
/// 判断两个张量是否可以广播（用于 +, -, *, /）
pub fn can_broadcast_with(&self, other: &Self) -> bool

/// 判断 other 是否可以广播到 self 的形状（用于 +=, -=, *=, /=）
pub fn can_assign_broadcast_from(&self, other: &Self) -> bool
```

**已修改的文件**：
- `tensor/ops/add.rs`, `tensor/ops/add_assign.rs`
- `tensor/ops/sub.rs`, `tensor/ops/sub_assign.rs`
- `tensor/ops/mul.rs`, `tensor/ops/mul_assign.rs`
- `tensor/ops/div.rs`, `tensor/ops/div_assign.rs`
- `tensor/property.rs`（新增广播检查方法）
- `tensor/tests/*.rs`（更新所有单元测试）

**新增的 Python 参考脚本**（`tests/python/tensor_reference/`）：
- `tensor_add_broadcast_reference.py`
- `tensor_add_assign_broadcast_reference.py`
- `tensor_sub_broadcast_reference.py`
- `tensor_sub_assign_broadcast_reference.py`
- `tensor_mul_broadcast_reference.py`
- `tensor_mul_assign_broadcast_reference.py`
- `tensor_div_broadcast_reference.py`
- `tensor_div_assign_broadcast_reference.py`

### 第 2 层：Node 层

#### Forward（前向传播）

**策略**：允许广播兼容的形状，计算广播后的输出形状。

```rust
impl Add {
    pub fn new(parents: &[&NodeHandle]) -> Result<Self, GraphError> {
        // 计算广播后的输出形状
        let shape = broadcast_shape(
            parents[0].value_expected_shape(),
            parents[1].value_expected_shape()
        )?;
        Ok(Self { shape, ... })
    }
}
```

#### Backward（反向传播）⚠️ 关键

**策略**：对被广播的维度求和。

```rust
impl TraitNode for Add {
    fn calc_grad_to_parent(
        &self,
        target_parent: &NodeHandle,
        upstream_grad: &Tensor,
        ...
    ) -> Result<Tensor, GraphError> {
        let target_shape = target_parent.value_expected_shape();

        if target_shape == upstream_grad.shape() {
            // 形状匹配，直接传递
            Ok(upstream_grad.clone())
        } else {
            // 被广播过，需要对广播维度求和
            Ok(sum_to_shape(upstream_grad, target_shape))
        }
    }
}
```

**梯度求和示例**：

```
Forward:  [32, 128] + [1, 128] → [32, 128]

Backward:
  dL/dA = upstream_grad           # [32, 128]（直接传递）
  dL/dB = sum(upstream_grad, axis=0, keepdims=true)  # [1, 128]
```

### 第 3 层：Layer 层

**策略**：简化实现，删除手动广播代码。

```rust
// 修改前（当前实现）
impl Linear {
    pub fn forward(&self, x: &Var) -> Var {
        let xw = x.matmul(&self.weights)?;

        // 手动广播 bias
        let ones = graph.ones(&[batch_size, 1])?;
        let bias_broadcast = ones.matmul(&self.bias)?;

        &xw + &bias_broadcast
    }
}

// 修改后
impl Linear {
    pub fn forward(&self, x: &Var) -> Var {
        let xw = x.matmul(&self.weights)?;
        &xw + &self.bias  // Add 节点自动广播
    }
}
```

---

## NEAT 兼容性分析

### 核心问题

> 广播是否影响 NEAT 节点层面的进化？

**答案：不影响。**

### 理由

| 概念 | 说明 | NEAT 关心？ |
|---|---|---|
| **结构语义** | 网络拓扑、节点类型、连接关系 | ✅ 是 |
| **执行语义** | batch 维度处理、形状适配 | ❌ 否 |

广播属于"执行语义"，是运算时的形状适配，不是网络结构本身。

### NEAT 进化示例

```
当前结构：Y = WX          （MatMul 节点）
进化目标：Y = WX + b       （增加 Add 节点 + Parameter 节点）

进化操作：
1. 新增 Parameter 节点 b，形状 [1, out]
2. 新增 Add 节点，连接 WX 和 b

结果：普通 Add 节点自动处理广播，无需特殊节点。
```

### 与原设计的对比

| 方面 | 原设计（显式节点） | 新设计（内置广播） |
|---|---|---|
| 进化加 bias | 需要加 ones + MatMul + Add | 只需加 Add + Parameter |
| 图结构 | 多余的 ones 节点 | 干净 |
| NEAT 感知 | 看到 ones/MatMul（无意义） | 只看到 Add（有意义） |

**新设计对 NEAT 更友好**——进化只需关心真正的结构节点。

---

## ChannelBiasAdd 的定位

### 当前状态

`ChannelBiasAdd` 是为 Conv2d 设计的专用节点：

```rust
// [batch, C, H, W] + [1, C] → [batch, C, H, W]
let output = graph.new_channel_bias_add_node(conv_out, bias)?;
```

### 新设计下的选择

| 选择 | 说明 |
|---|---|
| **保留** | 语义清晰，用户不需要手动 reshape bias |
| **删除** | 通用 Add 可以处理（需要 reshape bias 为 `[1, C, 1, 1]`） |

**建议：保留**。理由：

1. 语义更清晰（"通道级 bias 加法"）
2. 用户体验更好（Conv2d 层内部直接用）
3. 不增加复杂度（已实现）

---

## 实现计划

### 阶段 1：Tensor 层 ✅ 已完成

| 任务 | 文件 | 状态 |
|---|---|---|
| 移除 Add 形状检查 | `tensor/ops/add.rs` | ✅ |
| 移除 Sub 形状检查 | `tensor/ops/sub.rs` | ✅ |
| 移除 Mul 形状检查 | `tensor/ops/mul.rs` | ✅ |
| 移除 Div 形状检查 | `tensor/ops/div.rs` | ✅ |
| 移除 AddAssign 形状检查 | `tensor/ops/add_assign.rs` | ✅ |
| 移除 SubAssign 形状检查 | `tensor/ops/sub_assign.rs` | ✅ |
| 移除 MulAssign 形状检查 | `tensor/ops/mul_assign.rs` | ✅ |
| 移除 DivAssign 形状检查 | `tensor/ops/div_assign.rs` | ✅ |
| 添加广播兼容性检查方法 | `tensor/property.rs` | ✅ |
| 使用自定义错误信息 | 所有 ops 文件 | ✅ |
| 更新所有相关单元测试 | `tensor/tests/*.rs` | ✅ |
| 创建 Python 参考测试脚本 | `tests/python/tensor_reference/` | ✅ |

### 阶段 2：工具函数 ✅ 已完成

| 任务 | 文件 | 状态 |
|---|---|---|
| `broadcast_shape(a, b)` | `tensor/property.rs` | ✅ |
| `sum_to_shape(tensor, target_shape)` | `tensor/property.rs` | ✅ |
| `sum_axis_keepdims(axis)` | `tensor/ops/others.rs` | ✅ |
| Python 参考测试 | `tests/python/tensor_reference/broadcast_utils_reference.py` | ✅ |
| Rust 单元测试 | `tensor/tests/property.rs` | ✅ |

### 阶段 3：Node 层 ✅ 已完成

| 任务 | 文件 | 状态 |
|---|---|---|
| 修改 Add::new() | `nodes/raw_node/ops/add.rs` | ✅ 使用 broadcast_shape |
| 修改 Add::calc_grad_to_parent() | 同上 | ✅ 使用 sum_to_shape |
| 修改 Multiply | `nodes/raw_node/ops/multiply.rs` | ✅ 同上 |
| 修改 Divide | `nodes/raw_node/ops/divide.rs` | ✅ 同上 |
| **新增 Subtract 节点** | `nodes/raw_node/ops/subtract.rs` | ✅ 原生支持广播 |
| Node 层广播测试 | `nn/tests/node_*.rs` | ✅ 每个节点都有广播测试 |

### 阶段 4：Layer 层 ✅ 已完成

| 任务 | 文件 | 状态 |
|---|---|---|
| 简化 Linear::forward() | `layer/linear.rs` | ✅ 直接 xw + bias |
| 简化 RNN | `layer/rnn.rs` | ✅ 移除 ones @ b_h |
| 简化 LSTM | `layer/lstm.rs` | ✅ 移除 ones @ b_i/b_f/b_g/b_o |
| 简化 GRU | `layer/gru.rs` | ✅ 移除 ones @ b_r/b_z/b_n |
| 移除 ScalarMultiply | `nodes/raw_node/ops/` | ✅ 功能被 Multiply + Subtract 替代 |

### 阶段 5：测试和文档

| 任务 | 说明 | 状态 |
|---|---|---|
| 更新 Tensor 层测试 | 原有"形状不匹配应 panic"的测试需要修改 | ✅ |
| 新增 Tensor 层广播测试 | 使用 Python 参考数据验证正确性 | ✅ |
| 新增 Node 层广播测试 | Forward + Backward 正确性 | ✅ |
| 更新本文档 | 标记为"已实现" | ✅ |

---

## 风险和注意事项

### 1. Backward 的 sum_to_shape 逻辑

需要正确处理各种广播情况：

```
[32, 128] + [1, 128]  → dB 沿 axis=0 求和
[32, 128] + [128]     → dB 沿 axis=0 求和，然后 squeeze
[2, 3, 4] + [3, 1]    → dB 沿 axis=0 和 axis=2 求和
```

### 2. 现有测试

部分测试明确验证"形状不同应该 panic"，需要更新。

### 3. 性能

广播涉及内存分配和复制，但这是 ndarray 内部优化的范畴，且与 PyTorch 行为一致。

---

## 总结

### 设计决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 广播策略 | 完整 NumPy 广播 | ndarray 原生支持，简单高效 |
| Tensor 层 | 移除限制 | 直接使用 ndarray |
| Node 层 | Forward 广播 + Backward 求和 | 标准自动微分 |
| Layer 层 | 简化实现 | 删除手动广播代码 |
| ChannelBiasAdd | 保留 | 语义清晰 |
| NEAT 兼容 | 不影响 | 广播是执行语义，非结构语义 |

### 核心收益

| 收益 | 说明 |
|---|---|
| **简化代码** | 删除 ones @ b 等手动广播 |
| **减少图膨胀** | 不再每次 forward 创建新节点 |
| **NEAT 更友好** | 进化只需加普通 Add，无需特殊节点 |
| **与 PyTorch 一致** | 用户期望的行为 |

### Batch 训练的价值（保留）

广播机制使 batch 训练成为可能，这对训练效率至关重要：

| 层级 | 优化方式 | 收益 |
|---|---|---|
| 单个运算 | SIMD（ndarray/BLAS） | 4-8x |
| Batch 内 | 矩阵化 | 3-10x |
| Batch 间 | Rayon 并行 | 核心数 x |
| **组合** | **Batch × Rayon** | **可达 50-100x** |

---

## 变更历史

| 日期 | 变更 | 原因 |
|---|---|---|
| 2025-12-20 | 初版：显式节点广播 | NEAT 友好考虑 |
| 2026-01-19 | **重写：采用 NumPy 广播** | 发现 ndarray 原生支持；简化设计；NEAT 兼容性分析 |
| 2026-01-19 | **实现 Tensor 层广播** | 完成阶段 1：移除形状检查，启用 ndarray 原生广播，添加自定义错误信息 |
| 2026-01-19 | **实现广播工具函数** | 完成阶段 2：broadcast_shape, sum_to_shape, sum_axis_keepdims |
| 2026-01-19 | **实现 Node 层广播** | 完成阶段 3：Add, Subtract, Multiply, Divide 节点支持广播 |
| 2026-01-19 | **简化 Layer 层** | 完成阶段 4：移除 `ones @ bias` 手动广播，使用原生 Add 广播 |
| 2026-01-19 | **移除 ScalarMultiply** | 功能被 Multiply + Subtract 替代 |

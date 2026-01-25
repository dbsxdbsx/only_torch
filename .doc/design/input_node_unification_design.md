# Input 节点统一设计

> 最后更新: 2026-01-24
> 状态: **已实施**
> 影响范围: Node 层、可视化层、ModelState、Criterion

---

## 背景与动机

### 当前问题

从 GAN 示例的可视化图中发现以下反直觉的问题：

1. **Input 节点语义混淆**：图中的 `input_1` 实际上是 Loss 的 target，而不是模型的数据入口
2. **模型输入不可见**：模型的真正入口点（GradientRouter）使用虚线样式，看起来像"内部节点"
3. **两种输入类节点**：`Input` 和 `GradientRouter` 是平行的独立类型，概念上有冗余

### 设计目标

1. **概念统一**：所有"接收外部数据"的节点归类为 Input
2. **语义清晰**：用户能直观区分不同用途的输入
3. **可视化友好**：不同类型的输入有明确的视觉区分
4. **可扩展性**：未来易于添加新的输入类型

---

## 核心设计

### 类型定义

```rust
/// 节点类型（原有的，修改 Input 部分）
enum NodeType {
    Input(InputVariant),  // ← 统一的输入类型
    Parameter(...),
    State(...),
    Add(...),
    MatMul(...),
    // ... 其他节点类型
}

/// 输入节点的变体
enum InputVariant {
    /// 普通数据输入
    Data(BasicInput),
    /// Loss 目标值
    Target(BasicInput),
    /// 智能输入（动态 batch、梯度路由等）
    Smart(SmartInput),
}
```

### 结构体定义

```rust
/// 基本输入（Data 和 Target 共用）
struct BasicInput {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    shape: Vec<usize>,
}

/// 智能输入（当前的 GradientRouter 功能）
struct SmartInput {
    id: Option<NodeId>,
    name: Option<String>,
    value: Option<Tensor>,
    grad: Option<Tensor>,
    /// 动态形状：第一维是 None（动态 batch）
    dynamic_shape: DynamicShape,
    /// 用于 value_expected_shape 的固定形状缓存
    fixed_shape: Vec<usize>,
    /// 是否处于 detached 状态（阻止梯度传播）
    is_detached: RefCell<bool>,
    /// 梯度路由目标（backward 后将梯度累加到此节点）
    gradient_target: RefCell<Option<NodeId>>,
}
```

---

## 各变体的用途

| 变体 | 用途 | 创建方式 | 特殊功能 |
|------|------|---------|---------|
| **Data** | 用户手动创建的通用输入 | `graph.new_basic_input_node()` | 无 |
| **Target** | Loss 的目标值（真实标签） | `Criterion.forward()` 内部创建（使用 `new_target_input_node()`） | 无 |
| **Smart** | 模型入口（ModelState 使用） | `ModelState.forward()` 内部创建（使用 `new_smart_input_node()`） | 动态 batch、detached 控制、梯度路由 |

---

## 可视化设计

### 样式区分

| 变体 | 形状 | 边框 | 填充色 | 标签前缀 |
|------|------|------|-------|---------|
| **Data** | 椭圆 | 实线 | 浅蓝色 `#BBDEFB` | `Input` |
| **Target** | 椭圆 | 实线 | 浅橙色 `#FFE0B2` | `Target` |
| **Smart** | 椭圆 | 实线 | 浅绿色 `#C8E6C9` | `Input` |

### 示例效果（GAN）

修改前：
```
[虚线] router_1 (GradientRouter) ─→ ... ─→ Sigmoid
                                              │
[实线] input_1 (Input) ─────────────────────→ MSELoss
```

修改后：
```
[绿色] input_g (Input) ─→ ... ─→ Sigmoid    # Smart 变体
                                     │
[橙色] target_1 (Target) ───────────→ MSELoss  # Target 变体
```

---

## 实施计划

### 阶段 1：创建 Input 模块结构

**文件结构改动**：

```
src/nn/nodes/raw_node/ops/
├── input/                    # Input 作为独立 module
│   ├── mod.rs               # 导出 + InputVariant 枚举定义
│   ├── basic.rs             # BasicInput 结构体（Data 和 Target 共用）
│   └── smart.rs             # SmartInput 结构体（原 GradientRouter）
├── input.rs                  # 删除（迁移到 input/ 目录）
├── gradient_router.rs        # 删除（迁移到 input/smart.rs）
├── parameter.rs
├── add.rs
├── ...
```

**mod.rs 内容**：
```rust
mod basic;
mod smart;

pub use basic::BasicInput;
pub use smart::SmartInput;

/// 输入节点的变体
pub enum InputVariant {
    Data(BasicInput),
    Target(BasicInput),
    Smart(SmartInput),
}
```

**测试**：
- [ ] 基本类型创建测试

### 阶段 2：迁移代码到新模块

**文件改动**：
- [ ] `src/nn/nodes/raw_node/ops/input.rs` → 迁移到 `input/basic.rs`
- [ ] `src/nn/nodes/raw_node/ops/gradient_router.rs` → 迁移到 `input/smart.rs`
- [ ] `src/nn/graph.rs`：更新节点创建 API
- [ ] `src/nn/model_state.rs`：更新 ModelState 使用的节点类型

**测试**：
- [ ] 迁移 `tests/node_gradient_router.rs` 的所有测试
- [ ] 验证 GAN 示例仍然正常工作

### 阶段 3：Criterion 使用 Target 变体

**文件改动**：
- [ ] `src/nn/criterion.rs`：创建 Target 变体而非普通 Input
- [ ] `src/nn/graph.rs`：添加 `new_target_input_node()` API

**测试**：
- [ ] Target 节点基本功能测试
- [ ] Criterion 集成测试

### 阶段 4：更新可视化

**文件改动**：
- [ ] `src/nn/graph.rs` 中的 `to_dot()` 和相关可视化函数
- [ ] `src/nn/descriptor.rs`：更新 `NodeTypeDescriptor`

**测试**：
- [ ] 可视化输出验证（各变体样式正确）
- [ ] GAN 示例可视化验证

### 阶段 5：清理

**文件改动**：
- [ ] 移除旧的 `NodeType::GradientRouter`
- [ ] 更新所有相关文档

---

## 测试策略

### 共用测试（Data 和 Target）

由于 Data 和 Target 共用 `BasicInput`，核心功能测试可以共用：

```rust
#[test]
fn test_basic_input_set_get_value() {
    // 测试 BasicInput 的基本功能
    // 适用于 Data 和 Target
}

#[test]
fn test_input_variant_visualization() {
    // 测试不同变体的可视化样式区分
}
```

### Smart 变体测试（迁移自 GradientRouter）

```rust
#[test]
fn test_smart_input_dynamic_batch() { ... }

#[test]
fn test_smart_input_detached_control() { ... }

#[test]
fn test_smart_input_gradient_routing() { ... }
```

### 集成测试

```rust
#[test]
fn test_gan_with_unified_input() {
    // 验证 GAN 示例在新架构下正常工作
}
```

---

## 向后兼容性

### 内部 API 变化

| 原 API | 新 API | 说明 |
|--------|-------|------|
| `graph.new_input_node()` | `graph.new_basic_input_node()` | 创建 Data 变体 |
| `graph.new_gradient_router_node()` | `graph.new_smart_input_node()` | 创建 Smart 变体 |
| - | `graph.new_target_input_node()` | 创建 Target 变体（Criterion 内部使用） |
| `NodeType::Input` | `NodeType::Input(InputVariant::Data(...))` | |
| `NodeType::GradientRouter` | `NodeType::Input(InputVariant::Smart(...))` | |

### 迁移路径

1. 新 API 与旧 API 并存一段时间
2. 旧 API 标记为 `#[deprecated]`
3. 下一个版本移除旧 API

---

## 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|-------|------|---------|
| 引入回归 bug | 中 | 高 | 充分的单元测试 + 集成测试 |
| 性能下降 | 低 | 中 | 基准测试对比 |
| 改动范围超预期 | 中 | 中 | 分阶段实施，每阶段验证 |

---

## 开放问题

1. **Data 变体是否还需要？**
   - 如果用户总是通过 ModelState 使用模型，Data 可能很少直接使用
   - 保留它是为了给高级用户提供灵活性

2. **命名最终确认**
   - `SmartInput` vs 其他命名？
   - `Target` vs `Label` vs `GroundTruth`？

---

## 参考

- [广播机制设计](broadcast_mechanism_design.md)
- [Keras model_visualization.py](https://github.com/keras-team/keras/blob/master/keras/src/utils/model_visualization.py)

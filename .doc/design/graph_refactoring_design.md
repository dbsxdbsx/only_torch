# Graph 模块重构设计

> 本文档描述如何将单一的 `graph.rs`（5500+ 行）重构为模块化的 `graph/` 目录结构。
>
> **目标**：提高代码可维护性，为后续 Evolution 功能提供更清晰的扩展点。

---

## 1. 现状分析

### 1.1 当前结构

```
src/nn/
├── graph.rs          # 5523 行，过于庞大
├── nodes/            # 节点类型（已良好组织）
├── layer/            # 层封装（已良好组织）
├── var.rs            # Var 句柄
└── ...
```

### 1.2 graph.rs 内容分类

通过分析现有代码，`graph.rs` 包含以下逻辑分组：

| 分组 | 行数范围 | 功能 | 建议目标文件 |
|------|----------|------|-------------|
| 类型定义 | 1-160 | `GroupKind`, `LayerGroup`, `RecurrentLayerMeta`, `GraphInner` 结构体 | `types.rs` |
| Graph 句柄 | 164-503 | `Graph` 结构体、创建、执行、训练控制 | `mod.rs` |
| GraphInner 基础 | 505-1076 | 基本操作、名称生成、节点访问 | `core.rs` |
| Batch 模式 | 1077-1348 | 批量 forward/backward | `batch.rs` |
| 反向传播 | 1349-1504 | VJP 模式反向传播 | `backward.rs` |
| 序列化 | 1505-1703 | 参数保存/加载 | `serialization.rs` |
| 描述/摘要 | 1704-2101 | describe、summary | `describe.rs` |
| 可视化 | 2102-3760 | Graphviz DOT 生成 | `visualization.rs` |
| 模式控制 | 3763-3847 | train/eval、detach | `mode.rs` |
| 循环机制 | 3848-4039 | 循环边、step、reset | `recurrent.rs` |
| BPTT | 4040-4778 | BPTT 反向传播 | `bptt.rs` |
| 节点构建 | 4779-5477 | 各种 new_*_node 方法 | `node_builders.rs` |
| 可视化类型 | 5477-5523 | 可视化相关辅助类型 | `visualization.rs` |

---

## 2. 目标结构

### 2.1 重构后的目录布局

```
src/nn/
├── graph/
│   ├── mod.rs              # 公开 API：Graph, GraphInner, GraphError, 类型 re-export
│   ├── types.rs            # GroupKind, LayerGroup, RecurrentLayerMeta, StepSnapshot
│   ├── inner/
│   │   ├── mod.rs          # GraphInner 结构体定义 + re-export
│   │   ├── core.rs         # 基础操作：new, name, 节点访问, ID 生成
│   │   ├── forward.rs      # 前向传播逻辑
│   │   ├── backward.rs     # VJP 反向传播
│   │   ├── batch.rs        # Batch forward/backward
│   │   ├── mode.rs         # train/eval 模式、detach 机制
│   │   ├── recurrent.rs    # 循环机制：connect_recurrent, step, reset
│   │   ├── bptt.rs         # BPTT 相关
│   │   ├── node_builders.rs # new_*_node 方法
│   │   ├── serialization.rs # save/load 参数
│   │   ├── describe.rs     # describe, summary
│   │   └── visualization.rs # Graphviz DOT 生成
│   ├── error.rs            # GraphError 枚举
│   └── handle.rs           # Graph 句柄（Rc<RefCell<GraphInner>> 封装）
├── nodes/                  # 保持不变
├── layer/                  # 保持不变
└── ...
```

### 2.2 各文件职责

#### `graph/mod.rs`
```rust
//! Graph 模块：计算图的核心实现
//!
//! 公开 API：
//! - `Graph`: 用户级句柄（PyTorch 风格）
//! - `GraphInner`: 底层实现（高级用户/NEAT 使用）
//! - `GraphError`: 错误类型

mod error;
mod handle;
mod inner;
mod types;

pub use error::GraphError;
pub use handle::Graph;
pub use inner::GraphInner;
pub use types::{GroupKind, LayerGroup, RecurrentLayerMeta, RecurrentUnrollInfo, StepSnapshot};
```

#### `graph/error.rs`
```rust
/// Graph 操作错误类型
#[derive(Debug)]
pub enum GraphError {
    NodeNotFound(String),
    InvalidOperation(String),
    ComputationError(String),
    DuplicateNodeName(String),
    // ... 现有错误类型
}
```

#### `graph/types.rs`
```rust
//! Graph 相关的数据类型定义

use super::NodeId;
use crate::tensor::Tensor;

/// 分组类型（Layer/Model）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupKind {
    Layer,
    Model,
}

/// 层分组信息
#[derive(Debug, Clone)]
pub struct LayerGroup {
    pub name: String,
    pub layer_type: String,
    // ... 所有现有字段
}

/// 循环层元信息
#[derive(Debug, Clone)]
pub struct RecurrentLayerMeta {
    // ... 所有现有字段
}

/// BPTT 时间步快照
#[derive(Clone)]
pub(crate) struct StepSnapshot {
    pub value: Option<Tensor>,
}
```

#### `graph/handle.rs`
```rust
//! Graph 句柄：用户级 API

use super::inner::GraphInner;
use std::cell::RefCell;
use std::rc::Rc;

/// Graph - 计算图句柄（PyTorch 风格用户 API）
#[derive(Clone)]
pub struct Graph {
    inner: Rc<RefCell<GraphInner>>,
}

impl Graph {
    // ==================== 创建 ====================
    pub fn new() -> Self { ... }
    pub fn with_name(name: &str) -> Self { ... }
    pub fn new_with_seed(seed: u64) -> Self { ... }

    // ==================== 创建变量 ====================
    pub fn input(&self, ...) -> Result<Var, GraphError> { ... }
    pub fn parameter(&self, ...) -> Result<Var, GraphError> { ... }
    // ...

    // ==================== 执行 ====================
    pub fn forward(&self, var: &Var) -> Result<Tensor, GraphError> { ... }
    pub fn backward(&self, var: &Var) -> Result<(), GraphError> { ... }

    // ==================== 训练控制 ====================
    pub fn train(&self) { ... }
    pub fn eval(&self) { ... }
    pub fn zero_grad(&self) { ... }
    pub fn apply_gradients(&self, lr: f32) { ... }

    // ==================== 底层访问 ====================
    pub fn inner(&self) -> std::cell::Ref<GraphInner> { ... }
    pub fn inner_mut(&self) -> std::cell::RefMut<GraphInner> { ... }

    // ==================== 可视化 ====================
    pub fn save_visualization(&self, ...) -> Result<(), GraphError> { ... }
    pub fn to_dot(&self) -> String { ... }
}
```

#### `graph/inner/mod.rs`
```rust
//! GraphInner：计算图的底层实现

mod backward;
mod batch;
mod bptt;
mod core;
mod describe;
mod forward;
mod mode;
mod node_builders;
mod recurrent;
mod serialization;
mod visualization;

use super::types::*;
use super::NodeId;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// 图的完整定义（核心实现）
pub struct GraphInner {
    name: String,
    nodes: HashMap<NodeId, NodeHandle>,
    forward_edges: HashMap<NodeId, Vec<NodeId>>,
    backward_edges: HashMap<NodeId, Vec<NodeId>>,
    last_forward_pass_id: u64,
    last_backward_pass_id: u64,
    next_id: u64,
    is_eval_mode: bool,
    rng: Option<StdRng>,
    layer_groups: Vec<LayerGroup>,
    recurrent_layer_metas: Vec<RecurrentLayerMeta>,
    recurrent_edges: HashMap<NodeId, NodeId>,
    prev_values: HashMap<NodeId, Tensor>,
    time_step: u64,
    step_history: Vec<HashMap<NodeId, StepSnapshot>>,
    #[cfg(test)]
    bptt_debug: bool,
}
```

---

## 3. 实施步骤

### Phase 1: 准备工作（不改变外部 API）

1. **创建目录结构**
   ```bash
   mkdir -p src/nn/graph/inner
   ```

2. **提取类型定义**
   - 创建 `graph/types.rs`
   - 移动 `GroupKind`, `LayerGroup`, `RecurrentLayerMeta`, `RecurrentUnrollInfo`, `StepSnapshot`

3. **提取错误类型**
   - 创建 `graph/error.rs`
   - 移动 `GraphError` 枚举

### Phase 2: 拆分 GraphInner

按依赖关系顺序拆分：

| 顺序 | 文件 | 依赖 | 说明 |
|------|------|------|------|
| 1 | `inner/core.rs` | 无 | 基础操作 |
| 2 | `inner/forward.rs` | core | 前向传播 |
| 3 | `inner/backward.rs` | core, forward | 反向传播 |
| 4 | `inner/batch.rs` | core, forward, backward | 批量操作 |
| 5 | `inner/mode.rs` | core | 模式控制 |
| 6 | `inner/recurrent.rs` | core | 循环机制 |
| 7 | `inner/bptt.rs` | core, recurrent | BPTT |
| 8 | `inner/node_builders.rs` | core | 节点构建 |
| 9 | `inner/serialization.rs` | core | 序列化 |
| 10 | `inner/describe.rs` | core | 描述 |
| 11 | `inner/visualization.rs` | core, types | 可视化 |

### Phase 3: 提取 Graph 句柄

1. 创建 `graph/handle.rs`
2. 移动 `Graph` 结构体及其 impl 块
3. 更新 `graph/mod.rs` 的 re-export

### Phase 4: 更新外部引用

1. 修改 `src/nn/mod.rs` 的导入路径
2. 运行测试确保所有功能正常

---

## 4. 代码迁移示例

### 4.1 core.rs 示例

```rust
//! GraphInner 核心操作

use super::GraphInner;
use crate::nn::graph::error::GraphError;
use crate::nn::NodeId;

impl GraphInner {
    // ========== 创建 ==========

    pub fn new() -> Self {
        Self::with_name("default_graph")
    }

    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
            nodes: HashMap::new(),
            forward_edges: HashMap::new(),
            backward_edges: HashMap::new(),
            // ... 其他字段初始化
        }
    }

    pub fn new_with_seed(seed: u64) -> Self {
        // ... 现有实现
    }

    // ========== 基础访问器 ==========

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn get_node(&self, id: NodeId) -> Result<&NodeHandle, GraphError> {
        self.nodes.get(&id).ok_or_else(|| {
            GraphError::NodeNotFound(format!("Node {} not found", id))
        })
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Result<&mut NodeHandle, GraphError> {
        self.nodes.get_mut(&id).ok_or_else(|| {
            GraphError::NodeNotFound(format!("Node {} not found", id))
        })
    }

    // ========== ID 生成 ==========

    fn generate_valid_node_id(&mut self) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    fn check_duplicate_node_name(&self, name: &str) -> Result<(), GraphError> {
        // ... 现有实现
    }

    fn generate_valid_new_node_name(&self, base_name: &str, node_type: &str) -> Result<String, GraphError> {
        // ... 现有实现
    }
}
```

### 4.2 node_builders.rs 示例

```rust
//! GraphInner 节点构建方法

use super::GraphInner;
use crate::nn::graph::error::GraphError;
use crate::nn::nodes::NodeHandle;
use crate::nn::NodeId;

impl GraphInner {
    /// 添加节点到图中
    fn add_node_to_list(
        &mut self,
        mut node_handle: NodeHandle,
        name: Option<&str>,
        node_type: &str,
        parents: &[NodeId],
    ) -> Result<NodeId, GraphError> {
        // ... 现有实现
    }

    /// 创建基本输入节点
    pub fn new_basic_input_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
    ) -> Result<NodeId, GraphError> {
        let node = NodeHandle::new_basic_input(shape)?;
        self.add_node_to_list(node, name, "input", &[])
    }

    /// 创建参数节点
    pub fn new_parameter_node(
        &mut self,
        shape: &[usize],
        name: Option<&str>,
        init: Init,
    ) -> Result<NodeId, GraphError> {
        // ... 现有实现
    }

    // ... 其他 new_*_node 方法
}
```

---

## 5. 测试策略

### 5.1 增量验证

每个 Phase 完成后运行：

```bash
# 编译检查
cargo check

# 单元测试
cargo test

# 示例验证
cargo run --example xor
cargo run --example mnist
```

### 5.2 回归测试

确保以下功能正常：

| 功能 | 验证方式 |
|------|----------|
| 基本 forward/backward | `test_graph_basic.rs` |
| Batch 机制 | `test_batch_mechanism.rs` |
| 循环层 | `test_layer_rnn.rs`, `test_layer_lstm.rs` |
| 序列化 | `test_save_load.rs` |
| 可视化 | 手动验证 DOT 输出 |

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 循环依赖 | 编译失败 | 仔细规划依赖顺序，使用 `super::` 导入 |
| 可见性问题 | API 不可用 | 使用 `pub(crate)` 或 `pub(in crate::nn)` 精确控制 |
| 遗漏 impl 块 | 功能缺失 | 使用 IDE 的"查找所有引用"确认迁移完整 |
| 测试遗漏 | 隐藏 bug | 每个 Phase 后完整运行测试套件 |

---

## 7. 后续优化（可选）

重构完成后，可考虑进一步优化：

1. **Evolution 支持 API**
   - 在 `inner/` 中添加 `evolution.rs`
   - 提供 `add_edge()`, `remove_edge()`, `split_edge()` 等变异操作

2. **更细粒度拆分**
   - 如果 `visualization.rs` 仍然很大，可进一步拆分为 `dot.rs`, `summary.rs`

3. **文档完善**
   - 为每个子模块添加模块级文档
   - 更新 API 文档示例

---

## 8. 时间估计

| Phase | 工作内容 | 估计时间 |
|-------|----------|----------|
| Phase 1 | 创建目录、提取类型/错误 | 30 分钟 |
| Phase 2 | 拆分 GraphInner（11 个文件） | 2-3 小时 |
| Phase 3 | 提取 Graph 句柄 | 30 分钟 |
| Phase 4 | 更新引用、测试验证 | 1 小时 |
| **总计** | | **4-5 小时** |

---

## 9. 检查清单

- [ ] 创建 `graph/` 目录结构
- [ ] 迁移 `types.rs`（类型定义）
- [ ] 迁移 `error.rs`（错误类型）
- [ ] 迁移 `inner/core.rs`（基础操作）
- [ ] 迁移 `inner/forward.rs`（前向传播）
- [ ] 迁移 `inner/backward.rs`（反向传播）
- [ ] 迁移 `inner/batch.rs`（批量操作）
- [ ] 迁移 `inner/mode.rs`（模式控制）
- [ ] 迁移 `inner/recurrent.rs`（循环机制）
- [ ] 迁移 `inner/bptt.rs`（BPTT）
- [ ] 迁移 `inner/node_builders.rs`（节点构建）
- [ ] 迁移 `inner/serialization.rs`（序列化）
- [ ] 迁移 `inner/describe.rs`（描述）
- [ ] 迁移 `inner/visualization.rs`（可视化）
- [ ] 迁移 `handle.rs`（Graph 句柄）
- [ ] 更新 `nn/mod.rs` 导入
- [ ] 运行所有单元测试
- [ ] 运行所有示例
- [ ] 更新 README 中的 TODO 项

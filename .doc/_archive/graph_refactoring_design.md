# Graph 模块重构设计

> 本文档描述如何将单一的 `graph.rs`（5500+ 行）重构为模块化的 `graph/` 目录结构。
>
> **目标**：提高代码可维护性，为后续 Evolution 功能提供更清晰的扩展点。

---

## 1. 现状分析

### 1.1 当前结构

```
src/nn/
├── graph.rs          # 5525 行，过于庞大
├── nodes/            # 节点类型（已良好组织）
├── layer/            # 层封装（已良好组织）
├── var.rs            # Var 句柄
└── ...
```

### 1.2 graph.rs 内容分类

通过分析现有代码，`graph.rs` 包含以下逻辑分组：

| 分组 | 行数范围 | 约行数 | 建议目标文件 |
|------|----------|--------|-------------|
| 类型定义 | 1-160 | ~160 | `types.rs` |
| Graph 句柄 | 164-503 | ~340 | `handle.rs` |
| GraphInner 基础 + forward | 505-1076 | ~570 | `inner/core.rs` |
| VJP 反向传播 | 1077-1504 | ~430 | `inner/backward.rs` |
| 参数序列化 | 1505-1766 | ~260 | `inner/serialization.rs` |
| 模型 I/O | 1767-1832 | ~65 | `inner/model_io.rs` |
| 描述/摘要 | 1833-2100 | ~270 | `inner/describe.rs` |
| 可视化 | 2102-3760 | ~1660 | `inner/visualization.rs` |
| 模式/detach | 3762-3847 | ~85 | `inner/mode.rs` |
| 循环机制 | 3848-4039 | ~190 | `inner/recurrent.rs` |
| BPTT | 4040-4776 | ~740 | `inner/bptt.rs` |
| 节点构建 | 4778-5477 | ~700 | `inner/node_builders.rs` |
| GraphError | 5477-5525 | ~50 | `error.rs` |

> **注**：Forward 逻辑（~80 行）合并到 `inner/core.rs`；Backward 逻辑（~430 行）独立为 `inner/backward.rs`。
> 这参考了 Candle 框架的设计模式（见第 2 节）。

---

## 2. 主流框架参考

在确定目录结构前，分析了以下主流 ML 框架的代码组织方式：

### 2.1 框架对比

| 框架 | Forward 位置 | Backward 位置 | 特点 |
|------|------------|--------------|------|
| **PyTorch** | `function.py` (Function 类) | 同一个 Function 类中 | forward/backward 作为同一类的静态方法 |
| **Burn** | `ops/tensor.rs` | `ops/base.rs` + `runtime/server.rs` | Backward 分离到专门的模块 |
| **Candle** | `tensor.rs` | **`backprop.rs`** (独立文件) | Forward/Backward 明确分离到不同文件 |
| **Neuronika** | `node/*/mod.rs` | 同一个 `node/*/mod.rs` | 每个操作的 forward/backward 在一起 |

### 2.2 设计决策

**参考 Candle 的模式**（Hugging Face 的 Rust 深度学习框架）：

- **Forward 合并到 `inner/core.rs`**：
  - 代码量小（~80 行）
  - 与节点访问、图遍历紧密相关
  - Candle 的 forward 在 `tensor.rs`（核心文件）中

- **Backward 独立为 `inner/backward.rs`**：
  - 代码量大（~430 行）
  - 包含 VJP 核心、梯度路由、拓扑排序等多个子功能
  - Candle 有专门的 `backprop.rs` 文件
  - 便于未来扩展更多反向传播策略

---

## 3. 目标结构

### 3.1 重构后的目录布局

采用**分层结构**：顶层放公开 API，`inner/` 放 GraphInner 的所有实现。

```
src/nn/
├── graph/
│   ├── mod.rs              # 公开 API：re-export
│   ├── types.rs            # GroupKind, LayerGroup, RecurrentLayerMeta, StepSnapshot (~160 行)
│   ├── error.rs            # GraphError 枚举 (~50 行)
│   ├── handle.rs           # Graph 句柄（用户级 API）(~340 行)
│   └── inner/
│       ├── mod.rs          # GraphInner 结构体定义 + 子模块导入
│       ├── core.rs         # 基础操作 + forward (~150 行)
│       ├── backward.rs     # VJP 反向传播 (~430 行)
│       ├── mode.rs         # train/eval/detach (~85 行)
│       ├── recurrent.rs    # 循环机制 (~190 行)
│       ├── bptt.rs         # BPTT (~740 行)
│       ├── node_builders.rs # new_*_node (~700 行)
│       ├── serialization.rs # save_params/load_params 底层参数序列化 (~260 行)
│       ├── model_io.rs     # save_model/load_model 高层模型 I/O (~65 行)
│       ├── describe.rs     # describe/summary (~270 行)
│       ├── visualization.rs # DOT 可视化 (~1660 行)
│       └── evolution.rs    # Evolution API (骨架代码)
├── nodes/                  # 保持不变
├── layer/                  # 保持不变
└── ...
```

### 3.2 分层结构的优势

| 优势 | 说明 |
|------|------|
| 职责清晰 | 顶层只有 4 个文件：类型、错误、句柄、内部实现入口 |
| 逻辑分组 | 所有 `impl GraphInner` 的代码都在 `inner/` 下，一目了然 |
| 符合设计理念 | 用户只需关心 `Graph`，高级用户/NEAT 可以深入 `inner/` |
| 可维护性 | 当 `GraphInner` 的 impl 块增加时，不会污染顶层目录 |

### 3.3 各文件职责

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

#### `graph/types.rs`

```rust
//! Graph 相关的数据类型定义

/// 分组类型（Layer/Model）
pub enum GroupKind { ... }

/// 层分组信息
pub struct LayerGroup { ... }

/// 循环层元信息
pub struct RecurrentLayerMeta { ... }

/// 循环层展开信息
pub struct RecurrentUnrollInfo { ... }

/// BPTT 时间步快照
pub(crate) struct StepSnapshot { ... }
```

#### `graph/error.rs`

```rust
//! Graph 错误类型

/// Graph 操作错误
#[derive(Debug)]
pub enum GraphError {
    NodeNotFound(NodeId),
    InvalidOperation(String),
    ComputationError(String),
    DuplicateNodeName(String),
    ShapeMismatch(...),
    // ...
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
    pub fn new() -> Self { ... }
    pub fn input(&self, data: &Tensor) -> Result<Var, GraphError> { ... }
    pub fn forward(&self, output: &Var) -> Result<(), GraphError> { ... }
    pub fn backward(&self, loss: &Var) -> Result<f32, GraphError> { ... }
    pub fn inner(&self) -> Ref<GraphInner> { ... }
    pub fn inner_mut(&self) -> RefMut<GraphInner> { ... }
    // ...
}
```

#### `graph/inner/mod.rs`

```rust
//! GraphInner：计算图的底层实现
//!
//! 各 impl 块分散在子模块中：
//! - core.rs: 基础操作 + forward
//! - backward.rs: VJP 反向传播
//! - mode.rs: train/eval/detach
//! - 等等...

mod backward;
mod bptt;
mod core;
mod describe;
mod evolution;
mod mode;
mod node_builders;
mod recurrent;
mod serialization;
mod visualization;

use super::types::*;
use crate::nn::nodes::NodeHandle;
use crate::nn::NodeId;
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use std::collections::HashMap;

/// 图的完整定义（核心实现）
pub struct GraphInner {
    pub(in crate::nn::graph) name: String,
    pub(in crate::nn::graph) nodes: HashMap<NodeId, NodeHandle>,
    pub(in crate::nn::graph) forward_edges: HashMap<NodeId, Vec<NodeId>>,
    pub(in crate::nn::graph) backward_edges: HashMap<NodeId, Vec<NodeId>>,
    pub(in crate::nn::graph) last_forward_pass_id: u64,
    pub(in crate::nn::graph) last_backward_pass_id: u64,
    pub(in crate::nn::graph) next_id: u64,
    pub(in crate::nn::graph) is_eval_mode: bool,
    pub(in crate::nn::graph) rng: Option<StdRng>,
    pub(in crate::nn::graph) layer_groups: Vec<LayerGroup>,
    pub(in crate::nn::graph) recurrent_layer_metas: Vec<RecurrentLayerMeta>,
    pub(in crate::nn::graph) recurrent_edges: HashMap<NodeId, NodeId>,
    pub(in crate::nn::graph) prev_values: HashMap<NodeId, Tensor>,
    pub(in crate::nn::graph) time_step: u64,
    pub(in crate::nn::graph) step_history: Vec<HashMap<NodeId, StepSnapshot>>,
    #[cfg(test)]
    pub(in crate::nn::graph) bptt_debug: bool,
}
```

#### `graph/inner/core.rs`

```rust
//! GraphInner 核心操作 + 前向传播

impl GraphInner {
    // ========== 创建 ==========
    pub fn new() -> Self { ... }
    pub fn with_name(name: &str) -> Self { ... }
    pub fn new_with_seed(seed: u64) -> Self { ... }

    // ========== 基础访问器 ==========
    pub fn name(&self) -> &str { ... }
    pub fn get_node(&self, id: NodeId) -> Result<&NodeHandle, GraphError> { ... }
    pub fn get_node_mut(&mut self, id: NodeId) -> Result<&mut NodeHandle, GraphError> { ... }

    // ========== 前向传播 ==========
    pub fn forward(&mut self, node_id: NodeId) -> Result<(), GraphError> { ... }
    fn forward_node_internal(&mut self, node_id: NodeId, pass_id: u64) -> Result<(), GraphError> { ... }
}
```

#### `graph/inner/backward.rs`

```rust
//! GraphInner VJP 反向传播

impl GraphInner {
    pub fn backward(&mut self, loss: NodeId) -> Result<f32, GraphError> { ... }
    pub fn backward_ex(&mut self, loss: NodeId, retain_graph: bool) -> Result<f32, GraphError> { ... }
    fn backward_vjp_core(&mut self, loss_id: NodeId) -> Result<(), GraphError> { ... }
    fn propagate_grad_to_parents(...) -> Result<(), GraphError> { ... }
    fn topological_sort_backward(&self, loss_id: NodeId) -> Result<Vec<NodeId>, GraphError> { ... }
    pub fn clear_grad(&mut self) -> Result<(), GraphError> { ... }
    pub fn on_topology_changed(&mut self) { ... }
}
```

#### `graph/inner/serialization.rs`

```rust
//! GraphInner 底层参数序列化
//!
//! 职责：纯二进制序列化，只处理参数的读写

impl GraphInner {
    // ========== 底层参数序列化 ==========
    pub fn save_params<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> { ... }
    pub fn load_params<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> { ... }
}
```

#### `graph/inner/model_io.rs`

```rust
//! GraphInner 高层模型 I/O
//!
//! 职责：完整模型的保存/加载（拓扑 JSON + 参数 bin）
//! 依赖：describe() + save_params()/load_params()

impl GraphInner {
    // ========== 高层模型 I/O ==========
    pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), GraphError> { ... }
    pub fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<(), GraphError> { ... }
}
```

> **设计说明**：
> - `serialization.rs` 专注于**底层二进制序列化**（参数的原始读写）
> - `model_io.rs` 专注于**高层模型 I/O**（生成/解析 GraphDescriptor + 调用底层序列化）
> - 分离这两个职责可以让用户选择：只保存参数（轻量）或保存完整模型（拓扑+参数）

#### `graph/inner/evolution.rs`（骨架代码）

```rust
//! GraphInner Evolution API（NEAT 拓扑变异支持）
//!
//! 提供神经架构演化所需的图操作 API。
//! 详见：.doc/design/neural_architecture_evolution_design.md

impl GraphInner {
    // ========== 拓扑查询 ==========
    pub fn get_hidden_nodes(&self) -> Vec<NodeId> { todo!() }
    pub fn get_removable_edges(&self) -> Vec<(NodeId, NodeId)> { todo!() }
    pub fn get_possible_new_edges(&self) -> Vec<(NodeId, NodeId)> { todo!() }

    // ========== 拓扑修改 ==========
    pub fn add_edge(&mut self, src: NodeId, dst: NodeId) -> Result<(), GraphError> { todo!() }
    pub fn remove_edge(&mut self, src: NodeId, dst: NodeId) -> Result<(), GraphError> { todo!() }
    pub fn remove_node(&mut self, node_id: NodeId) -> Result<(), GraphError> { todo!() }
    pub fn remove_orphan_nodes(&mut self) -> Result<(), GraphError> { todo!() }

    // ========== 状态快照 ==========
    pub fn snapshot(&self) -> Result<GraphSnapshot, GraphError> { todo!() }
    pub fn restore(&mut self, snapshot: &GraphSnapshot) -> Result<(), GraphError> { todo!() }
}
```

---

## 4. 实施步骤

### Phase 1: 创建目录结构与基础文件

1. **创建目录**
   ```bash
   mkdir -p src/nn/graph/inner
   ```

2. **提取类型定义**
   - 创建 `graph/types.rs`
   - 移动 `GroupKind`, `LayerGroup`, `RecurrentLayerMeta`, `RecurrentUnrollInfo`, `StepSnapshot`

3. **提取错误类型**
   - 创建 `graph/error.rs`
   - 移动 `GraphError` 枚举

4. **创建 GraphInner 结构体定义**
   - 创建 `graph/inner/mod.rs`
   - 移动 `GraphInner` 结构体定义（仅字段，不含 impl）

### Phase 2: 拆分 GraphInner impl 块

按依赖关系顺序拆分：

| 顺序 | 文件 | 依赖 | 行数 | 说明 |
|------|------|------|------|------|
| 1 | `inner/core.rs` | mod, types, error | ~150 | 基础操作 + forward |
| 2 | `inner/backward.rs` | core | ~430 | VJP 反向传播 |
| 3 | `inner/mode.rs` | core | ~85 | train/eval/detach |
| 4 | `inner/recurrent.rs` | core | ~190 | 循环机制 |
| 5 | `inner/bptt.rs` | core, recurrent | ~740 | BPTT |
| 6 | `inner/node_builders.rs` | core | ~700 | new_*_node |
| 7 | `inner/serialization.rs` | core | ~260 | save_params/load_params |
| 8 | `inner/model_io.rs` | serialization, describe | ~65 | save_model/load_model |
| 9 | `inner/describe.rs` | core | ~270 | describe/summary |
| 10 | `inner/visualization.rs` | core, types | ~1660 | DOT 可视化 |
| 11 | `inner/evolution.rs` | core | 骨架 | Evolution API |

### Phase 3: 提取 Graph 句柄

1. 创建 `graph/handle.rs`
2. 移动 `Graph` 结构体及其 impl 块
3. 创建 `graph/mod.rs` 并设置 re-export

### Phase 4: 更新外部引用并验证

1. 修改 `src/nn/mod.rs` 的导入路径
2. 运行测试确保所有功能正常
3. 删除原 `graph.rs` 文件

---

## 5. 测试策略

### 5.1 测试组织方式

**保持现有按功能域组织的测试结构不变**：

```
tests/
├── graph_basic.rs      # 图创建、节点关系
├── graph_forward.rs    # 前向传播
├── graph_backward.rs   # 反向传播
├── graph_handle.rs     # Graph 句柄 API
├── graph_dynamic.rs    # NEAT 友好性测试
├── save_load.rs        # 序列化
├── recurrent_basic.rs  # 循环机制
├── recurrent_bptt.rs   # BPTT
└── ...
```

**理由**：
- 用户视角：开发者关心"反向传播是否正确"，而非 "backward.rs 是否工作"
- 覆盖良好：现有 74 个 graph 测试全部通过
- Rust 惯例：标准库和主流项目都按功能分组测试

### 5.2 增量验证

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

### 5.3 回归测试

| 功能 | 验证方式 |
|------|----------|
| 基本 forward/backward | `graph_basic.rs`, `graph_backward.rs` |
| Graph 句柄 API | `graph_handle.rs` |
| 循环层 | `layer_rnn.rs`, `layer_lstm.rs`, `layer_gru.rs` |
| 序列化 | `save_load.rs` |
| 可视化 | 手动验证 DOT 输出 |

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 循环依赖 | 编译失败 | 仔细规划依赖顺序，使用 `super::` 导入 |
| 可见性问题 | API 不可用 | 字段使用 `pub(in crate::nn::graph)`，跨模块方法使用 `pub(crate)` |
| 遗漏 impl 块 | 功能缺失 | 使用 IDE 的"查找所有引用"确认迁移完整 |
| 测试遗漏 | 隐藏 bug | 每个 Phase 后完整运行测试套件 |

---

## 7. 后续优化（可选）

重构完成后，可考虑进一步优化：

1. **完善 Evolution API**
   - 填充 `inner/evolution.rs` 中的骨架代码
   - 添加 `graph_evolution.rs` 测试文件

2. **拆分 visualization.rs**
   - 如果 `inner/visualization.rs`（~1660 行）仍然过大
   - 可拆分为 `dot.rs`（DOT 生成）、`summary.rs`（摘要输出）

3. **文档完善**
   - 为每个子模块添加模块级文档
   - 更新 API 文档示例

---

## 8. 检查清单

### Phase 1: 目录结构与基础文件
- [ ] 创建 `graph/` 目录
- [ ] 创建 `graph/inner/` 子目录
- [ ] 创建 `graph/mod.rs`（模块入口）
- [ ] 迁移 `graph/types.rs`（类型定义）
- [ ] 迁移 `graph/error.rs`（GraphError 枚举）

### Phase 2: 拆分 GraphInner
- [ ] 创建 `graph/inner/mod.rs`（GraphInner 结构体定义）
- [ ] 迁移 `graph/inner/core.rs`（基础操作 + forward）
- [ ] 迁移 `graph/inner/backward.rs`（VJP 反向传播）
- [ ] 迁移 `graph/inner/mode.rs`（train/eval/detach）
- [ ] 迁移 `graph/inner/recurrent.rs`（循环机制）
- [ ] 迁移 `graph/inner/bptt.rs`（BPTT）
- [ ] 迁移 `graph/inner/node_builders.rs`（new_*_node）
- [ ] 迁移 `graph/inner/serialization.rs`（save_params/load_params 底层参数序列化）
- [ ] 创建 `graph/inner/model_io.rs`（save_model/load_model 高层模型 I/O）
- [ ] 迁移 `graph/inner/describe.rs`（describe/summary）
- [ ] 迁移 `graph/inner/visualization.rs`（DOT 可视化）
- [ ] 创建 `graph/inner/evolution.rs`（Evolution API 骨架）

### Phase 3: 提取 Graph 句柄
- [ ] 迁移 `graph/handle.rs`（Graph 句柄）

### Phase 4: 验证与清理
- [ ] 更新 `nn/mod.rs` 导入路径
- [ ] 运行 `cargo check`
- [ ] 运行 `cargo test`
- [ ] 运行关键示例
- [ ] 删除原 `graph.rs` 文件
- [ ] 更新 README 中的 TODO 项

---

*最后更新：2026-01-27*

# 梯度流控制机制设计

## 概述

本文档描述 only_torch 中控制梯度计算和传播的三种核心机制：`no_grad`、`detach` 和 `retain_graph`。这三种机制在高级训练场景（如 GAN、强化学习、多任务学习）中经常组合使用。

## 机制对比总览

| 机制 | 作用域 | 目的 | 影响范围 | 典型场景 |
|------|--------|------|----------|----------|
| `no_grad` | 全局上下文 | 完全禁用梯度追踪 | 整个代码块 | 推理、评估、验证 |
| `detach` | 单个节点 | 截断特定路径的梯度流 | 局部路径 | GAN、Actor-Critic、Target Network |
| `retain_graph` | backward 调用 | 保留计算图供多次反向传播 | 计算图生命周期 | 多 Loss、高阶导数、TBPTT |

### 直观对比

```
┌─────────────────────────────────────────────────────────────────┐
│                        训练模式（默认）                          │
│  x → A → B → C → loss                                          │
│       ↑   ↑   ↑                                                │
│      梯度正常流动                                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        detach (局部截断)                         │
│  x → A → B.detach() → C → loss                                 │
│       ↑       ╳       ↑                                        │
│      无梯度  截断点   有梯度                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        no_grad (全局禁用)                        │
│  x → A → B → C → output                                        │
│      (无计算图构建，纯前向计算)                                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     retain_graph (保留计算图)                    │
│  x → A → B → C → loss1.backward(retain_graph=True)             │
│       ↑   ↑   ↑                                                │
│      图保留，可再次 backward                                     │
│              └───→ loss2.backward()                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. no_grad 上下文

### 1.1 设计目标

- **内存优化**：推理时不需要存储中间值用于反向传播
- **性能提升**：跳过梯度追踪相关的开销
- **语义明确**：明确标识"这段代码不需要梯度"

### 1.2 API 设计

```rust
impl Graph {
    /// 在 no_grad 上下文中执行闭包
    /// 在此上下文中，前向传播不会为反向传播缓存中间值
    pub fn no_grad_scope<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let was_train = self.is_train_mode();
        self.set_eval_mode();
        let result = f(self);
        if was_train {
            self.set_train_mode();
        }
        result
    }

    /// 检查是否在 no_grad 模式
    pub fn is_grad_enabled(&self) -> bool {
        self.is_train_mode()
    }
}
```

### 1.3 使用示例

```rust
// 训练循环
for epoch in 0..epochs {
    // 训练阶段
    graph.set_train_mode();
    for batch in train_loader {
        graph.forward_node(loss)?;
        graph.backward_nodes(&[w, b], loss)?;
        optimizer.step(&mut graph)?;
        graph.clear_jacobi()?;
    }

    // 验证阶段（no_grad）
    graph.no_grad_scope(|g| {
        let mut total_loss = 0.0;
        for batch in val_loader {
            g.forward_node(loss)?;
            total_loss += g.get_node_value(loss)?.unwrap().data()[0];
        }
        println!("Validation loss: {}", total_loss / val_loader.len());
        Ok(())
    })?;
}
```

### 1.4 实现要点

- 与现有 `is_train_mode()` / `set_eval_mode()` 集成
- `eval_mode` 下的 `forward_node` 可跳过为 backward 缓存的中间值
- 某些层（如未来的 Dropout、BatchNorm）在 eval 模式下行为不同

### 1.5 与 PyTorch/tch-rs 的对比

| 框架 | API | 行为 |
|------|-----|------|
| PyTorch | `with torch.no_grad():` | 上下文管理器 |
| tch-rs | `tch::no_grad(\|\| { ... })` | 闭包风格 |
| tch-rs | `tch::no_grad_guard()` | Guard 风格 |
| only_torch | `graph.no_grad_scope(\|g\| { ... })` | 闭包风格 |

### 1.6 为何暂不引入 `no_grad_guard` 形式

tch-rs 提供了两种 API：闭包形式和 Guard 形式。我们目前只实现闭包形式，原因如下：

#### 架构差异

| 框架 | 状态管理 | Guard 可行性 |
|------|----------|--------------|
| PyTorch/tch-rs | **全局/线程局部状态** | ✅ Guard 自然适配 |
| only_torch | **图绑定状态** | ⚠️ Guard 会导致借用冲突 |

```rust
// PyTorch/tch-rs 风格：全局状态
let _guard = tch::no_grad_guard();  // 修改全局状态
let output = model.forward(&input); // tensor 操作检查全局状态

// only_torch 若实现 Guard 会遇到问题
let _guard = graph.no_grad_guard();  // 借用了 &mut graph
graph.forward_node(output)?;         // ❌ 无法再借用 graph！
```

#### 闭包形式的优势

| 方面 | 闭包形式 | Guard 形式 |
|------|----------|------------|
| 作用域控制 | ✅ 自动、明确 | ⚠️ 依赖变量生命周期 |
| 状态恢复 | ✅ 保证恢复 | ⚠️ 需正确持有 guard |
| Rust 风格 | ✅ 更符合 RAII | ⚠️ 需额外注意 |
| 借用安全 | ✅ 闭包内 `&mut` 清晰 | ❌ 与图绑定架构冲突 |

#### 何时考虑引入 Guard 形式

当满足以下条件之一时，可考虑引入：

1. **架构演进为全局状态模式**：如果未来项目采用类似 PyTorch 的全局/线程局部状态管理梯度开关（而非绑定到 `Graph` 实例），Guard 形式将自然适配

2. **多图协同场景**：若需要跨多个 `Graph` 实例统一禁用梯度，全局 Guard 会比逐个调用 `no_grad_scope` 更便捷

3. **与外部 FFI 集成**：若需要在 C/FFI 边界控制梯度状态，Guard 模式可能更适合

#### 当前结论

**闭包形式 `no_grad_scope` 已足够满足需求**，且更符合 Rust 的借用规则和 RAII 原则。在当前图绑定架构下，这是更安全、更自然的选择。

### 1.7 no_grad 中调用 backward 的警告机制

#### 与 PyTorch 的行为差异

| 框架 | no_grad 内调用 backward | 原因 |
|------|------------------------|------|
| PyTorch | ❌ **运行时错误** | 动态图：no_grad 内创建的张量无 `grad_fn`，无法回溯 |
| only_torch | ⚠️ **警告但允许** | 静态图：图在节点创建时已构建，backward 技术上可行 |

#### 为何不阻止而是警告

1. **架构本质不同**：PyTorch 的错误是动态图的自然结果，而非显式检查。only_torch 若要阻止需人为添加限制。

2. **存在合法用例**（约 20%）：
   ```rust
   // 调试场景：在评估时查看梯度信息
   graph.no_grad_scope(|g| {
       g.forward_node(output)?;
       g.backward_nodes(&[w], output)?;
       println!("Debug grad: {:?}", g.get_node_jacobi(w));
       Ok(())
   });
   ```

3. **大多数情况是误用**（约 80%）：用户可能忘记在训练模式下调用 backward。

#### 实现

在 `backward_nodes_ex` 和 `backward_batch` 开头添加警告：

```rust
if !self.is_train_mode() {
    eprintln!(
        "[only_torch 警告] 在 no_grad/eval 模式下调用 backward，这通常是误用。\
        如确需此行为，请忽略此警告。"
    );
}
```

#### 对照测试

- Rust 测试: `test_no_grad_scope_backward_still_works`
- PyTorch 对照: `tests/calc_jacobi_by_pytorch/no_grad_scope_behavior.py`

---

## 2. detach 机制

### 2.1 设计目标

- **选择性梯度截断**：只阻止特定路径的梯度流，其他路径正常
- **支持高级训练模式**：GAN、Actor-Critic 等需要精细控制梯度流向

### 2.2 API 设计

```rust
impl Graph {
    /// 将节点标记为 detached，阻止梯度回流到其父节点
    pub fn detach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(true);
        Ok(())
    }

    /// 取消 detach 状态
    pub fn attach_node(&mut self, node_id: NodeId) -> Result<(), GraphError> {
        self.get_node_mut(node_id)?.set_detached(false);
        Ok(())
    }

    /// 检查节点是否被 detach
    pub fn is_node_detached(&self, node_id: NodeId) -> Result<bool, GraphError> {
        Ok(self.get_node(node_id)?.is_detached())
    }
}

// NodeHandle 扩展
impl NodeHandle {
    pub fn is_detached(&self) -> bool {
        self.is_detached
    }

    pub fn set_detached(&mut self, detached: bool) {
        self.is_detached = detached;
    }
}
```

### 2.3 实现方案

在现有 `pass_id` 机制下实现，修改 `backward_node_internal`：

```rust
fn backward_node_internal(
    &mut self,
    target_node_id: NodeId,
    result_node_id: NodeId,
) -> Result<(), GraphError> {
    let target_node = self.get_node(target_node_id)?;

    // 🆕 检查 detach 状态
    if target_node.is_detached() {
        // 视为叶子节点，不向父节点传播梯度
        // jacobi 不设置（保持 None）
        return Ok(());
    }

    // 原有逻辑保持不变...
    let parents_ids = self.get_node_parents(target_node_id)?;
    for parent_id in &parents_ids {
        self.backward_node_internal(*parent_id, result_node_id)?;
    }
    // ...
}
```

### 2.4 PyTorch 语义兼容性

**关键行为**：当节点被 detach 后，其上游参数节点的 jacobi 应为 `None`，而非零张量。

```
网络: x → w1 → h(detached) → w2 → y

backward(y) 后:
- w2.jacobi = Some(正常梯度)
- h.jacobi = None (被 detach)
- w1.jacobi = None (梯度被 h 阻断，符合 PyTorch 语义)
```

实现细节：
- 若目标节点的所有子节点都无 jacobi（因 detach 导致），则清除该节点的 jacobi
- 这确保了被 detach 阻断的上游节点不会残留零梯度
```

### 2.4 使用示例

#### GAN 训练

```rust
// 训练判别器
let fake = graph.forward_node(generator_output)?;
graph.detach_node(fake)?;  // 防止 D 的 loss 更新 G
let d_fake = graph.forward_node(discriminator_on_fake)?;
graph.backward_nodes(&[d_weights], d_loss)?;

// 训练生成器
graph.attach_node(fake)?;  // 恢复梯度流
graph.backward_nodes(&[g_weights], g_loss)?;
```

#### Actor-Critic (强化学习)

```rust
// Critic 的 value 估计传给 Actor 时需要 detach
let value = graph.forward_node(critic_output)?;
graph.detach_node(value)?;  // Actor 的 loss 不应更新 Critic
let advantage = compute_advantage(reward, value);
// ... 计算 actor_loss ...
graph.backward_nodes(&[actor_weights], actor_loss)?;
```

### 2.5 与 `value_version` 机制的关系

归档文档 `graph_execution_refactor.md` 提议用 `value_version` 替代 `pass_id`，并声称对 `detach` 更友好。

**结论**：`detach` 在当前 `pass_id` 机制下**完全可实现**，两种机制在功能上等价：

| 实现方式 | detach 处理 |
|----------|-------------|
| `pass_id` + 递归 | 递归时检查 `is_detached` flag，遇到则停止 |
| `value_version` + 拓扑排序 | 构建反向子图时排除 detached 分支 |

---

## 3. retain_graph 机制

### 3.1 设计目标

- **支持多次反向传播**：多个 Loss 共享计算路径时必需
- **支持高阶导数**：计算梯度的梯度需要保留计算图
- **内存控制**：默认释放以节省内存，需要时显式保留

### 3.2 API 设计

```rust
impl Graph {
    /// 反向传播（扩展版本）
    pub fn backward_nodes_ex(
        &mut self,
        target_nodes: &[NodeId],
        result_node_id: NodeId,
        retain_graph: bool,
    ) -> Result<(), GraphError> {
        // 执行反向传播...
        self.backward_nodes_internal(target_nodes, result_node_id)?;

        if !retain_graph {
            // 释放中间计算值以节省内存
            // 保留叶子节点（Input/Parameter）的值
            self.release_intermediate_values()?;
        }
        Ok(())
    }

    /// 简化版本，默认 retain_graph = false（与 PyTorch 一致）
    pub fn backward_nodes(
        &mut self,
        target_nodes: &[NodeId],
        result_node_id: NodeId,
    ) -> Result<(), GraphError> {
        self.backward_nodes_ex(target_nodes, result_node_id, false)
    }
}
```

### 3.3 必须使用 retain_graph 的场景

#### 场景 1：多 Loss 共享计算路径

```rust
// 多任务学习
let features = graph.forward_node(backbone_output)?;
let cls_loss = graph.forward_node(classification_loss)?;
let reg_loss = graph.forward_node(regression_loss)?;

// 第一个 loss backward，保留图
graph.backward_nodes_ex(&[cls_weights], cls_loss, true)?;
// 第二个 loss backward
graph.backward_nodes_ex(&[reg_weights], reg_loss, false)?;
```

#### 场景 2：强化学习多输出模型（Actor-Critic）

> **注意**：Actor-Critic 本质上是多任务学习的一种形式，结构与场景 1 相同。

```rust
// Actor-Critic 共享 backbone（与多任务学习结构相同）
let (actor_out, critic_out) = forward_shared_model(&mut graph)?;

let actor_loss = compute_actor_loss(actor_out, actions, advantages);
let critic_loss = compute_critic_loss(critic_out, returns);

// 两个 loss 都需要 backward
graph.backward_nodes_ex(&[actor_params], actor_loss, true)?;
graph.backward_nodes_ex(&[critic_params], critic_loss, false)?;
```

#### 场景 3：高阶导数

```rust
// 计算 Hessian（二阶导数）
// 需要保留一阶梯度的计算图
```

### 3.4 内存考虑

| retain_graph | 行为 | 内存 |
|--------------|------|------|
| `false`（默认） | backward 后释放中间值 | 低 |
| `true` | 保留所有中间值 | 高 |

---

## 4. 组合使用模式

### 4.1 GAN 训练完整示例

```rust
for epoch in 0..epochs {
    // === 训练判别器 ===
    // 真实样本
    let d_real = graph.forward_node(discriminator_on_real)?;

    // 生成样本（detach 防止更新生成器）
    let fake = graph.forward_node(generator_output)?;
    graph.detach_node(fake)?;
    let d_fake = graph.forward_node(discriminator_on_fake)?;

    let d_loss = compute_d_loss(d_real, d_fake);
    graph.backward_nodes(&[d_weights], d_loss)?;
    d_optimizer.step(&mut graph)?;
    graph.clear_jacobi()?;

    // === 训练生成器 ===
    graph.attach_node(fake)?;  // 恢复梯度流
    let g_loss = compute_g_loss(d_fake);
    graph.backward_nodes(&[g_weights], g_loss)?;
    g_optimizer.step(&mut graph)?;
    graph.clear_jacobi()?;
}
```

### 4.2 Actor-Critic (PPO 风格)

```rust
for epoch in 0..epochs {
    // 收集经验时使用 no_grad
    let trajectories = graph.no_grad_scope(|g| {
        collect_trajectories(g, env)
    })?;

    // 计算优势函数（Critic 输出 detach）
    let values = graph.forward_node(critic_output)?;
    graph.detach_node(values)?;
    let advantages = compute_gae(rewards, values);

    // 多次 PPO 更新
    for _ in 0..ppo_epochs {
        let actor_loss = compute_ppo_loss(actions, advantages);
        let critic_loss = compute_value_loss(values, returns);

        // 两个 loss 共享 backbone，需要 retain_graph
        graph.backward_nodes_ex(&[actor_params], actor_loss, true)?;
        graph.backward_nodes_ex(&[critic_params], critic_loss, false)?;

        optimizer.step(&mut graph)?;
        graph.clear_jacobi()?;
    }
}
```

### 4.3 多任务学习

```rust
// 共享 backbone 的多任务模型
let features = graph.forward_node(shared_backbone)?;

// 任务 1：分类
let cls_out = graph.forward_node(classification_head)?;
let cls_loss = graph.forward_node(ce_loss)?;

// 任务 2：检测
let det_out = graph.forward_node(detection_head)?;
let det_loss = graph.forward_node(detection_loss)?;

// 反向传播（注意 retain_graph）
graph.backward_nodes_ex(&[backbone, cls_head], cls_loss, true)?;
graph.backward_nodes_ex(&[backbone, det_head], det_loss, false)?;

optimizer.step(&mut graph)?;
graph.clear_jacobi()?;
```

---

## 5. 实现优先级

| 功能 | 优先级 | 依赖 | 触发条件 |
|------|--------|------|----------|
| `no_grad` / eval mode 增强 | 高 | 现有 `is_train_mode` | 推理/评估需求 |
| `detach` | 中 | `pass_id` 机制 | GAN/RL 示例 |
| `retain_graph` | 中 | backward 实现 | 多 Loss 场景 |

---

## 6. 与其他文档的关系

| 文档 | 关注点 |
|------|--------|
| **本文档** | 用户级梯度流控制 API |
| `gradient_clear_and_accumulation_design.md` | 训练循环中的梯度累积和清除时机 |
| `graph_execution_refactor.md`（已归档） | 底层执行机制（pass_id vs value_version） |

---

## 7. 实现注意事项

### 7.1 多次 forward 后的 backward

在多任务学习场景中，可能需要多次调用 `forward_node`：

```rust
graph.forward_node(out1)?;  // forward_pass_id = 1
graph.forward_node(out2)?;  // forward_pass_id = 2
```

**关键实现细节**：在 backward 时，不应严格检查节点的 `forward_pass_id` 是否等于图的当前 `last_forward_pass_id`。这会导致在多次 forward 后，早期 forward 的节点被错误跳过。

正确做法：只跳过**从未 forward 过**的节点（`forward_pass_id == 0`），而非 id 不匹配的节点。

### 7.2 梯度累积语义（PyTorch 兼容）

多次 backward 时，梯度累积遵循 PyTorch 语义：

| 节点类型 | 行为 | 说明 |
|----------|------|------|
| **参数节点** | jacobi **累积** | 支持梯度累积（如多任务学习、大 batch 模拟） |
| **中间节点** | jacobi **重新计算** | 每次 backward 独立计算，不累积 |

#### 核心机制：传播信号 vs 累加器

理解多次 backward 的关键是区分两种不同用途的梯度：

| 概念 | 用途 | 是否跨 backward 累积 |
|------|------|---------------------|
| **传播信号**（upstream grad） | 链式法则向上传递 | ❌ 必须是本次 backward 新算的 |
| **参数累加器**（param.jacobi） | 优化器更新用 | ✅ 跨 backward 累积 |

**关键规则**：
1. 每次 backward 都从 scratch 计算一条"本次梯度流"（传播信号只用本次的）
2. 参数节点维护一个跨 backward 的累加器（用于最终更新）
3. 非参数节点不维护跨 backward 的累加器（默认），因为它不是要更新的状态
4. ⚠️ **链式法则传播必须使用"本次新算的梯度"，而非任何累积后的值**（否则会 double count）

**规则 4 的重要补充**：即使下游节点也是需要累积梯度的参数节点，在计算上游节点的梯度时，也必须使用下游节点**本次 backward 新算的贡献**，而非其累加器中的累积值。

```
假设存在拓扑：u(param) → w(param) → out

第 1 次 backward:
  w.jacobi = ∂L1/∂w
  u.jacobi = ∂L1/∂w × ∂w/∂u  ← 使用本次新算的 ∂L1/∂w

第 2 次 backward:
  w.jacobi += ∂L2/∂w  → 累积后 = ∂L1/∂w + ∂L2/∂w
  u.jacobi += ∂L2/∂w × ∂w/∂u  ← 必须使用本次新算的 ∂L2/∂w，不能用累积后的！

正确结果：u.jacobi = (∂L1/∂w + ∂L2/∂w) × ∂w/∂u = ∂(L1+L2)/∂u ✓
错误结果（若用累积值）：u.jacobi = ∂L1/∂w×∂w/∂u + (∂L1/∂w+∂L2/∂w)×∂w/∂u
                                = 2×∂L1/∂w×∂w/∂u + ∂L2/∂w×∂w/∂u ✗ (L1 被算了两次)
```

#### 为什么中间节点不累积不影响参数的正确性？

```
多任务学习示例：
  x → w_shared → features → w1 → out1 (Loss1)
                    └────→ w2 → out2 (Loss2)
```

数学上，每次 backward 计算的是**独立的梯度流**：

```
第 1 次 backward(out1):
  features.jacobi = ∂L1/∂features  ← 本次新算
  w_shared.jacobi = ∂L1/∂w_shared  ← 使用上面的 features.jacobi

第 2 次 backward(out2):
  features.jacobi = ∂L2/∂features  ← 本次新算（不依赖第 1 次的值！）
  w_shared.jacobi += ∂L2/∂w_shared ← 累积到参数
```

**关键洞察**：计算 `w_shared` 的梯度时，只需要**当前这次 backward** 算出来的 `∂L/∂features`，不需要上一次 backward 留下来的值。所以清除中间节点的 jacobi 不会影响参数的累积正确性。

从"责任"的角度理解：
- **参数节点**：需要知道"我对所有 loss 负多少责任" → 累积
- **中间节点**：只是传递梯度的"管道"，每次 backward 可视为概念上不同的路径 → 不累积

#### 示例

```
backward(out1, retain_graph=True):
  - w_shared.jacobi = [1,2,3,4,...]  ✓ 保留（累加器）
  - features.jacobi = [[1],[1]]      本次传播信号

backward(out2):
  - w_shared.jacobi = [2,4,6,8,...]  累积 = task1 + task2
  - features.jacobi = [[1],[1]]      本次传播信号（重新计算，不是累积！）
```

#### 实现细节

**backward 开始时**：调用 `reset_intermediate_jacobi()` 清除中间节点的 jacobi，只保留参数节点的 jacobi。这确保：
1. 传播信号始终是"本次新算的"
2. 参数累加器正确累积多次 backward 的贡献

**backward 结束时（`retain_graph=false`）**：调用 `release_intermediate_results()` 同时释放中间节点的**值和梯度**：
- 值被释放：需要重新 forward 才能再次 backward
- 梯度也被释放：保持一致性，避免用户误以为中间节点的梯度是累积的

这更接近 PyTorch 的语义：中间节点的梯度默认不保留（除非显式调用 `retain_grad()`）。

若需要阻止参数节点的梯度累积，应在 backward 之间调用 `clear_jacobi()`。

### 7.3 为何不引入 `retain_grad` 功能

PyTorch 提供了 `retain_grad()` 方法，允许中间节点（非叶子节点）在多次 backward 时累积梯度。经过对主流框架的调研，我们决定**暂不引入**此功能。

#### 各框架对中间节点梯度的处理

| 框架 | 设计模式 | 中间节点梯度 | 类似 `retain_grad`? |
|------|----------|--------------|---------------------|
| **PyTorch** | 动态图 + 叶子节点区分 | 默认不保留，需显式 `retain_grad()` | ✅ 有 |
| **JAX** | 纯函数式 | **根本不暴露**（只返回输入参数的梯度） | ❌ 无此概念 |
| **TensorFlow/Keras** | GradientTape + watch | 只计算显式 `watch()` 的变量 | ❌ 无 |
| **MXNet** | `attach_grad()` 显式声明 | 只计算 `attach_grad()` 的变量 | ❌ 无 |

#### 不引入的理由

1. **内存效率**：中间特征（如 CNN 的 feature map）可能非常大，默认保留所有梯度会显著增加内存占用
2. **实用性低**：99% 的训练场景只需要参数梯度，`retain_grad` 主要用于调试和研究
3. **当前能力已足够**：在 `backward(..., retain_graph=true)` 后、下一次 backward 前，中间节点的 jacobi 是可以访问的，满足大多数调试需求
4. **API 简洁性**：避免引入额外概念，降低用户学习成本
5. **YAGNI 原则**：在没有明确需求前，不过早引入复杂功能

#### 当前的调试方式

```rust
// 第一次 backward 后，可以立即访问中间节点的 jacobi
graph.backward_nodes_ex(&[w], output, true)?;

// 这个时间窗口内，中间节点的 jacobi 是可访问的
let features_jacobi = graph.get_node(features_id)?.jacobi();
println!("中间特征的梯度: {:?}", features_jacobi);

// 下一次 backward 会重置中间节点的 jacobi
graph.backward_nodes_ex(&[w], output2, false)?;
```

#### 未来扩展

当前设计不阻碍未来添加 `retain_grad` 功能。如果确有需求，可以：
1. 在节点上添加 `retains_grad` 标志
2. 修改 `reset_intermediate_jacobi()` 跳过标记为 `retains_grad` 的节点

---

## 8. 参考资料

- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)

### 项目内对照测试

| Rust 测试 | PyTorch 对照脚本 |
|-----------|------------------|
| `test_retain_graph_multi_task_learning` | `tests/calc_jacobi_by_pytorch/multi_task_learning_retain_graph.py` |

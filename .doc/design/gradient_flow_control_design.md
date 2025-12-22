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

### 1.5 与 PyTorch 的对比

| 框架 | API | 行为 |
|------|-----|------|
| PyTorch | `with torch.no_grad():` | 上下文管理器 |
| only_torch | `graph.no_grad_scope(\|g\| { ... })` | 闭包风格 |

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
        // 可选：设置 jacobi 为 None 或不设置
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

    /// 原有 API 保持兼容（默认 retain_graph = false）
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

#### 场景 2：强化学习多输出模型

```rust
// Actor-Critic 共享 backbone
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

## 7. 参考资料

- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)


# 梯度流控制机制设计

## 概述

本文档描述 only_torch 中控制梯度计算和传播的核心机制：`no_grad`、`detach`、`retain_graph`，以及**可选的** `requires_grad` / 冻结机制。这些机制在高级训练场景（如 GAN、强化学习、多任务学习、迁移学习）中经常组合使用。

## 机制对比总览

| 机制 | 作用域 | 目的 | 影响范围 | 典型场景 |
|------|--------|------|----------|----------|
| `no_grad` | 全局上下文 | 完全禁用梯度追踪 | 整个代码块 | 推理、评估、验证 |
| `detach` | 单个节点 | 截断特定路径的梯度流 | 局部路径 | GAN、Actor-Critic、Target Network |
| `retain_graph` | backward 调用 | 保留计算图供多次反向传播 | 计算图生命周期 | 多 Loss、高阶导数、TBPTT |
| `requires_grad`* | 参数节点 | 控制参数是否参与梯度计算 | 单个参数 | 迁移学习（冻结层）、部分微调 |

> \* `requires_grad` / 冻结机制为**可选功能**，详见 [附录 B](#附录-brequires_grad--冻结机制可选功能)。

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
│  x → A → B.detach() → C → loss                                  │
│       ↑       ╳       ↑                                         │
│      无梯度  截断点   有梯度                                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        no_grad (全局禁用)                        │
│  x → A → B → C → output                                         │
│      (无计算图构建，纯前向计算)                                     │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     retain_graph (保留计算图)                    │
│  x → A → B → C → loss1.backward(retain_graph=True)              │
│       ↑   ↑   ↑                                                 │
│      图保留，可再次 backward                                      │
│              └───→ loss2.backward()                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              requires_grad=false (参数冻结) [可选功能]             │
│                                                                 │
│  [数据流 →]  data → w0 → w1[frozen] → w2 → loss                  │
│                    ↑         │         ↑                        │
│  [← 梯度流]    w0.grad ✅  穿过(不存)   w2.grad ✅                 │
│                                                                 │
│  关键：w1 冻结但梯度穿过，所以 w0 仍能训练                          │
│  对比 detach：如果 w1 之后 detach，w0 就收不到梯度了               │
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
        graph.zero_grad();
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
       println!("Debug grad: {:?}", g.get_node_grad(w));
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
- PyTorch 对照: `tests/no_grad_scope_behavior.py`

---

## 2. detach 机制

> **设计决策**：为何用 `detach()` 而非 `target_params` 控制梯度流？详见 [附录 A](#附录-a设计决策为什么用-detach-而非-target_params)。

### 2.1 设计目标

- **选择性梯度截断**：只阻止特定路径的梯度流，其他路径正常
- **支持高级训练模式**：GAN、Actor-Critic 等需要精细控制梯度流向
- **PyTorch 风格 API**：`var.detach()` 返回可用于 `ModelState::forward()` 的轻量包装

### 2.2 API 设计（PyTorch 风格）

only_torch 提供两种 detach API，适用于不同场景：

| 方法 | 返回类型 | 创建图节点 | 推荐场景 |
|------|---------|-----------|---------|
| `var.detach()` | `DetachedVar` | ❌ 否 | `ModelState::forward()` 输入 |
| `var.detach_node()` | `Var` | ✅ Identity 节点 | 直接图操作、可视化调试 |

```rust
impl Var {
    /// 创建 detached 视图（轻量级，不创建图节点）
    ///
    /// 返回 `DetachedVar`，用于传递给 `ModelState::forward()`。
    /// GradientRouter 会自动处理梯度路由。
    ///
    /// # 示例
    /// ```ignore
    /// // GAN 训练（推荐）
    /// let fake = G.forward(&noise)?;
    /// let d_fake = D.forward(&fake.detach())?;  // DetachedVar，无图节点创建
    /// d_loss.backward()?;  // D 的梯度不会流向 G
    /// ```
    pub fn detach(&self) -> DetachedVar {
        DetachedVar { inner: self.clone() }
    }

    /// 创建 detached 节点（在图中创建 Identity 节点）
    ///
    /// 返回新的 `Var`，指向一个 detached 的 Identity 节点。
    /// 用于需要在 detach 后继续进行图操作的场景。
    ///
    /// # 示例
    /// ```ignore
    /// let x_detached = x.detach_node();
    /// let y = x_detached.sigmoid();  // 可以继续构建图
    /// ```
    pub fn detach_node(&self) -> Self { ... }
}

/// DetachedVar: Var 的轻量级 detached 包装
///
/// 不创建图节点，仅作为语义标记。
/// 实现 ForwardInput trait，可直接传给 ModelState::forward()。
pub struct DetachedVar {
    inner: Var,
}
```

### 2.3 GradientRouter 梯度路由机制

当 `DetachedVar` 传入 `ModelState::forward()` 时，底层通过 `GradientRouter` 节点实现梯度路由：

```
┌─────────────────────────────────────────────────────────────────┐
│                    GradientRouter 工作原理                       │
│                                                                 │
│  Forward 阶段:                                                  │
│    G.forward(&noise) → fake_images                              │
│    fake_images.detach() → DetachedVar { inner: fake_images }    │
│    D.forward(&fake.detach())                                    │
│      └→ ModelState 创建/复用 GradientRouter:                    │
│         - 设置 value = fake_images 的值                         │
│         - 设置 is_detached = true                               │
│         - 设置 gradient_target = fake_images.node_id()          │
│                                                                 │
│  Backward 阶段:                                                 │
│    d_loss.backward()                                            │
│      └→ 梯度传播到 GradientRouter                               │
│         - is_detached = true → 梯度不从此节点继续向上传播        │
│         - gradient_target 存在 → 梯度被路由到 fake_images        │
│         - 从 fake_images 继续向上传播（到 G 的参数）             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 与 PyTorch 的对比

| 框架 | API | 语义 | 实现 |
|------|-----|------|------|
| **PyTorch** | `y = x.detach()` | 返回新张量，共享存储但无 grad_fn | 动态图，创建新节点 |
| **only_torch** | `var.detach()` | 返回 DetachedVar 轻量包装 | 静态图，不创建节点 |
| **only_torch** | `var.detach_node()` | 返回新 Var（Identity 节点） | 静态图，创建节点 |

### 2.5 使用示例

#### GAN 训练（PyTorch 风格）

```rust
// 创建模型
let generator = Generator::new(&graph)?;
let discriminator = Discriminator::new(&graph)?;
let criterion = MseLoss::new();

for epoch in 0..epochs {
    // === 训练 Discriminator ===
    d_optimizer.zero_grad()?;
    
    // D 对真实图像
    let real_out = discriminator.forward(&real_images)?;
    let d_real_loss = criterion.forward(&real_out, &real_labels)?;
    d_real_loss.backward()?;
    
    // D 对假图像（使用 detach 阻止梯度流向 G）
    let fake_images = generator.forward(&noise)?;
    let fake_out = discriminator.forward(&fake_images.detach())?;  // ✨ PyTorch 风格
    let d_fake_loss = criterion.forward(&fake_out, &fake_labels)?;
    d_fake_loss.backward()?;
    d_optimizer.step()?;
    
    // === 训练 Generator ===
    g_optimizer.zero_grad()?;
    let fake_for_g = generator.forward(&new_noise)?;
    let fake_out_g = discriminator.forward(&fake_for_g)?;  // 不 detach，梯度流向 G
    let g_loss = criterion.forward(&fake_out_g, &real_labels)?;
    g_loss.backward()?;
    g_optimizer.step()?;
}
```

#### Actor-Critic (强化学习)

```rust
// Critic 的 value 估计传给 advantage 计算时 detach
let value = critic.forward(&state)?;
let advantage = rewards - value.detach();  // 不需要 Critic 的梯度

// Actor 训练
let action_logits = actor.forward(&state)?;
let actor_loss = compute_policy_loss(&action_logits, &actions, &advantage)?;
actor_loss.backward()?;
actor_optimizer.step()?;
```

### 2.6 ModelState 与 ForwardInput trait

`ModelState` 提供了统一的模型前向传播接口，通过 `ForwardInput` trait 支持多种输入类型：

```rust
/// 模型前向输入 trait
pub trait ForwardInput {
    fn shape(&self) -> Vec<usize>;
    fn get_value(&self) -> Result<Tensor, GraphError>;
    fn is_detached(&self) -> bool;
    fn var_node_id(&self) -> Option<NodeId>;
}

// 已实现的类型：
impl ForwardInput for &Tensor { ... }      // Tensor 引用
impl ForwardInput for Tensor { ... }       // Tensor 值
impl ForwardInput for &Var { ... }         // Var 引用（非 detached）
impl ForwardInput for Var { ... }          // Var 值（非 detached）
impl ForwardInput for &DetachedVar { ... } // DetachedVar 引用（detached）
impl ForwardInput for DetachedVar { ... }  // DetachedVar 值（detached）
```

**ModelState 的智能缓存**：

| 输入类型 | is_detached | gradient_target | 行为 |
|----------|-------------|-----------------|------|
| `Tensor` | false | None | 缓存，无梯度路由 |
| `Var` | false | Some(var_id) | 缓存，梯度路由到原 Var |
| `DetachedVar` | true | Some(var_id) | 缓存，但阻止梯度向上传播 |

### 2.7 Criterion 损失函数封装

`Criterion` 提供 PyTorch 风格的损失函数 API，支持智能缓存：

```rust
let criterion = MseLoss::new();

// 自动按 output 节点 ID 缓存
let loss1 = criterion.forward(&output1, &target1)?;
let loss2 = criterion.forward(&output2, &target2)?;  // 不同 output → 新缓存
let loss3 = criterion.forward(&output1, &target3)?;  // 同 output → 复用缓存
```

### 2.8 与 `value_version` 机制的关系

归档文档 `graph_execution_refactor.md` 提议用 `value_version` 替代 `pass_id`，并声称对 `detach` 更友好。

**结论**：`detach` 在当前 `pass_id` 机制下**完全可实现**，两种机制在功能上等价：

| 实现方式 | detach 处理 |
|----------|-------------|
| `pass_id` + 递归 | 递归时检查 `is_detached` flag，遇到则停止 |
| `value_version` + 拓扑排序 | 构建反向子图时排除 detached 分支 |
| `GradientRouter` | 动态设置 is_detached 和 gradient_target |

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

### 4.1 GAN 训练完整示例（PyTorch 风格）

```rust
// 创建模型和优化器
let graph = Graph::new_with_seed(42);
let generator = Generator::new(&graph)?;
let discriminator = Discriminator::new(&graph)?;
let criterion = MseLoss::new();

let mut g_optimizer = Adam::new(&graph, &generator.parameters(), 0.001);
let mut d_optimizer = Adam::new(&graph, &discriminator.parameters(), 0.0005);

for epoch in 0..epochs {
    for batch in train_loader.iter() {
        let (real_images, _) = batch;
        let noise = Tensor::normal(0.0, 1.0, &[batch_size, latent_dim]);
        
        // === 训练 Discriminator ===
        d_optimizer.zero_grad()?;
        
        // D 对真实图像
        let real_out = discriminator.forward(&real_images)?;
        let real_labels = Tensor::ones(&[batch_size, 1]);
        let d_real_loss = criterion.forward(&real_out, &real_labels)?;
        d_real_loss.backward()?;
        d_optimizer.step()?;
        
        // D 对假图像
        d_optimizer.zero_grad()?;
        let fake_images = generator.forward(&noise)?;
        let fake_out = discriminator.forward(&fake_images.detach())?;  // ✨ detach
        let fake_labels = Tensor::zeros(&[batch_size, 1]);
        let d_fake_loss = criterion.forward(&fake_out, &fake_labels)?;
        d_fake_loss.backward()?;
        d_optimizer.step()?;
        
        // === 训练 Generator ===
        g_optimizer.zero_grad()?;
        let new_noise = Tensor::normal(0.0, 1.0, &[batch_size, latent_dim]);
        let fake_for_g = generator.forward(&new_noise)?;
        let fake_out_g = discriminator.forward(&fake_for_g)?;  // 不 detach
        let g_loss = criterion.forward(&fake_out_g, &real_labels)?;
        g_loss.backward()?;
        g_optimizer.step()?;
    }
}
```

### 4.2 Actor-Critic (PPO 风格)

```rust
let actor = Actor::new(&graph)?;
let critic = Critic::new(&graph)?;
let mut actor_optimizer = Adam::new(&graph, &actor.parameters(), 0.0003);
let mut critic_optimizer = Adam::new(&graph, &critic.parameters(), 0.001);

for epoch in 0..epochs {
    // 收集经验（no_grad 模式）
    let trajectories = collect_trajectories(&actor, &env)?;
    
    for _ in 0..ppo_epochs {
        // Critic 估计 value
        let values = critic.forward(&states)?;
        
        // 计算 advantage（使用 detach）
        let advantages = &returns - &values.detach();
        
        // Actor 更新
        actor_optimizer.zero_grad()?;
        let action_logits = actor.forward(&states)?;
        let actor_loss = compute_ppo_loss(&action_logits, &actions, &advantages)?;
        actor_loss.backward()?;
        actor_optimizer.step()?;
        
        // Critic 更新
        critic_optimizer.zero_grad()?;
        let values_new = critic.forward(&states)?;
        let critic_loss = values_new.mse_loss(&returns)?;
        critic_loss.backward()?;
        critic_optimizer.step()?;
    }
}
```

### 4.3 多任务学习

```rust
let backbone = Backbone::new(&graph)?;
let cls_head = ClassificationHead::new(&graph)?;
let det_head = DetectionHead::new(&graph)?;

// 所有参数共用一个优化器
let all_params = [
    backbone.parameters(),
    cls_head.parameters(),
    det_head.parameters(),
].concat();
let mut optimizer = Adam::new(&graph, &all_params, 0.001);

let cls_criterion = CrossEntropyLoss::new();
let det_criterion = MseLoss::new();

for batch in train_loader.iter() {
    optimizer.zero_grad()?;
    
    // 共享 backbone
    let features = backbone.forward(&images)?;
    
    // 任务 1：分类
    let cls_out = cls_head.forward(&features)?;
    let cls_loss = cls_criterion.forward(&cls_out, &labels)?;
    
    // 任务 2：检测
    let det_out = det_head.forward(&features)?;
    let det_loss = det_criterion.forward(&det_out, &boxes)?;
    
    // 总 loss（自动累积梯度）
    let total_loss = &cls_loss + &det_loss;
    total_loss.backward()?;
    optimizer.step()?;
}
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
| **参数节点** | grad **累积** | 支持梯度累积（如多任务学习、大 batch 模拟） |
| **中间节点** | grad **重新计算** | 每次 backward 独立计算，不累积 |

#### 核心机制：传播信号 vs 累加器

理解多次 backward 的关键是区分两种不同用途的梯度：

| 概念 | 用途 | 是否跨 backward 累积 |
|------|------|---------------------|
| **传播信号**（upstream grad） | 链式法则向上传递 | ❌ 必须是本次 backward 新算的 |
| **参数累加器**（param.grad） | 优化器更新用 | ✅ 跨 backward 累积 |

**关键规则**：
1. 每次 backward 都从 scratch 计算一条"本次梯度流"（传播信号只用本次的）
2. 参数节点维护一个跨 backward 的累加器（用于最终更新）
3. 非参数节点不维护跨 backward 的累加器（默认），因为它不是要更新的状态
4. ⚠️ **链式法则传播必须使用"本次新算的梯度"，而非任何累积后的值**（否则会 double count）

**规则 4 的重要补充**：即使下游节点也是需要累积梯度的参数节点，在计算上游节点的梯度时，也必须使用下游节点**本次 backward 新算的贡献**，而非其累加器中的累积值。

```
假设存在拓扑：u(param) → w(param) → out

第 1 次 backward:
  w.grad = ∂L1/∂w
  u.grad = ∂L1/∂w × ∂w/∂u  ← 使用本次新算的 ∂L1/∂w

第 2 次 backward:
  w.grad += ∂L2/∂w  → 累积后 = ∂L1/∂w + ∂L2/∂w
  u.grad += ∂L2/∂w × ∂w/∂u  ← 必须使用本次新算的 ∂L2/∂w，不能用累积后的！

正确结果：u.grad = (∂L1/∂w + ∂L2/∂w) × ∂w/∂u = ∂(L1+L2)/∂u ✓
错误结果（若用累积值）：u.grad = ∂L1/∂w×∂w/∂u + (∂L1/∂w+∂L2/∂w)×∂w/∂u
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
  features.grad = ∂L1/∂features  ← 本次新算
  w_shared.grad = ∂L1/∂w_shared  ← 使用上面的 features.grad

第 2 次 backward(out2):
  features.grad = ∂L2/∂features  ← 本次新算（不依赖第 1 次的值！）
  w_shared.grad += ∂L2/∂w_shared ← 累积到参数
```

**关键洞察**：计算 `w_shared` 的梯度时，只需要**当前这次 backward** 算出来的 `∂L/∂features`，不需要上一次 backward 留下来的值。所以清除中间节点的 grad 不会影响参数的累积正确性。

从"责任"的角度理解：
- **参数节点**：需要知道"我对所有 loss 负多少责任" → 累积
- **中间节点**：只是传递梯度的"管道"，每次 backward 可视为概念上不同的路径 → 不累积

#### 示例

```
backward(out1, retain_graph=True):
  - w_shared.grad = [1,2,3,4,...]  ✓ 保留（累加器）
  - features.grad = [[1],[1]]      本次传播信号

backward(out2):
  - w_shared.grad = [2,4,6,8,...]  累积 = task1 + task2
  - features.grad = [[1],[1]]      本次传播信号（重新计算，不是累积！）
```

#### 实现细节

**backward 开始时**：调用 `reset_intermediate_grad()` 清除中间节点的 grad，只保留参数节点的 grad。这确保：
1. 传播信号始终是"本次新算的"
2. 参数累加器正确累积多次 backward 的贡献

**backward 结束时（`retain_graph=false`）**：调用 `release_intermediate_results()` 同时释放中间节点的**值和梯度**：
- 值被释放：需要重新 forward 才能再次 backward
- 梯度也被释放：保持一致性，避免用户误以为中间节点的梯度是累积的

这更接近 PyTorch 的语义：中间节点的梯度默认不保留（除非显式调用 `retain_grad()`）。

若需要阻止参数节点的梯度累积，应在 backward 之间调用 `zero_grad()`。

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
3. **当前能力已足够**：在 `backward(..., retain_graph=true)` 后、下一次 backward 前，中间节点的 grad 是可以访问的，满足大多数调试需求
4. **API 简洁性**：避免引入额外概念，降低用户学习成本
5. **YAGNI 原则**：在没有明确需求前，不过早引入复杂功能

#### 当前的调试方式

```rust
// 第一次 backward 后，可以立即访问中间节点的 grad
graph.backward_ex(output, true)?;

// 这个时间窗口内，中间节点的 grad 是可访问的
let features_grad = graph.get_node_grad(features_id)?;
println!("中间特征的梯度: {:?}", features_grad);

// 下一次 backward 会重置中间节点的 grad
graph.backward_ex(output2, false)?;
```

#### 未来扩展

当前设计不阻碍未来添加 `retain_grad` 功能。如果确有需求，可以：
1. 在节点上添加 `retains_grad` 标志
2. 修改 `reset_intermediate_grad()` 跳过标记为 `retains_grad` 的节点

---

## 8. 与优化器的配合

梯度流控制机制通常与 `with_params` 优化器配合使用（如 GAN 训练中 `detach` + 独立优化器）。

详见 [优化器架构设计](optimizer_architecture_design.md#44-指定参数优化with_params) 和 `tests/test_mnist_gan.rs`。

---

## 9. 参考资料

- [PyTorch Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [JAX Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)

### 项目内对照测试

| Rust 测试 | PyTorch 对照脚本 |
|-----------|------------------|
| `test_retain_graph_multi_task_learning` | `tests/multi_task_learning_retain_graph.py` |
| `test_mnist_gan` | - (集成测试：验证 detach + with_params) |

---

## 10. 与高层 API 的集成

本文档描述的梯度流控制机制已在 [架构 V2 设计](architecture_v2_design.md) 中的高层 API 中得到支持：

| 功能 | PyTorch 风格 API | 说明 |
|------|-----------------|------|
| **detach（推荐）** | `var.detach()` → `DetachedVar` | 轻量级，不创建节点 |
| **detach（图操作）** | `var.detach_node()` → `Var` | 创建 Identity 节点 |
| **backward** | `loss.backward()?` | 自动前向（ensure-forward） |
| **ModelState** | `model.forward(&x)?` | 统一接受 Tensor/Var/DetachedVar |
| **Criterion** | `criterion.forward(&output, &target)?` | 智能缓存损失子图 |

### 10.1 核心组件

```rust
// ModelState: 封装模型的前向计算和缓存
pub struct ModelState { ... }

impl ModelState {
    pub fn forward(&self, x: impl ForwardInput) -> Result<Var, GraphError> { ... }
}

// Criterion: 封装损失函数和缓存
pub struct MseLoss { ... }
pub struct CrossEntropyLoss { ... }

impl MseLoss {
    pub fn forward(&self, output: &Var, target: &Tensor) -> Result<Var, GraphError> { ... }
}
```

### 10.2 GradientRouter 内部机制

`GradientRouter` 是 `ModelState` 的内部实现节点，用户无需直接使用：

| 属性 | 说明 |
|------|------|
| `value` | 动态设置的输入值 |
| `is_detached` | 是否阻止梯度向上传播 |
| `gradient_target` | 梯度路由目标节点 ID |

**可视化**：GradientRouter 在 Graphviz 中显示为椭圆形、虚线边框、浅灰色背景。

高层 API 的设计原则是**薄封装**：`Var` 只是 `NodeId` 的类型安全包装，所有梯度流控制的语义与底层完全一致。

---

## 附录 A：设计决策——为什么用 `detach()` 而非 `target_params`

> 本节解释为何 only_torch 移除了 `backward(target_params)` 参数，改用 `detach()` 控制梯度流。

### A.1 问题背景

在 GAN、Actor-Critic 等场景中，需要控制"哪些参数计算梯度"。存在两种设计方案：

| 方案 | API 形式 | 控制层面 |
|------|----------|----------|
| **方案 A: `target_params`** | `loss.backward(target_params=&[w1, w2])` | 反向传播时选择性计算 |
| **方案 B: `detach()`** | `fake.detach(); loss.backward();` | 前向时截断计算图拓扑 |

### A.2 两种方案的工作原理

```
场景：GAN 训练判别器 D（不想更新生成器 G）

计算图：
z ──→ [G] ──→ fake ──→ [D] ──→ d_loss
       ↑               ↑
      G 参数          D 参数

════════════════════════════════════════════════════════════════════

方案 A: target_params ─ 选择性计算
────────────────────────────────────
z ──→ [G] ──→ fake ──→ [D] ──→ d_loss
                              ↓
                    backward(target_params=[D 参数])

行为取决于实现：
├─ 实现 1：计算所有梯度，只返回 D 的（浪费计算）
└─ 实现 2：智能剪枝，只计算到达 D 参数的路径

问题：API 语义混乱——backward() 在做 optimizer 该做的事

════════════════════════════════════════════════════════════════════

方案 B: detach() ─ 拓扑截断
────────────────────────────
z ──→ [G] ──→ fake ──✂──→ [D] ──→ d_loss
                     ↑
              detach() 截断点

反向传播：d_loss → D → fake（停止，图被截断）
结果：G 的梯度根本不会被计算（图在前向时就截断了）

优势：语义清晰，性能最优
```

### A.3 为什么选择 `detach()`

#### 1. 语义更清晰

| 方式 | 语义 | 职责边界 |
|------|------|----------|
| `target_params` | "只计算这些参数的梯度" | ⚠️ `backward()` 在做 optimizer 的事 |
| `detach()` | "从这里切断梯度流" | ✅ 清晰的图拓扑控制 |

PyTorch 的职责分离：
```python
# PyTorch 风格（我们采用）
fake = G(z)
fake_detached = fake.detach()    # 控制图拓扑
d_loss = D(fake_detached)
d_loss.backward()                # 计算所有可达的梯度
d_optimizer.step()               # optimizer 决定更新谁

# target_params 风格（已弃用）
fake = G(z)
d_loss = D(fake)
d_loss.backward(target_params=D.parameters())  # backward 越权了
```

#### 2. 性能更优

| 方式 | 计算量 | 原因 |
|------|--------|------|
| `detach()` | ✅ 最少 | 前向时截断，被截断部分完全不计算 |
| `target_params` (实现 1) | ❌ 浪费 | 计算所有梯度再过滤 |
| `target_params` (实现 2) | ⚠️ 复杂 | 需要智能剪枝算法 |

#### 3. 与 PyTorch 一致

```python
# PyTorch 的 backward() 签名
Tensor.backward(
    gradient=None,      # 上游梯度（用于非标量 loss）
    retain_graph=None,  # 是否保留计算图
    create_graph=False  # 是否创建梯度的计算图（高阶导数）
)
# ❌ 没有 target_params 参数！
```

项目愿景是"媲美 PyTorch 的易用体验"，API 应与 PyTorch 保持一致。

#### 4. 更难出 Bug

- `detach()` 在前向时就截断图，如果图构建有问题会**立即暴露**
- `target_params` 可能**隐藏**图构建问题（反向时才发现某些梯度计算不对）

### A.4 是否存在 `detach()` 无法覆盖的场景？

**99% 的场景可以完全替代**。唯一可能的例外是：

```
复杂图（参数组共享中间节点）：

           ┌──→ [A] ──→ out_a ──→ loss_a
           │     ↑
x ──→ [shared] ──┤
           │     ↓
           └──→ [B] ──→ out_b ──→ loss_b

场景：只想计算 A 的梯度，不想计算 B 的梯度

detach() 方式：在 shared 和 B 之间 detach
  - 但这样 B 的梯度无法流回 shared
  - 如果 shared 需要从两条路径获得梯度，这就有问题

target_params 方式：只指定 A 的参数
  - 不影响图拓扑
```

**但这种场景极其罕见**。在实际 ML 场景中：
- GAN：`detach()` 是标准做法
- Actor-Critic：`detach()` 是标准做法
- 多任务学习：用 `retain_graph` + 分别 backward
- 迁移学习（冻结层）：用 `requires_grad=False`

### A.5 迁移指南

如果现有代码使用了旧版 `detach_node()`/`attach_node()` 状态式 API：

```rust
// ❌ 旧代码（状态式 detach/attach）
let fake = generator.forward(&noise)?;
graph.detach_node(fake.node_id())?;  // 状态式 detach
let d_fake = discriminator.forward(&fake)?;
d_loss.backward()?;
graph.attach_node(fake.node_id())?;  // 状态式 attach

// ✅ 新代码（PyTorch 风格，函数式 detach）
let fake = generator.forward(&noise)?;
let d_fake = discriminator.forward(&fake.detach())?;  // 函数式 detach
d_loss.backward()?;
// 无需 attach！下次使用 fake 时不带 .detach() 即可
```

**关键变化**：
1. `var.detach()` 返回 `DetachedVar`（轻量包装，不创建节点）
2. 无需 `attach()` 操作——detach 状态随 `DetachedVar` 生命周期结束而消失
3. 如需在图中创建显式 detach 边界，使用 `var.detach_node()`

### A.6 总结

| 维度 | `target_params` | `detach()` |
|------|-----------------|------------|
| **语义清晰度** | ⚠️ 混乱 | ✅ 清晰 |
| **性能** | ⚠️ 取决于实现 | ✅ 最优 |
| **PyTorch 兼容** | ❌ 不兼容 | ✅ 一致 |
| **错误暴露** | ⚠️ 可能隐藏问题 | ✅ 尽早暴露 |
| **场景覆盖** | 100% | 99%+ |

**结论**：采用 `detach()` + `optimizer.step()` 的 PyTorch 风格，移除 `target_params` 参数。

---

## 附录 B：`requires_grad` / 冻结机制（可选功能）

> **状态**：Optional TODO（不在当前迭代范围内）
>
> 本节描述 `requires_grad` / 参数冻结机制的设计草案。此功能在迁移学习、部分微调等场景中有用，但**对于绝大多数训练场景不是必需的**。当前 `detach` 机制已能覆盖 GAN、Actor-Critic 等复杂梯度流控制需求。

### B.1 问题背景

在某些训练场景中，用户希望"冻结"部分参数——不更新它们，但**仍然允许梯度流经这些参数**到达更早的可训练参数。

| 场景 | 需求 | `detach` 能解决？ | `requires_grad` 能解决？ |
|------|------|------------------|-------------------------|
| **GAN 训练 D 时不更新 G** | 梯度不流向 G | ✅ 是 | ✅ 是 |
| **迁移学习冻结 backbone** | backbone 不更新，但梯度穿过 | ⚠️ 看情况 | ✅ 是 |
| **冻结 embedding 但训练后续层** | embedding 不更新，梯度不需要穿过 | ✅ 是 | ✅ 是 |
| **共享 backbone + 只冻结一个分支的参数** | 复杂场景 | ⚠️ 可能困难 | ✅ 是 |

### B.2 `detach` vs `requires_grad` 关键区别

理解两者的区别需要明确两个方向：
- **前向传播方向**：数据从 input 流向 loss（input → ... → loss）
- **反向传播方向（梯度流方向）**：梯度从 loss 流向参数（loss → ... → parameters）

```
┌────────────────────────────────────────────────────────────────────────┐
│                           detach() 语义                                │
│                                                                        │
│  [数据流方向 →]                                                        │
│   data ──► layer_A[参数] ──► detach() ──► layer_B[参数] ──► loss       │
│                                │                                      │
│                           截断点：梯度完全停止                          │
│                                                                        │
│  [← 梯度流方向]                                                        │
│   loss → layer_B.grad ✅ → 停止 ╳                                      │
│                                                                        │
│   结果：layer_B 有梯度，layer_A 无梯度（被截断）                        │
├────────────────────────────────────────────────────────────────────────┤
│                      requires_grad=false 语义                          │
│                                                                        │
│  [数据流方向 →]                                                        │
│   data ──► layer_A[参数] ──► frozen_layer[冻结] ──► layer_B[参数] ──► loss │
│                                     │                                 │
│                            不累积自己的梯度，但梯度继续传播               │
│                                                                        │
│  [← 梯度流方向]                                                        │
│   loss → layer_B.grad ✅ → frozen_layer.grad=None → layer_A.grad ✅    │
│                              ↑ 不存储         ↑ 梯度穿过               │
│                                                                        │
│   结果：layer_A 和 layer_B 都有梯度，只有 frozen_layer 没有（但它让梯度穿过）│
└────────────────────────────────────────────────────────────────────────┘
```

**核心区别一句话总结**：
- `detach()` = 梯度**到这里就停**（阻断传播）
- `requires_grad=false` = 梯度**穿过但不存**（继续传播，但该参数不累积）

### B.3 何时需要 `requires_grad`（而非 `detach`）

**绝大多数场景（~99%）使用 `detach` 即可**。以下是少数需要 `requires_grad` 的场景：

1. **冻结中间层，但需要训练其"上游"参数**：
   ```
   [数据流方向 →]
   data ──► encoder[要训练] ──► adapter[要冻结] ──► head[要训练] ──► loss
                                    │
                               如果用 detach()：
                               - head 有梯度 ✅
                               - adapter 无梯度 ✅
                               - encoder 无梯度 ❌ ← 被截断了！

                               如果用 requires_grad=false：
                               - head 有梯度 ✅
                               - adapter 无梯度 ✅（冻结）
                               - encoder 有梯度 ✅ ← 梯度穿过了 adapter
   ```

2. **多任务学习 + 选择性冻结分支**：
   ```
                             ┌──► task_A_head[要训练] ──► loss_A
   data ──► shared_encoder ──┤
                             └──► task_B_head[要冻结] ──► loss_B

   需求：task_B_head 不训练，但 shared_encoder 需要从 loss_B 获得梯度

   如果用 detach() 放在 shared_encoder 和 task_B_head 之间：
   - task_B_head 不训练 ✅
   - 但 loss_B 的梯度无法流回 shared_encoder ❌

   如果用 requires_grad=false 冻结 task_B_head：
   - task_B_head 不训练 ✅
   - loss_B 的梯度可以穿过 task_B_head 到达 shared_encoder ✅
   ```

**重要结论**：
- 如果你要冻结的参数**更靠近 loss 端**（即：冻结点和 loss 之间没有其他需要训练的参数），用 `detach` 完全足够
- 如果你要冻结的参数**位于中间**，且其"上游"（更靠近 input 端）还有需要训练的参数，则需要 `requires_grad=false`

### B.4 API 设计（草案）

```rust
impl Graph {
    /// 冻结参数（不累积梯度，但允许梯度流经）
    pub fn freeze_param(&mut self, param_id: NodeId) -> Result<(), GraphError>;

    /// 解冻参数
    pub fn unfreeze_param(&mut self, param_id: NodeId) -> Result<(), GraphError>;

    /// 检查参数是否被冻结
    pub fn is_param_frozen(&self, param_id: NodeId) -> Result<bool, GraphError>;
}

// Var 扩展（高层 API）
impl Var {
    /// 冻结此参数
    pub fn freeze(&self) -> Result<&Self, GraphError>;

    /// 解冻此参数
    pub fn unfreeze(&self) -> Result<&Self, GraphError>;

    /// 检查是否被冻结
    pub fn is_frozen(&self) -> Result<bool, GraphError>;
}

// 使用示例
let backbone_params = backbone.parameters();
for param in &backbone_params {
    param.freeze()?;  // 冻结 backbone
}

// 训练时，backbone 的参数不会被更新，但梯度会穿过它们
loss.backward()?;
optimizer.step()?;  // 只更新非冻结参数
```

### B.5 实现要点

1. **在 `Parameter` 节点上增加 `requires_grad` 标志**
2. **backward 时**：对于 `requires_grad=false` 的参数，不累积其 `.grad`，但继续向其父节点传播梯度
3. **optimizer.step() 时**：跳过 `requires_grad=false` 的参数

### B.6 与其他机制的对比总结

| 机制 | 阻止梯度流？ | 阻止参数更新？ | 适用场景 |
|------|-------------|---------------|----------|
| `detach()` | ✅ 是 | ✅ 是（间接） | GAN、Actor-Critic、梯度隔离 |
| `requires_grad=false` | ❌ 否 | ✅ 是 | 迁移学习冻结、部分微调 |
| `no_grad` | ✅ 是（全局） | ✅ 是（全局） | 推理、评估 |
| `optimizer` 不包含参数 | ❌ 否 | ✅ 是 | 选择性训练 |

**重要：这三个维度是正交的**

- `detach` 管"梯度能不能过去"（图拓扑）
- `requires_grad` 管"这个参数要不要累积梯度"
- `optimizer参数列表` 管"step 时更新谁"

在 PyTorch 中，它们可以独立组合使用。

### B.7 为什么是 Optional TODO

1. **`detach` 已覆盖 99% 场景**：GAN、Actor-Critic、Target Network 等都用 `detach`
2. **optimizer 选择性绑定**：`SGD::with_params(&partial_params, lr)` 也能实现"只更新部分参数"
3. **实现成本**：需要修改 backward 逻辑，增加复杂度
4. **用户学习成本**：多一个概念需要理解

**⚠️ 重要提醒：optimizer 选择性绑定的局限性**

"optimizer 不包含参数"**只能替代"冻结更新"，不能等价替代 `requires_grad=false`**：
1. **梯度仍会被计算/占内存**：如果参数 `requires_grad=true`（默认），即使不在 optimizer 中，backward 时仍会计算其 `.grad`
2. **`zero_grad()` 覆盖范围不同**：optimizer 的 `zero_grad()` 只清除其管理的参数，不会清除"不在 optimizer 中"的参数梯度，可能导致梯度意外累积
3. **生态工具链依赖 `requires_grad` 语义**：如 `filter(p.requires_grad, params)`、梯度裁剪等

**建议**：仅当有明确的迁移学习 / 复杂冻结需求时，再实现此功能。

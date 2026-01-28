# ModelState 多输入 Forward 架构设计（v2）

> 本文档描述 `ModelState::forward` 支持多输入的架构设计。
> 
> **v2 变更**：采用「独立参数」方案，移除 `VarArgs` 关联类型，更接近 PyTorch 风格。

---

## 一、设计目标

1. **API 简洁**：用户只需写 `impl ForwardInput`，无需任何后缀或约束
2. **PyTorch 风格**：多输入使用独立参数 `x, y, z`，而非元组
3. **类型安全**：编译时知道输入数量和类型
4. **可扩展**：轻松支持 2、3、4+ 输入
5. **向后兼容**：现有单输入代码无需改动

---

## 二、核心设计

### 2.1 简化的 ForwardInput Trait

**关键改变**：移除 `VarArgs` 关联类型，trait 专注于单输入。

```rust
/// 模型前向输入类型（单输入）
///
/// 实现此 trait 的类型可以作为 `ModelState::forward()` 的输入。
/// 所有实现类型在闭包内都会转换为 `Var`。
pub trait ForwardInput {
    /// 输入的形状（用于缓存键）
    fn shape(&self) -> Vec<usize>;
    
    /// 获取输入值
    fn get_value(&self) -> Result<Tensor, GraphError>;
    
    /// 是否 detached
    /// - `None`: 无梯度流概念（Tensor）
    /// - `Some(true)`: 被显式 detach（DetachedVar）
    /// - `Some(false)`: 正常传播梯度（Var）
    fn is_detached(&self) -> Option<bool>;
    
    /// 获取 NodeId（用于梯度路由）
    /// - `None`: 不需要梯度路由（Tensor 或 DetachedVar）
    /// - `Some(id)`: 需要将梯度路由到该节点（Var）
    fn var_node_id(&self) -> Option<NodeId>;
}
```

### 2.2 单输入实现

```rust
impl ForwardInput for &Tensor {
    fn shape(&self) -> Vec<usize> {
        Tensor::shape(self).to_vec()
    }
    
    fn get_value(&self) -> Result<Tensor, GraphError> {
        Ok((*self).clone())
    }
    
    fn is_detached(&self) -> Option<bool> {
        None  // Tensor 没有梯度流概念
    }
    
    fn var_node_id(&self) -> Option<NodeId> {
        None  // 不需要梯度路由
    }
}

impl ForwardInput for &Var {
    fn shape(&self) -> Vec<usize> {
        self.value_expected_shape()
    }
    
    fn get_value(&self) -> Result<Tensor, GraphError> {
        // 获取或计算值...
    }
    
    fn is_detached(&self) -> Option<bool> {
        Some(false)  // 正常传播梯度
    }
    
    fn var_node_id(&self) -> Option<NodeId> {
        Some(self.id())  // 梯度路由到此 Var
    }
}

impl ForwardInput for &DetachedVar {
    fn is_detached(&self) -> Option<bool> {
        Some(true)  // 被显式 detach
    }
    
    fn var_node_id(&self) -> Option<NodeId> {
        None  // 不需要梯度路由
    }
    // ...
}

// 同样实现：Tensor, Var, DetachedVar（owned 版本）
```

---

## 三、ModelState 多方法设计

### 3.1 方法族

**关键设计**：闭包接收 `&Var` 引用，用户无需手动加 `&`。

```rust
impl ModelState {
    /// 单输入 forward
    pub fn forward<X, R, F>(&self, x: X, f: F) -> Result<R, GraphError>
    where
        X: ForwardInput,
        F: FnOnce(&Var) -> Result<R, GraphError>,  // ← 引用
    { ... }
    
    /// 双输入 forward
    pub fn forward2<X, Y, R, F>(&self, x: X, y: Y, f: F) -> Result<R, GraphError>
    where
        X: ForwardInput,
        Y: ForwardInput,
        F: FnOnce(&Var, &Var) -> Result<R, GraphError>,  // ← 引用
    { ... }
    
    /// 三输入 forward
    pub fn forward3<X, Y, Z, R, F>(&self, x: X, y: Y, z: Z, f: F) -> Result<R, GraphError>
    where
        X: ForwardInput,
        Y: ForwardInput,
        Z: ForwardInput,
        F: FnOnce(&Var, &Var, &Var) -> Result<R, GraphError>,  // ← 引用
    { ... }
    
    // 可扩展：forward4, forward5, ...
}
```

### 3.2 缓存结构

```rust
struct StateCache {
    /// 多个 GradientRouter（每个输入一个）
    routers: Vec<Var>,
    /// 输出节点
    output: Var,
}

/// 缓存键：所有输入的 feature shape（忽略 batch）
type CacheKey = Vec<Vec<usize>>;
```

### 3.3 forward2 实现示例

```rust
pub fn forward2<X, Y, R, F>(&self, x: X, y: Y, f: F) -> Result<R, GraphError>
where
    X: ForwardInput,
    Y: ForwardInput,
    F: FnOnce(Var, Var) -> Result<R, GraphError>,
{
    // 1. 收集所有输入信息
    let inputs = vec![
        InputInfo::from(&x),
        InputInfo::from(&y),
    ];
    
    // 2. 构建缓存键
    let cache_key: Vec<Vec<usize>> = inputs.iter()
        .map(|i| i.feature_shape())
        .collect();
    
    // 3. 检查缓存
    let mut cache = self.cache.borrow_mut();
    
    if let Some(c) = cache.get(&cache_key) {
        // 缓存命中：更新 router 值和状态
        for (i, router) in c.routers.iter().enumerate() {
            router.set_value(&inputs[i].value)?;
            inputs[i].apply_to_router(router)?;
        }
        c.output.forward()?;
        return Ok(/* 从 output 提取结果 */);
    }
    
    // 4. 缓存未命中：为每个输入创建 GradientRouter
    let routers: Vec<Var> = inputs.iter().map(|i| {
        let router = self.create_router(&i.shape)?;
        router.set_value(&i.value)?;
        i.apply_to_router(&router)?;
        Ok(router)
    }).collect::<Result<_, _>>()?;
    
    // 5. 调用用户逻辑（传递引用）
    let output = f(&routers[0], &routers[1])?;
    
    output.forward()?;
    
    // 6. 缓存结果
    cache.insert(cache_key, StateCache { 
        routers, 
        output: output.clone() 
    });
    
    Ok(/* 结果 */)
}
```

---

## 四、用户使用示例

### 4.1 单输入（完全不变）

```rust
pub struct SimpleModel {
    fc: Linear,
    state: ModelState,
}

impl SimpleModel {
    pub fn forward(&self, x: impl ForwardInput) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {  // input: &Var
            Ok(self.fc.forward(input).relu())  // 直接使用，无需 &
        })
    }
}

// 调用
model.forward(&tensor)?;      // &Tensor
model.forward(&var)?;         // &Var
model.forward(&detached)?;    // &DetachedVar
```

### 4.2 双输入（新支持）

```rust
/// 多模态融合模型
pub struct MultiModalModel {
    image_encoder: CNN,
    text_encoder: Linear,
    fusion: Linear,
    state: ModelState,
}

impl MultiModalModel {
    /// 接收图像和文本两个输入
    pub fn forward(
        &self, 
        image: impl ForwardInput, 
        text: impl ForwardInput
    ) -> Result<Var, GraphError> {
        self.state.forward2(image, text, |img_var, txt_var| {  // 都是 &Var
            let img_feat = self.image_encoder.forward(img_var);
            let txt_feat = self.text_encoder.forward(txt_var);
            
            // 特征融合
            let combined = Var::stack(&[&img_feat, &txt_feat], 1, StackMode::Concat)?;
            Ok(self.fusion.forward(&combined))
        })
    }
}

// 调用
model.forward(&image_tensor, &text_tensor)?;
model.forward(&image_var, &text_tensor)?;
model.forward(&image_tensor, &text_var.detach())?;
```

### 4.3 三输入

```rust
/// 三模态模型
impl TriModalModel {
    pub fn forward(
        &self,
        audio: impl ForwardInput,
        video: impl ForwardInput,
        text: impl ForwardInput,
    ) -> Result<Var, GraphError> {
        self.state.forward3(audio, video, text, |a, v, t| {  // 都是 &Var
            let a_feat = self.audio_enc.forward(a);
            let v_feat = self.video_enc.forward(v);
            let t_feat = self.text_enc.forward(t);
            
            let combined = Var::stack(&[&a_feat, &v_feat, &t_feat], 1, StackMode::Concat)?;
            Ok(self.output.forward(&combined))
        })
    }
}
```

### 4.4 GAN 场景

```rust
/// Discriminator 需要接收真实图像或生成图像
impl Discriminator {
    pub fn forward(&self, x: impl ForwardInput) -> Result<Var, GraphError> {
        self.state.forward(x, |input| {  // input: &Var
            let h = self.fc1.forward(input).leaky_relu(0.2);
            Ok(self.fc2.forward(&h).sigmoid())
        })
    }
}

// 训练时的各种调用方式
let d_real = discriminator.forward(&real_images)?;        // &Tensor
let d_fake = discriminator.forward(&fake.detach())?;      // DetachedVar
let d_fake_for_g = discriminator.forward(&fake)?;         // &Var
```

---

## 五、与旧方案对比

| 方面 | 旧方案（VarArgs 关联类型） | 新方案（独立参数） |
|------|---------------------------|-------------------|
| 单输入写法 | `impl ForwardInput<VarArgs = Var>` | `impl ForwardInput` ✅ |
| 多输入写法 | `(x, y): (impl ..., impl ...)` | `x: impl ..., y: impl ...` ✅ |
| 闭包参数 | `\|(a, b)\|` 元组解构 | `\|a, b\|` 独立参数 ✅ |
| 与 PyTorch 风格 | 较远 | 一致 ✅ |
| 新增输入数量 | 添加元组实现 | 添加 `forward_n` 方法 |
| Rust 类型推断 | 需要约束 | 自动推断 ✅ |

---

## 六、Breaking Changes

**需要回滚之前的改动**：

| 场景 | v1 改动后 | v2 新方案 | 修改方式 |
|------|-----------|-----------|----------|
| 闭包参数类型 | `Var`（所有权） | `&Var`（引用） | 回滚 |
| 闭包内使用 | `&input` | `input` | 去掉 `&` |
| ForwardInput trait | 有 `VarArgs` | 无 `VarArgs` | 简化 |

**修复示例**：

```rust
// v1 改动后（需要 &）
|input| { self.fc.forward(&input).relu() }

// v2 新方案（不需要 &）
|input| { self.fc.forward(input).relu() }
```

---

## 七、风险分析

### 7.1 Detach 操作 ✅ 安全

每个输入对应独立的 `GradientRouter`，各自管理：
- `is_detached`: 当前是否 detach
- `gradient_target`: 梯度路由目标

`process_gradient_routing()` 遍历所有 `SmartInput` 节点独立处理。

### 7.2 可视化 ⚠️ 需要修改

`register_model_group` 需要支持多个 `router_id`：

```rust
// 修改前
pub fn register_model_group(&mut self, name: &str, router_id: NodeId, output_id: NodeId)

// 修改后
pub fn register_model_group(&mut self, name: &str, router_ids: &[NodeId], output_id: NodeId)
```

### 7.3 缓存机制 ✅ 已适配

缓存键从 `Vec<usize>` 改为 `Vec<Vec<usize>>`，每个输入一个 shape。

---

## 八、实现清单

### 8.1 ForwardInput trait 改动

- [x] **回滚** trait 定义：移除 `VarArgs`、`COUNT`，返回单值而非 `Vec`
- [x] 更新 `&Tensor`, `Tensor` 实现
- [x] 更新 `&Var`, `Var` 实现
- [x] 更新 `&DetachedVar`, `DetachedVar` 实现

### 8.2 ModelState 改动

- [x] 保持 `StateCache` 结构（`routers: Vec<Var>`）
- [x] 保持缓存键类型（`Vec<Vec<usize>>`）
- [x] 简化 `forward` 方法签名（移除 `build_var_args` 调用）
- [x] 新增 `forward2` 方法
- [x] 新增 `forward3` 方法

### 8.3 可视化改动

- [x] 修改 `register_model_group` 签名（支持 `&[NodeId]`）
- [x] 更新 `ModelState` 调用

### 8.4 示例和测试

- [x] 验证所有现有示例正常运行
- [x] 添加双输入单元测试（7 个测试用例）
- [x] 创建双输入 example（`examples/dual_input_add`）
- [x] 创建共享编码器 example（`examples/siamese_similarity`）

---

## 九、扩展性

### 9.1 添加更多输入数量

只需添加新方法：

```rust
impl ModelState {
    pub fn forward4<A, B, C, D, R, F>(&self, a: A, b: B, c: C, d: D, f: F) -> Result<R, GraphError>
    where
        A: ForwardInput,
        B: ForwardInput,
        C: ForwardInput,
        D: ForwardInput,
        F: FnOnce(&Var, &Var, &Var, &Var) -> Result<R, GraphError>,  // 引用
    { ... }
}
```

### 9.2 使用宏简化

```rust
macro_rules! impl_forward_n {
    ($name:ident, $($T:ident),+) => {
        pub fn $name<$($T,)+ R, F>(&self, $($T: $T,)+ f: F) -> Result<R, GraphError>
        where
            $($T: ForwardInput,)+
            F: FnOnce($(&Var),+) -> Result<R, GraphError>,  // 引用
        {
            // ...
        }
    };
}

impl_forward_n!(forward2, A, B);
impl_forward_n!(forward3, A, B, C);
impl_forward_n!(forward4, A, B, C, D);
```

---

## 十、可选扩展：forward_n（动态数量输入）

> **状态**：Optional，用于 NEAT/强化学习等输入数量不固定的场景。

### 10.1 动机

`forward2`, `forward3` 等方法适用于**编译时已知输入数量**的场景。但某些场景需要**运行时动态数量**：

- **NEAT 神经进化**：网络结构演化，输入节点数量可变
- **Graph Neural Network**：节点数量不固定
- **多智能体强化学习**：智能体数量动态变化
- **Attention 机制**：Query/Key/Value + 多个 context

### 10.2 API 设计

```rust
impl ModelState {
    /// 动态数量输入 forward
    /// 
    /// # Arguments
    /// * `inputs` - 输入切片，所有输入必须实现 ForwardInput
    /// * `f` - 用户逻辑闭包，接收 `&[&Var]` 切片
    /// 
    /// # Example
    /// ```rust
    /// let inputs: Vec<&Tensor> = vec![&t1, &t2, &t3];
    /// model.forward_n(&inputs, |vars| {
    ///     // vars: &[&Var]
    ///     let sum = vars.iter().fold(vars[0].clone(), |acc, v| acc + *v);
    ///     Ok(sum)
    /// })?;
    /// ```
    pub fn forward_n<X, R, F>(&self, inputs: &[X], f: F) -> Result<R, GraphError>
    where
        X: ForwardInput,
        F: FnOnce(&[&Var]) -> Result<R, GraphError>,
    { ... }
}
```

### 10.3 实现要点

```rust
pub fn forward_n<X, R, F>(&self, inputs: &[X], f: F) -> Result<R, GraphError>
where
    X: ForwardInput,
    F: FnOnce(&[&Var]) -> Result<R, GraphError>,
{
    // 1. 收集所有输入信息
    let input_infos: Vec<InputInfo> = inputs.iter()
        .map(InputInfo::from)
        .collect();
    
    // 2. 构建缓存键（所有输入的 feature shape）
    let cache_key: Vec<Vec<usize>> = input_infos.iter()
        .map(|i| i.feature_shape())
        .collect();
    
    // 3. 检查缓存或创建新图
    let mut cache = self.cache.borrow_mut();
    
    if let Some(c) = cache.get(&cache_key) {
        // 缓存命中：更新所有 router
        for (i, router) in c.routers.iter().enumerate() {
            router.set_value(&input_infos[i].value)?;
            input_infos[i].apply_to_router(router)?;
        }
        c.output.forward()?;
        return Ok(/* 从 output 提取结果 */);
    }
    
    // 4. 缓存未命中：为每个输入创建 GradientRouter
    let routers: Vec<Var> = input_infos.iter().map(|i| {
        let router = self.create_router(&i.shape)?;
        router.set_value(&i.value)?;
        i.apply_to_router(&router)?;
        Ok(router)
    }).collect::<Result<_, _>>()?;
    
    // 5. 转换为引用切片并调用用户逻辑
    let router_refs: Vec<&Var> = routers.iter().collect();
    let output = f(&router_refs)?;
    
    output.forward()?;
    
    // 6. 缓存结果
    cache.insert(cache_key, StateCache { routers, output: output.clone() });
    
    Ok(/* 结果 */)
}
```

### 10.4 使用示例

#### NEAT 动态输入

```rust
/// NEAT 网络：输入数量由基因决定
pub struct NeatNetwork {
    genome: Genome,
    state: ModelState,
}

impl NeatNetwork {
    pub fn forward(&self, inputs: &[impl ForwardInput]) -> Result<Var, GraphError> {
        self.state.forward_n(inputs, |vars| {
            // 根据基因拓扑计算
            self.genome.evaluate(vars)
        })
    }
}

// 调用
let sensory_inputs: Vec<&Tensor> = vec![&vision, &audio, &touch];
neat_network.forward(&sensory_inputs)?;
```

#### 多智能体聚合

```rust
/// 多智能体：聚合所有智能体的观测
impl MultiAgentModel {
    pub fn aggregate_observations(
        &self, 
        agent_obs: &[impl ForwardInput]
    ) -> Result<Var, GraphError> {
        self.state.forward_n(agent_obs, |vars| {
            // 使用 attention 或简单求和聚合
            let stacked = Var::stack(vars, 0, StackMode::Stack)?;
            Ok(self.attention.forward(&stacked))
        })
    }
}
```

### 10.5 与静态方法的对比

| 方面 | forward2/forward3（静态） | forward_n（动态） |
|------|--------------------------|------------------|
| **输入数量** | 编译时固定 | 运行时决定 |
| **类型安全** | 强（每个参数独立类型） | 弱（所有输入同类型） |
| **性能** | 略优（无 Vec 分配） | 略慢（需要 Vec） |
| **适用场景** | 常规多模态、GAN | NEAT、GNN、多智能体 |
| **推荐度** | 默认首选 | 特殊场景使用 |

### 10.6 实现优先级

- **当前**：先完成 `forward`, `forward2`, `forward3`
- **后续**：根据 NEAT/强化学习需求再实现 `forward_n`

---

## 十一、更新记录

| 日期 | 内容 |
|------|------|
| 2026-01-28 | 初始设计文档（VarArgs 方案） |
| 2026-01-28 | 添加风险分析与验证 |
| 2026-01-28 | 完成 VarArgs 方案实现，所有示例和测试通过 |
| 2026-01-28 | **v2 重构**：采用「独立参数」方案，移除 `VarArgs`，更接近 PyTorch 风格 |
| 2026-01-28 | 添加「第十节：forward_n 可选扩展」，支持 NEAT/强化学习动态输入场景 |
| 2026-01-28 | **实现完成**：forward2/forward3、多 router 可视化、7 个单元测试、dual_input_add 示例 |
| 2026-01-28 | **验证完成**：新增 siamese_similarity 共享编码器示例，Stack/Concat 可视化区分 |

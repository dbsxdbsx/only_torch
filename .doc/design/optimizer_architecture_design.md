# Optimizer 架构设计

## 1. 设计目标

基于 MatrixSlow Python 版本的 optimizer 设计，为 only_torch 项目设计一个可扩展、可维护的优化器架构，支持多种优化算法。

## 2. 核心设计原则

- **可扩展性**: 易于添加新的优化算法
- **类型安全**: 利用 Rust 的类型系统确保安全性
- **性能优化**: 避免不必要的内存分配和拷贝
- **API 一致性**: 与 MatrixSlow Python 版本保持相似的使用方式
- **梯度累积**: 支持 mini-batch 训练的梯度累积机制

## 3. 架构概览

```
Optimizer Trait (优化器特征)
├── 核心方法:
│   ├── one_step()     # 单步训练（前向+反向传播+梯度累积）
│   ├── update()       # 参数更新（执行具体优化算法）
│   └── reset()        # 重置累积状态
├── 已实现:
│   ├── SGD              # 随机梯度下降
│   └── Adam             # Adam优化器
├── 待实现:
│   ├── Momentum         # 动量法
│   ├── AdaGrad          # AdaGrad
│   └── RMSProp          # RMSProp
└── 辅助结构:
    ├── OptimizerState    # 优化器状态管理
    └── GradientAccumulator # 梯度累积器
```

## 4. 核心接口设计

### 4.1 Optimizer Trait

```rust
pub trait Optimizer {
    /// 执行一步训练：前向传播 + 反向传播 + 梯度累积
    fn one_step(&mut self, graph: &mut Graph, target_node: NodeId) -> Result<(), GraphError>;

    /// 更新参数（执行具体的优化算法）
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError>;

    /// 重置累积状态
    fn reset(&mut self);

    /// 获取学习率
    fn learning_rate(&self) -> f32;

    /// 设置学习率
    fn set_learning_rate(&mut self, lr: f32);
}
```

### 4.2 梯度累积器

```rust
pub struct GradientAccumulator {
    /// 累积的梯度：NodeId -> 累积梯度
    accumulated_gradients: HashMap<NodeId, Tensor>,
    /// 累积的样本数量
    sample_count: usize,
}

impl GradientAccumulator {
    /// 累积单个样本的梯度
    pub fn accumulate(&mut self, node_id: NodeId, gradient: &Tensor) -> Result<(), GraphError>;

    /// 获取平均梯度
    pub fn get_average_gradient(&self, node_id: NodeId) -> Option<Tensor>;

    /// 清除累积状态
    pub fn clear(&mut self);

    /// 获取累积的样本数量
    pub fn sample_count(&self) -> usize;
}
```

### 4.3 优化器状态管理

```rust
pub struct OptimizerState {
    /// 可训练参数的节点ID列表
    trainable_nodes: Vec<NodeId>,
    /// 梯度累积器
    gradient_accumulator: GradientAccumulator,
    /// 学习率
    learning_rate: f32,
}

impl OptimizerState {
    /// 自动获取图中所有可训练节点
    pub fn new(graph: &Graph, learning_rate: f32) -> Result<Self, GraphError>;

    /// 使用指定参数创建（用于 GAN、迁移学习等场景）
    pub fn with_params(params: Vec<NodeId>, learning_rate: f32) -> Self;
}
```

### 4.4 指定参数优化（with_params）

支持为不同参数组创建独立优化器，用于以下场景：

| 场景 | 说明 |
|------|------|
| **GAN 训练** | Generator 和 Discriminator 使用不同优化器 |
| **迁移学习** | 预训练层和新层使用不同学习率 |
| **分层学习率** | 不同网络层使用不同学习率 |
| **参数冻结** | 只优化部分参数，其余保持不变 |

**API 对比**：

| 方法 | 用途 | 参数来源 |
|------|------|----------|
| `new(&graph, ...)` | 优化图中所有可训练节点 | 自动获取 |
| `with_params(&params, ...)` | 只优化指定参数 | 用户指定 |

**代码示例**：

```rust
// GAN 训练：分别优化 G 和 D
let optimizer_g = Adam::with_params(&g_params, 0.0002, 0.5, 0.999, 1e-8);
let optimizer_d = Adam::with_params(&d_params, 0.0001, 0.5, 0.999, 1e-8);

// 迁移学习：分层学习率
let optimizer_pretrained = SGD::with_params(&pretrained_params, 0.001);
let optimizer_new = SGD::with_params(&new_params, 0.01);
```

## 5. 具体优化器实现

### 5.1 SGD 优化器

```rust
pub struct SGD {
    state: OptimizerState,
}

impl Optimizer for SGD {
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        for &node_id in self.state.trainable_nodes() {
            if let Some(avg_gradient) = self.state.gradient_accumulator().get_average_gradient(node_id) {
                let current_value = graph.get_node_value(node_id)?.unwrap();
                // θ = θ - α * ∇θ
                let new_value = current_value - self.state.learning_rate() * &avg_gradient;
                graph.set_node_value(node_id, Some(&new_value))?;
            }
        }
        self.state.reset();
        Ok(())
    }
}
```

### 5.2 Adam 优化器

```rust
pub struct Adam {
    state: OptimizerState,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    /// 一阶矩估计
    m: HashMap<NodeId, Tensor>,
    /// 二阶矩估计
    v: HashMap<NodeId, Tensor>,
    /// 时间步
    t: usize,
}

impl Optimizer for Adam {
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        self.t += 1;

        for &node_id in &self.state.trainable_nodes {
            if let Some(gradient) = self.state.gradient_accumulator.get_average_gradient(node_id) {
                // 更新一阶矩估计
                let m_t = self.beta1 * self.m.get(&node_id).unwrap_or(&Tensor::zeros(gradient.shape()))
                         + (1.0 - self.beta1) * &gradient;

                // 更新二阶矩估计
                let v_t = self.beta2 * self.v.get(&node_id).unwrap_or(&Tensor::zeros(gradient.shape()))
                         + (1.0 - self.beta2) * &gradient.element_wise_multiply(&gradient);

                // 偏差修正
                let m_hat = &m_t / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = &v_t / (1.0 - self.beta2.powi(self.t as i32));

                // 参数更新
                let current_value = graph.get_node_value(node_id)?.unwrap();
                let denominator = v_hat.element_wise_sqrt() + self.epsilon;
                let new_value = current_value - self.state.learning_rate * &m_hat.element_wise_divide(&denominator);

                graph.set_node_value(node_id, Some(&new_value))?;

                // 保存状态
                self.m.insert(node_id, m_t);
                self.v.insert(node_id, v_t);
            }
        }

        self.state.gradient_accumulator.clear();
        Ok(())
    }
}
```

## 6. 使用示例

### 6.1 基本使用方式

```rust
// 创建计算图和网络结构
let mut graph = Graph::new();
let x = graph.new_input_node(&[3, 1], Some("x"))?;
let w = graph.new_parameter_node(&[1, 3], Some("w"))?;
let b = graph.new_parameter_node(&[1, 1], Some("b"))?;
let output = graph.new_add_node(&[graph.new_mat_mul_node(w, x, None)?, b], None)?;
let loss = graph.new_perception_loss_node(output, Some("loss"))?;

// 创建优化器
let mut optimizer = Adam::new(&graph, loss, 0.01)?;

// 训练循环
for epoch in 0..50 {
    for (features, label) in train_data {
        // 设置输入数据
        graph.set_node_value(x, Some(&features))?;
        graph.set_node_value(label_node, Some(&label))?;

        // 执行一步训练（前向+反向传播+梯度累积）
        optimizer.one_step(&mut graph, loss)?;
    }

    // 更新参数
    optimizer.update(&mut graph)?;
}
```

### 6.2 Mini-batch 训练

```rust
let mini_batch_size = 8;
let mut current_batch_size = 0;

for (features, label) in train_data {
    graph.set_node_value(x, Some(&features))?;
    graph.set_node_value(label_node, Some(&label))?;

    optimizer.one_step(&mut graph, loss)?;
    current_batch_size += 1;

    // 当积累到一个mini batch时，执行参数更新
    if current_batch_size == mini_batch_size {
        optimizer.update(&mut graph)?;
        current_batch_size = 0;
    }
}
```

### 6.3 GAN 训练（with_params + detach）

```rust
// 构建 GAN 网络
let (fake_images, g_params) = build_generator(&mut graph, z)?;
let (d_output, d_params) = build_discriminator(&mut graph, input)?;

// 为 G 和 D 创建独立优化器
let mut optimizer_g = Adam::with_params(&g_params, 0.0002, 0.5, 0.999, 1e-8);
let mut optimizer_d = Adam::with_params(&d_params, 0.0001, 0.5, 0.999, 1e-8);

for epoch in 0..epochs {
    // === 训练 Discriminator ===
    graph.detach_node(fake_images)?;  // 阻止 D 的 loss 更新 G
    graph.forward_batch(d_loss)?;
    graph.clear_grad()?;
    graph.backward_batch(d_loss)?;
    optimizer_d.update_batch(&mut graph)?;

    // === 训练 Generator ===
    graph.attach_node(fake_images)?;  // 恢复梯度流
    graph.forward_batch(g_loss)?;
    graph.clear_grad()?;
    graph.backward_batch(g_loss)?;
    optimizer_g.update_batch(&mut graph)?;
}
```

详见 `tests/test_mnist_gan.rs` 完整示例。

### 6.4 迁移学习（分层学习率）

```rust
// 加载预训练模型参数
let pretrained_params = vec![conv1_w, conv2_w, conv3_w];
let new_params = vec![fc1_w, fc2_w];

// 预训练层使用小学习率，新层使用大学习率
let mut optimizer_pretrained = SGD::with_params(&pretrained_params, 0.001);
let mut optimizer_new = SGD::with_params(&new_params, 0.01);

for batch in data_loader {
    graph.forward_batch(loss)?;
    graph.clear_grad()?;
    graph.backward_batch(loss)?;

    // 两个优化器都更新
    optimizer_pretrained.update_batch(&mut graph)?;
    optimizer_new.update_batch(&mut graph)?;
}
```

## 7. 实现状态

| 组件                | 状态 | 说明                           |
| ------------------- | ---- | ------------------------------ |
| Optimizer trait     | ✅   | 核心接口                       |
| GradientAccumulator | ✅   | 梯度累积器                     |
| OptimizerState      | ✅   | 状态管理                       |
| with_params         | ✅   | 指定参数优化（GAN/迁移学习）   |
| SGD                 | ✅   | 随机梯度下降                   |
| Adam                | ✅   | 自适应矩估计                   |
| Momentum            | ❌   | 待实现                         |
| AdaGrad             | ❌   | 待实现                         |
| RMSProp             | ❌   | 待实现                         |

## 8. 文件结构

```
src/nn/optimizer/
├── mod.rs          # 模块导出
├── base.rs         # Optimizer trait、GradientAccumulator、OptimizerState
├── sgd.rs          # SGD优化器
└── adam.rs         # Adam优化器

src/nn/tests/optimizer/
├── mod.rs          # 优化器单元测试模块
├── adam.rs         # Adam 单元测试（含 with_params 测试）
├── sgd.rs          # SGD 单元测试（含 with_params 测试）
└── ...

tests/
├── optimizer_example.rs   # 优化器集成测试
├── test_ada_line.rs       # 单样本ADALINE测试
├── test_adaline_batch.rs  # 批量ADALINE测试
└── test_mnist_gan.rs      # GAN 集成测试（with_params + detach）
```

# Optimizer架构设计

## 1. 设计目标

基于MatrixSlow Python版本的optimizer设计，为only_torch项目设计一个可扩展、可维护的优化器架构，支持多种优化算法（SGD、Momentum、AdaGrad、RMSProp、Adam等）。

## 2. 核心设计原则

- **可扩展性**: 易于添加新的优化算法
- **类型安全**: 利用Rust的类型系统确保安全性
- **性能优化**: 避免不必要的内存分配和拷贝
- **API一致性**: 与MatrixSlow Python版本保持相似的使用方式
- **梯度累积**: 支持mini-batch训练的梯度累积机制

## 3. 架构概览

```
Optimizer Trait (优化器特征)
├── 核心方法:
│   ├── one_step()     # 单步训练（前向+反向传播+梯度累积）
│   ├── update()       # 参数更新（执行具体优化算法）
│   └── reset()        # 重置累积状态
├── 具体实现:
│   ├── GradientDescent    # 梯度下降
│   ├── Momentum          # 动量法
│   ├── AdaGrad           # AdaGrad
│   ├── RMSProp           # RMSProp
│   └── Adam              # Adam优化器
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
```

## 5. 具体优化器实现

### 5.1 梯度下降优化器

```rust
pub struct GradientDescent {
    state: OptimizerState,
}

impl Optimizer for GradientDescent {
    fn update(&mut self, graph: &mut Graph) -> Result<(), GraphError> {
        for &node_id in &self.state.trainable_nodes {
            if let Some(avg_gradient) = self.state.gradient_accumulator.get_average_gradient(node_id) {
                let current_value = graph.get_node_value(node_id)?.unwrap();
                let new_value = current_value - self.state.learning_rate * &avg_gradient;
                graph.set_node_value(node_id, Some(&new_value))?;
            }
        }
        self.state.gradient_accumulator.clear();
        Ok(())
    }
}
```

### 5.2 Adam优化器

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

### 6.2 Mini-batch训练

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

## 7. 实现计划

### 阶段1: 基础架构 ✅
- [x] 实现`Optimizer` trait
- [x] 实现`GradientAccumulator`
- [x] 实现`OptimizerState`

### 阶段2: 基础优化器 🔄
- [x] 实现`SGD` (重命名自GradientDescent)
- [x] 创建`optimizer_example.rs`集成测试
- [ ] **修复梯度计算问题** (当前所有梯度为0)

### 阶段3: 高级优化器 🔄
- [ ] 实现`Momentum`
- [ ] 实现`AdaGrad`
- [ ] 实现`RMSProp`
- [x] 实现`Adam` (框架完成，需修复梯度问题)

### 阶段4: 优化和扩展
- [x] 创建batch版本测试 (`test_adaline_batch.rs`)
- [ ] 修复梯度计算，确保optimizer正常工作
- [ ] 性能优化
- [ ] 完善文档和测试

## 8. 当前问题和解决方案

### 问题1: 梯度计算返回0 🚨
**现象**: 所有参数节点的梯度都是0.0，导致参数无法更新
**可能原因**:
- 损失函数输入计算方式不正确
- 反向传播链路有问题
- 梯度转换逻辑错误

**解决方案**:
1. 对比原始单样本测试的损失函数计算方式
2. 检查`get_node_grad`方法的实现
3. 验证反向传播是否正确执行

### 问题2: 优化器算法命名 ✅
**解决**: 将`GradientDescent`重命名为`SGD`，更准确地反映其实现

## 8. 文件结构

```
src/nn/
├── mod.rs
├── graph.rs
├── nodes/
├── optimizer/           # 新增优化器模块
│   ├── mod.rs
│   ├── base.rs         # Optimizer trait和基础结构
│   ├── gradient_descent.rs
│   ├── momentum.rs
│   ├── adagrad.rs
│   ├── rmsprop.rs
│   └── adam.rs
└── tests/

tests/
└── optimizer_example.rs  # 集成测试
```

这个设计确保了代码的可扩展性和可维护性，同时与MatrixSlow Python版本保持API一致性。

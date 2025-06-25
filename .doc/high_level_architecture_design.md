# Only Torch 架构设计方案

## 1. 设计目标

- **PyTorch风格API**：支持类似PyTorch的模块化模型定义和训练
- **节点级演化**：支持NEAT算法在基础操作层面（add, matmul）进行结构演化
- **隐藏底层复杂性**：用户无需直接操作Graph对象
- **CPU专用**：无设备概念，专注CPU计算

## 2. 架构层次设计

```
graph TB
    subgraph "用户API层 (PyTorch风格)"
        A[Module Trait] --> B[Linear]
        A --> C[Conv2d]
        A --> D[Dropout]
        E[functional模块] --> F[relu, softmax, etc.]
        G[optim模块] --> H[SGD, Adam, etc.]
    end

    subgraph "中间抽象层"
        I[ModelGraph] --> J[封装Graph管理]
        K[AutoGrad] --> L[自动微分引擎]
        M[Evolvable Trait] --> N[NEAT演化接口]
    end

    subgraph "底层计算层 (现有)"
        O[Graph] --> P[计算图管理]
        Q[NodeHandle] --> R[节点操作]
        S[Tensor] --> T[张量运算]
    end

    B --> I
    C --> I
    D --> I
    I --> O
    K --> O
    M --> O

    style A fill:#e1f5fe
    style I fill:#f3e5f5
    style O fill:#e8f5e8
```

### 2.1 用户API层 (`nn::high_level`)

```rust
// 模块基类 - 类似PyTorch nn.Module
pub trait Module {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor, GraphError>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn train(&mut self);
    fn eval(&mut self);
}

// 具体层实现
pub struct Linear {
    weight: Parameter,
    bias: Option<Parameter>,
    // 内部持有ModelGraph引用
}

pub struct Conv2d { /* ... */ }
pub struct Dropout { /* ... */ }

// 函数式API - 类似PyTorch F
pub mod functional {
    pub fn relu(input: &Tensor) -> Tensor;
    pub fn log_softmax(input: &Tensor, dim: usize) -> Tensor;
    pub fn max_pool2d(input: &Tensor, kernel_size: usize) -> Tensor;
}
```

### 2.2 中间抽象层 (`nn::core`)

```rust
// 模型图管理器 - 封装底层Graph
pub struct ModelGraph {
    graph: Graph,
    input_nodes: HashMap<String, NodeId>,
    parameter_nodes: HashMap<String, NodeId>,
    output_node: Option<NodeId>,
}

// 自动微分引擎
pub struct AutoGrad {
    model_graph: ModelGraph,
}

// 演化接口 - 为NEAT预留
pub trait Evolvable {
    fn mutate_structure(&mut self) -> Result<(), GraphError>;
    fn crossover(&self, other: &Self) -> Result<Self, GraphError>;
    fn get_node_genes(&self) -> Vec<NodeGene>;
}
```

### 2.3 底层计算层 (现有代码)

保持现有的`Graph`、`NodeHandle`、`Tensor`不变，作为底层计算引擎。

## 3. 使用示例对比

### 3.1 PyTorch风格使用 (目标API)

```rust
use only_torch::nn::{Module, Linear, functional as F};

struct Net {
    conv1: Conv2d,
    conv2: Conv2d,
    dropout1: Dropout,
    dropout2: Dropout,
    fc1: Linear,
    fc2: Linear,
}

impl Net {
    fn new() -> Self {
        Self {
            conv1: Conv2d::new(1, 32, 3, 1),
            conv2: Conv2d::new(32, 64, 3, 1),
            dropout1: Dropout::new(0.25),
            dropout2: Dropout::new(0.5),
            fc1: Linear::new(9216, 128),
            fc2: Linear::new(128, 10),
        }
    }
}

impl Module for Net {
    fn forward(&mut self, x: &Tensor) -> Result<Tensor, GraphError> {
        let x = self.conv1.forward(x)?;
        let x = F::relu(&x);
        let x = self.conv2.forward(&x)?;
        let x = F::relu(&x);
        let x = F::max_pool2d(&x, 2);
        let x = self.dropout1.forward(&x)?;
        let x = x.flatten(1);
        let x = self.fc1.forward(&x)?;
        let x = F::relu(&x);
        let x = self.dropout2.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        let output = F::log_softmax(&x, 1);
        Ok(output)
    }
}

// 训练代码
fn main() -> Result<(), GraphError> {
    let mut model = Net::new();
    let optimizer = SGD::new(model.parameters(), 0.01);

    for epoch in 0..10 {
        for (data, target) in train_loader {
            let output = model.forward(&data)?;
            let loss = F::nll_loss(&output, &target);

            optimizer.zero_grad();
            loss.backward()?;
            optimizer.step();
        }
    }
    Ok(())
}
```

### 3.2 当前底层使用方式 (保持兼容)

```rust
// 现有的test_ada_line.rs风格仍然可用
let mut graph = Graph::new();
let x = graph.new_input_node(&[3, 1], Some("x"))?;
let w = graph.new_parameter_node(&[1, 3], Some("w"))?;
// ... 其余代码保持不变
```

## 4. 实现路径

### 阶段1：基础模块系统
1. 实现`Module` trait和基础层（Linear, Conv2d等）
2. 实现`ModelGraph`封装现有Graph
3. 实现基础的`functional`模块

### 阶段2：自动微分优化
1. 实现`AutoGrad`引擎
2. 优化反向传播性能
3. 添加优化器支持

### 阶段3：演化接口
1. 设计`Evolvable` trait
2. 实现节点级变异操作
3. 支持结构演化算法

## 5. 关键设计决策

### 5.1 Graph生命周期管理
- 每个`Module`内部持有一个`ModelGraph`
- `ModelGraph`封装底层`Graph`，管理节点生命周期
- 用户无需直接操作Graph

### 5.2 参数管理
- `Parameter`类型封装参数节点
- 自动收集模型参数用于优化器
- 支持参数共享和冻结

### 5.3 演化兼容性
- 底层节点保持可变性
- 高层API通过抽象层访问底层节点
- 演化算法可直接操作Graph结构

## 6. 下一步行动

1. **确认设计方案**：用户确认整体架构方向
2. **实现基础Module系统**：从Linear层开始
3. **创建示例项目**：验证API易用性
4. **逐步完善功能**：添加更多层类型和功能

这个设计既保持了现有底层代码的灵活性，又提供了PyTorch风格的高层API，同时为NEAT演化预留了接口。

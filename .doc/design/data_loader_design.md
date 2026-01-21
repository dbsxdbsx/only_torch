# DataLoader 使用指南

> 创建日期：2025-12-21
> 更新日期：2025-01-21
> 状态：✅ 已实现

## 1. 概述

`DataLoader` 是 PyTorch 风格的数据批量加载器，提供统一的数据迭代 API。

### 核心组件

| 组件 | 说明 |
|------|------|
| `TensorDataset` | 持有特征和标签的数据集容器 |
| `DataLoader` | 批量迭代器，支持 shuffle、drop_last 等选项 |

### 支持的功能

- ✅ 自动分批 (`batch_size`)
- ✅ 随机打乱 (`shuffle`)
- ✅ 丢弃不完整批次 (`drop_last`)
- ✅ 可重复性种子 (`seed`)
- ✅ 任意维度张量（支持 MLP、CNN、RNN 等场景）

---

## 2. 快速开始

### 2.1 基础用法

```rust
use only_torch::data::{DataLoader, TensorDataset};
use only_torch::tensor::Tensor;

// 准备数据
let features = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]); // [3 samples, 2 features]
let labels = Tensor::new(&[0.0, 1.0, 0.0], &[3, 1]);                   // [3 samples, 1 label]

// 创建数据集
let dataset = TensorDataset::new(features, labels);
println!("样本数: {}", dataset.len()); // 输出: 样本数: 3

// 创建 DataLoader
let loader = DataLoader::new(dataset, 2); // batch_size = 2

// 迭代
for (x_batch, y_batch) in loader.iter() {
    println!("特征形状: {:?}, 标签形状: {:?}", x_batch.shape(), y_batch.shape());
}
// 输出:
// 特征形状: [2, 2], 标签形状: [2, 1]  (第一批)
// 特征形状: [1, 2], 标签形状: [1, 1]  (第二批，不完整)
```

### 2.2 完整训练循环

```rust
use only_torch::data::{DataLoader, TensorDataset};
use only_torch::nn::{Graph, Adam, Module, Optimizer, VarLossOps};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 准备数据
    let (train_x, train_y) = generate_data();
    let train_dataset = TensorDataset::new(train_x, train_y);
    let train_loader = DataLoader::new(train_dataset, 32)
        .shuffle(true)
        .seed(42);

    // 2. 构建模型
    let graph = Graph::new_with_seed(42);
    let model = MyModel::new(&graph);
    let loss = model.output().cross_entropy(&labels_node);
    
    // 3. 优化器
    let mut optimizer = Adam::new(&graph, &model.parameters(), 0.001);

    // 4. 训练循环
    for epoch in 0..10 {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        for (x_batch, y_batch) in train_loader.iter() {
            // 前向传播
            input_node.set_value(&x_batch)?;
            labels_node.set_value(&y_batch)?;
            
            // 计算损失并反向传播
            epoch_loss += loss.value()?.unwrap()[[0, 0]];
            num_batches += 1;
            
            loss.backward()?;
            optimizer.step()?;
        }

        println!("Epoch {}: loss = {:.4}", epoch, epoch_loss / num_batches as f32);
    }

    Ok(())
}
```

---

## 3. API 详解

### 3.1 TensorDataset

```rust
impl TensorDataset {
    /// 创建新的数据集
    /// 
    /// # Panics
    /// 如果 features 和 labels 的第一维（样本数）不一致
    pub fn new(features: Tensor, labels: Tensor) -> Self;
    
    /// 获取样本数量
    pub fn len(&self) -> usize;
    
    /// 检查是否为空
    pub fn is_empty(&self) -> bool;
    
    /// 获取特征张量引用
    pub fn features(&self) -> &Tensor;
    
    /// 获取标签张量引用
    pub fn labels(&self) -> &Tensor;
}
```

### 3.2 DataLoader

```rust
impl DataLoader {
    /// 创建新的 DataLoader
    /// 
    /// # 参数
    /// - `dataset`: TensorDataset 实例
    /// - `batch_size`: 每批样本数（必须 > 0）
    pub fn new(dataset: TensorDataset, batch_size: usize) -> Self;
    
    /// 设置是否随机打乱数据（默认 false）
    pub fn shuffle(self, shuffle: bool) -> Self;
    
    /// 设置是否丢弃最后一个不完整批次（默认 false）
    pub fn drop_last(self, drop_last: bool) -> Self;
    
    /// 设置随机种子（用于 shuffle 的可重复性）
    pub fn seed(self, seed: u64) -> Self;
    
    /// 获取批次数量
    pub fn num_batches(&self) -> usize;
    
    /// 获取数据集大小
    pub fn len(&self) -> usize;
    
    /// 创建迭代器
    pub fn iter(&self) -> DataLoaderIterator<'_>;
}
```

---

## 4. 使用场景

### 4.1 MLP（全连接网络）

```rust
// 特征: [samples, feature_dim]
let features = Tensor::new(&data, &[1000, 784]);
let labels = Tensor::new(&targets, &[1000, 10]);

let dataset = TensorDataset::new(features, labels);
let loader = DataLoader::new(dataset, 64)
    .shuffle(true)
    .drop_last(true);  // 确保所有批次大小一致
```

### 4.2 CNN（卷积网络）

```rust
// 图像: [samples, channels, height, width]
let images = Tensor::new(&pixels, &[60000, 1, 28, 28]);
let labels = Tensor::new(&targets, &[60000, 10]);

let dataset = TensorDataset::new(images, labels);
let loader = DataLoader::new(dataset, 32).shuffle(true);

for (batch_images, batch_labels) in loader.iter() {
    // batch_images.shape() == [32, 1, 28, 28]
    // batch_labels.shape() == [32, 10]
}
```

### 4.3 RNN（循环网络）

```rust
// 序列: [samples, seq_len, input_size]
let sequences = Tensor::new(&seq_data, &[500, 10, 8]);  // 500 个长度为 10 的序列
let labels = Tensor::new(&targets, &[500, 2]);          // 二分类

let dataset = TensorDataset::new(sequences, labels);
let loader = DataLoader::new(dataset, 16).shuffle(true);

for (x_batch, y_batch) in loader.iter() {
    // x_batch.shape() == [16, 10, 8]
    // y_batch.shape() == [16, 2]
    model.forward(&x_batch)?;
}
```

---

## 5. 设计决策

### 5.1 为什么是 `iter()` 而不是直接实现 `Iterator`？

`DataLoader` 需要在每次迭代时可能重新打乱数据，因此采用 `iter()` 方法创建新的迭代器，允许多次遍历且每次可以有不同的顺序。

```rust
// 每次调用 iter() 都可能产生不同的顺序（如果 shuffle=true 且无 seed）
for _ in 0..epochs {
    for batch in loader.iter() {  // 每个 epoch 重新创建迭代器
        // ...
    }
}
```

### 5.2 为什么 `shuffle` 每次 `iter()` 调用都重新打乱？

这是 PyTorch 的标准行为，有助于：
- 减少过拟合（每个 epoch 看到不同的批次组合）
- 提高训练稳定性

如需可重复性，请使用 `seed()` 方法。

### 5.3 `drop_last` 的使用场景

某些网络（如 BatchNorm、固定 batch 维度的图）要求所有批次大小一致，此时应使用 `drop_last(true)`：

```rust
let loader = DataLoader::new(dataset, 32)
    .drop_last(true);  // 丢弃最后不足 32 个样本的批次
```

---

## 6. 与 PyTorch 对比

| 功能 | PyTorch | only_torch |
|------|---------|------------|
| 基础迭代 | `for x, y in loader:` | `for (x, y) in loader.iter()` |
| Shuffle | `DataLoader(..., shuffle=True)` | `.shuffle(true)` |
| Drop Last | `DataLoader(..., drop_last=True)` | `.drop_last(true)` |
| 固定种子 | `torch.manual_seed(42)` | `.seed(42)` |
| 多进程加载 | `num_workers=4` | ❌ 暂不支持 |
| 自定义采样器 | `sampler=...` | ❌ 暂不支持 |

---

## 7. 非目标（当前阶段）

以下功能暂不在实现范围内：

- ❌ 多线程/异步数据加载
- ❌ 数据增强（augmentation）
- ❌ 自定义采样器（Sampler trait）
- ❌ 分布式数据加载

---

## 8. 参考资料

- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
- [tf.data 指南](https://www.tensorflow.org/guide/data)

# DataLoader 设计文档

> 创建日期：2025-12-21
> 更新日期：2025-01-29
> 状态：✅ 已完成 Phase 1

## 1. 概述

`DataLoader` 是 PyTorch 风格的数据批量加载器，提供统一的数据迭代 API。

### 当前组件

| 组件 | 说明 | 适用场景 |
|------|------|----------|
| `TensorDataset` | 持有特征和标签的数据集容器 | 固定长度数据 |
| `DataLoader` | 批量迭代器 | 固定长度数据 |
| `VarLenDataset` | 变长序列数据集 | 变长序列 |
| `BucketedDataLoader` | 分桶加载器 | 变长序列 |

### 支持的功能

- ✅ 自动分批 (`batch_size`)
- ✅ 随机打乱 (`shuffle`)
- ✅ 丢弃不完整批次 (`drop_last`)
- ✅ 可重复性种子 (`seed`)
- ✅ 任意维度张量（支持 MLP、CNN、RNN 等场景）
- ✅ 变长序列分桶 (`VarLenDataset` + `BucketedDataLoader`)

---

## 2. 架构改进计划

### 2.1 当前问题

当前存在**两个独立的 DataLoader**，API 不统一：

```rust
// 固定长度数据
let loader = DataLoader::new(dataset, 64).shuffle(true);

// 变长序列 —— 完全不同的类，没有 batch_size 参数
let loader = BucketedDataLoader::new(&var_len_dataset).shuffle(true);
```

| 问题 | 说明 |
|------|------|
| API 碎片化 | 用户需要根据数据类型选择不同的类 |
| 无共享抽象 | 两个类没有公共 trait，无法泛型编程 |
| 扩展性差 | 添加新策略（如优先级采样）需要再创建新类 |

### 2.2 改进目标（✅ 已实现）

统一成**单一 DataLoader**，通过策略模式支持不同的采样方式：

```rust
// 固定长度数据 —— 使用 SequentialSampling（默认）
let loader = DataLoader::new(dataset, 64)
    .shuffle(true)
    .drop_last(true);

// 变长序列 —— 使用 BucketedSampling
let loader = DataLoader::from_var_len(&var_len_dataset)
    .shuffle(true)
    .seed(42);
```

### 2.3 目标架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    SamplingStrategy Trait                       │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │ SequentialSampling │  │ BucketedSampling │  │ 未来扩展...    │    │
│  │   (顺序/打乱)    │  │   (分桶)       │  │               │    │
│  └────────────────┘  └────────────────┘  └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              DataLoader<D: Dataset, S: SamplingStrategy>        │
│                                                                 │
│  - new(dataset, batch_size)                                     │
│  - shuffle(bool)                                                │
│  - drop_last(bool)                                              │
│  - seed(u64)                                                    │
│  - bucketed() -> DataLoader<D, BucketedSampling>                │
│  - iter() -> impl Iterator<Item = (Tensor, Tensor)>             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. TODO 清单

### Phase 1：统一监督学习 DataLoader（✅ 已完成）

- [x] 定义 `Dataset` trait
- [x] 定义 `SamplingStrategy` trait
- [x] 实现 `SequentialSampling`（顺序/打乱采样）
- [x] 实现 `BucketedSampling`（分桶采样）
- [x] 重构 `DataLoader` 为泛型结构
- [x] 添加 `DataLoader::from_var_len()` 便捷方法
- [x] 删除旧 `BucketedDataLoader`
- [x] 更新 examples 中的用法
- [x] 补充单元测试（21 个测试用例）

### Phase 2：强化学习扩展（待定）

> ⚠️ 以下内容待强化学习框架确定后再规划

- [ ] 设计 `Experience` 样本类型
- [ ] 设计 `ReplayBuffer` 结构
- [ ] 实现 `UniformSampling`（均匀随机采样）
- [ ] 实现 `PrioritizedSampling`（优先级采样 / PER）
- [ ] 实现 `HardSampling`（困难样本优先）
- [ ] 实现 `HindsightSampling`（HER）
- [ ] 评估是否复用 `SamplingStrategy` trait 或需要新抽象

---

## 4. 快速开始

### 4.1 基础用法

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

### 4.2 完整训练循环

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

## 5. API 详解

### 5.1 TensorDataset

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

### 5.2 DataLoader

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

### 5.3 VarLenDataset（变长序列）

```rust
impl VarLenDataset {
    /// 创建新的变长数据集
    pub fn new(feature_size: usize, label_size: usize) -> Self;

    /// 添加样本
    pub fn push(&mut self, sample: VarLenSample);

    /// 获取样本数量
    pub fn len(&self) -> usize;
}
```

### 5.4 DataLoader + VarLenDataset（分桶加载）

```rust
impl DataLoader<&VarLenDataset, BucketedSampling> {
    /// 从变长数据集创建 DataLoader（使用 BucketedSampling）
    pub fn from_var_len(dataset: &VarLenDataset) -> Self;

    /// 设置是否打乱（桶内打乱）
    pub fn shuffle(self, shuffle: bool) -> Self;

    /// 设置随机种子
    pub fn seed(self, seed: u64) -> Self;
}
```

---

## 6. 使用场景

### 6.1 MLP（全连接网络）

```rust
// 特征: [samples, feature_dim]
let features = Tensor::new(&data, &[1000, 784]);
let labels = Tensor::new(&targets, &[1000, 10]);

let dataset = TensorDataset::new(features, labels);
let loader = DataLoader::new(dataset, 64)
    .shuffle(true)
    .drop_last(true);  // 确保所有批次大小一致
```

### 6.2 CNN（卷积网络）

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

### 6.3 RNN（循环网络 - 固定长度）

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

### 6.4 RNN（循环网络 - 变长序列）

```rust
// 变长序列数据集
let mut dataset = VarLenDataset::new(1, 2);  // feature_size=1, label_size=2
dataset.push(VarLenSample::new(vec![1.0, 0.0, 1.0], 3, 1, vec![0.0, 1.0]));
dataset.push(VarLenSample::new(vec![1.0, 1.0, 0.0, 1.0, 0.0], 5, 1, vec![1.0, 0.0]));

// 分桶加载器 —— 同一批次内序列长度相同
let loader = DataLoader::from_var_len(&dataset).shuffle(true);

for (x_batch, y_batch) in loader.iter() {
    // x_batch: [batch, seq_len, feature_size]
    // 同一批次内 seq_len 相同，不同批次可能不同
}
```

---

## 7. 设计决策

### 7.1 为什么是 `iter()` 而不是直接实现 `Iterator`？

`DataLoader` 需要在每次迭代时可能重新打乱数据，因此采用 `iter()` 方法创建新的迭代器，允许多次遍历且每次可以有不同的顺序。

```rust
// 每次调用 iter() 都可能产生不同的顺序（如果 shuffle=true 且无 seed）
for _ in 0..epochs {
    for batch in loader.iter() {  // 每个 epoch 重新创建迭代器
        // ...
    }
}
```

### 7.2 为什么 `shuffle` 每次 `iter()` 调用都重新打乱？

这是 PyTorch 的标准行为，有助于：
- 减少过拟合（每个 epoch 看到不同的批次组合）
- 提高训练稳定性

如需可重复性，请使用 `seed()` 方法。

### 7.3 `drop_last` 的使用场景

某些网络（如 BatchNorm、固定 batch 维度的图）要求所有批次大小一致，此时应使用 `drop_last(true)`：

```rust
let loader = DataLoader::new(dataset, 32)
    .drop_last(true);  // 丢弃最后不足 32 个样本的批次
```

---

## 8. 与 PyTorch 对比

| 功能 | PyTorch | only_torch |
|------|---------|------------|
| 基础迭代 | `for x, y in loader:` | `for (x, y) in loader.iter()` |
| Shuffle | `DataLoader(..., shuffle=True)` | `.shuffle(true)` |
| Drop Last | `DataLoader(..., drop_last=True)` | `.drop_last(true)` |
| 固定种子 | `torch.manual_seed(42)` | `.seed(42)` |
| 多进程加载 | `num_workers=4` | ❌ 暂不支持 |
| 自定义采样器 | `sampler=...` | 🔄 Phase 1 目标 |

---

## 9. 非目标（当前阶段）

以下功能暂不在实现范围内：

- ❌ 多线程/异步数据加载
- ❌ 数据增强（augmentation）
- ❌ 分布式数据加载

---

## 10. 参考资料

- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
- [tf.data 指南](https://www.tensorflow.org/guide/data)

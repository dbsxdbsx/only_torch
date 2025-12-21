# DataLoader 设计文档

> 创建日期：2025-12-21
> 状态：设计完成，待实现

## 1. 背景与目标

### 1.1 当前状态

- 无数据加载模块，数据在测试代码中内嵌
- 现有示例（XOR、Adaline）使用手动 `Tensor::stack` + `slice` 处理
- MNIST 示例需要：加载 IDX 二进制文件、归一化、one-hot 编码

### 1.2 设计目标

1. **MVP 优先**：让 MNIST 能跑起来
2. **可扩展**：预留 trait 抽象，未来可支持更多数据集
3. **Rust 风格**：利用迭代器模式，零成本抽象
4. **简单易用**：API 直观，减少样板代码

### 1.3 非目标（当前阶段）

- ❌ 多线程/异步数据加载
- ❌ 数据增强（augmentation）
- ❌ 分布式数据加载
- ❌ 真正的 batch forward（框架暂不支持）

---

## 2. 主流框架对比

| 框架           | 核心抽象                 | 批处理          | 打乱              | 特点                   |
| -------------- | ------------------------ | --------------- | ----------------- | ---------------------- |
| **PyTorch**    | `Dataset` + `DataLoader` | DataLoader 参数 | DataLoader 参数   | 灵活，用户实现 Dataset |
| **TensorFlow** | `tf.data.Dataset`        | `.batch()` 链式 | `.shuffle()` 链式 | 函数式，惰性求值       |
| **JAX**        | 无官方，用 grain/tfds    | 外部库          | 外部库            | 极简核心               |
| **MatrixSlow** | 无                       | 手动循环        | 手动              | 外部 sklearn 加载      |

### 我们的选择

采用 **PyTorch 风格**（trait 抽象）+ **Rust 迭代器模式**，因为：

- 与现有代码风格一致（trait-based）
- 迭代器是 Rust 的惯用模式
- 比 tf.data 的链式调用更简单直观

---

## 3. 架构设计

### 3.1 模块结构

```
src/data/
├── mod.rs              # 模块入口，re-export 公共 API
├── error.rs            # DataError 错误类型
├── dataset.rs          # Dataset trait 定义
├── sampler.rs          # Sampler trait + 实现
├── dataloader.rs       # DataLoader 结构体
├── transforms.rs       # 数据变换函数
└── datasets/
    ├── mod.rs          # 内置数据集入口
    ├── mnist.rs        # MNIST 数据集
    └── in_memory.rs    # 通用内存数据集（未来）
```

### 3.2 核心 Trait

```rust
// ===== dataset.rs =====

/// 数据集 trait
///
/// 类似 PyTorch 的 Dataset，但更 Rust 化。
/// 所有数据集必须支持随机访问（Map-style）。
pub trait Dataset {
    /// 返回数据集中的样本数量
    fn len(&self) -> usize;

    /// 获取第 index 个样本
    /// 返回 (input, label) 元组
    /// - input: 单样本 Tensor，形状取决于数据类型
    ///   - 图像（原始）: [C, H, W] 如 [1, 28, 28]
    ///   - 图像（flatten）: [D] 如 [784]
    /// - label: one-hot 或标量，如 [num_classes] 或 [1]
    fn get(&self, index: usize) -> (Tensor, Tensor);

    /// 数据集是否为空
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 输入的形状（不含 batch 维度）
    /// 如 MNIST: [1, 28, 28] 或 [784]
    fn input_shape(&self) -> &[usize];

    /// 标签的形状（不含 batch 维度）
    /// 如 MNIST one-hot: [10]
    fn label_shape(&self) -> &[usize];
}
```

```rust
// ===== sampler.rs =====

/// 采样器 trait
///
/// 控制数据访问顺序，支持顺序、随机、加权等策略。
pub trait Sampler: Iterator<Item = usize> {
    /// 采样器覆盖的总索引数
    fn len(&self) -> usize;

    /// 重置采样器到初始状态（用于新 epoch）
    fn reset(&mut self);
}

/// 顺序采样器
pub struct SequentialSampler { ... }

/// 随机采样器（可设置种子）
pub struct RandomSampler { ... }
```

```rust
// ===== dataloader.rs =====

/// 数据加载器
///
/// 组合 Dataset 和 Sampler，提供批量数据迭代。
pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    seed: Option<u64>,
    // 内部状态
    indices: Vec<usize>,
    current_position: usize,
}

impl<D: Dataset> DataLoader<D> {
    /// 创建新的 DataLoader
    pub fn new(dataset: D, batch_size: usize) -> Self;

    /// 设置是否打乱数据
    pub fn shuffle(self, shuffle: bool) -> Self;

    /// 设置随机种子（用于可重复性）
    pub fn seed(self, seed: u64) -> Self;

    /// 重置到 epoch 开始，可选重新打乱
    pub fn reset(&mut self);

    /// 返回总批次数
    pub fn num_batches(&self) -> usize;
}

impl<D: Dataset> Iterator for DataLoader<D> {
    type Item = (Tensor, Tensor);  // (batch_inputs, batch_labels)

    fn next(&mut self) -> Option<Self::Item>;
}
```

### 3.3 MNIST 数据集

```rust
// ===== datasets/mnist.rs =====

/// MNIST 手写数字数据集
///
/// 自动处理：
/// - IDX 二进制格式解析（支持 .gz 压缩）
/// - 像素归一化 (0-255 → 0-1)
/// - 标签 one-hot 编码
/// - 可选自动下载
pub struct MnistDataset {
    images: Tensor,     // [N, 1, 28, 28] 或 flatten 后 [N, 784]
    labels: Tensor,     // [N, 10]
    len: usize,
    is_flattened: bool,
}

impl MnistDataset {
    /// 完整加载 API
    ///
    /// - root: 数据目录，None 则使用默认 (~/.cache/only_torch/datasets/mnist)
    /// - train: true=训练集(60000), false=测试集(10000)
    /// - download: true=自动下载缺失文件
    pub fn load(root: Option<&str>, train: bool, download: bool) -> Result<Self, DataError>;

    /// 便捷 API：加载训练集（默认路径，自动下载）
    pub fn train() -> Result<Self, DataError> {
        Self::load(None, true, true)
    }

    /// 便捷 API：加载测试集（默认路径，自动下载）
    pub fn test() -> Result<Self, DataError> {
        Self::load(None, false, true)
    }

    /// 将图像展平为 [N, 784]（用于 MLP）
    pub fn flatten(self) -> Self;

    /// 从原始字节加载（用于嵌入式或测试）
    pub fn from_bytes(images_bytes: &[u8], labels_bytes: &[u8]) -> Result<Self, DataError>;
}

impl Dataset for MnistDataset {
    fn len(&self) -> usize { self.len }
    fn get(&self, index: usize) -> (Tensor, Tensor) { ... }
    fn input_shape(&self) -> &[usize] {
        if self.is_flattened { &[784] } else { &[1, 28, 28] }
    }
    fn label_shape(&self) -> &[usize] { &[10] }
}
```

### 3.4 数据变换

```rust
// ===== transforms.rs =====

/// 将 0-255 像素值归一化到 0-1
pub fn normalize_pixels(tensor: &Tensor) -> Tensor;

/// 将类别索引转换为 one-hot 编码
///
/// 输入: [N] 或 [N, 1]，值为 0..num_classes
/// 输出: [N, num_classes]
pub fn one_hot(labels: &Tensor, num_classes: usize) -> Tensor;

/// 展平图像
///
/// 输入: [N, H, W] 或 [N, C, H, W]
/// 输出: [N, H*W] 或 [N, C*H*W]
pub fn flatten_images(tensor: &Tensor) -> Tensor;
```

---

## 4. 使用示例

### 4.1 MNIST 训练示例（MLP，flatten 版本）

```rust
use only_torch::data::{MnistDataset, DataLoader};
use only_torch::nn::{Graph, optimizer::SGD};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 加载数据（使用 flatten 版本，适合 MLP）
    let train_data = MnistDataset::train()?.flatten();  // [60000, 784]
    let test_data = MnistDataset::test()?.flatten();    // [10000, 784]

    // 2. 创建 DataLoader
    let mut train_loader = DataLoader::new(train_data, 32)
        .shuffle(true)
        .seed(42);

    // 3. 构建网络（2 层 MLP: 784 -> 128 -> 10）
    let mut graph = Graph::new();
    let x = graph.new_input_node(&[1, 784], Some("input"))?;
    let y = graph.new_input_node(&[1, 10], Some("label"))?;

    // 隐藏层: 784 -> 128
    let w1 = graph.new_parameter_node(&[784, 128], Some("w1"))?;
    let b1 = graph.new_parameter_node(&[1, 128], Some("b1"))?;
    let h1 = graph.new_add_node(&[
        graph.new_mat_mul_node(x, w1, None)?,
        b1
    ], None)?;
    let a1 = graph.new_sigmoid_node(h1, None)?;

    // 输出层: 128 -> 10
    let w2 = graph.new_parameter_node(&[128, 10], Some("w2"))?;
    let b2 = graph.new_parameter_node(&[1, 10], Some("b2"))?;
    let logits = graph.new_add_node(&[
        graph.new_mat_mul_node(a1, w2, None)?,
        b2
    ], None)?;

    // 损失函数
    let loss = graph.new_softmax_cross_entropy_node(logits, y, Some("loss"))?;

    // 4. 优化器
    let mut optimizer = SGD::new(&graph, 0.01)?;

    // 5. 训练循环
    for epoch in 0..10 {
        train_loader.reset();
        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for (batch_images, batch_labels) in &mut train_loader {
            // 当前框架不支持 batch forward，需逐样本处理
            for i in 0..batch_images.shape()[0] {
                let image = batch_images.slice(&[&i, &(..)]);  // [1, 784]
                let label = batch_labels.slice(&[&i, &(..)]);  // [1, 10]

                graph.set_node_value(x, Some(&image))?;
                graph.set_node_value(y, Some(&label))?;

                optimizer.one_step(&mut graph, loss)?;
            }

            optimizer.update(&mut graph)?;
            total_loss += graph.get_node_value(loss)?.unwrap()[[0, 0]];
            batch_count += 1;
        }

        println!("Epoch {}: avg_loss = {:.4}", epoch + 1, total_loss / batch_count as f32);
    }

    Ok(())
}
```

### 4.2 加载原始形状（用于未来 CNN）

```rust
// 保持原始 NCHW 格式 [N, 1, 28, 28]
let train_data = MnistDataset::train()?;  // 不调用 flatten()

// 单样本形状为 [1, 28, 28]
let (image, label) = train_data.get(0);
assert_eq!(image.shape(), &[1, 28, 28]);
assert_eq!(label.shape(), &[10]);
```

### 4.3 自定义数据路径

```rust
// 指定自定义路径，不自动下载
let train_data = MnistDataset::load(
    Some("./my_custom_data/mnist"),
    true,   // train
    false,  // download=false，数据必须已存在
)?;
```

---

## 5. 实现计划

### Phase 1: MNIST MVP ✅ 优先

**目标**：能跑通 MNIST MLP 训练示例

- [ ] 添加依赖到 `Cargo.toml`

  ```toml
  [dependencies]
  flate2 = "1.0"       # gzip 解压
  dirs = "5.0"         # 跨平台目录
  ureq = "2.9"         # HTTP 下载（blocking）

  [dev-dependencies]
  indicatif = "0.17"   # 进度条（可选）
  ```

- [ ] `src/data/mod.rs` - 模块入口
- [ ] `src/data/error.rs` - DataError 定义
- [ ] `src/data/datasets/mnist.rs`
  - [ ] IDX 二进制格式解析
  - [ ] gzip 解压支持
  - [ ] 自动下载功能
  - [ ] 像素归一化 (0-255 → 0-1)
  - [ ] one-hot 编码
  - [ ] flatten() 方法
- [ ] `src/data/transforms.rs` - `normalize_pixels`, `one_hot`, `flatten_images` 函数
- [ ] `tests/test_mnist_loading.rs` - MNIST 数据加载测试
- [ ] `examples/mnist_mlp.rs` - MNIST MLP 训练示例

### Phase 2: DataLoader 抽象

**目标**：提供通用的数据加载器

- [ ] `src/data/dataset.rs` - Dataset trait
- [ ] `src/data/sampler.rs` - Sampler trait + SequentialSampler + RandomSampler
- [ ] `src/data/dataloader.rs` - DataLoader 实现
- [ ] 重构 MnistDataset 实现 Dataset trait
- [ ] `tests/test_dataloader.rs` - DataLoader 单元测试

### Phase 3: 扩展（未来）

- [ ] `src/data/datasets/in_memory.rs` - 通用内存数据集
- [ ] 更多变换函数（标准化、随机裁剪等）
- [ ] FashionMNIST, CIFAR-10 等数据集
- [ ] 多线程数据预加载（可选）

---

## 6. IDX 文件格式参考

MNIST 使用 IDX 二进制格式：

### Images 文件 (train-images-idx3-ubyte)

```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
...
```

### Labels 文件 (train-labels-idx1-ubyte)

```
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
...
```

**注意**：所有整数均为大端序（Big Endian）。

---

## 7. 错误处理

```rust
// ===== error.rs =====

#[derive(Debug)]
pub enum DataError {
    /// 文件未找到
    FileNotFound(String),

    /// IO 错误
    IoError(std::io::Error),

    /// 格式错误（如 magic number 不匹配）
    FormatError(String),

    /// 索引越界
    IndexOutOfBounds { index: usize, len: usize },

    /// 形状不匹配
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    /// 下载错误
    DownloadError(String),

    /// 校验和不匹配
    ChecksumMismatch { expected: String, got: String },

    /// 解压错误
    DecompressionError(String),
}

impl std::fmt::Display for DataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileNotFound(path) => write!(f, "文件未找到: {}", path),
            Self::IoError(e) => write!(f, "IO 错误: {}", e),
            Self::FormatError(msg) => write!(f, "格式错误: {}", msg),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "索引越界: {} >= {}", index, len)
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "形状不匹配: 期望 {:?}, 实际 {:?}", expected, got)
            }
            Self::DownloadError(msg) => write!(f, "下载错误: {}", msg),
            Self::ChecksumMismatch { expected, got } => {
                write!(f, "校验和不匹配: 期望 {}, 实际 {}", expected, got)
            }
            Self::DecompressionError(msg) => write!(f, "解压错误: {}", msg),
        }
    }
}

impl std::error::Error for DataError {}

impl From<std::io::Error> for DataError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}
```

---

## 8. 设计决策

### 8.1 数据存放位置

**决策**：采用 PyTorch 惯例，默认 `~/.cache/only_torch/datasets/`

| 框架         | 默认路径                         |
| ------------ | -------------------------------- |
| PyTorch      | `~/.cache/torch/hub/` 或用户指定 |
| TensorFlow   | `~/.keras/datasets/`             |
| Hugging Face | `~/.cache/huggingface/`          |

```rust
/// 获取默认数据目录
pub fn default_data_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("only_torch")
        .join("datasets")
}

/// MNIST 加载，支持自定义路径
pub fn load(root: Option<&str>, train: bool, download: bool) -> Result<Self, DataError>
```

### 8.2 gzip 压缩支持

**决策**：原生支持 `.gz` 格式

- MNIST 官方仅提供 `.gz` 压缩文件
- 使用 `flate2` crate 解压（纯 Rust，无外部依赖）
- 自动检测：若同时存在 `.gz` 和解压文件，优先使用解压版本（加载更快）

```toml
# Cargo.toml
[dependencies]
flate2 = "1.0"
```

### 8.3 自动下载功能

**决策**：支持自动下载，参考 PyTorch 的 `download=True` 模式

```rust
let mnist = MnistDataset::load(None, true, true)?;  // download=true
```

- 下载源：官方镜像 `http://yann.lecun.com/exdb/mnist/`
- 使用 `ureq` 或 `reqwest`（blocking）进行 HTTP 请求
- 显示下载进度（可选，用 `indicatif`）
- 下载后自动验证 MD5/SHA256 校验和

### 8.4 Tensor 形状约定 ⭐

**决策**：遵循 **NCHW** 格式（PyTorch 风格），同时提供 flatten 选项

#### 主流框架对比

| 框架                 | 图像格式 | MNIST 形状       | 说明                 |
| -------------------- | -------- | ---------------- | -------------------- |
| **PyTorch**          | NCHW     | `[N, 1, 28, 28]` | 卷积层默认格式       |
| **TensorFlow/Keras** | NHWC     | `[N, 28, 28, 1]` | 默认 `channels_last` |
| **JAX**              | 灵活     | 用户自定义       | 常用 NHWC            |
| **OpenCV**           | HWC      | `[28, 28, 1]`    | 无 batch 维度        |

#### 我们的方案

1. **原始形状**：`[N, 1, 28, 28]` (NCHW)

   - 与 PyTorch 一致，未来支持 CNN 时无需转换
   - 单样本为 `[1, 28, 28]`

2. **提供 flatten 选项**：

   ```rust
   // 方式 1：加载时指定
   let mnist = MnistDataset::load(None, true, true)?
       .flatten();  // 形状变为 [N, 784]

   // 方式 2：使用变换函数
   let flat_images = transforms::flatten_images(&images);
   ```

3. **单样本 vs 批量**：
   - `Dataset::get(index)` 返回单样本：`([1, 28, 28], [10])` 或 `([784], [10])`
   - `DataLoader::next()` 返回批量：`([B, 1, 28, 28], [B, 10])` 或 `([B, 784], [B, 10])`

#### 设计理由

```
为什么选择 NCHW 而非 NHWC？

1. PyTorch 兼容性：用户从 PyTorch 迁移时体验一致
2. CNN 友好：卷积层的标准输入格式（未来扩展）
3. 内存布局：NCHW 在 CPU 上的局部性更好（通道连续）
4. 简单转换：flatten 比 reshape 更直观

权衡：
- TensorFlow 用户需适应（但我们项目定位更接近 PyTorch）
- 纯 MLP 场景下 [N, 784] 更简洁（因此提供 flatten 选项）
```

### 8.5 API 设计汇总

```rust
use only_torch::data::{MnistDataset, DataLoader};

// 完整 API（带所有选项）
let train_data = MnistDataset::load(
    Some("./my_data"),  // 自定义路径，None 则使用默认
    true,               // train=true 加载训练集
    true,               // download=true 自动下载
)?;

// 简洁 API（使用默认值）
let train_data = MnistDataset::train()?;      // 训练集，默认路径，自动下载
let test_data = MnistDataset::test()?;        // 测试集

// 获取 flatten 版本（用于 MLP）
let train_flat = train_data.flatten();        // [N, 784]

// 创建 DataLoader
let loader = DataLoader::new(train_flat, 32)
    .shuffle(true)
    .seed(42);
```

### 8.6 缓存管理

**决策**：暂不提供专门的清理 API，通过文档说明手动清理方式

**理由**：

- MNIST 数据集较小（~11MB），清理需求不强烈
- 用户可直接删除目录，无需额外 API
- 符合 MVP 原则，减少 API 表面积
- 主流框架（TensorFlow/Keras）也采用类似方式

**缓存位置**：

| 平台        | 默认路径                              |
| ----------- | ------------------------------------- |
| Linux/macOS | `~/.cache/only_torch/datasets/`       |
| Windows     | `%LOCALAPPDATA%\only_torch\datasets\` |

**手动清理方式**：

```bash
# Linux/macOS
rm -rf ~/.cache/only_torch/datasets/mnist/

# Windows (PowerShell)
Remove-Item -Recurse -Force "$env:LOCALAPPDATA\only_torch\datasets\mnist"

# Windows (cmd)
rmdir /s /q "%LOCALAPPDATA%\only_torch\datasets\mnist"
```

> 💡 **未来扩展**：如有实际需求，可考虑添加 `MnistDataset::clear_cache()` API。

---

## 9. 参考资料

- [MNIST 官方网站](http://yann.lecun.com/exdb/mnist/)
- [PyTorch DataLoader 文档](https://pytorch.org/docs/stable/data.html)
- [tf.data 指南](https://www.tensorflow.org/guide/data)
- [IDX 文件格式说明](http://yann.lecun.com/exdb/mnist/)

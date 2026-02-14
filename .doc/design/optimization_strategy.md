# 性能优化策略

> 最后更新: 2025-12-23
> 状态: **规划中**
> 适用范围: Only Torch 全局

---

## 项目特点

在讨论优化策略之前，必须明确 Only Torch 的独特定位：

| 特点              | 说明                       | 优化影响                  |
| :---------------- | :------------------------- | :------------------------ |
| **CPU-only**      | 不支持 GPU，专注 CPU 计算  | 无 CUDA，依赖 SIMD/多线程 |
| **NEAT 演化网络** | 网络结构由进化产生，非预设 | 稀疏连接，不规则拓扑      |
| **网络规模较小**  | 进化网络通常几十到几百节点 | 小矩阵运算为主            |
| **结构动态变化**  | 进化过程中拓扑不断改变     | 难以预编译优化            |
| **混合训练模式**  | 进化 + 梯度下降交替        | 需要两种场景都高效        |

---

## 计算图执行模式

### 静态图 vs 动态图

| 特性         | 静态图（TensorFlow 1.x / XLA） | 动态图（PyTorch） | Only Torch   |
| ------------ | ------------------------------ | ----------------- | ------------ |
| **构建方式** | 先定义完整图，后执行           | 边执行边构建      | 先定义后执行 |
| **图结构**   | 编译时完全已知                 | 运行时动态变化    | 运行时数据   |
| **优化时机** | 编译时全局优化                 | 运行时 JIT        | 仅代码级优化 |
| **算子融合** | ✅ 可做                        | ⚠️ JIT 部分支持   | ❌ 难以实现  |

### Only Torch 的执行模式

虽然代码风格看起来像静态图：

```rust
// 构建阶段（看起来像静态图）
let mut graph = Graph::new();
let x = graph.new_input_node(&[batch, 784], Some("x"))?;
let fc1 = linear(&mut graph, x, 784, 128, batch, Some("fc1"))?;
// ...

// 执行阶段
graph.forward_batch(loss)?;
```

但实际上是**运行时解释执行**：

```
Rust 编译器看到的：               实际运行时：
┌─────────────────────┐          ┌─────────────────────┐
│ "创建一些节点"        │          │ 遍历节点列表         │
│ "建立一些连接"        │   →      │ 动态分发到具体类型   │
│ "调用 forward"       │          │ 逐个计算节点值       │
└─────────────────────┘          └─────────────────────┘
     编译器不知道                      运行时才知道
     图的具体结构                      具体执行什么
```

### 为什么传统静态图优化难以应用

**传统框架的图级优化**（如 TensorFlow XLA、PyTorch TorchScript）：

```
识别模式:  Conv → BatchNorm → ReLU
              ↓ 编译时融合
生成代码:  ConvBNReLU_fused_kernel()  ← 一个优化后的函数
```

**Only Torch 的挑战**：

| 挑战               | 原因                                     | 影响                          |
| ------------------ | ---------------------------------------- | ----------------------------- |
| **无标准算子模式** | NEAT 使用原始节点而非 Conv/FC 等高层算子 | 无法匹配已知的融合模式        |
| **不规则拓扑**     | 跳跃连接、多输入多输出、稀疏连接         | 难以预测内存访问模式          |
| **结构动态变化**   | 每代进化可能产生不同拓扑                 | 无法预编译固定执行计划        |
| **编译器不可见**   | 图结构是运行时数据                       | rustc/LLVM 无法针对特定图优化 |

### 有效的优化层次

```
┌─────────────────────────────────────────────────────────────┐
│  图级优化（XLA 风格）                                        │
│  ├─ 算子融合                           ❌ 不适用             │
│  ├─ 内存布局全局规划                    ❌ 不适用             │
│  └─ 静态执行计划编译                    ❌ 不适用             │
├─────────────────────────────────────────────────────────────┤
│  代码级优化（LLVM）                                          │
│  ├─ 循环优化、内联                      ✅ 有效（opt-level）  │
│  ├─ SIMD 自动向量化                     ✅ 有效              │
│  └─ 寄存器分配、指令调度                 ✅ 有效              │
├─────────────────────────────────────────────────────────────┤
│  并行化优化（Rayon）                                         │
│  ├─ Batch 维度并行                      ✅ 有效              │
│  ├─ 种群个体并行                        ✅ 有效（NEAT 核心）  │
│  └─ 节点内部并行                        ✅ 有效              │
└─────────────────────────────────────────────────────────────┘
```

### Cargo.toml 配置建议

针对"代码级优化"层次，推荐配置：

```toml
# 对本项目代码开启基本优化（循环优化、部分内联）
# 保持一定调试能力，同时显著提升计算密集型操作的速度
[profile.dev]
opt-level = 1

# 对所有依赖开启最大优化
# ndarray、rayon 等库的性能对整体影响很大
[profile.dev.package."*"]
opt-level = 3
```

> **注意**：此配置仅影响本项目开发。如果用户将 `only_torch` 作为依赖使用，
> 需要在**用户自己的项目**中添加相同配置才能获得同样的优化效果。

---

## 优化维度分析

### 传统深度学习 vs Only Torch

```
传统深度学习优化重点：
┌─────────────────────────────────────────┐
│  GPU 并行  ████████████████████  最重要  │
│  BLAS 优化 ██████████████       高       │
│  Batch 大小 ████████████        高       │
│  算子融合  ████████             中       │
│  多线程    ████                 低       │
└─────────────────────────────────────────┘

Only Torch 优化重点：
┌─────────────────────────────────────────┐
│  个体并行  ████████████████████  最重要  │
│  Batch 训练 ████████████        高       │
│  多线程    ████████████         高       │
│  SIMD      ████████             中       │
│  BLAS      ████                 低       │
└─────────────────────────────────────────┘
```

### 三个关键场景的优化策略

| 场景          | 主要瓶颈               | 最佳优化方向         |
| :------------ | :--------------------- | :------------------- |
| NEAT 进化评估 | 评估大量不同拓扑的个体 | **Rayon 个体间并行** |
| 固定结构训练  | 权重更新的梯度计算     | **Batch + SIMD**     |
| 推理部署      | 单样本延迟             | **编译优化 + 缓存**  |

---

## 优化策略详解

### 策略 1：个体间并行（NEAT 进化阶段）

**场景**：评估种群中的数百个不同网络拓扑。

```rust
// NEAT 进化的核心循环
population.par_iter_mut()  // Rayon 并行
    .for_each(|individual| {
        let fitness = evaluate(individual, &test_data);
        individual.set_fitness(fitness);
    });
```

**为什么有效**：

- 每个个体的网络结构不同，无法 batch 化
- 但个体之间**完全独立**，可完美并行
- 利用多核 CPU，收益 = 核心数 ×

**优先级**：⭐⭐⭐⭐⭐（最高）

---

### 策略 2：Batch 样本间向量化（固定结构训练阶段）

**场景**：网络结构确定后，用梯度下降训练权重。

```rust
// 固定结构的 batch 训练
for batch in data_loader.batches(batch_size) {
    let output = model.forward(&batch)?;  // [batch, ...] 维度向量化
    optimizer.step()?;
}
```

**为什么有效**：

即使 NEAT 网络的单个节点只有少量连接，**batch 维度**的多个样本可以同时计算：

```
传统思维（节点内）：
  node_5 = w_1*x_1 + w_2*x_2  ← 只有2次运算

Batch 思维（样本间）：
  node_5[batch] = w_1*x_1[batch] + w_2*x_2[batch]
                  └── batch 个样本同时计算，可向量化
```

**收益估计**（NEAT 小网络）：3-10x

**优先级**：⭐⭐⭐⭐（高）

---

### 策略 3：SIMD 自动向量化（底层计算）

**场景**：Tensor 层的基础运算。

**当前状态**：

```toml
# Cargo.toml - 当前配置
ndarray = {version="^0.15", features=["serde"]}
# 未启用 BLAS
```

**优化路径**：

| 级别 | 配置                  | 收益  | 复杂度 |
| :--- | :-------------------- | :---- | :----- |
| 基础 | LLVM 自动优化（当前） | 1-3x  | 零     |
| 中级 | 启用 OpenBLAS         | 3-10x | 低     |
| 高级 | Intel MKL（仅 x86）   | 5-20x | 中     |

**建议**：

- MVP 阶段保持现状（LLVM 自动优化）
- 性能调优阶段按需添加 BLAS
- 提供 feature flag 让用户选择

```toml
# 未来可选配置
[features]
default = []
blas-openblas = ["ndarray/blas", "blas-src/openblas"]
blas-mkl = ["ndarray/blas", "blas-src/intel-mkl"]
```

**优先级**：⭐⭐⭐（中）—— 对小矩阵收益有限

---

### 策略 4：内存布局优化

**场景**：NEAT 网络的节点访问模式。

**问题**：进化网络的连接是稀疏且不规则的，导致缓存命中率低。

**优化方向**：

1. **拓扑排序执行**：按依赖顺序计算节点，减少随机访问
2. **节点值连续存储**：所有节点的值放在连续内存中
3. **批量读取连接权重**：同一节点的所有输入权重连续存放

```rust
// 优化前：每个节点单独存储
struct Node {
    value: Tensor,
    weights: Vec<f32>,
}

// 优化后：所有节点值连续存储
struct OptimizedNetwork {
    all_values: Tensor,      // [num_nodes, batch_size]
    all_weights: Vec<f32>,   // 连续存放所有权重
    connection_indices: Vec<(usize, usize)>,  // 连接关系
}
```

**优先级**：⭐⭐（低）—— 后期优化，当前可忽略

---

### 策略 5：编译期优化（远期）

**场景**：网络结构固定后的极致优化。

**思路**：当网络结构确定（进入部署或长期训练），可以将计算图"编译"为优化的执行代码。

```rust
// 概念性代码
let compiled_model = model.compile()?;  // 生成优化的执行路径
for batch in data {
    let output = compiled_model.forward_optimized(&batch)?;
}
```

**可能的优化**：

- 预计算拓扑排序
- 融合连续的线性操作
- 消除中间变量
- 预分配内存

**优先级**：⭐（最低）—— 远期目标

---

## 优化优先级路线图

### MVP 阶段（当前）

| 优化           |   状态    | 说明                                                        |
| :------------- | :-------: | :---------------------------------------------------------- |
| Rayon 个体并行 | 🔲 待实现 | NEAT 进化的核心                                             |
| Batch 训练     | 🔲 待实现 | 详见 [batch_mechanism_design.md](batch_mechanism_design.md) |
| LLVM 自动优化  |  ✅ 已有  | release 模式自动                                            |

### 性能调优阶段

| 优化          | 触发条件   | 说明              |
| :------------ | :--------- | :---------------- |
| BLAS 可选支持 | 性能瓶颈   | 添加 feature flag |
| 内存布局优化  | 大规模网络 | 连续存储节点值    |

### 远期阶段

| 优化        | 触发条件 | 说明           |
| :---------- | :------- | :------------- |
| 编译期优化  | 部署需求 | 固定结构后编译 |
| 自定义 SIMD | 极致性能 | 手写关键路径   |

---

## NEAT 网络的向量化收益分布

```
NEAT 网络各部分的向量化潜力：

输入处理:     ████████████  高（batch 输入）
中间节点:     ████          中等（每个节点的 batch 计算）
权重计算:     ██            低（稀疏连接，小矩阵）
输出处理:     ████████████  高（batch 输出）
激活函数:     ████████      高（逐元素，易向量化）
```

**核心认知**：

1. NEAT 网络的向量化收益**主要来自 Batch 维度**
2. 层内/节点内的向量化收益**远低于传统网络**
3. **个体间并行**是 NEAT 场景的最大优化点

---

## 总结

### Only Torch 的优化哲学

```
不追求单个操作的极致性能，
而是追求整体工作流的高效：

  NEAT 进化: 个体并行 >> Batch >> SIMD
  权重训练: Batch >> SIMD >> 个体并行
  推理部署: 编译优化 >> 缓存 >> SIMD
```

### 与传统框架的差异

| 维度     | PyTorch/TensorFlow | Only Torch       |
| :------- | :----------------- | :--------------- |
| 核心假设 | 大规模规则网络     | 小规模不规则网络 |
| 主要并行 | GPU CUDA           | CPU Rayon        |
| 矩阵规模 | 大（数百到数千）   | 小（几个到几十） |
| 结构特点 | 固定层叠加         | 动态进化拓扑     |
| 优化重点 | BLAS/cuBLAS        | 个体并行 + Batch |

### 行动原则

1. **正确性优先**：先保证功能正确，再考虑性能
2. **测量驱动**：用 benchmark 证明瓶颈，而非猜测
3. **渐进优化**：从高收益低复杂度的优化开始
4. **保持简单**：不为假设的未来需求过度工程化

---

## 附录：CPU 内核优化技术参考

> 来源：PyTorch ATen CPU 内核 + Intel oneDNN + Intel CPU 训练论文
> 详细论文笔记见 [Intel CPU Training 2022](../paper/Intel_CPU_Training_2022/summary.md)

### 1. 并行化技术

**PyTorch 实现** (`at::parallel_for`)：

```cpp
// PyTorch: AvgPoolKernel.cpp
at::parallel_for(0, channels, 0, [&](int64_t begin, int64_t end) {
    for (const auto c : c10::irange(begin, end)) {
        // 每个线程处理一部分 channels
    }
});
```

**Rust 对应方案**：

| PyTorch                                   | Rust 等价                                  | 说明                   |
| ----------------------------------------- | ------------------------------------------ | ---------------------- |
| `at::parallel_for`                        | [`rayon::par_iter`](https://docs.rs/rayon) | 数据并行，自动负载均衡 |
| `at::parallel_for(0, n, grain_size, ...)` | `rayon::iter::with_min_len()`              | 控制最小分块大小       |

```rust
// Rust 等价实现
use rayon::prelude::*;

// 基础用法
(0..channels).into_par_iter().for_each(|c| {
    // 每个线程处理一部分 channels
});

// 带最小分块大小（类似 grain_size）
(0..channels)
    .into_par_iter()
    .with_min_len(64)  // 每个任务至少处理 64 个元素
    .for_each(|c| { /* ... */ });
```

---

### 2. SIMD 向量化技术

**PyTorch 实现** (`vec::Vectorized`)：

```cpp
// PyTorch: MaxPoolKernel.cpp
using Vec = vec::Vectorized<scalar_t>;

int64_t d = 0;
for (; d < len; d += Vec::size()) {
    Vec val_vec = Vec::loadu(in + d);        // SIMD 加载
    Vec max_vec = Vec::loadu(out + d);
    Vec result = Vec::blendv(max_vec, val_vec, val_vec > max_vec);
    result.store(out + d);                    // SIMD 存储
}
// 处理尾部（不足一个 SIMD 宽度）
for (; d < size; d++) {
    out[d] = std::max(out[d], in[d]);
}
```

**Rust 对应方案**：

| PyTorch              | Rust 等价                                                    | 说明                  |
| -------------------- | ------------------------------------------------------------ | --------------------- |
| `vec::Vectorized<T>` | [`std::simd`](https://doc.rust-lang.org/std/simd/) (nightly) | 标准库 SIMD（实验性） |
|                      | [`wide`](https://docs.rs/wide) crate                         | 稳定版跨平台 SIMD     |
|                      | [`packed_simd`](https://docs.rs/packed_simd)                 | 更底层的 SIMD 控制    |
|                      | [`pulp`](https://docs.rs/pulp)                               | 自动 SIMD 分发        |

```rust
// 方案 1: wide crate（推荐，稳定版可用）
use wide::f32x8;

let mut d = 0;
while d + 8 <= len {
    let val = f32x8::from(&input[d..d+8]);
    let max = f32x8::from(&output[d..d+8]);
    let result = val.max(max);
    result.store(&mut output[d..d+8]);
    d += 8;
}
// 尾部标量处理
for i in d..len {
    output[i] = output[i].max(input[i]);
}

// 方案 2: std::simd（nightly，未来标准）
#![feature(portable_simd)]
use std::simd::{f32x8, SimdFloat};

let val = f32x8::from_slice(&input[d..]);
let max = f32x8::from_slice(&output[d..]);
let result = val.simd_max(max);
```

**常用 SIMD 操作对照表**：

| 操作     | PyTorch `vec::Vectorized` | Rust `wide` / `std::simd` |
| -------- | ------------------------- | ------------------------- |
| 加载     | `Vec::loadu(ptr)`         | `f32x8::from(slice)`      |
| 存储     | `vec.store(ptr)`          | `vec.store(slice)`        |
| 加法     | `a + b`                   | `a + b`                   |
| 乘法     | `a * b`                   | `a * b`                   |
| 最大值   | `Vec::max(a, b)`          | `a.max(b)`                |
| 条件选择 | `Vec::blendv(a, b, mask)` | `mask.select(b, a)`       |
| 水平求和 | `vec.reduce_add()`        | `vec.reduce_add()`        |

---

### 3. 内存布局优化

**PyTorch 策略**：

```cpp
// PyTorch 支持多种内存布局
auto input = input_.contiguous();                      // NCHW (默认)
auto input = input_.contiguous(MemoryFormat::ChannelsLast);  // NHWC

// Channels Last 对 SIMD 更友好（连续访问 channel 维度）
```

**Rust 对应方案**：

```rust
// ndarray 支持不同内存布局
use ndarray::{Array4, Axis};

// C 顺序（NCHW，行优先）—— 默认
let tensor = Array4::<f32>::zeros((batch, channels, height, width));

// Fortran 顺序（列优先）
let tensor = Array4::<f32>::zeros((batch, channels, height, width).f());

// 转换布局
let contiguous = tensor.as_standard_layout().to_owned();
```

---

### 4. 数据类型优化

**PyTorch 实现**：

```cpp
// PyTorch 对 BFloat16/Half 使用 float 累加，避免精度损失
using opmath_t = at::opmath_type<scalar_t>;  // scalar_t=bf16 → opmath_t=f32

opmath_t sum = 0;
for (...) {
    sum += opmath_t(input[i]);  // 累加时提升精度
}
output[i] = scalar_t(sum / count);  // 输出时降回原精度
```

**Rust 对应方案**：

```rust
// 使用 half crate 处理半精度
use half::{bf16, f16};

// 累加时使用 f32
let sum: f32 = input.iter()
    .map(|&x| f32::from(x))  // bf16 → f32
    .sum();
let avg = bf16::from_f32(sum / count as f32);  // f32 → bf16
```

---

### 5. 索引优化技术

**PyTorch 实现**（避免多维索引计算）：

```cpp
// PyTorch: 预计算偏移量，避免重复索引计算
int64_t index = id * input_height * input_width + ih * input_width + iw;
const scalar_t* in = input_data + n * input_depth * input_height * input_width;
```

**Rust 对应方案**：

```rust
// 使用 unsafe 指针运算（性能敏感路径）
let base_offset = n * depth * height * width;
let idx = base_offset + d * height * width + h * width + w;

// 或使用 ndarray 的高效索引
use ndarray::s;
let slice = tensor.slice(s![n, .., h0..h1, w0..w1]);
```

---

### 6. 尾部处理模式

**通用模式**（PyTorch 和 Rust 通用）：

```rust
// SIMD 宽度对齐 + 尾部标量处理
let simd_width = 8;  // 如 f32x8
let aligned_len = (len / simd_width) * simd_width;

// SIMD 处理对齐部分
for i in (0..aligned_len).step_by(simd_width) {
    // SIMD 操作
}

// 标量处理尾部
for i in aligned_len..len {
    // 标量操作
}
```

---

### 相关 Rust Crate 汇总

| 用途        | Crate                  | 说明                  |
| ----------- | ---------------------- | --------------------- |
| 并行迭代    | `rayon`                | 数据并行，类似 OpenMP |
| SIMD (稳定) | `wide`                 | 跨平台 SIMD 抽象      |
| SIMD (底层) | `packed_simd`          | 更细粒度控制          |
| SIMD (实验) | `std::simd`            | 未来标准（nightly）   |
| 半精度浮点  | `half`                 | f16/bf16 支持         |
| BLAS        | `ndarray` + `blas-src` | 线性代数加速          |
| 内存对齐    | `aligned`              | 对齐内存分配          |

---

### 7. oneDNN CPU 内核优化参考

> 来源：[oneDNN](https://github.com/uxlfoundation/oneDNN)（C++，Apache 2.0）
> oneDNN 是 Intel 开源的深度学习原语库，其 `src/cpu/` 目录包含所有 CPU 优化实现。

#### 关键源码目录对照

| oneDNN 路径 | 内容 | 对应 only_torch 模块 |
|-------------|------|---------------------|
| `src/cpu/gemm/` | GEMM（矩阵乘法）分块实现 | `Tensor` 矩阵运算 |
| `src/cpu/x64/jit_*_conv*` | 卷积 kernel（direct/Winograd/im2col） | `Conv2d` 节点 |
| `src/cpu/x64/jit_uni_batch_normalization.*` | BatchNorm SIMD 实现 | `BatchNorm` 节点 |
| `src/cpu/x64/jit_uni_eltwise.*` | 激活函数 SIMD 实现 | 激活函数节点 |
| `src/cpu/x64/jit_uni_pool*` | 池化 SIMD 实现 | `MaxPool2d` 节点 |
| `src/common/memory_desc.hpp` | 内存格式定义（nchw/nhwc/blocked） | Tensor 内存布局 |

#### Cache Tiling（缓存分块）

将大矩阵分块为适合 L1/L2 cache 的小块，逐块计算以最大化缓存命中率。

```rust
// 概念性示例：GEMM 分块
// C[M,N] = A[M,K] * B[K,N]
let tile_m = 64;  // 适合 L1 cache
let tile_n = 64;
let tile_k = 256; // 适合 L2 cache

for m in (0..M).step_by(tile_m) {
    for n in (0..N).step_by(tile_n) {
        for k in (0..K).step_by(tile_k) {
            // 在 tile 内计算，数据全在 cache 中
            micro_kernel(&a[m..m+tile_m, k..k+tile_k],
                        &b[k..k+tile_k, n..n+tile_n],
                        &mut c[m..m+tile_m, n..n+tile_n]);
        }
    }
}
```

#### Blocked Memory Format（分块内存格式）

oneDNN 使用 `nChw16c` 等 blocked 格式，让 16 个 channel 连续存放，对 AVX-512（16 个 f32 lane）天然友好。

```
NCHW（默认）：
  data[n][c][h][w]  →  同一 channel 的所有空间位置连续
                       SIMD 处理 channel 维度时需要跳跃访问

nChw16c（blocked）：
  data[n][C/16][h][w][16]  →  16 个 channel 连续存放
                               SIMD 一次加载 16 个 channel 值
```

**对 only_torch 的意义**：当前我们使用 NCHW 格式。当性能成为瓶颈时，可考虑在 Conv2d 内部将输入转为 blocked 格式，计算后再转回——这正是 oneDNN 的策略。

#### 训练特有优化（来自 Intel 论文）

| 优化 | 说明 | 适用时机 |
|------|------|----------|
| **层融合（训练模式）** | Conv + FrozenBN + ReLU 可融合；含可训练权重的层需保留中间值 | 固定结构训练 |
| **优化器融合** | 将参数遍历+梯度应用合并为单次操作，减少内存遍历 | Optimizer 实现 |
| **in-place 反向传播** | 反向传播中尽量原地修改梯度 Tensor，避免分配临时变量 | 已在 `optimization_candidates.md` 中有相关讨论 |
| **BF16 混合精度** | 计算用 BF16（速度翻倍），loss 保持 f32（避免精度崩溃） | 远期特性 |

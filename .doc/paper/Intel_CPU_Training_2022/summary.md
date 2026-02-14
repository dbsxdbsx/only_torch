# Intel CPU 训练优化论文笔记

> **主题**：在 Intel CPU 上高效训练深度学习模型的方法论与工程实践
> **年份**：2022
> **核心参考仓库**：[oneDNN](https://github.com/uxlfoundation/oneDNN)（`src/cpu/` 目录）

---

## 论文一：Deep Learning Models on CPUs — A Methodology for Efficient Training

> **arXiv**：[2206.10034](https://arxiv.org/abs/2206.10034)
> **本地 PDF**：[deep_learning_on_cpus_2206.10034.pdf](./deep_learning_on_cpus_2206.10034.pdf)
> **作者**：Quchen Fu (Vanderbilt Univ.), Ramesh Chukka 等 (Intel)
> **成果**：RetinaNet-ResNext50 训练获得整体 2x 加速

### 核心贡献

提出 CPU 训练的系统化优化方法论（Profile → Trace → Project → Optimize），并开发 ProfileDNN 工具定位瓶颈。

### 关键优化技术

| 技术 | 做法 | 收益 |
|------|------|------|
| **瓶颈定位** | 用 ProfileDNN 可视化各 primitive 操作（conv/matmul/bnorm 等）的耗时占比 | 精准找到优化目标 |
| **内存布局** | NHWC（channel last）比 NCHW 对 CPU SIMD 更友好 | 训练效率提升 |
| **算子替换** | 用 IPEX 的 FrozenBatchNorm2d 替换 torchvision 版本 | 29.8% 提升 |
| **低精度训练** | BF16 训练（与 f32 同范围但 7 位小数），计算时间减半 | ~2x（但 loss 计算需保持 f32） |
| **层融合** | conv + FrozenBatchNorm + ReLU 融合为单一 kernel | ~30% 提升 |
| **优化器融合** | 将 SGD/Lamb 参数更新融合为单一操作 | 参数更新 5.5x 加速 |
| **自定义 kernel** | 手写 focal loss 前向/反向 kernel，大量使用 in-place 操作 | 前向 2.6x、反向 1.3x |

### 关键发现

1. **反向传播 > 前向传播**：CNN/RNN 模型的反向传播耗时大于前向，应优先优化反向
2. **loss 计算不可忽视**：当类别数增多（如 OpenImage 600+ 类），focal loss 耗时可达总反向传播的 1/3
3. **BF16 的陷阱**：loss 中的 reduction 操作在 BF16 下精度损失严重，应保持 f32
4. **in-place 操作**：自定义 kernel 中尽量使用 in-place 操作，减少临时 Tensor 分配

### 对 only_torch 的启示

| 启示 | 具体行动 |
|------|----------|
| **瓶颈定位方法论** | benchmark 时分别测量各 primitive 操作，而非只看整体 |
| **内存布局** | Conv2d 实现考虑 NHWC 支持（参考 oneDNN 的 blocked format） |
| **in-place 优化** | 反向传播中减少临时 Tensor 分配（已在 `optimization_candidates.md` 中记录类似问题） |
| **层融合思路** | 将来固定结构训练时，可融合 Conv + BN + Activation |
| **低精度训练** | 远期考虑 BF16 支持，但 loss 计算需保持高精度 |

---

## 论文二：Strategies for Optimizing E2E AI Pipelines on Intel Xeon Processors

> **arXiv**：[2211.00286](https://arxiv.org/abs/2211.00286)
> **本地 PDF**：[e2e_ai_pipeline_optimization_2211.00286.pdf](./e2e_ai_pipeline_optimization_2211.00286.pdf)
> **作者**：Meena Arunachalam 等 (Intel)
> **成果**：8 个 E2E 管线获得 1.8x ~ 81.7x 提升

### 核心贡献

强调端到端管线优化——不能只优化模型本身，数据预处理/后处理可占总时间的 4% ~ 98%。

### 优化层次

```
┌──────────────────────────────────────────────┐
│  应用层     数据格式、批处理策略、管线编排     │
├──────────────────────────────────────────────┤
│  框架层     oneDNN 集成、IPEX、多实例并行     │
├──────────────────────────────────────────────┤
│  库层       Intel MKL、Modin、TBB             │
├──────────────────────────────────────────────┤
│  系统层     NUMA 绑定、内存策略、核心分配     │
├──────────────────────────────────────────────┤
│  硬件层     AVX-512、DL Boost (VNNI)、AMX    │
└──────────────────────────────────────────────┘
```

### 关键优化策略

| 策略 | 说明 |
|------|------|
| **多实例并行** | 在多核 Xeon 上同时运行多个训练/推理实例，比单实例充分利用核心 |
| **数据管线并行** | 预处理与训练重叠执行，避免 GPU 式的"等数据"瓶颈 |
| **NUMA 感知** | 将工作负载绑定到特定 NUMA 节点，减少跨节点内存访问 |
| **量化压缩** | INT8 推理（Neural Compressor），训练保持更高精度 |

### 对 only_torch 的启示

| 启示 | 具体行动 |
|------|----------|
| **DataLoader 不能拖后腿** | DataLoader 设计需支持多线程预加载、prefetch（参考 `data_loader_design.md`） |
| **NUMA 感知** | Rayon 线程池未来可考虑 NUMA 亲和性配置 |
| **多实例并行** | NEAT 种群评估天然适合多实例模式——每个个体一个实例 |

---

## 核心参考仓库

### oneDNN（uxlfoundation/oneDNN）

> CPU 上深度学习原语的参考实现，对我们实现高效 kernel 最有价值。

**关键目录对照**：

| oneDNN 路径 | 内容 | 对应 only_torch 模块 |
|-------------|------|---------------------|
| `src/cpu/gemm/` | GEMM 实现（cache tiling、分块策略） | `Tensor` 矩阵乘法 |
| `src/cpu/x64/jit_*_conv*` | 卷积 kernel（direct/Winograd/im2col） | `Conv2d` 节点 |
| `src/cpu/x64/jit_avx*.hpp` | AVX-512/AVX2 SIMD 原语封装 | 底层向量化参考 |
| `src/common/memory_desc.hpp` | 内存格式定义（nchw/nhwc/blocked） | Tensor 内存布局 |
| `src/cpu/x64/jit_uni_batch_normalization.*` | BatchNorm 优化实现 | BatchNorm 节点 |
| `src/cpu/x64/jit_uni_eltwise.*` | 激活函数 SIMD 实现 | 激活函数节点 |
| `src/cpu/x64/jit_uni_pool*` | 池化 SIMD 实现 | Pooling 节点 |

**关键优化思路**：

1. **Cache tiling**：将大矩阵分块为适合 L1/L2 cache 的小块，逐块计算
2. **Blocked memory format**：nChw16c 格式让 16 个 channel 连续存放，对 SIMD 友好
3. **JIT 代码生成**：运行时根据具体问题尺寸生成最优汇编（我们暂不需要，但思路可借鉴）

### Intel Neural Compressor（intel/neural-compressor）

> 主攻推理端量化/剪枝/蒸馏。与我们的训练优化关系不大，但其中的 QAT（量化感知训练）策略可作远期参考。

---

## 引用格式

```bibtex
@article{fu2022deep,
  title={Deep Learning Models on CPUs: A Methodology for Efficient Training},
  author={Fu, Quchen and Chukka, Ramesh and Achorn, Keith and Atta-fosu, Thomas and Canchi, Deepak R and Teng, Zhongwei and White, Jules and Schmidt, Douglas C},
  journal={arXiv preprint arXiv:2206.10034},
  year={2022}
}

@article{arunachalam2022strategies,
  title={Strategies for Optimizing End-to-End Artificial Intelligence Pipelines on Intel Xeon Processors},
  author={Arunachalam, Meena and Sanghavi, Vrushabh and Yao, Yi A and Zhou, Yi A and Wang, Lifeng A and Wen, Zongru and Ammbashankar, Niroop and Wang, Ning W and Mohammad, Fahim},
  journal={arXiv preprint arXiv:2211.00286},
  year={2022}
}
```

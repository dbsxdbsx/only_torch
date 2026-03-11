# 神经架构演化（Neural Architecture Evolution）

> **设计理念**：用户只提供数据和目标，系统从最小网络出发，通过层级变异 + 梯度训练的混合策略自动发现最优架构。
>
> **核心原则**：进化负责结构搜索，梯度负责权重优化——各取所长（EXAMM 策略）。

---

## 1. 快速上手

### 1.1 最简用法（零模型代码）

```rust
use only_torch::nn::evolution::{Evolution, gene::TaskMetric};
use only_torch::tensor::Tensor;

// 准备数据
let train = (inputs, labels);  // (Vec<Tensor>, Vec<Tensor>)
let test = train.clone();

// 一行启动演化
let result = Evolution::supervised(train, test, TaskMetric::Accuracy)
    .with_target_metric(0.95)
    .with_seed(42)
    .run()
    .expect("演化出错");

println!("架构: {}", result.architecture());
println!("准确率: {:.1}%", result.fitness.primary * 100.0);

// 推理：一行
let predictions = result.predict(&new_data)?;

// 可视化：一行
result.visualize("output/my_model")?;
```

### 1.1.1 序列 / 记忆单元

当每个样本是 2D 张量 `[seq_len, input_dim]` 时，系统自动检测为序列模式，从最小 RNN 结构出发演化：

```rust
use only_torch::nn::evolution::{Evolution, gene::TaskMetric};
use only_torch::tensor::Tensor;

// 每个样本 [seq_len, 1] 的二进制序列
let train_data = generate_parity_data(200, 8, 42);
let test_data = generate_parity_data(50, 8, 99);

let result = Evolution::supervised(train_data, test_data, TaskMetric::Accuracy)
    .with_target_metric(0.90)
    .with_seed(42)
    .run()?;
```

无需手动指定序列模式——数据形状决定一切。系统自动选择记忆单元（RNN/LSTM/GRU）并演化网络拓扑。支持变长序列（自动零填充至最大长度）。

### 1.1.2 CNN / 图像分类

当每个样本是 3D 张量 `[C, H, W]` 时，系统自动检测为空间模式，从最小 CNN 结构出发演化：

```rust
use only_torch::data::MnistDataset;
use only_torch::nn::evolution::{Evolution, gene::TaskMetric};

let train_data = collect_samples(&train_set, 1000); // 每个样本 [1, 28, 28]
let test_data = collect_samples(&test_set, 500);

let result = Evolution::supervised(train_data, test_data, TaskMetric::Accuracy)
    .with_target_metric(0.95)
    .with_seed(42)
    .run()?;
```

无需手动指定 CNN 模式——数据形状决定一切。

### 1.2 可调参数

```rust
Evolution::supervised(train, test, TaskMetric::Accuracy)
    .with_target_metric(0.95)         // 目标指标值（默认 1.0）
    .with_seed(42)                    // 随机种子（可复现）
    .with_max_generations(200)        // 最大代数（默认 100）
    .with_batch_size(64)              // 显式 batch size（默认自动策略）
    .with_verbose(false)              // 关闭日志（默认 true）
    .with_convergence(config)         // 收敛检测配置
    .with_constraints(constraints)    // 网络规模约束
    .with_stagnation_patience(30)     // 停滞检测耐心值（默认 20）
    .with_eval_runs(3)                // 多次评估取保守值（默认 1）
    .with_mutation_registry(registry) // 自定义变异注册表
    .with_callback(my_callback)       // 自定义回调
    .run()
```

### 1.3 支持的任务指标

| TaskMetric | 自动推断的 Loss | 适用场景 |
|---|---|---|
| `Accuracy` | BCE (output_dim=1) / CrossEntropy (output_dim>1) | 分类 |
| `R2` | MSE | 回归 |
| `MultiLabelAccuracy` | BCE | 多标签分类 |

### 1.4 Batch Size 自动策略

用户无需关心 batch size——`SupervisedTask` 根据数据量自动选择：

| 数据量 | 策略 | 典型场景 |
|---|---|---|
| ≤ 128 | full-batch | XOR 等小任务 |
| 129–10000 | batch_size = 64 | Iris 等中等数据集 |
| > 10000 | batch_size = 256 | 大规模数据集 |

可通过 `.with_batch_size()` 覆盖。

---

## 2. 演化主循环

### 2.1 Genome-Centric 架构

系统采用 **Genome-Centric** 路线：每代从基因组（`NetworkGenome`）重建计算图，而非直接修改图。

```
每一代 (generation):

  1. build()            从 Genome 构建计算图 → BuildResult
  2. restore_weights()  从 weight_snapshots 恢复权重（Lamarckian 继承）
  3. train()            梯度训练直到收敛 → loss
  4. capture_weights()  将训练后的权重保存回 Genome
  5. evaluate()         计算 fitness（如 Accuracy）
  6. 接受 / 回滚        fitness >= best → 接受；否则回滚到 best_genome
  7. 变异               修改 Genome 拓扑或参数 → 下一代
```

### 2.2 信号职责分离

| 信号 | 产生者 | 消费者 | 语义 |
|---|---|---|---|
| **loss** | `train()` 内部 | `ConvergenceDetector` | 当前架构训练够了吗？（架构内信号） |
| **grad_norm** | `train()` 内部 | `ConvergenceDetector` | 辅助判断收敛 |
| **fitness** | `evaluate()` | `Evolution::run()` | 这个架构好不好？（架构间信号） |

收敛 ≠ 达标：收敛只说明"当前拓扑已榨干"，需要变异换结构继续尝试。

### 2.3 Seed 传播

单一 `StdRng` 贯穿所有随机操作，相同 seed 产生完全相同的演化轨迹：

```
Evolution.seed → StdRng
    ├─ build(rng)        → 新层参数初始化
    ├─ train(rng)        → Dropout mask、mini-batch 顺序
    ├─ evaluate(rng)     → 评估时随机性
    └─ apply_random(rng) → 变异选择和执行
```

### 2.4 停止条件与返回值

`run()` 返回 `Result<EvolutionResult, EvolutionError>`。演化搜索未达标不是错误，通过 `status` 标识停止原因：

| EvolutionStatus | 含义 |
|---|---|
| `TargetReached` | primary ≥ target_metric，达标 |
| `MaxGenerations` | 到达最大代数限制 |
| `CallbackStopped` | 自定义回调请求停止 |
| `NoApplicableMutation` | 所有变异均不可用（搜索空间耗尽） |

`Err(EvolutionError)` 用于数据验证失败（`InvalidData`）或系统错误（`Graph`）。数据验证延迟到 `run()` 执行时，`supervised()` 构造器本身不会失败。

---

## 3. 接受策略与回滚

### 3.1 非严格接受（允许中性漂移）

```
fitness >= best → 接受（含中性漂移）
fitness <  best → 回滚到 best_genome
```

**为什么用 `>=` 而非 `>`？** 许多有价值的结构变化需要多步积累才能提升 fitness（"stepping stone" 问题）。以 XOR 为例，从 `Linear(1)` 到能解 XOR 至少需要两步变异（GrowHidden + InsertActivation），但每步单独都不涨分。`>=` 允许 fitness 不变的变异留下来，为后续突破性变异创造条件。

### 3.2 离散指标的 Tiebreaker

`FitnessScore` 采用字典序比较，`primary` 保持纯粹的指标值，不融合任何 tiebreak 信息：

```rust
pub struct FitnessScore {
    pub primary: f32,             // 主目标（越高越好）
    pub inference_cost: Option<f32>, // 预留：推理成本
    pub tiebreak_loss: Option<f32>,  // 离散指标：test loss（越低越好）
}
```

比较规则：
1. `primary` 更高 → 接受
2. `primary` 更低 → 拒绝
3. `primary` 相等 → 比 `tiebreak_loss`（都有则越低越好，否则直接接受）

连续指标（R²）的 `tiebreak_loss` 为 `None`，天然进入规则 3 的"直接接受"分支。

### 3.3 停滞检测

连续 `stagnation_patience`（默认 20）代 primary 未严格提升后，强制从结构性变异（`InsertLayer` / `RemoveLayer` / `AddSkipEdge` / `RemoveSkipEdge`）中选择，打破参数变异空转。

---

## 4. 基因数据结构

### 4.1 NetworkGenome

```rust
pub struct NetworkGenome {
    pub layers: Vec<LayerGene>,          // 层列表（最后一层 = 输出头，受保护）
    pub skip_edges: Vec<SkipEdge>,       // 跳跃连接列表
    pub input_dim: usize,
    pub output_dim: usize,
    pub seq_len: Option<usize>,              // None=平坦, Some(n)=序列（每时间步 input_dim 维）
    pub input_spatial: Option<(usize, usize)>, // None=非空间, Some((H,W))=空间
    pub training_config: TrainingConfig, // lr / optimizer / loss 等
    pub generated_by: String,            // 变异来源（调试用）
    weight_snapshots: HashMap<u64, Vec<Tensor>>,  // Lamarckian 继承
}
```
**最小初始网络**（按数据形状自动选择）：
- 平坦数据 `[D]`：`Input(D) → [Linear(output_dim)]`
- 序列数据 `[T, D]`：`Input(seq×D) → Rnn(output_dim) → [Linear(output_dim)]`
- 空间数据 `[C, H, W]`：`Input(C@H×W) → Flatten → [Linear(out_dim)]`（Conv2d/Pool2d 由演化自主插入）

序列初始网络以最简单的 Rnn 为起点（而非 LSTM/GRU），后续 `MutateCellType` 可升级记忆单元类型，`InsertLayer` 可在序列域插入更多循环层。

**输出头保护**：

**权重快照自包含**：`clone()` 时权重一并复制，回滚到 `best_genome` 自带权重，无需保留旧 Graph。

### 4.2 LayerConfig

```rust
pub enum LayerConfig {
    Linear { out_features: usize },
    Activation { activation_type: ActivationType },
    Rnn { hidden_size: usize },
    Lstm { hidden_size: usize },
    Gru { hidden_size: usize },
    Dropout { p: f32 },
    Conv2d { out_channels: usize, kernel_size: usize }, // stride=1, same padding
    Pool2d { pool_type: PoolType, kernel_size: usize, stride: usize },
    Flatten, // Spatial(C,H,W) → Flat(C*H*W)
}
```

只存输出侧参数，输入维度由 `resolve_dimensions()` 自动推导。循环层（Rnn / Lstm / Gru）仅在 `seq_len` 非 None 时使用，空间层（Conv2d / Pool2d / Flatten）仅在 `input_spatial` 非 None 时使用。

### 4.3 ShapeDomain 域系统

`ShapeDomain` 描述张量在网络中的维度语义，用于验证层链合法性和约束 skip edge 范围：

```rust
pub enum ShapeDomain {
    Flat,      // 2D [batch, features]
    Sequence,  // 3D [batch, seq_len, features]
    Spatial,   // 4D [batch, channels, H, W]
}
```

**序列模式域链规则**（`is_domain_valid()`）：
- `Sequence → Sequence`：循环层输出 `return_sequences=true`（下一个实质层也是循环层时自动启用）
- `Sequence → Flat`：循环层输出 `return_sequences=false`（仅返回最后一个时间步的隐藏状态）
- `Flat → Flat`：Linear、Activation 等平坦层
- `Flat → Sequence`：**非法**（不允许回溯至序列域）
- 终态必须为 `Flat`（输出头需要 2D 输入）

**空间模式域链规则**：`Spatial* → Flatten → Flat*`（详见 11.5 节）。

`compute_domain_map()` 为每个层节点映射到其输出形状域，供 skip edge 约束和变异操作的域感知逻辑使用。

### 4.4 SkipEdge 与聚合

跳跃连接不作为独立层存在于 `layers` 中。`SkipEdge` 携带聚合策略，`build()` 时自动在目标层输入处生成聚合操作：

```rust
pub struct SkipEdge {
    pub innovation_number: u64,
    pub from_innovation: u64,   // 源层
    pub to_innovation: u64,     // 目标层（聚合发生在此层输入处）
    pub strategy: AggregateStrategy,
    pub enabled: bool,
}

pub enum AggregateStrategy {
    Add,                  // ResNet 风格，要求维度相同
    Concat { dim: i32 },  // DenseNet 风格，允许不同维度
    Mean,                 // 要求维度相同
    Max,                  // 要求维度相同
}
```

示例：

```
变异前：  Input → Linear(4) → ReLU → [Linear(1)]

AddSkipEdge(Input → 输出头, strategy=Add) 后：

          Input ─────────────────────────┐
            │                            │
            └─> Linear(4) → ReLU ──(+)─> [Linear(1)]
                                    ↑
                          build 时自动生成 Add 聚合
```

### 4.4 TrainingConfig

```rust
pub struct TrainingConfig {
    pub optimizer_type: OptimizerType,  // Adam（默认）/ SGD
    pub learning_rate: f32,             // 默认 0.01
    pub batch_size: Option<usize>,      // None = 自动策略
    pub weight_decay: f32,              // 默认 0.0（尚未支持非零值）
    pub loss_override: Option<LossType>, // None = 自动推断，Some = 显式指定
}
```

`effective_loss()` 优先使用 `loss_override`，否则按 TaskMetric + output_dim 自动推断。

### 4.5 维度推导

`resolve_dimensions()` 从 `input_dim` 出发沿 `layers` 顺序遍历，为每层计算 `(in_dim, out_dim)`。遇到 skip edge 目标层时，按聚合策略计算有效输入维度（Add/Mean/Max 要求维度相同，Concat 求和）。

空间模式下，维度推导同时维护 `current_spatial: Option<(H, W)>`：
- Conv2d (same padding, stride=1)：`(H, W)` 不变，`out_dim = out_channels`
- Pool2d (kernel_size=k, stride=s)：`(H/s, W/s)`，channels 不变
- Flatten：`out_dim = channels × H × W`，空间维度归 None
`compute_spatial_map()` 为每个层节点（含 INPUT）映射到其输出空间尺寸，Flatten 之后为 None。

`total_params()`
`total_params()` 和 `build()` 共享同一套维度推导逻辑，避免重复实现。变异操作的 `is_applicable()` 通过试探式调用 `resolve_dimensions()` 统一检测维度合法性。

---

## 5. 变异操作

### 5.1 Mutation trait + MutationRegistry

```rust
pub trait Mutation: Send + Sync {
    fn name(&self) -> &str;
    fn apply(&self, genome: &mut NetworkGenome, constraints: &SizeConstraints, rng: &mut StdRng)
        -> Result<(), MutationError>;
    fn is_applicable(&self, genome: &NetworkGenome, constraints: &SizeConstraints) -> bool;
    fn is_structural(&self) -> bool { false }
}
```

`MutationRegistry` 按权重随机选择可用变异并执行。`apply` 失败时自动排除并重试，直到成功或所有候选耗尽。

### 5.2 默认注册表（12 + 条件变异）

基础 12 种变异始终注册，`MutateCellType`（序列模式）和 `MutateKernelSize`（空间模式）按数据形状条件注册。

**结构/参数变异（7 + 条件）**：

| 变异 | 权重 | 结构性 | 核心逻辑 |
|---|---|---|---|
| `InsertLayer` | 0.15 | ✅ | 域感知：Flat 域选 Linear/Activation，Sequence 域选 RNN/LSTM/GRU，Spatial 域选 Conv2d(80%)/Pool2d(20%) |
| `RemoveLayer` | 0.15 | ✅ | 随机移除非输出头的隐藏层（验证域链合法性） |
| `ReplaceLayerType` | 0.10 | | Activation 内部互换（ReLU↔Tanh↔Sigmoid…共 13 种） |
| `GrowHiddenSize` | 0.24 | | 增大 Linear/RNN/LSTM/GRU/Conv2d 的尺寸（+1 或 ×2） |
| `ShrinkHiddenSize` | 0.29 | | 缩小 Linear/RNN/LSTM/GRU/Conv2d 的尺寸（-1 或 /2） |
| `MutateLayerParam` | 0.05 | | LeakyReLU/ELU alpha，Dropout p |
| `MutateLossFunction` | 0.02 | | 切换兼容 loss（如 BCE↔MSE） |
| `MutateCellType` | 0.10 | | Rnn↔Lstm↔Gru 循环层类型切换（仅序列模式注册） |
| `MutateKernelSize` | 0.10 | | Conv2d kernel_size 在 {1,3,5,7} 中切换（仅空间模式注册） |

> Shrink 权重 > Grow，鼓励演化出更紧凑的网络（奥卡姆剃刀）。

**SkipEdge 变异（3 种）**：

| 变异 | 权重 | 结构性 | 核心逻辑 |
|---|---|---|---|
| `AddSkipEdge` | 0.08 | ✅ | DAG 前向约束 + 策略继承 + 试探式维度验证 |
| `RemoveSkipEdge` | 0.05 | ✅ | 从 skip_edges 移除 |
| `MutateAggregateStrategy` | 0.03 | | 目标层组级原子操作，同一目标所有边统一切换 |

**训练超参数变异（2 种）**：

| 变异 | 权重 | 核心逻辑 |
|---|---|---|
| `MutateLearningRate` | 0.05 | Log ladder 13 级 [1e-5, 1e-1]，80% 移 1 步 / 20% 移 2 步 |
| `MutateOptimizer` | 0.02 | Adam↔SGD 切换 + lr band snap（避免裸切换必回滚） |

### 5.3 合法性保障

所有变异在 `is_applicable()` 和 `apply()` 中确保：

- **输出头保护**：不删除、不修改、不替换、不在其之后插入
- **维度兼容**：通过试探式 `resolve_dimensions()` 统一检测
- **规模约束**：`max_layers` / `max_hidden_size` / `max_total_params`
- **连续 Activation 禁止**：不允许两个 Activation 相邻
- **DAG 约束**：skip edge 只能前向（from 在 to 之前）
- **域链合法性**：`is_domain_valid()` 确保合法的域转换序列
  - 序列模型：`Sequence* → Flat*`（不允许 Flat→Sequence 回溯）
  - 空间模型：`Spatial* → Flatten → Flat*`
- **记忆单元原子性**：序列模式下 skip edge 仅允许在 Flat 域内，不穿透循环层
- **最小保护**：至少保留输出头一层

### 5.4 自定义变异

通过 `MutationRegistry` 注册自定义变异：

```rust
let mut registry = MutationRegistry::default_registry(&metric);
registry.register(0.10, MyCustomMutation);

Evolution::supervised(train, test, metric)
    .with_mutation_registry(registry)
    .run()
```

---

## 6. 训练与收敛检测

### 6.1 ConvergenceDetector

按优先级判断是否应停止当前架构的训练：

1. **NaN / Infinity loss** → 立即停止（训练已爆炸）
2. **FixedEpochs 模式** → 到达指定 epoch 数
3. **max_epochs 安全上限** → 防止无限循环（默认 100）
4. **Loss 稳定** → patience 窗口内相对变化 `(max-min)/(min.abs()+1e-8)` < tolerance
5. **梯度消失** → 连续 patience 次 grad_norm < grad_tolerance

默认配置：`patience=5, loss_tolerance=1e-4, grad_tolerance=1e-5, max_epochs=100`。

### 6.2 TrainingBudget

| 模式 | 用途 |
|---|---|
| `UntilConverged` | 默认，训练到收敛或 max_epochs |
| `FixedEpochs(n)` | 预留：快速筛选候选架构（BOHB/Successive Halving 策略） |

---

## 7. Lamarckian 权重继承

训练后的权重保存在 `NetworkGenome.weight_snapshots` 中，按 `innovation_number` 索引。下一代 `build()` 后、训练前调用 `restore_weights()`：

| 情况 | 行为 |
|---|---|
| 同 innovation_number 且形状相同 | 直接复制（继承） |
| 形状不匹配（如 GrowHiddenSize 后） | 保留新初始化值 |
| 无快照（新插入的层） | 保留新初始化值 |

`InheritReport` 返回 `inherited` / `reinitialized` 计数。

回滚机制利用此特性：`best_genome.clone()` 自带权重快照，恢复后 `restore_weights()` 直接还原到 best 时期的权重状态。

---

## 8. 可观测性

### 8.1 EvolutionCallback

```rust
pub trait EvolutionCallback {
    fn on_generation(&mut self, gen: usize, genome: &NetworkGenome, loss: f32, score: &FitnessScore) {}
    fn on_new_best(&mut self, gen: usize, genome: &NetworkGenome, score: &FitnessScore) {}
    fn on_mutation(&mut self, gen: usize, mutation_name: &str, genome: &NetworkGenome) {}
    fn should_stop(&self, gen: usize) -> bool { false }
}
```

`on_new_best` 仅在 primary **严格提升**时触发（tiebreak 改善和中性漂移均不触发）。

### 8.2 DefaultCallback 日志

`verbose=true` 时每代输出一行：

```
[Gen  0] Input(2) → [Linear(1)]                         | fitness=0.501 | minimal *
[Gen  1] Input(2) → Linear(1) → [Linear(1)]             | fitness=0.502 | InsertLayer
[Gen  5] Input(2) → Linear(4) → ReLU → [Linear(1)]      | fitness=1.000 | GrowHiddenSize *
```

`*` 表示 primary 严格提升。`verbose=false` 时完全静默。

### 8.3 计算图可视化

`EvolutionResult` 提供 `visualize()` 方法，内部委托已有的 Graphviz 管线：

```rust
let vis = result.visualize("output/evolution")?;
// 生成 .dot + .png（含 Loss/Target 节点、模型聚类、激活菱形样式）
```

演化主循环在达标和非达标路径均自动调用 `snapshot_once_from()`，用户无需手动触发。

### 8.4 推理

`EvolutionResult` 提供 `predict()` 方法，封装了 Graph / Var 等内部细节：

```rust
// 平坦：单样本 [input_dim] 或批量 [batch, input_dim]
// 序列：单样本 [seq_len, input_dim] 或批量 [batch, seq_len, input_dim]
// 空间：单样本 [C, H, W] 或批量 [batch, C, H, W]
let predictions = result.predict(&input)?;  // 返回 [batch, output_dim]
```

用户无需接触 `Graph`、`Var`、`BuildResult` 等计算图层类型。

---

## 9. 模块结构

```
src/nn/evolution/
├── mod.rs              Evolution + run() + TaskSpec + EvolutionResult + EvolutionStatus
├── error.rs            EvolutionError（InvalidData / InvalidConfig / Graph）
├── gene.rs             NetworkGenome, LayerGene, LayerConfig, SkipEdge, TrainingConfig, TaskMetric
├── mutation.rs         Mutation trait + MutationRegistry + 12+条件变异操作
├── builder.rs          Genome → Graph 转换 + BuildResult + Lamarckian 权重管理
├── model_io.rs         模型序列化/反序列化（save/load .otm 文件）
├── convergence.rs      ConvergenceDetector + ConvergenceConfig + TrainingBudget
├── task.rs             EvolutionTask trait + SupervisedTask + FitnessScore
├── callback.rs         EvolutionCallback trait + DefaultCallback
└── tests/
    ├── gene.rs         基因数据结构单元测试（含序列域/空间层维度测试）
    ├── mutation.rs     变异操作单元测试（含 MutateCellType）
    ├── builder.rs      构建与权重继承单元测试（含 RNN/LSTM/GRU/Conv2d 前向测试）
    ├── convergence.rs  收敛检测单元测试
    ├── task.rs         训练与评估单元测试
    └── evolution.rs    主循环集成测试（含序列演化/空间演化端到端测试）
```

---

## 10. 关键设计决策
| 决策 | 理由 |
||---|---|
| Genome-Centric（每代重建图） | 基因组自包含，clone 即回滚，避免 Graph 内部状态纠缠 |
| Mutation trait + 注册表 | 可插拔设计，添加新变异不修改 Evolution（EXAMM/LayerNAS 标准做法） |
| `>=` 非严格接受 | 解决 stepping stone 问题（XOR 等任务需多步结构变化才能突破） |
| Fitness 驱动（非 loss） | 通用性：fitness 在所有范式中有意义；loss 不可跨架构比较 |
| Tiebreak 分离（独立字段） | primary 保持纯指标值，避免 epsilon 融合污染日志和 target_metric 比较 |
| 聚合操作 build 时派生 | layers 纯粹包含计算层，变异操作不需要处理 Aggregate 特殊情况 |
| TrainingConfig 绑定 Genome | 架构与训练超参数耦合（NAS-HPO-Bench-II 证实），联合搜索优于分离 |
| Batch size 是 Task 层职责 | EvolutionTask::train() 不感知 batch；监督/RL 各自管理喂数据策略 |
| 延迟实例化（TaskSpec） | `supervised()` 无错构造，数据验证和 Task 创建延迟到 `run()`，保持 builder 链零 boilerplate |
| 单一 StdRng 贯穿 | seed 完整控制所有随机性，可复现；未来可无痛升级为 per-phase seed |
| `run()` 演化未达标不是 Err | 搜索未达标用 EvolutionStatus 区分，Err 仅用于数据验证失败和系统错误 |
| Log ladder 学习率变异 | 单 genome + rollback 下，离散台阶避免冗余值、便于回访、日志可读 |
| Optimizer 切换 + lr band snap | Adam/SGD 有效 lr 范围不同，裸切换几乎必被回滚 |
| Graph 不暴露给用户 | 抽象一致性：演化是 AutoML 层 API，Graph 是计算图层；封装后内部可自由重构 |

---

## 11. 扩展指南

### 11.1 添加新层类型

1. 在 `LayerConfig` 枚举添加新变体
2. 在 `gene.rs` 的 `compute_output_dim()` 和 `compute_layer_params()` 添加分支
3. 在 `builder.rs` 的 `build()` 添加构建逻辑
4. 在 `InsertLayerMutation` 中纳入新类型（或创建专用 Mutation）

### 11.2 添加新变异操作

1. 实现 `Mutation` trait
2. 在 `default_registry()` 中注册（或由用户通过 `with_mutation_registry()` 注册）

### 11.3 支持新学习范式

实现 `EvolutionTask` trait：

```rust
pub trait EvolutionTask {
    fn train(&self, genome: &NetworkGenome, build: &BuildResult,
             convergence: &ConvergenceConfig, rng: &mut StdRng) -> Result<f32, GraphError>;
    fn evaluate(&self, genome: &NetworkGenome, build: &BuildResult,
                rng: &mut StdRng) -> Result<FitnessScore, GraphError>;
}
```

例如 `RLTask` 可在 `train()` 中执行策略梯度，在 `evaluate()` 中跑 Episode 取奖励。

内部扩展时，还需在 `mod.rs` 的 `TaskSpec` 枚举添加新变体（如 `RL { env, metric }`），
并在 `materialize_task()` 添加对应分支 + 公开构造器（如 `Evolution::reinforcement()`）。

### 11.4 序列 / 记忆单元演化

**已完成**。序列模式自动启用——当输入样本为 2D 张量 `[seq_len, input_dim]` 时，系统检测到序列结构并切换到记忆单元演化策略。

**记忆单元类型**：
- **Rnn** `{ hidden_size }`：最简单的循环层，`h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_h)`
- **Lstm** `{ hidden_size }`：长短期记忆，输入门/遗忘门/候选细胞/输出门四路门控
- **Gru** `{ hidden_size }`：门控循环单元，更新门/重置门两路门控

**最小序列架构** (minimal_sequential)：

```
Input(seq×D) → Rnn(output_dim) → [Linear(output_dim)]
```

初始使用最简单的 Rnn，`MutateCellType` 可升级为 LSTM/GRU，`InsertLayer` 可在序列域再插入更多循环层。

**build() 中的 `needs_return_sequences` 逻辑**：builder 在构建循环层时自动判断是否需要返回完整序列——向后扫描 resolved 层列表（跳过 Activation/Dropout），若下一个实质层也是循环层则调用 `forward_seq()`（返回 `[batch, seq_len, hidden]`），否则调用 `forward()`（仅返回最后一步 `[batch, hidden]`）。

**域约束**：序列模式下 skip edge 仅允许在 Flat 域内。记忆单元（RNN/LSTM/GRU）作为原子单元，不允许 skip edge 跨越或穿透 Sequence 域——避免 3D/2D 形状不兼容和 concat dim 语义混乱（dim=1 对 3D 是 seq_len 而非 features）。

**变长序列支持**：`SupervisedTask` 在构造时自动检测变长输入（各样本 seq_len 不同），零填充至全局最大长度。用户无需手动处理 padding。

**变异支持**：`InsertLayer` 在序列域随机选择 RNN/LSTM/GRU（各 1/3 概率），`MutateCellType` 切换循环层类型（保持 hidden_size 不变，旧权重快照失效后重新初始化），`Grow/ShrinkHiddenSize` 调整循环层 hidden_size。

**使用示例**：参见 `examples/evolution_parity_seq/main.rs`（固定长度）和 `examples/evolution_parity_seq_var_len/main.rs`（变长序列）。

### 11.5 卷积神经网络（CNN）演化

**已完成**。空间模式自动启用——当输入样本为 3D 张量 `[C, H, W]` 时，系统检测到空间结构并切换到 CNN 演化策略。

**空间层类型**：
- **Conv2d** `{ out_channels, kernel_size }`：2D 卷积，stride=1，same padding（padding=kernel_size/2，不改变 H/W）
- **Pool2d** `{ pool_type: Max|Avg, kernel_size, stride }`：2D 池化，空间降维（H/stride, W/stride），channels 不变
- **Flatten**：空间域到平坦域的过渡层，将 `(C, H, W)` 展平为 `C*H*W`

**最小空间架构** (minimal_spatial)：

```
Input(C@H×W) → Flatten → [Linear(out_dim)]
```

从最简单的 Flatten+FC 结构出发，Conv2d/Pool2d 由演化通过 `InsertLayer` 自主发现和插入。对于小图像（如 MNIST 28×28），纯 FC 方案参数量更少、训练更快；对于大图像，演化会在 Flatten 前插入 Conv2d/Pool2d 来降维（触及 `max_total_params` 约束时被迫探索）。

**域系统**：层序列被划分为 Spatial 域（Conv2d、Pool2d，4D 张量）和 Flat 域（Linear、Activation，2D 张量）。Flatten 是唯一的域过渡层。`is_domain_valid()` 确保 `Spatial* → Flatten → Flat*` 结构（允许 0 个空间层，即 `Flatten → Flat*` 也合法）。

**变异支持**：`InsertLayer` 域感知插入（Spatial 域 Conv2d 80% / Pool2d 20%），`MutateKernelSize` 切换 Conv2d kernel，`Grow/ShrinkHiddenSize` 调整 Conv2d out_channels。

**使用示例**：参见 `examples/evolution_mnist/main.rs`。

### 11.6 未来方向

| 方向 | 难度 | 说明 |
|---|---|---|
| 种群演化 + 交叉 | 高 | 当前为单网络迭代，需引入种群管理和基因组对齐 |
| Budget-based 训练（Successive Halving） | 低 | TrainingBudget::FixedEpochs 已预留 |

---

## 附录 A：典型演化轨迹

### XOR (input_dim=2, output_dim=1)

```
[Gen  0] Input(2) → [Linear(1)]                         | fitness=0.501 | minimal *
         ↓ InsertLayer（在输出头前插入 Linear(1)）
[Gen  1] Input(2) → Linear(1) → [Linear(1)]             | fitness=0.502 | InsertLayer
         ↓ GrowHiddenSize
[Gen  3] Input(2) → Linear(4) → [Linear(1)]             | fitness=0.503 | GrowHiddenSize
         ↓ InsertLayer（在 Linear(4) 和输出头间插入 ReLU）
[Gen  5] Input(2) → Linear(4) → ReLU → [Linear(1)]      | fitness=1.000 | InsertLayer *
         → 达标！系统自动发现了解决 XOR 问题的架构。
```

`[Linear(1)]` 表示输出头（受保护）。中间几代的"中性漂移"（fitness 持平但结构变化被接受）为最终的突破性变异创造了条件。

### Iris 三分类 (input_dim=4, output_dim=3, mini-batch)

```
[Gen  0] Input(4) → [Linear(3)]                         | fitness=0.953 | minimal *
         → 1 代即达标（线性可分数据集，最小结构已足够）。
```

### 奇偶性序列检测 (input=seq×1, output_dim=1, 200 train / 50 test)

```
[Gen  0] Input(seq×1) → RNN(1) → [Linear(1)]                    | fitness=0.520 | minimal_sequential *
         ↓ GrowHiddenSize
[Gen  3] Input(seq×1) → RNN(2) → [Linear(1)]                    | fitness=0.540 | GrowHiddenSize
         ↓ MutateCellType
[Gen  8] Input(seq×1) → GRU(2) → [Linear(1)]                    | fitness=0.680 | MutateCellType *
         ↓ GrowHiddenSize
[Gen 12] Input(seq×1) → GRU(4) → [Linear(1)]                    | fitness=0.900 | GrowHiddenSize *
         → 系统自动发现 GRU 比初始 RNN 更适合此任务。
```

记忆单元类型由 `MutateCellType` 自动探索，`GrowHiddenSize` 扩展容量——联合搜索结构与参数。

### MNIST CNN (input=1@28×28, output_dim=10, 1000 train / 500 test)

```
[Gen  0] Input(1@28×28) → Flatten → [Linear(10)]                                     | fitness=0.xxx | minimal_spatial *
         ↓ InsertLayer / GrowHiddenSize / ...
         → 系统从纯 FC 出发，自主决定是否插入 Conv2d/Pool2d 或增加 FC 层。
```

初始架构 Flatten → [Linear(10)] 参数量约 7.9K（784×10 + 10），在 `max_total_params` 默认值 10K 下仍有增长空间。演化可自由探索纯 FC 或 Conv+FC 混合架构。

---

## 附录 B：参考文献

1. **NEAT**（2002）：Evolving Neural Networks through Augmenting Topologies
2. **EXAMM**（2019）：Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution
3. **LayerNAS**（2023）：Neural Architecture Search in Polynomial Complexity
4. **NAS-HPO-Bench-II**：联合架构-超参数搜索基准
5. **BOHB**：Robust and Efficient Hyperparameter Optimization at Scale
6. **Stitching for Neuroevolution**（2024）：权重复用加速演化训练

---

*本文档描述 only_torch 的神经架构演化模块实际实现。最后更新：2026-03-10*

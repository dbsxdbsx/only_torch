# 神经架构演化（Neural Architecture Evolution）

> **设计理念**：用户只提供数据和目标，系统从最小网络出发，通过自动变异 + 梯度训练的混合策略自动发现最优架构。
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
    .with_population_size(16)         // NSGA-II 种群大小（默认 auto = rayon threads clamped 12..32）
    .with_offspring_batch_size(12)    // 每代新候选数（默认 auto = max(12, rayon threads)）
    .with_parallelism(8)              // 并行评估线程数（默认 auto = rayon threads）
    .with_pareto_patience(40)         // Pareto archive 收敛耐心值（默认 auto = max(20, pop*2)）
    .with_complexity_metric(ComplexityMetric::FLOPs) // inference_cost 计算方式（默认 FLOPs）
    .with_batch_size(64)              // 显式 batch size（默认自动策略）
    .with_verbose(false)              // 关闭日志（默认 true）
    .with_convergence(config)         // 收敛检测配置（Phase 2 使用）
    .with_constraints(constraints)    // 网络规模约束（默认 auto 推导）
    .with_stagnation_patience(30)     // 停滞检测耐心值（默认 20）
    .with_eval_runs(3)                // 多次评估取保守值（默认 1）
    .with_mutation_registry(registry) // 自定义变异注册表（覆盖两阶段）
    .with_report_metrics([ReportMetric::F1]) // 追加评估报告指标（不参与选择）
    .with_primary_proxy(ProxyKind::LossSlope) // 学习速度代理（默认启用，传 None 关闭）
    .with_asha(AshaConfig::default()) // ASHA 多保真评估（默认启用 rung_epochs=[1,2,4], eta=3）
    .with_callback(my_callback)       // 自定义回调
    .run()
```

### 1.3 支持的任务指标

| TaskMetric | 自动推断的 Loss | 适用场景 |
|---|---|---|
| `Accuracy` | BCE (output_dim=1) / CrossEntropy (output_dim>1) | 分类 |
| `R2` | MSE | 回归 |
| `MultiLabelAccuracy` | BCE | 多标签分类 |

### 1.4 评估报告指标

`TaskMetric` 仍是主目标：它驱动 loss 推断、`target_metric` 判断、NSGA-II 选择和离散指标的 `tiebreak_loss`。`ReportMetric` 是评估后的附加报告指标，只用于日志、回调和 `EvolutionResult::fitness.report`，不改变演化选择结果。

默认报告指标按任务类型自动开启：

| TaskMetric | 默认 ReportMetric |
|---|---|
| `Accuracy` | `Accuracy` / `Precision` / `Recall` / `F1` |
| `R2` | `R2` / `MeanSquaredError` / `MeanAbsoluteError` / `RootMeanSquaredError` |
| `MultiLabelAccuracy` | `MultiLabelLooseAccuracy` / `MultiLabelStrictAccuracy` |

可通过 `.with_report_metrics([...])` 在默认集合基础上追加指标；不兼容当前任务类型的指标会被忽略，重复指标自动去重。底层计算复用通用 `src/metrics/` 模块，因此手写模型也可以直接调用同一套指标函数。

### 1.5 Batch Size 自动策略

用户无需关心 batch size——`SupervisedTask` 根据数据量自动选择：

| 数据量 | 策略 | 典型场景 |
|---|---|---|
| ≤ 128 | full-batch | XOR 等小任务 |
| 129–10000 | batch_size = 64 | Iris 等中等数据集 |
| > 10000 | batch_size = 256 | 大规模数据集 |

可通过 `.with_batch_size()` 覆盖。

---

## 2. 演化主循环

### 2.1 Genome-Centric 架构 + Pareto 种群搜索

系统采用 **Genome-Centric** 路线：每代从基因组（`NetworkGenome`）重建计算图，而非直接修改图。

采用 **Pareto 种群搜索 + NSGA-II 选择**策略：维护一个种群（默认 12–32 个体，auto = rayon 线程数 clamped 到 [12,32]），每代通过二元锦标赛从种群中采样父代、变异生成 offspring，然后用 NSGA-II 环境选择从 parents ∪ offspring 中保留最优 `population_size` 个。相比旧版 (1+λ) 单谱系搜索，种群策略维护多样性、天然支持多目标优化（primary ↑ + inference_cost ↓），跳出局部最优概率大幅提升。

**全局 Pareto Archive**：独立于种群维护所有非支配解。Archive 中的成员在目标空间互不支配（不存在 A 在所有目标上 ≥ B 的情况）。达标检查从 archive 而非种群中查找，确保不遗漏高质量解。

**并行评估**：offspring 的 build→train→evaluate 流水线通过 rayon 并行化。`with_parallelism(n)` 创建独立的 `rayon::ThreadPool`（`ThreadPoolBuilder::num_threads(n)`），真正控制评估并发度。训练/测试数据在 `SupervisedTask` 构造时一次性 stack 为 `Arc<Tensor>`，`TaskRuntime::clone()` 只复制 Arc 指针（O(1)），每个 rayon 工作线程通过 `map_init` 克隆轻量 task 副本，避免每代每 worker 重复堆叠数据。`Graph` 使用 `Rc<RefCell<>>` (!Send)，但并行安全因为每个 worker 通过 `genome.build()` 创建独立的本地 Graph。

```
自适应约束：用户未指定 with_constraints() 时，自动从数据维度推导 SizeConstraints
随机爆发初始化：对 minimal genome 施加 K 次随机变异（K = max(2, min(8, max_layers/2))）生成初始种群
两阶段训练预算：Phase 1（前 40% 代数）用 FixedEpochs 快速筛选；Phase 2（后 60%）用 UntilConverged 精炼
两阶段变异权重：Phase 1 偏向结构探索（InsertLayer↑, Grow↑）；Phase 2 偏向超参调优（MutateLR↑, MutateOptimizer↑）
Primary plateau 提前切换：若 primary fitness 连续 stagnation_patience 代未提升，即使未到 Phase 2 代数也立即切换

初始化:
  for i in 0..population_size:
    genome_i = minimal_genome + K 次随机变异
  evaluate_batch(初始种群)  // rayon 并行
  初始化 archive = 非支配解集合

每一代 (generation):
  1. 二元锦标赛采样 + 变异 → offspring_batch_size 个候选
     (按 pareto_rank 和 crowding_distance 选亲本)
  2. evaluate_batch(offspring)  // rayon 并行
  3. pool = parents ∪ offspring
  4. update_archive(pool)  // 合并新非支配解
  5. parents = nsga2_select(pool, population_size)  // 保留最优种群
  6. 检查 archive 中是否有达标成员
  7. 检查 archive 是否收敛（连续 pareto_patience 代未变化）
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
    ├─ build(rng)        → 新层参数初始化（Graph RNG）
    ├─ train(rng)        → Dropout mask（Graph next_seed）、mini-batch 顺序（shuffle_mut_seeded）
    ├─ evaluate(rng)     → 评估时随机性
    └─ apply_random(rng) → 变异选择和执行
```

**确定性保证**（v0.14 起）：当 `with_seed()` 被调用时，系统自动固定 `population_size`（20）和 `offspring_batch_size`（12），消除因不同机器线程数导致的 RNG 消耗序列差异。用户显式调用 `with_population_size()` / `with_offspring_batch_size()` 可覆盖此默认值。

`rebuild_pareto_member()` 使用保存的 `evolution_seed` 派生子 seed（`seed ^ index`），而非 `from_entropy()`，确保同一演化结果的 Pareto 成员重建可复现。

### 2.4 停止条件与返回值

`run()` 返回 `Result<EvolutionResult, EvolutionError>`。演化搜索未达标不是错误，通过 `status` 标识停止原因：

| EvolutionStatus | 含义 |
|---|---|
| `TargetReached` | archive 中存在 primary ≥ target_metric 的成员（选 inference_cost 最小的达标成员） |
| `MaxGenerations` | 到达最大代数限制 |
| `CallbackStopped` | 自定义回调请求停止 |
| `NoApplicableMutation` | 所有变异均不可用（搜索空间耗尽） |
| `ParetoConverged` | 全局 Pareto archive 连续 `pareto_patience` 代未发生实质变化 |

`Err(EvolutionError)` 用于数据验证失败（`InvalidData`）或系统错误（`Graph`）。数据验证延迟到 `run()` 执行时，`supervised()` 构造器本身不会失败。

---

## 3. 选择策略

### 3.1 NSGA-II 多目标选择

种群搜索采用 NSGA-II 环境选择，双目标优化：
- **Objective 1**：`primary`（越高越好）——任务指标（Accuracy / R² 等）
- **Objective 2**：`inference_cost`（越低越好）——模型复杂度（默认 FLOPs，可通过 `with_complexity_metric()` 切换为 ParamCount）

选择排序规则（优先级从高到低）：
1. Pareto rank 越小越好（非支配前沿优先）
2. 同 rank 时 crowding distance 越大越好（保持多样性）
3. 同 rank 同 distance 时用 `tiebreak_loss` 决胜

**Crowding distance 按 Pareto front 分层计算**：每一层非支配前沿（rank=0, rank=1, ...）内部独立计算拥挤度距离，确保各 front 的边界解获得正确的 ∞ 距离，不被跨 front 排序污染。

每代从 `parents ∪ offspring` 合并池中选出 `population_size` 个幸存者。Pareto 前沿（rank=0）的个体自动进入全局 archive。

### 3.2 FitnessScore 与多目标

```rust
pub struct FitnessScore {
    pub primary: f32,                // 主目标（越高越好）
    pub inference_cost: Option<f32>, // 推理成本（越低越好，用于 Pareto 第二目标）
    pub tiebreak_loss: Option<f32>,  // 离散指标：test loss（NSGA-II 同 rank 同 distance 时的决胜条件）
    pub primary_proxy: Option<f32>,  // 学习速度代理（plateau 时用于打破平局）
    pub report: MetricReport,        // 附加报告指标（不参与 NSGA-II objective）
}
```

`inference_cost` 由 `compute_inference_cost()` 根据 `ComplexityMetric` 计算（默认 FLOPs——直接反映训练/推理耗时；对非空间任务 FLOPs 与 ParamCount 成比例，行为等价）。`inference_cost` 为 `None` 时退化为单目标排序。

`report` 只承载可观测性信息，例如 `accuracy` / `f1` / `mse` 等，不进入 `objective_point()`，因此不会影响 Pareto rank、crowding distance、达标判断或 archive 收敛。

### 3.3 停滞检测

连续 `stagnation_patience`（默认 20）代 best primary 未严格提升后，强制从结构性变异（`InsertLayer` / `RemoveLayer` / `AddSkipEdge` / `RemoveSkipEdge`）中选择，打破参数变异空转。

### 3.4 Pareto 收敛与主目标平台期检测

**Archive 收敛**：全局 archive 的代表成员（primary 最高或满足 target 的最小 cost 成员）连续 `pareto_patience`（默认 `max(20, population_size * 2)`）代的 FitnessScore 未发生实质变化时（primary / inference_cost / tiebreak_loss 三个选择相关字段均在 tolerance=1e-6 内），判定 Pareto 前沿收敛，返回 `ParetoConverged` 状态。`MetricReport` 不参与该判断。

**Primary 平台期提前切换**：即使 archive 仍有细微 trade-off 变化（如新成员 inference_cost 更低但 primary 相同），primary fitness 连续 `stagnation_patience` 代未严格提升时也会触发 Phase 2 切换和结构变异强制——避免"archive 在变但主指标不涨"的隐性停滞。

### 3.5 EvolutionResult 中的 Pareto 信息

`EvolutionResult` 包含完整的 Pareto 前沿信息：
- `pareto_front: Vec<ParetoSummary>`：轻量摘要（fitness + 架构描述）
- `rebuild_pareto_member(index)`：按索引 lazy rebuild 完整 EvolutionResult（含权重）
- `smallest_meeting_target_index(target)`：找到满足 target 且 inference_cost 最小的成员

---

## 4. 基因数据结构

### 4.1 NetworkGenome

```rust
pub struct NetworkGenome {
    pub input_dim: usize,
    pub output_dim: usize,
    pub seq_len: Option<usize>,                // None=平坦, Some(n)=序列
    pub input_spatial: Option<(usize, usize)>,  // None=非空间, Some((H,W))=空间
    pub training_config: TrainingConfig,        // lr / optimizer / loss 等
    pub generated_by: String,                   // 变异来源（调试用）
    pub(crate) repr: GenomeRepr,                // 内部表示（NodeLevel / LayerLevel）
}

pub(crate) enum GenomeRepr {
    /// 节点级表示（唯一正式内核表示）
    NodeLevel {
        nodes: Vec<NodeGene>,
        next_innovation: u64,
        weight_snapshots: HashMap<u64, Tensor>,  // param_innovation → Tensor
    },
    /// 层级表示（仅作为用户 DSL 入口保留，运行前自动迁移到 NodeLevel）
    LayerLevel {
        layers: Vec<LayerGene>,
        skip_edges: Vec<SkipEdge>,
        next_innovation: u64,
        weight_snapshots: HashMap<u64, Vec<Tensor>>,
    },
}
```

**内核统一**：演化主循环 `run()` 在启动时对所有类型（Flat/Spatial/Sequential）调用 `migrate_to_node_level()`，确保内部始终以 NodeLevel 运行。LayerLevel 仅在用户通过 `from_flat`/`from_spatial`/`from_sequential` 等 DSL 入口传入初始配置时暂存，随即被迁移。

**最小初始网络**（按数据形状自动选择）：
- 平坦数据 `[D]`：`Input(D) → [Linear(output_dim)]`
- 序列数据 `[T, D]`：`Input(seq×D) → Rnn(output_dim) → [Linear(output_dim)]`
- 空间数据 `[C, H, W]`：`Input(C@H×W) → Flatten → [Linear(out_dim)]`（Conv2d/Pool2d 由演化自主插入）

序列初始网络以最简单的 Rnn 为起点（而非 LSTM/GRU），后续 `MutateCellType` 可升级记忆单元类型，`InsertLayer` 可在序列域插入更多循环层。

**输出头保护**：最后一个主路径块（输出头）不可删除、不可替换。

**权重快照自包含**：`clone()` 时权重一并复制，回滚到 `best_genome` 自带权重，无需保留旧 Graph。权重按 Parameter 节点 innovation number 索引（`HashMap<u64, Tensor>`），Grow/Shrink/Replace 时按节点是否保留、形状是否兼容判断继承。

### 4.2 NodeGene

```rust
pub struct NodeGene {
    pub innovation_number: u64,            // NEAT 风格创新号
    pub node_type: NodeTypeDescriptor,     // 直接对齐图 IR 层
    pub output_shape: Vec<usize>,          // 输出形状
    pub parents: Vec<u64>,                 // 父节点创新号列表
    pub enabled: bool,                     // NEAT 风格禁用机制
    pub block_id: Option<u64>,             // 模板组标识
}
```

`NodeGene` 是演化系统的最小可操作单元。`node_type` 直接使用 `NodeTypeDescriptor`（定义在 `src/nn/descriptor.rs`），实现演化层和图 IR 层的 1:1 对齐。

**block_id** 用于模板组操作：同一个高层模板（如 Linear = MatMul + Parameter + Add + Parameter）展开后的节点共享相同 `block_id`。Grow/Shrink/Remove 以模板组为单位操作。`block_id = None` 表示独立节点（如单独的激活函数）。

### 4.3 GenomeAnalysis

```rust
pub struct GenomeAnalysis {
    pub topo_order: Vec<u64>,              // 拓扑排序后的节点 ID 序列
    pub shape_map: HashMap<u64, Vec<usize>>, // 每个节点的推导形状
    pub domain_map: HashMap<u64, ShapeDomain>, // 每个节点的域
    pub param_count: usize,                // 总参数量
    pub reachable: HashSet<u64>,           // 从输出可达的节点集
    // ...
}
```

`GenomeAnalysis` 是不可变快照，通过 `genome.analyze()` 生成。任何 mutation、migration、builder 修正之后都必须重新分析，不允许共享可变 analysis 状态。统一承担拓扑排序、环检测、形状推导、域推导、参数统计、输出可达性检查、连接合法性检查。

### 4.4 LayerConfig（用户入口 DSL）

```rust
pub enum LayerConfig {
    Linear { out_features: usize },
    Activation { activation_type: ActivationType },
    Rnn { hidden_size: usize },
    Lstm { hidden_size: usize },
    Gru { hidden_size: usize },
    Dropout { p: f32 },
    Conv2d { out_channels: usize, kernel_size: usize },
    Pool2d { pool_type: PoolType, kernel_size: usize, stride: usize },
    Flatten,
}
```

LayerConfig 不再是内核表示，仅作为用户入口 DSL 保留。`migrate_to_node_level()` 通过 `TemplateExpander` 将每个 `LayerConfig` 展开为对应的 `NodeGene` 模板组（如 `Linear` → MatMul + Parameter + Add + Parameter）。

### 4.5 NodeLevel 跨层连接

NodeLevel 中不再维护独立的 `SkipEdge` 概念。任何跨层连接（残差、绕连等）都是 DAG 中的普通前向父边，由 `AddConnection`/`RemoveConnection` 变异直接操作。

新增连接时的自动保障：
- 新增连接必须满足拓扑序，不能成环
- 如果目标已有主输入，自动插入聚合节点（`Add`/`Concat`/`Maximum`）
- 如果形状不兼容，自动插入投影节点（Flat 域 `Linear`、Spatial 域 `1×1 Conv2d`）
- 所有连接合法性统一由 `GenomeAnalysis` 校验

旧版 `SkipEdge` 仅在 LayerLevel DSL 入口中保留兼容。

### 4.6 ShapeDomain 域系统

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

`GenomeAnalysis` 对 NodeLevel 基因组统一执行域推导（`domain_map`），为连接合法性检查和变异约束提供基础。

### 4.7 TrainingConfig

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

### 4.8 维度推导与形状分析

NodeLevel 基因组的形状推导统一由 `GenomeAnalysis` 完成：
- 按拓扑序遍历 `NodeGene` 列表，从 `BasicInput` 出发逐节点推导输出形状
- Parameter / State 节点的 `output_shape` 为权威值，计算节点的 `output_shape` 为声明值（Analysis 验证一致性）
- 空间域维护 `(H, W)` 信息：Conv2d 保持尺寸（same padding, stride=1），Pool2d 缩减，Flatten 归零
- `total_params()` 从 Analysis 的 `param_count` 获取，与 `build()` 共享同一套推导逻辑

变异操作的 `is_applicable()` 通过试探式 `analyze()` 统一检测形状合法性。

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

### 5.2 两阶段变异注册表

演化主循环自动切换两套变异注册表：Phase 1（前 70% 代数）偏向结构探索，Phase 2（后 30%）偏向超参调优。用户通过 `with_mutation_registry()` 传入自定义注册表时，两阶段均使用该单一注册表。

`MutateCellType`（序列模式）和 `MutateKernelSize`（空间模式）按数据形状条件注册，两阶段权重相同（0.10）。

**Phase 1：拓扑搜索（`phase1_registry`）**

| 变异 | 权重 | 结构性 | 核心逻辑 |
|---|---|---|---|
| `InsertLayer` | **0.20** | ✅ | 域感知：Flat 域选 Linear/Activation/BatchNorm/LayerNorm/RMSNorm，Sequence 域选 RNN/LSTM/GRU/LayerNorm/RMSNorm，Spatial 域选 Conv2d(80%)/Pool2d(15%)/BatchNorm(5%)。归一化层以 10% 概率独立触发，不与同类连续 |
| `InsertAtomicNode` | 0.10 | ✅ | NEAT "Add Node"：在主路径两块之间插入单个激活节点（15 种激活函数随机选择，85%）或 Dropout（15%，p∈{0.1,0.2,0.3,0.5}）。保护输出头、避免连续激活/连续 Dropout |
| `RemoveLayer` | 0.08 | ✅ | 随机移除非输出头的隐藏层（早期偏向增长） |
| `ReplaceLayerType` | 0.04 | | Activation 内部互换（ReLU↔Tanh↔Sigmoid…共 13 种） |
| `GrowHiddenSize` | **0.12** | | 增大尺寸（40% +step, 40% ×1.5, 20% ×2，step = max(1, current/4)）。权重从 0.25 降至 0.12，将探索预算让渡给 FM 级别细粒度变异 |
| `ShrinkHiddenSize` | 0.08 | | 缩小尺寸（40% -step, 40% ×0.67, 20% ÷2） |
| `MutateLayerParam` | 0.05 | | LeakyReLU/ELU alpha，Dropout p |
| `MutateLossFunction` | 0.02 | | 切换兼容 loss（如 BCE↔MSE） |
| `MutateLearningRate` | 0.05 | | Log ladder 13 级 [1e-5, 1e-1] |
| `MutateOptimizer` | 0.02 | | Adam↔SGD 切换 + lr band snap |

> Phase 1 特点：InsertLayer 和 Grow 权重大幅提升，鼓励快速探索网络拓扑；Remove 和 Shrink 权重压低，避免早期裁剪。InsertAtomicNode 以 0.10 权重提供 NEAT 级精细探索。

**Phase 2：精炼（`phase2_registry`）**

| 变异 | 权重 | 结构性 | 核心逻辑 |
|---|---|---|---|
| `InsertLayer` | 0.08 | ✅ | 同上（含归一化层插入） |
| `InsertAtomicNode` | 0.10 | ✅ | 同上（含 Dropout 插入） |
| `RemoveLayer` | 0.08 | ✅ | 同上 |
| `ReplaceLayerType` | 0.08 | | 同上 |
| `GrowHiddenSize` | 0.15 | | 同上 |
| `ShrinkHiddenSize` | 0.15 | | 同上 |
| `MutateLayerParam` | 0.05 | | 同上 |
| `MutateLossFunction` | 0.02 | | 同上 |
| `MutateLearningRate` | **0.15** | | 同上 |
| `MutateOptimizer` | **0.08** | | 同上 |

> Phase 2 特点：结构变异权重降低，超参数调优权重大幅提升（MutateLR 0.05→0.15，MutateOptimizer 0.02→0.08）。Grow/Shrink 对称化（各 0.15）以微调尺寸。

**连接变异（NodeLevel 上操作 DAG 父边）**：

| 变异 | Phase 1 | Phase 2 | 结构性 | 核心逻辑 |
|---|---|---|---|---|
| `AddConnection` | 0.08 | 0.06 | ✅ | 选择两个满足拓扑序的节点，添加父边；形状不兼容时自动插入投影/聚合节点 |
| `RemoveConnection` | 0.05 | 0.05 | ✅ | 移除非关键前向父边（保持图连通性） |

**循环边变异（序列模式专属）**：

| 变异 | Phase 1 | Phase 2 | 结构性 | 核心逻辑 |
|---|---|---|---|---|
| `AddRecurrentEdge` | 0.08 | 0.08 | ✅ | EXAMM 风格：在两个非叶计算节点间添加循环连接 + 权重参数节点。与 cell-based 循环互斥 |
| `RemoveRecurrentEdge` | 0.04 | 0.04 | ✅ | 移除循环边及其孤立权重参数节点 |

所有变异在 NodeLevel 上以模板组（`block_id`）为操作单位：InsertLayer 展开完整模板组，RemoveLayer 删除整个模板组，Grow/Shrink 修改模板组内的 Parameter 形状。

### 5.3 合法性保障

所有变异在 `is_applicable()` 和 `apply()` 中确保：

- **输出头保护**：不删除、不修改、不替换、不在其之后插入
- **形状兼容**：通过试探式 `analyze()` 统一检测（替代旧 `resolve_dimensions()`）
- **规模约束**：`max_layers` / `max_hidden_size` / `max_total_params`
- **循环边不变量**（8 条）：
  1. 悬空源引用检测（`RecurrentMissingSource`）
  2. 权重参数节点存在性检查（`RecurrentInvalidWeight`）
  3. 权重参数必须为 `Parameter` 类型
  4. 权重形状 `[target_dim, source_dim]` 兼容性（`RecurrentShapeMismatch`）
  5. 仅 Flat/Sequence 域允许循环边（不支持 Spatial）
  6. 范式互斥：edge-based 与 cell-based（CellRnn/LSTM/GRU）不共存（`RecurrentParadigmConflict`）
  7. 删除节点后级联清理引用已删节点的循环边
  8. 孤立权重参数节点随循环边删除同步回收
- **连续 Activation 禁止**：不允许两个 Activation 相邻
- **拓扑约束**：新增连接必须满足 DAG 拓扑序（不成环），由 `GenomeAnalysis` 环检测保障
- **域链合法性**：序列模型 `Sequence* → Flat*`（不允许回溯），空间模型 `Spatial* → Flatten → Flat*`
- **最小保护**：至少保留输出头一个模板组

### 5.4 自定义变异

通过 `MutationRegistry` 注册自定义变异：

```rust
let mut registry = MutationRegistry::phase1_registry(&metric, false, false);
registry.register(0.10, MyCustomMutation);

Evolution::supervised(train, test, metric)
    .with_mutation_registry(registry) // 覆盖两阶段，统一使用此注册表
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
| `UntilConverged` | 训练到收敛或 max_epochs（Phase 2 使用） |
| `FixedEpochs(n)` | 快速筛选候选架构（Phase 1 使用，`n = max(3, min(10, n_train/(batch_size*5)))`） |

演化主循环自动分配两阶段训练预算：
- **Phase 1（前 40% 代数）**：`FixedEpochs(fast_epochs)` — 低成本快速评估大量结构候选
- **Phase 2（后 60% 代数）**：`UntilConverged`（用户 `ConvergenceConfig`）— 对有潜力拓扑做充分训练

分界点：`phase1_gens = (max_generations * 0.4).ceil()`。阶段切换时不丢弃权重快照，Phase 2 直接在 Phase 1 最佳基因组上继续。另外，primary fitness 连续 `stagnation_patience` 代停滞也会提前触发 Phase 2 切换，即使尚未到达 40% 代数分界点。

**Mini-batch Shuffle**：mini-batch 训练路径在每个 epoch 开始时对训练数据进行 seeded shuffle（`shuffle_mut_seeded`），确保 mini-batch 组成每 epoch 不同，改善梯度估计质量。Shuffle seed 由演化主 rng 分配，保持可复现性。

---

## 7. Lamarckian 权重继承

训练后的权重保存在 `NetworkGenome.weight_snapshots` 中。NodeLevel 基因组按 Parameter 节点的 innovation number 索引（`HashMap<u64, Tensor>`，每个 Parameter 节点独立一个 Tensor）。

下一代 `build()` 后、训练前调用 `restore_weights()`：

| 情况 | 行为 |
|---|---|
| 同 Parameter innovation_number 且形状相同 | 直接复制（继承） |
| 形状不匹配（如 GrowHiddenSize 后） | 保留新初始化值 |
| 无快照（新插入的模板组带来的新 Parameter） | 保留新初始化值 |

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
    fn on_population_evaluated(&mut self, gen: usize, pop_size: usize, offspring_evaluated: usize,
                               archive_size: usize, front_size: usize, best_primary: f32, best_cost: f32) {}
    fn should_stop(&self, gen: usize) -> bool { false }
}
```

- `on_new_best` 仅在 archive 中 best primary **严格提升**时触发。
- `on_population_evaluated` 每代在 NSGA-II 选择后调用，报告种群/archive/前沿统计。

### 8.2 DefaultCallback 日志

`verbose=true` 时每代输出种群级别统计：

```
[Gen  0] arch=nodes=4 active=4 params=2 | metrics=accuracy=0.501 precision=0.500 recall=0.500 f1=0.500
[Gen  0] pop=12 | off=10 | archive=8 | best=0.501 | cost=784
[Gen  5] pop=12 | off=11 | archive=5 | best=1.000 | cost=396 *
```

第一行来自 `on_generation`，展示当前代表成员架构与 `MetricReport`；第二行来自 `on_population_evaluated`，展示种群级统计。`*` 表示 best primary 严格提升。`verbose=false` 时完全静默。

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
├── mod.rs              Evolution + run() + TaskSpec + EvolutionResult + EvolutionStatus + ParetoSummary + ComplexityMetric
├── error.rs            EvolutionError（InvalidData / InvalidConfig / Graph）
├── gene.rs             NetworkGenome, GenomeRepr, LayerGene, LayerConfig, SkipEdge, TrainingConfig, TaskMetric
├── node_gene.rs        NodeGene 数据结构（innovation_number, node_type, output_shape, parents, block_id）
├── node_ops.rs         NodeBlock / NodeBlockKind / node_main_path()：节点级基因组的模板组分析
├── mutation.rs         Mutation trait + MutationRegistry + 15+条件变异操作（NodeLevel 上以模板组/原子节点/循环边为单位）
├── fm_mutation.rs      FM 级别变异操作（10 种：AddFeatureMap/RemoveFeatureMap/AddFMEdge/RemoveFMEdge/SplitFMEdge + 5 种参数变异）
├── fm_ops.rs           FM 辅助数据结构和查询函数（FMNodeInfo/FMEdgeInfo/FMSubgraphAnalysis/全连接检测/可连接对查询）
├── migration.rs        LayerLevel → NodeLevel 迁移（migrate_to_node_level + TemplateExpander）+ Conv2d → FM 分解（migrate_conv2d_to_feature_maps）
├── selection.rs        NSGA-II 多目标选择 + Pareto Archive 管理（pareto_rank, crowding_distance, nsga2_select, update_archive）
├── builder.rs          Genome → GraphDescriptor → Graph 转换 + to/from_graph_descriptor + backfill_node_group_tags + Lamarckian 权重管理
├── model_io.rs         模型序列化/反序列化（save/load .otm 文件，仅 NodeLevel）+ ONNX 导出（export_onnx）
├── convergence.rs      ConvergenceDetector + ConvergenceConfig + TrainingBudget
├── task.rs             EvolutionTask trait + SupervisedTask + FitnessScore + MetricReport
├── callback.rs         EvolutionCallback trait + DefaultCallback（含 on_population_evaluated）
└── tests/
    ├── gene.rs         基因数据结构单元测试（含序列域/空间层维度测试）
    ├── mutation.rs     变异操作单元测试（含 MutateCellType / NodeLevel 变异）
    ├── fm_mutation.rs  FM 级别变异单元测试（10 种变异的 is_applicable + apply + 稳定性测试，27 个）
    ├── builder.rs      构建与权重继承单元测试（含 RNN/LSTM/GRU/Conv2d 前向 + NodeGroupTag 回填验证）
    ├── model_io.rs     模型序列化测试（含手写⇄演化互通三角测试）
    ├── convergence.rs  收敛检测单元测试
    ├── task.rs         训练与评估单元测试（含 mini-batch shuffle 可复现性验证）
    ├── selection.rs    NSGA-II 选择 + Archive 管理单元测试（含 per-front crowding distance 验证）
    └── evolution.rs    主循环集成测试（含 Pareto 种群/并行评估/fitness_changed/stagnation/序列/空间端到端测试）

src/nn/graph/
├── onnx_import.rs      ONNX → GraphDescriptor 四层导入流水线
├── onnx_export.rs      GraphDescriptor → ONNX 三层导出流水线
├── onnx_ops.rs         ONNX OpType ↔ NodeTypeDescriptor 双向算子映射表
├── onnx_error.rs       OnnxError 错误类型（算子/数据类型/图结构/权重等分类报错）
└── model_save.rs       Graph 的 .otm / ONNX 保存与加载便捷接口

src/nn/tests/onnx/
├── mod.rs              ONNX 测试子模块入口
├── error.rs            OnnxError 错误类型覆盖测试（11 个）
├── ops.rs              双向算子映射测试 + 往返一致性测试（含 ConvTranspose2d，62 个）
├── import.rs           ONNX 导入流水线测试（20 个，含端到端 Graph/NetworkGenome 测试）
└── export.rs           ONNX 导出流水线测试（16 个，含数值往返 + PyTorch 交叉验证）
```

---

## 10. 关键设计决策
| 决策 | 理由 |
|---|---|
| **NodeLevel 为唯一内核表示** | 消除演化层与图 IR 层的抽象断层，`NodeGene.node_type` 直接对齐 `NodeTypeDescriptor`，1:1 映射 |
| **LayerLevel 降级为用户入口 DSL** | 保持用户 API 不变（`from_flat(layers)`），内部立即迁移到 NodeLevel |
| **构图统一走 GraphDescriptor** | `NetworkGenome → GraphDescriptor → Graph::from_descriptor()`，演化和手写模型共用同一条构图管线 |
| **模型互通（.otm）** | 统一存储 NodeLevel 表示，手写模型可通过 `from_graph_descriptor()` 转为 NetworkGenome，再演化或加载 |
| Genome-Centric（每代重建图） | 基因组自包含，clone 即回滚，避免 Graph 内部状态纠缠 |
| Mutation trait + 注册表 | 可插拔设计，添加新变异不修改 Evolution（EXAMM/LayerNAS 标准做法） |
| Pareto 种群 + NSGA-II | 多目标（primary↑ + cost↓）种群搜索，维护全局 Pareto archive，天然支持多样性和 complexity-accuracy 权衡 |
| rayon 并行评估 | offspring 的 build→train→evaluate 通过 map_init 并行化，每个 worker 独立 materialize task 避免数据冗余复制 |
| 数据驱动 auto_constraints | 用户无需配置约束，系统根据任务维度自动推导合理搜索空间 |
| 两阶段训练+变异切换 | 解决搜索效率 vs 评估精度矛盾；Phase1 快速探索拓扑，Phase2 精炼超参 |
| 随机爆发初始化 | 复用现有变异基础设施，零新增代码产生多样化初始候选 |
| `>=` 非严格接受 | 解决 stepping stone 问题（XOR 等任务需多步结构变化才能突破） |
| Fitness 驱动（非 loss） | 通用性：fitness 在所有范式中有意义；loss 不可跨架构比较 |
| Tiebreak 分离（独立字段） | primary 保持纯指标值，避免 epsilon 融合污染日志和 target_metric 比较 |
| TrainingConfig 绑定 Genome | 架构与训练超参数耦合（NAS-HPO-Bench-II 证实），联合搜索优于分离 |
| Batch size 是 Task 层职责 | EvolutionTask::train() 不感知 batch；监督/RL 各自管理喂数据策略 |
| 延迟实例化（TaskSpec） | `supervised()` 无错构造，数据验证和 Task 创建延迟到 `run()`，保持 builder 链零 boilerplate |
| 单一 StdRng 贯穿 | seed 完整控制所有随机性，可复现；未来可无痛升级为 per-phase seed |
| `run()` 演化未达标不是 Err | 搜索未达标用 EvolutionStatus 区分，Err 仅用于数据验证失败和系统错误 |
| Log ladder 学习率变异 | 单 genome + rollback 下，离散台阶避免冗余值、便于回访、日志可读 |
| Optimizer 切换 + lr band snap | Adam/SGD 有效 lr 范围不同，裸切换几乎必被回滚 |
| Graph 不暴露给用户 | 抽象一致性：演化是 AutoML 层 API，Graph 是计算图层；封装后内部可自由重构 |
| **ONNX 转换经过 GraphDescriptor 中心 IR** | ONNX 不直接与 NetworkGenome 或 Graph 交互，所有转换统一经过 GraphDescriptor，保持单一转换枢纽 |
| **ONNX 模块位于 graph 层而非 evolution 层** | ONNX 是通用模型格式，不专属演化；手动训练模型同样可以导入导出 |
| **`onnx-rs` 零依赖解析** | 纯 Rust protobuf 解析，不引入 C++ ONNX Runtime 或 prost 代码生成，符合项目"无 C++ 绑定"原则 |
| **不支持的 ONNX 算子必须明确报错** | 不允许静默忽略或降级替换，返回包含 op_type 和位置信息的 `OnnxError` |
| **训练节点导出时自动剔除** | 导出推理图时自动过滤 loss/target 节点，无需用户手动构建推理子图 |

---

## 11. 扩展指南

### 11.1 添加新层类型

1. 在 `NodeTypeDescriptor` 中添加新变体（已有：BatchNormOp、LayerNormOp、RMSNormOp、Dropout 等）
2. 在 `migration.rs` 中定义 `expand_xxx()` 模板展开函数（如 `expand_batch_norm` = gamma + BN_op + Mul + beta + Add）
3. 在 `NodeBlockKind` 中添加对应变体，并在 `infer_block_kind()` 中添加识别规则
4. 在 `create_insert_nodes()` 中纳入新类型的随机生成逻辑（或在 `InsertAtomicNodeMutation` 中添加原子级插入）
5. 在 `backfill_node_group_tags()` 中添加可视化标签
6. 在 `repair_param_input_dims_inner()` 中处理级联形状修复（若新层含可学习参数）
5. （可选）在 `LayerConfig` 中添加用户 DSL 入口，并在 `migrate_to_node_level()` 中处理迁移

### 11.2 添加新变异操作

1. 实现 `Mutation` trait
2. 在 `phase1_registry()`/`phase2_registry()` 中注册（或由用户通过 `with_mutation_registry()` 注册）

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
- **Conv2d** `{ out_channels, kernel_size, stride, dilation }`：2D 卷积，支持 stride（空间降维）和 dilation（空洞卷积，扩大感受野而不增加参数）。same padding 策略，padding 自动推导
- **ConvTranspose2d** `{ out_channels, kernel_size, stride, output_padding }`：2D 转置卷积（上采样），将特征图空间尺寸放大。支持 ONNX 双向映射和 descriptor rebuild
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

### 11.6 ONNX 双向桥接（导入 + 导出）

`only_torch` 支持 ONNX（Open Neural Network Exchange）双向互操作，使得模型可在不同框架之间无缝流转。

**完整的模型互通全景图**：

```
PyTorch / TensorFlow / 其他框架
        ↓ 导出 .onnx
  ┌─────────────────────────────────────────────────┐
  │              only_torch 统一 IR 层               │
  │                                                 │
  │   .onnx ──→ GraphDescriptor ←── .otm（原生格式） │
  │                   ↕                              │
  │     ┌─────────────┴─────────────┐                │
  │     ↓                           ↓                │
  │  Graph（手动训练/推理）    NetworkGenome（演化）    │
  │     ↓                           ↓                │
  │  GraphDescriptor ──→ .onnx / .otm 导出           │
  └─────────────────────────────────────────────────┘
        ↓ 导出 .onnx
  ONNX Runtime / TensorRT / 其他部署环境
```

**导入 API**（从外部框架导入）：

```rust
// 导入为可推理的 Graph（手动模式用户）
let rebuild = Graph::from_onnx("model.onnx")?;
rebuild.inputs[0].1.set_value(&input)?;
rebuild.graph.forward(&rebuild.outputs[0])?;

// 导入为 NetworkGenome（演化种子）
let genome = NetworkGenome::from_onnx("model.onnx")?;
```

**导出 API**（导出供外部部署）：

```rust
// 从手动训练的 Graph 导出
graph.export_onnx("my_model.onnx", &[&output])?;

// 从演化结果导出
result.export_onnx("evolved_model.onnx")?;
```

**架构**：所有 ONNX 转换都经过 `GraphDescriptor` 中心 IR。ONNX 模块位于 `src/nn/graph/`（通用模型格式，不专属演化），包含四个文件：

| 文件 | 职责 |
|------|------|
| `onnx_import.rs` | 四层流水线：解析 → 符号表 → 算子映射 → 装配 |
| `onnx_export.rs` | 三层流水线：分类过滤 → AST 构建 → 编码输出 |
| `onnx_ops.rs` | 双向算子映射表（ONNX OpType ↔ NodeTypeDescriptor） |
| `onnx_error.rs` | `OnnxError` 错误类型（不支持的算子、数据类型等明确报错） |

**依赖**：使用 `onnx-rs`（零外部依赖，纯 Rust ONNX protobuf 解析/编码），符合项目"纯 Rust、无 C++ 绑定"设计理念。

**支持范围**：覆盖 opset 13–21（PyTorch 1.x–2.x 主流导出范围），支持 30+ 种算子的双向映射，包括所有主要激活函数、算术运算、卷积/池化、归一化、循环单元等。训练专用节点（loss/target）在导出时自动剔除，仅导出推理子图。不支持的算子返回包含 op_type 名称和位置信息的明确错误。

**互操作验证**：往返一致性测试（`only_torch → ONNX → only_torch` 推理数值一致）和 PyTorch 交叉验证测试（PyTorch 导出 ONNX → `only_torch` 导入 → 推理结果对齐）均已通过。

### 11.7 未来方向

#### 变异粒度说明与演化模式

当前系统在存储层统一使用节点粒度表示（`Vec<NodeGene>`），但所有变异操作的实际粒度仍为 **模板块（Block）级别**——`InsertLayer` 调用 `expand_linear()` / `expand_conv2d()` / `expand_rnn()` 等函数，一次性插入整个多节点模板（如 Linear = MatMul + Add + 2×Parameter）。这相当于在"器官"级别做移植，而非在"细胞"级别做增减。

**唯一内核表示**：系统只有 NodeLevel 一种运行时表示。所谓"层级变异"和"节点级变异"并非两套独立系统，而是同一个 NodeLevel DAG 上不同粒度的操作——模板块变异（InsertLayer 等）一次操作多个节点的模板组，原子节点变异（InsertAtomicNode）一次操作单个节点，循环边变异（AddRecurrentEdge / RemoveRecurrentEdge）操作节点间的时延连接。三者在同一个 `Vec<NodeGene>` 上共存，通过 `MutationRegistry` 的权重配比控制使用比例。无需维护"层级模式"与"节点模式"的切换——粒度是变异操作的属性，不是基因组的属性。

下述路线图旨在：（1）增强 Spatial 域能力解决当前 CNN 演化瓶颈；（2）将变异粒度逐步扩展到真正的单节点级别。

#### 阶段 A：Spatial 域增强 + 通用度量 ✅ 已完成

| 方向 | 状态 | 实现说明 |
|---|---|---|
| Conv-BN-ReLU 复合模板 | ❌ 已移除 | 实测表明演化更倾向于独立发现最优激活组合（如 LeakyReLU + Swish），硬编码 Conv+BN+ReLU 限制了搜索多样性。改为依赖现有的独立插入机制：30% 概率插入任意激活函数 + 100% Conv2d 作为 Spatial 域默认插入。BatchNorm 的独立插入可在后续按需加入 |
| Spatial 域初始种子增强 | ✅ | `minimal_spatial` 从 `[Flatten, Linear]` 改为 `[Conv2d(in_ch→8,k=3), Pool2d(Max,2,2), Flatten, Linear]`——从已知有效的 CNN 起点出发，Pool2d 将空间尺寸减半控制 Flatten 后特征维度 |
| Conv 超参变异扩展（stride） | ✅ | `MutateStrideMutation`：在 stride (1,1) 和 (2,2) 之间切换，允许卷积层自身进行空间降维。已在 Phase 1/2 注册表中注册（权重 0.06）。Dilation 暂缓——底层 Conv2d 节点尚未实现 dilation 前向 |
| SizeConstraints 空间域调优 | ✅ | `auto()` 空间模型改用"双层 Conv + 2×Pool + FC(64) 头"参考基线：MNIST `max_total_params` ≈ 220K、`max_hidden_size` = 256（channels 上限）、`max_layers` = 20。`fc_base` 假设至少 2 次 stride-2 pool 降低 Flatten 维度，避免 Linear 参数爆炸 |
| FLOPs 作为 complexity metric | ✅ | `ComplexityMetric::FLOPs` + `NetworkGenome::total_flops()`：per-node FLOPs 估算覆盖 MatMul（2×out×in）、Conv2d（2×out×Cin×kH×kW）、BatchNorm（4×elements）、LayerNorm（5×elements）、RMSNorm（3×elements）、Pool、激活等 |
| Conv2d resize 修复（额外发现） | ✅ | 修复 `resize_conv2d_out` 和 `repair_param_input_dims_inner` 中 Conv2d bias/gamma/beta 参数形状更新错误——通过 Conv2d 节点的父边关系精确定位 kernel 参数，避免误修改同 block 内的 BN 参数 |

#### 阶段 B：NEAT/EXAMM 级别——Flat & Sequence 域的单节点演化 ✅

覆盖非空间世界（Flat + Sequence），目标是达到与 NEAT（2002）/ EXAMM（2019）同等的变异粒度。模板块变异（InsertLayer 等）继续保留，作为"大步跳跃"的探索手段；原子节点变异作为"精细雕刻"的新增能力，两者通过注册表权重共存。

| 方向 | 状态 | 说明 |
|---|---|---|
| InsertAtomicNode | ✅ | 在主路径两块之间插入**单个**激活函数节点（ReLU/Tanh/Sigmoid 等 15 种，85%）或 **Dropout 节点**（15%，p∈{0.1,0.2,0.3,0.5}）。NEAT "Add Node" 的等价操作。保护输出头、避免连续激活/连续 Dropout、形状自动推导。Phase 1/2 权重均为 0.10 |
| 归一化层插入（B 补强） | ✅ | InsertLayer 新增 10% 概率插入归一化模板块：**BatchNorm**（Flat/Spatial 域，5 节点：gamma + BN_op + Mul + beta + Add）、**LayerNorm**（Flat/Sequence 域，5 节点：gamma + LN_op + Mul + beta + Add）、**RMSNorm**（Flat/Sequence 域，3 节点：gamma + RMSNorm_op + Mul）。归一化块为 shape passthrough（不改变维度），不可 resize（Grow/Shrink 自动跳过），参数形状随上游维度自动级联修复。避免连续归一化块 |
| 通用循环边（RecurrentEdge） | ✅ | EXAMM 风格 edge-based 循环连接——`NodeGene::recurrent_parents` 列表记录 `(source_id, weight_param_id)` 对。运行时语义：`target_input += W @ prev_activation[source]`。支持自回路和跨节点循环。与 cell-based 循环（CellRnn/LSTM/GRU）互斥（范式排他） |
| AddRecurrentEdge / RemoveRecurrentEdge | ✅ | 序列模式专属变异：添加/删除循环边及其权重参数节点。Phase 1 权重 0.08/0.04，Phase 2 权重 0.08/0.04。非序列基因组自动跳过。删除时级联清理孤立权重参数 |
| GenomeAnalysis 循环边验证 | ✅ | 8 条合法性不变量：悬空源引用、无效权重参数、形状兼容性（`[target_dim, source_dim]`）、序列模式要求、范式互斥（edge-based vs cell-based）。所有变异的 `apply()` 均通过 post-analysis 检查 |
| build_recurrent_from_nodes | ✅ | 时间步展开构图路径：创建 `[batch, seq_len, features]` 输入 → 共享 Parameter Var → 逐时间步按拓扑序计算 + 注入循环贡献 → 堆叠输出。支持 BPTT 梯度传播 |
| 循环边 FLOPs 估算 | ✅ | 每条循环边：`steps × (2 × target_dim × source_dim + target_dim)` FLOPs。已集成到 `total_flops()` |
| ONNX 导出屏蔽 | ✅ | 含 edge-based 循环边的基因组不支持 ONNX 导出（展开图含动态时间步共享权重），返回明确错误提示使用 .otm 格式 |
| 级联清理 | ✅ | `remove_block()` 删除引用已删节点的循环边，同时回收孤立权重参数节点及其快照 |

#### 阶段 C：EXACT 级别——Spatial 域的 Feature Map 粒度演化 ✅

覆盖空间世界（Spatial），达到与 EXACT（2017）同等的变异粒度。

| 方向 | 状态 | 说明 |
|---|---|---|
| ConvTranspose2d（转置卷积） | ✅ | 新 RawNode + Layer + ONNX 映射，支持上采样结构演化 |
| Conv2d Dilation（空洞卷积） | ✅ | 扩展 Conv2d 参数，im2col 支持 dilation 采样间隔 |
| Feature Map 粒度表示 | ✅ | `NodeGene.fm_id` 字段 + FM 辅助函数（`fm_ops.rs`） |
| Conv2d → FM 分解迁移 | ✅ | `migrate_conv2d_to_feature_maps` + `migrate_to_fm_level` 增量迁移 |
| FM-aware Builder | ✅ | 稀疏 FM 子图构图 + 全连接优化（合并为单个 Conv2d）+ Concat→Flatten 过渡 |
| 10 种 FM 级别变异 | ✅ | 结构变异 5 种 + 参数变异 5 种，已注册到 Phase 1/2 注册表 |
| 集成 | ✅ | InsertLayer FM 迁移、Grow/Shrink FM 边过滤、旧模型自动兼容 |

**FM 级别变异表**：

| # | 变异 | 类型 | Phase 1 权重 | Phase 2 权重 |
|---|------|------|------------|------------|
| 1 | AddFeatureMap | 结构 | 0.08 | 0.04 |
| 2 | RemoveFeatureMap | 结构 | 0.04 | 0.04 |
| 3 | AddFMEdge | 结构 | 0.06 | 0.06 |
| 4 | RemoveFMEdge | 结构 | 0.04 | 0.04 |
| 5 | SplitFMEdge | 结构 | 0.06 | 0.04 |
| 6 | ChangeFMEdgeType | 参数 | 0.04 | 0.04 |
| 7 | MutateFMEdgeKernelSize | 参数 | 0.04 | 0.06 |
| 8 | MutateFMEdgeStride | 参数 | 0.04 | 0.04 |
| 9 | MutateFMEdgeDilation | 参数 | 0.02 | 0.04 |
| 10 | ChangeFeatureMapSize | 参数 | 0.04 | 0.02 |

#### 两个世界的粒度差异

```
非空间世界（Flat / Sequence）            空间世界（Spatial）
┌──────────────────────────┐      ┌──────────────────────────┐
│  可统一到单神经元粒度        │      │  最低只能到 feature map 粒度 │
│                          │      │  （因卷积的权重共享约束）     │
│  • 权重 = 独立的标量边      │      │                          │
│  • 节点 = 单个激活/计算单元  │      │  • 卷积核 = 边（含权重共享） │
│  • 循环边 = 时序反馈        │      │  • Feature map = 节点     │
│                          │      │  • 池化 = 节点属性         │
│  参考：NEAT, EXAMM        │      │  参考：EXACT              │
│  阶段 B 目标              │      │  阶段 C 目标              │
└──────────────────────────┘      └──────────────────────────┘
         通过 Flatten 连接：Spatial 输出展平后进入 Flat 域
```

#### 阶段 D（候选）：新算子扩展——丰富可演化算子集

阶段 D 与阶段 C **完全正交**——它不修改 FM 表示或变异框架，只是向系统注入新的"零件"类型。阶段 C 完成后的 `MutationRegistry` + `NodeTypeDescriptor` 可插拔设计天然支持新算子接入。按实际需求选择性实施。

| 方向 | 难度 | 与 Phase C 关系 | 说明 |
|------|------|----------------|------|
| **Deformable Conv2d** | 中 | 正交，新 NodeTypeDescriptor 变体 | 采样位置可学习的卷积。需要一个"偏移预测子网络"为每个采样点生成 (dx, dy) 偏移。实现为新 RawNode + FM 边类型扩展（`ChangeFMEdgeType` 变异自动覆盖）。应用：目标检测、语义分割中的几何变形建模 |
| **Dynamic Conv2d** | 中 | 正交，新节点类型 | kernel 权重由输入动态生成（通过一个小型 attention 或线性层），而非静态参数。实现为新 RawNode + 专属模板扩展。应用：轻量级模型中用少量参数表达更丰富的卷积行为 |
| **Attention / Transformer 算子集** | 高 | 正交，新算子集 | MultiHeadAttention（已有 Layer 实现，但未纳入演化算子集）、Positional Encoding、LayerNorm（已有）。需要定义新的域链规则或扩展 Flat 域的语义，使 Transformer block 可被演化拓扑搜索发现。应用：NLP、Vision Transformer |
| **Segmentation 任务支持** | 中 | 部分交集，需新 loss 类型 | 语义分割是任务类型扩展，不是架构搜索能力扩展。Phase C 的 ConvTranspose2d + FM 跨层连接已提供 U-Net/FPN 所需的架构搜索能力。额外需要：Dice Loss / Focal Loss 等像素级 loss、TaskMetric::IoU / mIoU、输出格式从 `[batch, classes]` 扩展到 `[batch, classes, H, W]` |

**接入模式**（以 Deformable Conv2d 为例）：

```rust
// 1. 新增 NodeTypeDescriptor 变体
DeformableConv2d { stride: (usize, usize), padding: (usize, usize), deformable_groups: usize }

// 2. 实现 RawNode 前向/反向（可变采样位置的 im2col）

// 3. 在 AddFMEdge 变异中作为可选边类型自动可用
//    ChangeFMEdgeType 变异可在 conv ↔ deformable_conv ↔ pool ↔ deconv 间切换

// Phase C 的 FM 表示、builder、其他变异完全不变
```

#### 阶段 E（候选）：搜索效率优化——加速演化评估

阶段 E 关注"搜索速度"而非"搜索能力"。与阶段 A–D 的功能扩展正交，是性能优化方向。

| 方向 | 难度 | 与 Phase C 关系 | 说明 |
|------|------|----------------|------|
| **ENAS 式权重共享** | 高 | 解耦，效率优化 | 不同基因组共享部分权重（当重叠子图结构相同时复用已训练权重），减少每代 offspring 的训练成本。当前每个 offspring 独立 build→train→evaluate，权重共享可将训练开销降低一个数量级。需要维护一个"超网络权重库"，按子图结构哈希索引 |
| **Surrogate 模型** | 中 | 解耦，效率优化 | 用一个轻量级模型（如 GBM / 小型 NN）预测基因组的 fitness，跳过昂贵的 train+evaluate。仅对预测 fitness 有潜力的候选执行完整评估。需要积累足够的 (genome_features → fitness) 训练数据 |
| **分布式并行演化** | 中 | 解耦，工程优化 | 当前 rayon 并行限于单机多核。扩展到多机：每台机器运行一个子种群（island model），定期交换精英个体。需要序列化/反序列化 genome + fitness 的网络通信层 |
| **DARTS 混合搜索** | 高 | 完全不同的搜索范式 | 可微分架构搜索：用连续松弛 + 梯度下降搜索架构（不经过 mutation/selection）。与演化路线完全不同的技术方向。可作为互补方案——DARTS 快速收敛到局部最优，演化跳出局部最优。实现需要全新的搜索引擎模块 |

**优先级建议**：

```
阶段 C（已完成）   →  阶段 F（已完成）    →  阶段 D（按需）     →  阶段 E（长期）
EXACT 级 FM 演化       信号与预算修复           新算子多样性扩展         搜索效率优化
├─ ConvTranspose2d     ├─ Net2Net 变异 ✅      ├─ Deformable Conv     ├─ 权重共享加速
├─ Dilation            ├─ Cell 权重迁移 ✅     ├─ Attention 算子集     ├─ Surrogate 模型
├─ FM 分解             ├─ Learning-speed ✅    ├─ Segmentation 任务    ├─ 分布式演化
├─ 10 种变异           └─ ASHA 多保真 ✅       └─ Dynamic Conv        └─ DARTS 混合搜索
└─ Builder 优化

已完成                已完成                   按需添加                长期方向
（空间域基础设施）       （已修复流程内在缺陷）     （扩展算子多样性）        （扩展搜索效率）
```

> **说明**：阶段 A–C 建立各域变异基础设施；阶段 F 修复了评估与继承两条链路的内在缺陷（破坏性变异 + 弱选择信号 + 均匀低预算）。A–C、F 全部落地并默认启用后，当前演化流程已自洽。阶段 D 的所有新算子都是"插件"，通过 `MutationRegistry` 和 `NodeTypeDescriptor` 自然接入；阶段 E 的效率优化不影响搜索能力，可在任何时间点独立引入。

#### 阶段 F：流程修复——function-preserving 变异 + 多保真评估 ✅ 已完成

阶段 F 不引入新算子、不改变搜索空间，而是修复演化**评估与继承**两条链路的内在缺陷。动机来自 parity-8 序列演化实验：传统 `Rnn(16)+150 epoch` 可以稳定到达 95%+，但演化早期版本卡在 50–60% 随机猜测区间。根因不是算力不足，而是：

1. **破坏性变异**：`GrowHiddenSize` 把新增维度初始化为随机值，父代已学到的权重被部分重置，Lamarckian 继承失效。
2. **悬崖型 fitness landscape**：parity 类任务在越过临界点前 accuracy ≈ 随机猜，Phase 1 的 3–10 epoch 快速评估看到的全是噪声，NSGA-II 选择压力失效。
3. **均匀低预算评估**：所有候选获得同等小预算，头部候选没机会拿到"越过悬崖"所需的训练量。

| 方向 | 状态 | 适用域 | 实现说明 |
|------|------|--------|------|
| **Net2Net function-preserving 变异** | ✅ | Flat / Sequential / Spatial 三域通用 | `src/nn/evolution/net2net.rs` 实现 `apply_widen_to_snapshots`——`GrowHiddenSize` 扩容时新增列复制已有列 + 小扰动，下游消费者行按复制次数缩放，RNN/GRU/LSTM 的 `W_hh` 用 `gather_along_axis_scaled(axis=0, counts)` + `gather_along_axis(axis=1)` 的两轴变换保持前向等价。覆盖 Linear / Conv2d / RNN / LSTM / GRU + 下游 Linear/Conv2d/RNN 消耗端 + BN/LN/RMS pass-through + Flatten 跨域 + SkipAgg 放弃。参数预算回滚路径不污染快照 |
| **Cell 类型切换权重迁移** | ✅ | Sequential 域 | `src/nn/evolution/cell_migration.rs` 实现 `migrate_cell_weights()`，覆盖 6 种迁移（RNN↔LSTM / RNN↔GRU / LSTM↔GRU）。特征门保留权重（g/n gate），饱和门用 `W=0 + bias=±6` 使 σ 饱和为 1/0，信号路径逼近原 cell。`MutateCellType` 在删除旧节点前采集旧快照，新节点 commit 后调用迁移并合并进新 param_ids，避免全部权重重初始化 |
| **Learning-speed fitness proxy** | ✅ | 悬崖型任务（序列 / RL）优先 | `FitnessScore` 新增 `primary_proxy: Option<f32>` 字段，`ProxyKind::LossSlope` 由 `SupervisedTask` 在训练过程中记录 loss 曲线并计算下降斜率。NSGA-II 选择时主指标处于 plateau（区分度低于阈值）则用 proxy 打破平局。默认启用，可 `.with_primary_proxy(None)` 关闭 |
| **ASHA 多保真评估** | ✅ | 三域通用 | `AshaConfig` 默认 `rung_epochs=[1,2,4], eta=3`：rung 0 所有 offspring 训练 1 epoch，保留 top 1/3；rung 1 幸存者再训练 2 epoch，保留 top 1/3；rung 2 最后训练 4 epoch。Lamarckian 权重快照天然支持跨 rung 继续训练。头部候选累积获得 ≥10× 原均匀预算的训练量，总预算基本持平。默认启用，可 `.with_asha(None)` 关闭 |
| **Task-aware minimal genome** | ✅ | 序列域优先 | `minimal_sequential` 从 hidden = output_dim 提升为 `max(4, output_dim)`，解决 parity 类问题状态容量不足 |
| **MetricReport 评估报告** | ✅ | 三域通用 | `FitnessScore::report` 保存附加指标；默认按任务类型报告 Accuracy/Precision/Recall/F1、R²/MSE/MAE/RMSE 或多标签 loose/strict accuracy。报告指标只用于观测，不进入 `objective_point()`、target 或 archive 收敛判断 |

**优先级与完成记录**：

1. ✅ Net2Net `GrowHiddenSize`（F1）——为所有带维度扩展的变异解锁 Lamarckian 继承
2. ✅ Cell 迁移（F2）——`MutateCellType` 不再是"丢掉权重重新开始"
3. ✅ Learning-speed proxy（F3）——让悬崖型任务立即拿到可用选择信号
4. ✅ ASHA 多保真评估（F4）——系统性提升头部候选的训练预算

阶段 F 完成后，fitness 信号已足够可靠，后续可进入阶段 D（新算子扩展）或阶段 E（效率优化 / surrogate）。

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

### MNIST (input=1@28×28, output_dim=10, 1000 train / 500 test)

```
[Gen  0] 爆发初始化：λ=4 个候选，每个对 minimal_spatial 施加 K=5 次随机变异
         初始种子：Conv2d(1→8,k=3) → Pool2d(Max,2,2) → Flatten → [Linear(10)]
         候选可能产生：
         - Conv2d(1→8,k=3) → Pool2d → Conv2d(8→16,k=3) → Flatten → [Linear(10)]
         - Conv2d(1→16,k=5) → Pool2d → Flatten → Linear(32) → [Linear(10)]
         - Conv2d(1→8,k=3) → Pool2d → Conv2d(8→8,k=3,s=2) → Flatten → [Linear(10)]
         → 选 fitness 最高者作为 best_genome

  Phase 1（Gen 1~70）：Pareto 种群搜索 + FixedEpochs 快速训练 + 结构探索变异权重
  Phase 2（Gen 71~100）：Pareto 种群搜索 + UntilConverged 充分训练 + 超参调优变异权重
```

自适应约束（`SizeConstraints::auto()`）为 MNIST 推导：`max_total_params` ≈ 220K、`max_hidden_size=256`（channels 上限）、`max_layers=20`、`min_hidden_size=16`。`fc_base` 假设至少 2 次 stride-2 pool（空间尺寸 /4），FC 隐藏层宽度 64，防止 Flatten 后 Linear 参数爆炸。独立的 Conv2d + 激活函数 + Pool2d 插入 + stride 变异 + kernel size 变异联合搜索高效 CNN 架构。

---

## 附录 B：参考文献

1. **NEAT**（2002）：Evolving Neural Networks through Augmenting Topologies
2. **EXACT**（2017）：Large Scale Evolution of Convolutional Neural Networks Using Volunteer Computing — Feature map 粒度的 CNN 拓扑演化（节点=feature map，边=卷积核）
3. **EXALT**（2019）：Evolving Recurrent Neural Networks for Time Series Data Prediction — LSTM 拓扑演化，EXAMM 的前身
4. **EXAMM**（2019）：Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution — 6 种记忆单元的 RNN 拓扑演化，EXALT 的扩展
5. **LayerNAS**（2023）：Neural Architecture Search in Polynomial Complexity
6. **NAS-HPO-Bench-II**：联合架构-超参数搜索基准
7. **BOHB**：Robust and Efficient Hyperparameter Optimization at Scale
8. **Stitching for Neuroevolution**（2024）：权重复用加速演化训练

> 注：EXACT / EXALT / EXAMM 三者出自同一研究组（Travis Desell），代码均在 [travisdesell/exact](https://github.com/travisdesell/exact) 仓库中。EXACT 针对空间域（CNN），EXALT/EXAMM 针对序列域（RNN）。

---

## 已知问题

| 问题 | 状态 | 说明 |
|---|---|---|
| MNIST 准确率上限 ~91.5% | ✅ 已修复 | 阶段 A 已完成全部修复：Conv-BN-ReLU 模板、Conv2d+Pool2d 种子、stride 变异、SizeConstraints 空间域调优、Conv2d resize 修复 |
| Spatial 域演化运行缓慢 | 部分改善 | 已完成多项优化：(1) FM 掩码融合——构图时检测同构 FM 边并合并为单个 Conv2d 操作（`fm_ops.rs`），大幅减少 FM 级别拓扑的计算图节点数；(2) 收紧 `SizeConstraints::auto()` 的 `fc_base`（2×Pool 降维 + FC 隐藏层 64），防止 Flatten→Linear 参数爆炸；(3) `ComplexityMetric` 默认切为 FLOPs，NSGA-II 选择压力直接对准计算耗时；(4) GrowHiddenSize 权重从 0.25 降到 0.12，将探索预算让渡给 FM 级别细粒度变异；(5) BLAS 线程守卫防止多线程超订阅。当前 MNIST Debug 模式下仍偏慢（每代约 10-15 秒），进一步提速方向见阶段 E |

---

*本文档描述 only_torch 的神经架构演化模块实际实现。最后更新：2026-04-20（v14: 阶段 F 标记为已完成——Net2Net `GrowHiddenSize`、Cell 类型迁移、Learning-speed proxy、ASHA 多保真评估全部落地并默认启用；更新优先级图与可调参数清单）*

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

数据形状决定演化范式：

- 平坦 `[D]`：从 `Input(D) → [Linear(output_dim)]` 出发。
- 序列 `[seq_len, input_dim]`：从最简 RNN 出发，可演化为 LSTM / GRU，并支持变长样本（自动零填充至最大长度）。
- 空间 `[C, H, W]`：从 `Input(C@H×W) → Conv2d(8,k=3) → Pool2d → Flatten → [Linear(out_dim)]` 出发，可演化为完整 CNN。

完整示例分别见 `examples/evolution/parity_seq`、`examples/evolution/parity_seq_var_len`、`examples/evolution/mnist`。

### 1.2 多输出 / 多头监督任务

固定数量多头任务从快捷入口升级到显式 `SupervisedSpec`：

```rust
use only_torch::nn::evolution::{Evolution, SupervisedSpec, TaskMetric};

let result = Evolution::supervised_task(
    SupervisedSpec::new(train_inputs, test_inputs)
        .head_targets("quadrant", train_quadrants, test_quadrants, TaskMetric::Accuracy)
        .head_targets("radius", train_radii, test_radii, TaskMetric::R2)
        .primary_head("quadrant"),
)
.with_target_metric(0.80)
.with_seed(42)
.run()?;

let quadrant = result.predict_head("quadrant", &point)?;
let both = result.predict_heads(&["quadrant", "radius"], &point)?;
```

完整示例见 `examples/evolution/multi_head_quadrant_radius`。当前多头限定为共享输入、所有 head target 与 inputs 一一对齐；不支持不同 head 使用不同 dataset、缺标注 masking、检测/实例 matching 等协议。

### 1.3 可调参数

```rust
Evolution::supervised(train, test, TaskMetric::Accuracy)
    .with_target_metric(0.95)         // 目标指标值（默认 1.0）
    .with_seed(42)                    // 随机种子（可复现）
    .with_max_generations(200)        // 最大代数（默认 100）
    .with_population_size(16)         // NSGA-II 种群大小（默认 auto；空间分类为 8）
    .with_offspring_batch_size(12)    // 每代新候选数（默认 auto；空间分类为 8）
    .with_parallelism(8)              // 并行评估线程数（默认 auto = rayon threads）
    .with_pareto_patience(40)         // Pareto archive 收敛耐心值（默认 auto = max(20, pop*2)）
    .with_complexity_metric(ComplexityMetric::FLOPs) // inference_cost 计算方式（默认 FLOPs）
    .with_batch_size(64)              // 显式 batch size（默认自动策略）
    .with_initial_burst(2)            // 初始种群随机爆发变异次数（默认按约束自动推导）
    .with_initial_portfolio(InitialPortfolioConfig::vision_classification()) // 覆盖初始候选族
    .with_candidate_scoring(CandidateScoringConfig::heuristic()) // 覆盖启发式候选预筛
    .with_verbose(true)               // 打开详细搜索日志（默认 false）
    .with_convergence(config)         // 收敛检测配置（Phase 2 使用）
    .with_constraints(constraints)    // 网络规模约束（默认 auto 推导）
    .with_stagnation_patience(30)     // 停滞检测耐心值（默认 20）
    .with_eval_runs(3)                // 多次评估取保守值（默认 1）
    .with_mutation_registry(registry) // 自定义变异注册表（覆盖两阶段）
    .with_report_metrics([ReportMetric::F1]) // 追加评估报告指标（不参与选择）
    .with_primary_proxy(ProxyKind::LossSlope) // 学习速度代理（默认启用，传 None 关闭）
    .with_asha(AshaConfig::default()) // ASHA 多保真评估（非序列任务默认启用）
    .with_callback(my_callback)       // 自定义回调
    .run()
```

### 1.4 任务指标与报告指标

`TaskMetric` 是主目标：驱动 loss 推断、`target_metric` 判断、NSGA-II 选择和离散指标的 `tiebreak_loss`。

| TaskMetric | 自动推断的 Loss | 适用场景 |
|---|---|---|
| `Accuracy` | BCE（output_dim=1）/ CrossEntropy（output_dim>1） | 分类 |
| `R2` | MSE | 回归 |
| `MultiLabelAccuracy` | BCE | 多标签分类 |
| `BinaryIoU` / `MeanIoU` / `Dice` | BCEWithLogits / CrossEntropy | 语义分割 |

`ReportMetric` 是评估后的附加报告指标，只用于日志、回调和 `EvolutionResult::fitness.report`，不改变演化选择结果。默认报告指标按任务类型自动开启（如 `Accuracy` 默认附带 `Precision` / `Recall` / `F1`）。可通过 `.with_report_metrics([...])` 在默认集合基础上追加；不兼容当前任务类型的指标会被忽略，重复指标自动去重。底层计算复用通用 `src/metrics/` 模块，手写模型也可以直接调用同一套指标函数。

### 1.5 Batch Size 自动策略

用户无需关心 batch size——`SupervisedTask` 根据数据量自动选择：

| 数据量 | 策略 |
|---|---|
| ≤ 128 | full-batch |
| 129–10000 | batch_size = 64 |
| > 10000 | batch_size = 256 |

可通过 `.with_batch_size()` 覆盖。

---

## 2. 演化主循环

### 2.1 Genome-Centric 架构 + Pareto 种群搜索

系统采用 **Genome-Centric** 路线：每代从基因组（`NetworkGenome`）重建计算图，而非直接修改图。

采用 **Pareto 种群搜索 + NSGA-II 选择**策略：维护一个种群（默认 12–32 个体，auto = rayon 线程数 clamp 到 [12, 32]），每代通过二元锦标赛从种群中采样父代、变异生成 offspring，然后用 NSGA-II 环境选择从 parents ∪ offspring 中保留最优 `population_size` 个。种群策略维护多样性、天然支持多目标优化（primary ↑ + inference_cost ↓），跳出局部最优概率显著高于单谱系搜索。

**全局 Pareto Archive**：独立于种群维护所有非支配解。Archive 中的成员在目标空间互不支配。达标检查从 archive 而非种群中查找，确保不遗漏高质量解。

**并行评估**：offspring 的 build → train → evaluate 流水线通过 rayon 并行化。`with_parallelism(n)` 创建独立的 `rayon::ThreadPool`，真正控制评估并发度。训练 / 测试数据在 `SupervisedTask` 构造时一次性 stack 为 `Arc<Tensor>`，`TaskRuntime::clone()` 只复制 Arc 指针（O(1)），每个 rayon 工作线程通过 `map_init` 克隆轻量 task 副本。`Graph` 使用 `Rc<RefCell<>>`（!Send），但并行安全因为每个 worker 通过 `genome.build()` 创建独立的本地 Graph。

```
自适应约束：用户未指定 with_constraints() 时，自动从数据维度推导 SizeConstraints
初始结构：默认从 minimal genome 出发；空间任务默认混入多个合理初始候选族
随机爆发初始化：对非候选族种子个体施加 K 次随机变异（默认 K = max(2, min(8, max_layers/2))）
初始候选达标直返：初始候选族评估后若已有 primary ≥ target_metric，直接返回 TargetReached
两阶段训练预算：Phase 1（前 40% 代数）非序列任务默认 ASHA 多保真，序列任务默认 FixedEpochs；Phase 2（后 60%）UntilConverged 精炼
两阶段变异权重：Phase 1 偏向结构探索（InsertLayer↑, Grow↑）；Phase 2 偏向超参调优（MutateLR↑, MutateOptimizer↑）
Primary plateau 提前切换：若 primary fitness 连续 stagnation_patience 代未提升，立即切换到 Phase 2

初始化:
  seeds = initial_portfolio.unwrap_or([minimal_genome])
  for i in 0..population_size:
    genome_i = seeds[i % seeds.len()]
    若 i >= seeds.len(): genome_i += K 次随机变异
  evaluate_batch(初始种群)  // rayon 并行
  初始化 archive = 非支配解集合
  若 archive 中存在达标成员：直接返回最小 cost 的达标成员

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
    ├─ train(rng)        → Dropout mask、mini-batch 顺序
    ├─ evaluate(rng)     → 评估时随机性
    └─ apply_random(rng) → 变异选择和执行
```

**确定性保证**：当 `with_seed()` 被调用时，系统自动固定 `population_size`（20）和 `offspring_batch_size`（12），消除因不同机器线程数导致的 RNG 消耗序列差异。用户显式调用 `with_population_size()` / `with_offspring_batch_size()` 可覆盖此默认值。`rebuild_pareto_member()` 使用保存的 `evolution_seed` 派生子 seed（`seed ^ index`），确保同一演化结果的 Pareto 成员重建可复现。

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

- **Objective 1**：`primary`（越高越好）——任务指标（Accuracy / R² / Mean IoU 等）。
- **Objective 2**：`inference_cost`（越低越好）——模型复杂度（默认 FLOPs，可通过 `with_complexity_metric()` 切换为 ParamCount）。

选择排序规则（优先级从高到低）：

1. Pareto rank 越小越好（非支配前沿优先）。
2. 同 rank 时 crowding distance 越大越好（保持多样性）。
3. 同 rank 同 distance 时用 `tiebreak_loss` 决胜。

**Crowding distance 按 Pareto front 分层计算**：每一层非支配前沿（rank=0, rank=1, ...）内部独立计算拥挤度距离，确保各 front 的边界解获得正确的 ∞ 距离，不被跨 front 排序污染。

每代从 `parents ∪ offspring` 合并池中选出 `population_size` 个幸存者。Pareto 前沿（rank=0）的个体自动进入全局 archive。

### 3.2 FitnessScore 与多目标

```rust
pub struct FitnessScore {
    pub primary: f32,                // 主目标（越高越好）
    pub inference_cost: Option<f32>, // 推理成本（越低越好，Pareto 第二目标）
    pub tiebreak_loss: Option<f32>,  // 离散指标：test loss（NSGA-II 同 rank 同 distance 时的决胜条件）
    pub primary_proxy: Option<f32>,  // 学习速度代理（plateau 时用于打破平局）
    pub report: MetricReport,        // 附加报告指标（不参与 NSGA-II objective）
}
```

`inference_cost` 由 `compute_inference_cost()` 根据 `ComplexityMetric` 计算（默认 FLOPs——直接反映训练 / 推理耗时）。`inference_cost` 为 `None` 时退化为单目标排序。`report` 只承载可观测性信息，不进入 `objective_point()`，因此不会影响 Pareto rank、crowding distance、达标判断或 archive 收敛。

### 3.3 Initial Portfolio（初始候选族）

默认演化会从任务类型推导一个 `minimal_genome`，再通过随机爆发变异生成初始种群。空间任务会默认混入多个合理结构族；dense segmentation 任务默认混入最小分割头与稍深 dense conv head，让它们进入同一套训练、评估、Pareto 选择流程。高级用户可通过 `with_initial_portfolio(...)` 覆盖或关闭。

Initial Portfolio 只是**结构候选族初始化**，不是预训练权重 warm start。权重层面的复用仍由 Lamarckian 权重继承负责。

当前 portfolio 包含：

| Seed | 结构 | 定位 |
|---|---|---|
| `spatial_flat_mlp` | `Flatten → Linear(hidden) → Softplus → Linear(output)` | 低成本强 baseline，适合 MNIST / Fashion-MNIST |
| `minimal_spatial` | `Conv2d → Pool2d → Flatten → Linear` | TinyCNN 起点 |
| `spatial_lenet_tiny` | `Conv → ReLU → Pool → Conv → ReLU → Pool → FC → ReLU → FC` | LeNet 量级强先验 |
| `minimal_spatial_segmentation` | `Conv2d → 1×1 Conv2d head` | dense spatial-to-spatial 最小分割头 |
| `spatial_segmentation_tiny` | `Conv → ReLU → Conv → ReLU → 1×1 Conv head` | 稍深 dense 分割候选族 |
| `spatial_segmentation_unet_lite` | `Conv → Pool → Conv → ConvTranspose2d → Concat(skip) → Conv → 1×1 head` | encoder-decoder + skip concat 分割候选族 |
| `spatial_segmentation_deformable_tiny` | `Conv → ReLU → DeformableConv2d → ReLU → 1×1 head` | offset-only DeformableConv2d 分割候选族 |

Portfolio 不是写死答案：候选仍然必须经过训练和评估，只有 primary / cost 表现足够好的分支才会进入 parents 或 archive。若初始候选族已经存在达标候选，`run()` 会直接返回 `TargetReached`，避免把用户默认路径拖入随机长搜索；若未达标，则继续进入常规 mutation loop。

### 3.4 启发式候选预筛（Heuristic Candidate Scoring）

`CandidateScoringConfig::heuristic()` 是低耦合的启发式候选预筛层。空间分类任务与 segmentation 默认启用；其他任务域保持关闭，直到有稳定候选族和验证数据。它不预测最终 fitness，也不替代训练评估，只基于结构特征和 FLOPs 对候选排序，并将完整训练预算集中到 top-k。

> **术语澄清**：本节描述的预筛严格意义上**不是 surrogate**——surrogate model 在 NAS 学术语境中通常指带参数、可训练的预测模型（如 BANANAS / NASBench predictor）。本预筛只用人工启发式规则给候选排序，不预测 fitness 数值，所以命名为 `heuristic` 而非 `surrogate`。

启发式特征包括：

- depth、Conv2d / Pool2d / Linear 块数量、是否包含 Flatten；
- FLOPs；
- 是否匹配 FlatMLP、TinyCNN、LeNetTiny、dense segmentation head、encoder-decoder segmentation、DeformableConv2d segmentation 这类结构族特征。

预筛使用 family-diverse top-k：先按启发式 score 排序，再确保 `FlatMLP`、`TinyCNN`、`LenetLike`、`Hybrid`、`DenseSegHead`、`DenseSegDeep`、`EncoderDecoderSeg`、`Other` 等结构族尽量各保留至少一个候选，避免低 FLOPs 的浅层 head 在早期把其他结构族挤出完整训练评估。

ASHA 同样做轻量多样性保护：中间 rung 先保留全局 top elite，再尽量保留不同结构族代表，并通过 `min_survivors` 防止小候选池在末轮只剩单一结构族。

调试日志（`with_verbose(true)`）会输出：

- 启发式预筛生成池与保留池的结构族分布；
- 全量候选 score 分布与 kept score 分布；
- 真实评估候选的结构族分布；
- 本批最佳候选来自哪个结构族。

### 3.5 停滞检测

连续 `stagnation_patience`（默认 20）代 best primary 未严格提升后，强制从结构性变异（`InsertLayer` / `RemoveLayer` / `AddConnection` / `RemoveConnection`）中选择，打破参数变异空转。

### 3.6 Pareto 收敛与主目标平台期检测

**Archive 收敛**：全局 archive 的代表成员（primary 最高或满足 target 的最小 cost 成员）连续 `pareto_patience`（默认 `max(20, population_size * 2)`）代的 FitnessScore 未发生实质变化时（primary / inference_cost / tiebreak_loss 三个选择相关字段均在 tolerance=1e-6 内），判定 Pareto 前沿收敛，返回 `ParetoConverged` 状态。`MetricReport` 不参与该判断。

**Primary 平台期提前切换**：即使 archive 仍有细微 trade-off 变化（如新成员 inference_cost 更低但 primary 相同），primary fitness 连续 `stagnation_patience` 代未严格提升时也会触发 Phase 2 切换和结构变异强制——避免"archive 在变但主指标不涨"的隐性停滞。

### 3.7 EvolutionResult 中的 Pareto 信息

`EvolutionResult` 包含完整的 Pareto 前沿信息：

- `pareto_front: Vec<ParetoSummary>`：轻量摘要（fitness + 架构描述）。
- `rebuild_pareto_member(index)`：按索引 lazy rebuild 完整 EvolutionResult（含权重）。
- `smallest_meeting_target_index(target)`：找到满足 target 且 inference_cost 最小的成员。

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
    pub(crate) repr: GenomeRepr,                // 内部表示（NodeLevel）
}

pub(crate) enum GenomeRepr {
    /// 节点级表示（唯一正式内核表示）
    NodeLevel {
        nodes: Vec<NodeGene>,
        next_innovation: u64,
        weight_snapshots: HashMap<u64, Tensor>,  // param_innovation → Tensor
    },
}
```

**内核统一**：演化系统只以 NodeLevel 作为运行时内核。用户通过任务 API 提供数据、指标与约束；Linear / Conv2d / DeformableConv2d / RNN / Dropout 等高层结构在内部展开为带 `block_id` 的 `NodeGene` 子图。

**最小初始网络**（按数据形状自动选择）：

- 平坦数据 `[D]`：`Input(D) → [Linear(output_dim)]`
- 序列数据 `[T, D]`：`Input(seq×D) → Rnn(output_dim) → [Linear(output_dim)]`
- 空间数据 `[C, H, W]`：`Input(C@H×W) → Conv2d(8,k=3) → Pool2d(Max,2,2) → Flatten → [Linear(out_dim)]`

序列初始网络以最简单的 RNN 为起点（而非 LSTM/GRU），后续 `MutateCellType` 可升级记忆单元类型，`InsertLayer` 可在序列域插入更多循环层块。

**输出头保护**：最后一个主路径块（输出头）不可删除、不可替换。

**输出语义**：`NetworkGenome` 可记录命名 `OutputHead`，`BuildResult` 同时保留旧的 `output: Var` 默认输出和新的 `outputs: Vec<Var>` 多输出列表。旧单输出 genome 仍回退到最后一个非参数节点；多头 supervised genome 则显式写入多个 output head，避免 FM 分解 / 融合产生的无后继中间端点被误判为输出。

**多头 supervised 协议**：旧入口 `Evolution::supervised(train, test, metric)` 完全兼容，内部包装成单 head `SupervisedSpec`。多头任务使用 `Evolution::supervised_task(SupervisedSpec::new(...).head_targets(...).primary_head(...))`；每个 head 拥有独立 targets、`TaskMetric`、loss / metric 权重和 inference 标记。训练时逐 head 创建 target / loss 并按 `loss_weight` 聚合；评估时逐 head 生成 `HeadMetricReport`，`FitnessScore.primary` 默认取 primary head，未指定时回退为 weighted metric 平均。`EvolutionResult::predict()` 返回默认 head，`predict_head()` / `predict_heads()` 支持只推理所需 head。

**权重快照自包含**：`clone()` 时权重一并复制，回滚到 `best_genome` 自带权重，无需保留旧 Graph。权重按 Parameter 节点 innovation number 索引（`HashMap<u64, Tensor>`），Grow / Shrink / Replace 时按节点是否保留、形状是否兼容判断继承。

### 4.2 NodeGene

```rust
pub struct NodeGene {
    pub innovation_number: u64,            // NEAT 风格创新号
    pub node_type: NodeTypeDescriptor,     // 直接对齐图 IR 层
    pub output_shape: Vec<usize>,          // 输出形状
    pub parents: Vec<u64>,                 // 父节点创新号列表
    pub enabled: bool,                     // NEAT 风格禁用机制
    pub block_id: Option<u64>,             // 层块标识
}
```

`NodeGene` 是演化系统的最小可操作单元。`node_type` 直接使用 `NodeTypeDescriptor`（定义在 `src/nn/descriptor.rs`），实现演化层和图 IR 层的 1:1 对齐。

**block_id** 用于层块操作：同一个高层层规格（如 Linear = MatMul + Parameter + Add + Parameter）展开后的节点共享相同 `block_id`。Grow / Shrink / Remove 以层块为单位操作。`block_id = None` 表示独立节点（如单独的激活函数）。

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

### 4.4 LayerSpec（内部层规格）

```rust
pub enum LayerSpec {
    Linear { out_features: usize },
    Activation { activation_type: ActivationType },
    Rnn { hidden_size: usize },
    Lstm { hidden_size: usize },
    Gru { hidden_size: usize },
    Dropout { p: f32 },
    Conv2d { out_channels: usize, kernel_size: usize },
    DeformableConv2d { out_channels: usize, kernel_size: usize },
    Pool2d { pool_type: PoolType, kernel_size: usize, stride: usize },
    Flatten,
}
```

层规格不作为用户可见 DSL 保留。`InsertLayerMutation` 通过内部层展开函数将 `LayerSpec` 展开为对应的 `NodeGene` 层块（如 `Linear` → MatMul + Parameter + Add + Parameter）。

### 4.5 NodeLevel 跨层连接

NodeLevel 中不维护独立的跳跃边概念。任何跨层连接（残差、绕连等）都是 DAG 中的普通前向父边，由 `AddConnection` / `RemoveConnection` 变异直接操作。

新增连接时的自动保障：

- 新增连接必须满足拓扑序，不能成环。
- 如果目标已有主输入，自动插入聚合节点（`Add` / `Concat` / `Maximum`）。
- 如果形状不兼容，自动插入投影节点（Flat 域 `Linear`、Spatial 域 `1×1 Conv2d`）。
- 所有连接合法性统一由 `GenomeAnalysis` 校验。

### 4.6 ShapeDomain 域系统

`ShapeDomain` 描述张量在网络中的维度语义，用于验证层链合法性和约束变异范围：

```rust
pub enum ShapeDomain {
    Flat,      // 2D [batch, features]
    Sequence,  // 3D [batch, seq_len, features]
    Spatial,   // 4D [batch, channels, H, W]
}
```

**序列模式域链规则**（`is_domain_valid()`）：

- `Sequence → Sequence`：循环层输出 `return_sequences=true`（下一个实质层也是循环层时自动启用）。
- `Sequence → Flat`：循环层输出 `return_sequences=false`（仅返回最后一个时间步的隐藏状态）。
- `Flat → Flat`：Linear、Activation 等平坦层。
- `Flat → Sequence`：**非法**（不允许回溯至序列域）。
- 终态必须为 `Flat`（输出头需要 2D 输入）。

**空间模式域链规则**：`Spatial* → Flatten → Flat*`（详见 11.4 节）。

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

- 按拓扑序遍历 `NodeGene` 列表，从 `BasicInput` 出发逐节点推导输出形状。
- Parameter / State 节点的 `output_shape` 为权威值，计算节点的 `output_shape` 为声明值（Analysis 验证一致性）。
- 空间域维护 `(H, W)` 信息：Conv2d / DeformableConv2d 保持尺寸（same padding, stride=1），Pool2d 缩减，Flatten 归零。
- `total_params()` 从 Analysis 的 `param_count` 获取，与 `build()` 共享同一套推导逻辑。

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
| `InsertLayer` | **0.20** | ✅ | 域感知：Flat 域选 Linear / Activation / BatchNorm / LayerNorm / RMSNorm；Sequence 域选 RNN / LSTM / GRU / LayerNorm / RMSNorm；Spatial 域选 Conv2d / Pool2d / BatchNorm；segmentation 的空间保持插入有概率生成 DeformableConv2d block。归一化层以 10% 概率独立触发，不与同类连续 |
| `InsertEncoderDecoderSkip` | 0.08 | ✅ | Segmentation 专用：一次性插入 `Pool2d → Conv2d → ConvTranspose2d → Concat(skip) → Conv2d`，保持输出 H/W 与通道数不变 |
| `InsertAtomicNode` | 0.10 | ✅ | NEAT "Add Node"：在主路径两块之间插入单个激活节点（15 种激活函数随机选择，85%）或 Dropout（15%，p∈{0.1, 0.2, 0.3, 0.5}）。保护输出头、避免连续激活 / 连续 Dropout |
| `RemoveLayer` | 0.08 | ✅ | 随机移除非输出头的隐藏层 |
| `ReplaceLayerType` | 0.04 | | Activation 内部互换（13 种） |
| `GrowHiddenSize` | **0.12** | | 增大尺寸（40% +step, 40% ×1.5, 20% ×2，step = max(1, current/4)） |
| `ShrinkHiddenSize` | 0.08 | | 缩小尺寸（40% -step, 40% ×0.67, 20% ÷2） |
| `MutateLayerParam` | 0.05 | | LeakyReLU / ELU alpha，Dropout p |
| `MutateLossFunction` | 0.02 | | 切换兼容 loss（如 BCE↔MSE） |
| `MutateLearningRate` | 0.05 | | Log ladder 13 级 [1e-5, 1e-1] |
| `MutateOptimizer` | 0.02 | | Adam ↔ SGD 切换 + lr band snap |

**Phase 2：精炼（`phase2_registry`）**

| 变异 | 权重 | 结构性 | 核心逻辑 |
|---|---|---|---|
| `InsertLayer` | 0.08 | ✅ | 同上（含归一化层插入） |
| `InsertEncoderDecoderSkip` | 0.04 | ✅ | Segmentation 专用，同 Phase 1 |
| `InsertAtomicNode` | 0.10 | ✅ | 同上 |
| `RemoveLayer` | 0.08 | ✅ | 同上 |
| `ReplaceLayerType` | 0.08 | | 同上 |
| `GrowHiddenSize` | 0.15 | | 同上 |
| `ShrinkHiddenSize` | 0.15 | | 同上 |
| `MutateLayerParam` | 0.05 | | 同上 |
| `MutateLossFunction` | 0.02 | | 同上 |
| `MutateLearningRate` | **0.15** | | 同上 |
| `MutateOptimizer` | **0.08** | | 同上 |

**连接变异（NodeLevel 上操作 DAG 父边）**：

| 变异 | Phase 1 | Phase 2 | 结构性 | 核心逻辑 |
|---|---|---|---|---|
| `AddConnection` | 0.08 | 0.06 | ✅ | 选择两个满足拓扑序的节点添加父边；形状不兼容时自动插入投影 / 聚合节点 |
| `RemoveConnection` | 0.05 | 0.05 | ✅ | 移除非关键前向父边（保持图连通性） |

**循环边变异（序列模式专属）**：

| 变异 | Phase 1 | Phase 2 | 结构性 | 核心逻辑 |
|---|---|---|---|---|
| `AddRecurrentEdge` | 0.08 | 0.08 | ✅ | EXAMM 风格：在两个非叶计算节点间添加循环连接 + 权重参数节点。与 cell-based 循环互斥 |
| `RemoveRecurrentEdge` | 0.04 | 0.04 | ✅ | 移除循环边及其孤立权重参数节点 |

**FM 级别变异（Spatial 域专属，作用于 feature map 子图）**：

| # | 变异 | 类型 | Phase 1 | Phase 2 |
|---|---|---|---|---|
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

所有变异在 NodeLevel 上以层块（`block_id`）为操作单位：InsertLayer 展开完整层块，RemoveLayer 删除整个层块，Grow / Shrink 修改层块内的 Parameter 形状。

### 5.3 合法性保障

所有变异在 `is_applicable()` 和 `apply()` 中确保：

- **输出头保护**：不删除、不修改、不替换、不在其之后插入。
- **形状兼容**：通过试探式 `analyze()` 统一检测。
- **规模约束**：`max_layers` / `max_hidden_size` / `max_total_params`。
- **循环边不变量**：悬空源引用检测、权重参数节点存在性、权重形状 `[target_dim, source_dim]` 兼容性、仅 Flat / Sequence 域允许循环边、edge-based 与 cell-based 互斥、删除节点时级联清理引用、孤立权重参数节点同步回收。
- **连续 Activation 禁止**：不允许两个 Activation 相邻。
- **拓扑约束**：新增连接必须满足 DAG 拓扑序（不成环）。
- **域链合法性**：序列模型 `Sequence* → Flat*`，空间模型 `Spatial* → Flatten → Flat*`。
- **最小保护**：至少保留输出头一个层块。

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

1. **NaN / Infinity loss** → 立即停止（训练已爆炸）。
2. **FixedEpochs 模式** → 到达指定 epoch 数。
3. **max_epochs 安全上限** → 防止无限循环（默认 100）。
4. **Loss 稳定** → patience 窗口内相对变化 `(max-min)/(min.abs()+1e-8)` < tolerance。
5. **梯度消失** → 连续 patience 次 grad_norm < grad_tolerance。

默认配置：`patience=5, loss_tolerance=1e-4, grad_tolerance=1e-5, max_epochs=100`。

### 6.2 TrainingBudget

| 模式 | 用途 |
|---|---|
| `UntilConverged` | 训练到收敛或 max_epochs（Phase 2 使用） |
| `FixedEpochs(n)` | 快速筛选候选架构（序列任务 Phase 1 默认路径，或关闭 ASHA 后使用） |

演化主循环自动分配两阶段训练预算：

- **Phase 1（前 40% 代数）**：非序列任务默认 ASHA 多保真评估（`rung_epochs=[1,2,4], eta=3`）；序列任务默认 `FixedEpochs(fast_epochs)`，避免极短 early rung 在悬崖型 loss 上误淘汰候选。可通过 `.with_asha(None)` 关闭 ASHA。
- **Phase 2（后 60% 代数）**：`UntilConverged`（用户 `ConvergenceConfig`）——对有潜力拓扑做充分训练。

> **为什么序列 / 记忆任务默认跳过 ASHA**：ASHA 的前提是"低预算排序能预测高预算表现"。Flat / Spatial 任务通常较满足这一点；但 RNN / LSTM / GRU 等记忆任务在早期 epoch 往往所有候选都接近随机表现，尤其 parity / 变长序列这类悬崖型任务，1 epoch rung 容易把慢热但正确的结构提前淘汰。后续可通过 sequence-aware rung（更长 warmup、延迟淘汰、结合 loss slope proxy）重新启用。

分界点：`phase1_gens = (max_generations * 0.4).ceil()`。阶段切换时不丢弃权重快照，Phase 2 直接在 Phase 1 最佳基因组上继续。`primary` fitness 连续 `stagnation_patience` 代停滞也会提前触发 Phase 2 切换。

**Mini-batch Shuffle**：mini-batch 训练路径在每个 epoch 开始时对训练数据进行 seeded shuffle（`shuffle_mut_seeded`），确保 mini-batch 组成每 epoch 不同，改善梯度估计质量。Shuffle seed 由演化主 rng 分配，保持可复现性。

---

## 7. Lamarckian 权重继承

训练后的权重保存在 `NetworkGenome.weight_snapshots` 中。NodeLevel 基因组按 Parameter 节点的 innovation number 索引（`HashMap<u64, Tensor>`，每个 Parameter 节点独立一个 Tensor）。

下一代 `build()` 后、训练前调用 `restore_weights()`：

| 情况 | 行为 |
|---|---|
| 同 Parameter innovation_number 且形状相同 | 直接复制（继承） |
| 形状不匹配（如 GrowHiddenSize 后） | 由 `Net2Net` function-preserving 逻辑扩容，保持前向等价；不可扩容时保留新初始化值 |
| 无快照（新插入的层块带来的新 Parameter） | 保留新初始化值 |

`InheritReport` 返回 `inherited` / `reinitialized` 计数。

回滚机制利用此特性：`best_genome.clone()` 自带权重快照，恢复后 `restore_weights()` 直接还原到 best 时期的权重状态。

**Net2Net function-preserving 扩容**：`GrowHiddenSize` 扩容时新增列复制已有列 + 小扰动，下游消费者行按复制次数缩放；RNN / GRU / LSTM 的 `W_hh` 用 `gather_along_axis_scaled(axis=0, counts)` + `gather_along_axis(axis=1)` 的两轴变换保持前向等价。覆盖 Linear / Conv2d / RNN / LSTM / GRU + 下游 Linear / Conv2d / RNN 消耗端 + BN / LN / RMS pass-through + Flatten 跨域。

**Cell 类型切换权重迁移**：`MutateCellType` 在 RNN ↔ LSTM ↔ GRU 6 种切换间，特征门保留权重（g/n gate），饱和门用 `W=0 + bias=±6` 使 σ 饱和为 1/0，信号路径逼近原 cell。在删除旧节点前采集旧快照，新节点 commit 后调用迁移并合并进新 param_ids，避免全部权重重初始化。

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
- `on_population_evaluated` 每代在 NSGA-II 选择后调用，报告种群 / archive / 前沿统计。

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
├── gene.rs             NetworkGenome, GenomeRepr, TrainingConfig, TaskMetric
├── node_gene.rs        NodeGene 数据结构（innovation_number, node_type, output_shape, parents, block_id）
├── node_ops.rs         NodeBlock / NodeBlockKind / node_main_path()：节点级基因组的层块分析
├── mutation.rs         Mutation trait + MutationRegistry + 15+ 条件变异操作（NodeLevel 上以层块/原子节点/循环边为单位）
├── fm_mutation.rs      FM 级别变异操作（10 种：AddFeatureMap/RemoveFeatureMap/AddFMEdge/RemoveFMEdge/SplitFMEdge + 5 种参数变异）
├── fm_ops.rs           FM 辅助数据结构和查询函数（FMNodeInfo/FMEdgeInfo/FMSubgraphAnalysis/全连接检测/可连接对查询）
├── migration.rs        内部层展开函数 + Conv2d → FM 分解（migrate_conv2d_to_feature_maps）
├── net2net.rs          Net2Net function-preserving 扩容快照变换
├── cell_migration.rs   RNN ↔ LSTM ↔ GRU 权重迁移
├── selection.rs        NSGA-II 多目标选择 + Pareto Archive 管理（pareto_rank, crowding_distance, nsga2_select, update_archive）
├── builder.rs          Genome → GraphDescriptor → Graph 转换 + to/from_graph_descriptor + backfill_node_group_tags + Lamarckian 权重管理
├── model_io.rs         模型序列化 / 反序列化（save/load .otm 文件，仅 NodeLevel）+ ONNX 导出（export_onnx）
├── convergence.rs      ConvergenceDetector + ConvergenceConfig + TrainingBudget
├── task.rs             EvolutionTask trait + SupervisedTask + FitnessScore + MetricReport
├── callback.rs         EvolutionCallback trait + DefaultCallback（含 on_population_evaluated）
└── tests/              单元测试与集成测试

src/nn/graph/
├── onnx_import.rs      ONNX → GraphDescriptor 四层导入流水线
├── onnx_export.rs      GraphDescriptor → ONNX 三层导出流水线
├── onnx_ops.rs         ONNX OpType ↔ NodeTypeDescriptor 双向算子映射表
├── onnx_error.rs       OnnxError 错误类型（算子 / 数据类型 / 图结构 / 权重等分类报错）
└── model_save.rs       Graph 的 .otm / ONNX 保存与加载便捷接口
```

---

## 10. 关键设计决策

| 决策 | 理由 |
|---|---|
| **NodeLevel 为唯一内核表示** | 消除演化层与图 IR 层的抽象断层，`NodeGene.node_type` 直接对齐 `NodeTypeDescriptor`，1:1 映射 |
| **不暴露旧层级表示** | 用户 API 只保留任务级入口；层级演化在内部展开为 NodeLevel 子图 |
| **构图统一走 GraphDescriptor** | `NetworkGenome → GraphDescriptor → Graph::from_descriptor()`，演化和手写模型共用同一条构图管线 |
| **模型互通（.otm）** | 统一存储 NodeLevel 表示，手写模型可通过 `from_graph_descriptor()` 转为 NetworkGenome，再演化或加载 |
| Genome-Centric（每代重建图） | 基因组自包含，clone 即回滚，避免 Graph 内部状态纠缠 |
| Mutation trait + 注册表 | 可插拔设计，添加新变异不修改 Evolution（EXAMM / LayerNAS 标准做法） |
| Pareto 种群 + NSGA-II | 多目标（primary↑ + cost↓）种群搜索，维护全局 Pareto archive，天然支持多样性和 complexity-accuracy 权衡 |
| rayon 并行评估 | offspring 的 build → train → evaluate 通过 map_init 并行化，每个 worker 独立 materialize task 避免数据冗余复制 |
| 数据驱动 auto_constraints | 用户无需配置约束，系统根据任务维度自动推导合理搜索空间 |
| 两阶段训练 + 变异切换 | 解决搜索效率 vs 评估精度矛盾；Phase1 快速探索拓扑，Phase2 精炼超参 |
| 初始候选族 + 随机爆发初始化 | 保留 minimal seed 的开放性，同时允许任务合理结构族参与竞争；未达标时再通过随机爆发生成多样化候选 |
| `>=` 非严格接受 | 解决 stepping stone 问题（XOR 等任务需多步结构变化才能突破） |
| Fitness 驱动（非 loss） | 通用性：fitness 在所有范式中有意义；loss 不可跨架构比较 |
| Tiebreak 分离（独立字段） | primary 保持纯指标值，避免 epsilon 融合污染日志和 target_metric 比较 |
| TrainingConfig 绑定 Genome | 架构与训练超参数耦合（NAS-HPO-Bench-II 证实），联合搜索优于分离 |
| Batch size 是 Task 层职责 | EvolutionTask::train() 不感知 batch；监督 / RL 各自管理喂数据策略 |
| 延迟实例化（TaskSpec） | `supervised()` 无错构造，数据验证和 Task 创建延迟到 `run()`，保持 builder 链零 boilerplate |
| 单一 StdRng 贯穿 | seed 完整控制所有随机性，可复现 |
| `run()` 演化未达标不是 Err | 搜索未达标用 EvolutionStatus 区分，Err 仅用于数据验证失败和系统错误 |
| Log ladder 学习率变异 | 单 genome + rollback 下，离散台阶避免冗余值、便于回访、日志可读 |
| Optimizer 切换 + lr band snap | Adam / SGD 有效 lr 范围不同，裸切换几乎必被回滚 |
| Graph 不暴露给用户 | 抽象一致性：演化是 AutoML 层 API，Graph 是计算图层；封装后内部可自由重构 |
| **ONNX 转换经过 GraphDescriptor 中心 IR** | ONNX 不直接与 NetworkGenome 或 Graph 交互，所有转换统一经过 GraphDescriptor，保持单一转换枢纽 |
| **ONNX 模块位于 graph 层而非 evolution 层** | ONNX 是通用模型格式，不专属演化；手动训练模型同样可以导入导出 |
| **`onnx-rs` 零依赖解析** | 纯 Rust protobuf 解析，不引入 C++ ONNX Runtime 或 prost 代码生成，符合项目"无 C++ 绑定"原则 |
| **不支持的 ONNX 算子必须明确报错** | 不允许静默忽略或降级替换，返回包含 op_type 和位置信息的 `OnnxError` |
| **训练节点导出时自动剔除** | 导出推理图时自动过滤 loss / target 节点，无需用户手动构建推理子图 |
| **启发式预筛而非 learned surrogate** | 学习型 surrogate 在 toy benchmark 下样本不足、跨任务不通用、易引入预测噪声；启发式 + family-diverse top-k 已覆盖主要价值，且通用、可解释 |

---

## 11. 扩展指南

### 11.1 添加新层类型

1. 在 `NodeTypeDescriptor` 中添加新变体（已有：BatchNormOp、LayerNormOp、RMSNormOp、Dropout 等）。
2. 在 `migration.rs` 中定义 `expand_xxx()` 层展开函数（如 `expand_batch_norm` = gamma + BN_op + Mul + beta + Add）。
3. 在 `NodeBlockKind` 中添加对应变体，并在 `infer_block_kind()` 中添加识别规则。
4. 在 `create_insert_nodes()` 中纳入新类型的随机生成逻辑（或在 `InsertAtomicNodeMutation` 中添加原子级插入）。
5. 在 `backfill_node_group_tags()` 中添加可视化标签。
6. 在 `repair_param_input_dims_inner()` 中处理级联形状修复（若新层含可学习参数）。
7. （可选）在内部 `LayerSpec` 枚举中添加层规格入口，并在层展开函数中处理展开。

### 11.2 添加新变异操作

1. 实现 `Mutation` trait。
2. 在 `phase1_registry()` / `phase2_registry()` 中注册（或由用户通过 `with_mutation_registry()` 注册）。

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

例如 `RLTask` 可在 `train()` 中执行策略梯度，在 `evaluate()` 中跑 Episode 取奖励。内部扩展时，还需在 `mod.rs` 的 `TaskSpec` 枚举添加新变体（如 `RL { env, metric }`），并在 `materialize_task()` 添加对应分支 + 公开构造器（如 `Evolution::reinforcement()`）。

### 11.4 已支持的演化范式

#### 平坦数据（Flat）

最小架构 `Input(D) → [Linear(output_dim)]`。变异覆盖：层块（Linear / Activation / BatchNorm / LayerNorm / RMSNorm / Dropout）、原子节点（NEAT "Add Node"）、连接（DAG 父边）、超参（lr / optimizer）。参数粒度可达单神经元（NEAT 级别）。

#### 序列 / 记忆单元（Sequence）

最小架构 `Input(seq×D) → Rnn(output_dim) → [Linear(output_dim)]`。系统从最简 RNN 出发，`MutateCellType` 可升级为 LSTM / GRU；`AddRecurrentEdge` / `RemoveRecurrentEdge` 支持 EXAMM 风格 edge-based 循环连接（与 cell-based 互斥）。变长序列自动零填充至最大长度。

`build()` 中的 `needs_return_sequences` 逻辑：builder 在构建循环层时自动判断是否需要返回完整序列——向后扫描 resolved 层列表（跳过 Activation / Dropout），若下一个实质层也是循环层则调用 `forward_seq()`（返回 `[batch, seq_len, hidden]`），否则调用 `forward()`（仅返回最后一步 `[batch, hidden]`）。

域约束：序列模式下 skip edge 仅允许在 Flat 域内。记忆单元（RNN / LSTM / GRU）作为原子单元，不允许 skip edge 跨越或穿透 Sequence 域。

#### 卷积神经网络（Spatial）

最小架构 `Input(C@H×W) → Conv2d(8,k=3) → Pool2d(Max,2,2) → Flatten → [Linear(out_dim)]`。从一个已知有效的 CNN 起点出发，避免演化从纯 Flatten + FC 结构被迫先"发现"卷积价值。

空间层类型：

- **Conv2d** `{ out_channels, kernel_size, stride, dilation }`：2D 卷积，支持 stride（空间降维）和 dilation（空洞卷积）。same padding 策略，padding 自动推导。
- **DeformableConv2d** `{ out_channels, kernel_size, deformable_groups }`：offset-only 可变形卷积，offset 由同尺寸卷积分支预测。已接入 raw node / Layer / descriptor rebuild / NodeLevel block / segmentation portfolio；ONNX 导出暂标记为 unsupported。
- **ConvTranspose2d** `{ out_channels, kernel_size, stride, output_padding }`：2D 转置卷积（上采样）。支持 ONNX 双向映射和 descriptor rebuild。
- **Pool2d** `{ pool_type: Max | Avg, kernel_size, stride }`：2D 池化，空间降维（H/stride, W/stride），channels 不变。
- **Flatten**：空间域到平坦域的过渡层，将 `(C, H, W)` 展平为 `C*H*W`。

域系统：层序列被划分为 Spatial 域（Conv2d、DeformableConv2d、ConvTranspose2d、Pool2d；4D 张量）和 Flat 域（Linear、Activation；2D 张量）。Flatten 是唯一的域过渡层。`is_domain_valid()` 确保 `Spatial* → Flatten → Flat*` 结构（允许 0 个空间层，即 `Flatten → Flat*` 也合法）。

变异：`InsertLayer` 域感知插入（Spatial 域 Conv2d 80% / Pool2d 20%；segmentation 路径有概率生成 DeformableConv2d block），`MutateKernelSize` 切换 Conv2d kernel，`Grow / ShrinkHiddenSize` 调整 Conv2d out_channels，`MutateStride` 在 (1,1) ↔ (2,2) 之间切换，`InsertEncoderDecoderSkip` 一次性插入保持分辨率的 U-Net 风格 Pool / ConvTranspose2d / Concat / fuse Conv 局部结构。

参数粒度：空间域最低到 Feature Map 级别（受卷积权重共享约束限制）。`NodeGene.fm_id` 字段 + `fm_ops.rs` 提供 FM 子图分析，10 种 FM 级别变异（结构 5 种 + 参数 5 种）作用于 feature map 子图。

#### 多输出 / 多头任务

`SupervisedSpec` / `HeadSpec` 支持固定数量命名 head，每个 head 拥有独立 targets、`TaskMetric`、loss / metric 权重和 inference 标记。当前限制为平坦共享输入、所有 head target 与 inputs 一一对齐；空间 / 序列多头、检测变长实例、matching / NMS / mAP 协议为后续扩展方向。

### 11.5 ONNX 双向桥接（导入 + 导出）

`only_torch` 支持 ONNX 双向互操作，使得模型可在不同框架之间无缝流转。

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
  │  Graph（手动训练 / 推理）   NetworkGenome（演化）  │
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
|---|---|
| `onnx_import.rs` | 四层流水线：解析 → 符号表 → 算子映射 → 装配 |
| `onnx_export.rs` | 三层流水线：分类过滤 → AST 构建 → 编码输出 |
| `onnx_ops.rs` | 双向算子映射表（ONNX OpType ↔ NodeTypeDescriptor） |
| `onnx_error.rs` | `OnnxError` 错误类型（不支持的算子、数据类型等明确报错） |

**依赖**：使用 `onnx-rs`（零外部依赖，纯 Rust ONNX protobuf 解析 / 编码），符合项目"纯 Rust、无 C++ 绑定"设计理念。

**支持范围**：覆盖 opset 13–21（PyTorch 1.x–2.x 主流导出范围），支持 30+ 种算子的双向映射，包括所有主要激活函数、算术运算、卷积 / 池化、归一化、循环单元等。训练专用节点（loss / target）在导出时自动剔除，仅导出推理子图。不支持的算子返回包含 op_type 名称和位置信息的明确错误。

**互操作验证**：往返一致性测试（`only_torch → ONNX → only_torch` 推理数值一致）和 PyTorch 交叉验证测试（PyTorch 导出 ONNX → `only_torch` 导入 → 推理结果对齐）均已通过。

**限制**：含 edge-based 循环边的基因组不支持 ONNX 导出（展开图含动态时间步共享权重），需使用 `.otm` 格式；DeformableConv2d 暂未在 ONNX 算子映射表中支持。

### 11.6 长期可选研究方向

以下方向不是当前主线，仅在出现明确需求 / 真实大型 benchmark 暴露相应瓶颈时再启动评估：

- **ENAS 风格权重共享**：不同基因组共享部分权重，可显著降低单 offspring 训练成本；代价是会重塑 Lamarckian 继承 / 权重快照 / rollback 三套机制，并引入 supernet ranking bias。
- **分布式 island model**：多机多 island 独立演化、周期性交换 elite，需要 genome + fitness 序列化的网络通信层。
- **学习型 performance predictor**：在已有启发式预筛之上训练任务本地的小型回归 / 排名模型（即真正意义上的 learned surrogate），需先沉淀稳定可信的 audit 数据集 + 相关性验证。

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

### MNIST (input=1@28×28, output_dim=10)

```
5000 train / 1000 test
  用户代码：Evolution::supervised(...).with_target_metric(0.95).run()
  框架默认：Initial portfolio = FlatMLP + TinyCNN + LeNetTiny
  启发式预筛：生成候选池后按结构特征 + FLOPs 预筛 family-diverse top-k
  若初始候选族未达标，再进入：
    Phase 1：Pareto 种群搜索 + ASHA 多保真评估 + 结构探索变异权重
    Phase 2：Pareto 种群搜索 + UntilConverged 充分训练 + 超参调优变异权重
```

自适应约束（`SizeConstraints::auto()`）为 MNIST 推导：`max_total_params` ≈ 220K、`max_hidden_size=256`（channels 上限）、`max_layers=20`、`min_hidden_size=16`。`fc_base` 假设至少 2 次 stride-2 pool（空间尺寸 /4），FC 隐藏层宽度 64，防止 Flatten 后 Linear 参数爆炸。

### Segmentation Evolution (input=1@H×W, output=classes@H×W)

```
用户代码：Evolution::supervised(..., TaskMetric::MeanIoU).run()
框架默认：Initial portfolio = minimal dense head + dense segmentation tiny + U-Net-lite encoder-decoder
启发式预筛：按 dense / encoder-decoder segmentation 结构族 + FLOPs 预筛 family-diverse top-k
Phase 1：ASHA 多保真评估 + spatial-to-spatial / encoder-decoder skip 变异
```

DeformableConv2d 通过 `spatial_segmentation_deformable_tiny` 和 `InitialPortfolioConfig::vision_segmentation_with_deformable()` 接入演化主流程；`evolution_deformable_conv2d_segmentation` 示例显式启用 DeformableConv2d 初始族验证算子能进入搜索路径。

---

## 附录 B：参考文献

1. **NEAT**（2002）：Evolving Neural Networks through Augmenting Topologies
2. **EXACT**（2017）：Large Scale Evolution of Convolutional Neural Networks Using Volunteer Computing — Feature map 粒度的 CNN 拓扑演化（节点 = feature map，边 = 卷积核）
3. **EXALT**（2019）：Evolving Recurrent Neural Networks for Time Series Data Prediction — LSTM 拓扑演化，EXAMM 的前身
4. **EXAMM**（2019）：Investigating Recurrent Neural Network Memory Structures using Neuro-Evolution — 6 种记忆单元的 RNN 拓扑演化，EXALT 的扩展
5. **LayerNAS**（2023）：Neural Architecture Search in Polynomial Complexity
6. **NAS-HPO-Bench-II**：联合架构-超参数搜索基准
7. **BOHB**：Robust and Efficient Hyperparameter Optimization at Scale
8. **Stitching for Neuroevolution**（2024）：权重复用加速演化训练

> 注：EXACT / EXALT / EXAMM 三者出自同一研究组（Travis Desell），代码均在 [travisdesell/exact](https://github.com/travisdesell/exact) 仓库中。EXACT 针对空间域（CNN），EXALT / EXAMM 针对序列域（RNN）。

---

## 附录 C：术语对照

| 术语 | 含义 |
|---|---|
| Genome | 基因组，演化的最小可操作单元，包含完整的网络结构和权重快照 |
| NodeLevel | 节点级表示，演化系统的唯一内核表示 |
| LayerSpec | 内部层规格，用于指导节点展开，不暴露给用户 |
| Innovation Number | NEAT 风格的全局唯一编号，用于跨基因组节点匹配 |
| Block | 层块，由展开后共享 `block_id` 的多个节点构成 |
| FM (Feature Map) | 空间域中的最小可演化单元，对应一个通道 |
| Lamarckian Inheritance | 拉马克主义遗传：训练后的权重可被子代继承（与达尔文随机初始化对立） |
| ASHA | Asynchronous Successive Halving Algorithm，多保真评估策略 |
| NSGA-II | 非支配排序遗传算法 II，多目标优化的标准选择算子 |
| Pareto Archive | 全局非支配解集合，独立于种群维护 |
| Heuristic Prefilter | 启发式候选预筛层，基于结构特征 + FLOPs 排序 + family-diverse top-k；通过 `CandidateScoringConfig::heuristic()` 配置 |
| Net2Net | function-preserving 网络扩容技术，扩容前后前向输出等价 |
| Phase 1 / Phase 2 | 演化两阶段：拓扑探索 vs 超参精炼 |

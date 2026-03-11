/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化模块
 *
 * Genome-centric 层级进化：
 * - gene.rs: 基因数据结构（LayerGene, NetworkGenome, LayerConfig 等）
 * - mutation.rs: 变异操作（Mutation trait + MutationRegistry）
 * - builder.rs: Genome → Graph 转换 + Lamarckian 权重继承
 * - convergence.rs: 训练收敛检测（ConvergenceDetector）
 * - task.rs: EvolutionTask trait + SupervisedTask
 * - callback.rs: 回调接口（EvolutionCallback + DefaultCallback）
 * - selection.rs: NSGA-II 多目标选择与 Pareto Archive 管理
 *
 * Evolution 结构体是用户入口，驱动 genome-centric 主循环：
 * build → restore_weights → train → capture_weights → evaluate → accept/rollback → mutate
 */

pub mod builder;
pub mod callback;
pub mod convergence;
pub mod error;
pub mod gene;
pub mod model_io;
pub mod mutation;
pub mod selection;
pub mod task;

#[cfg(test)]
mod tests;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::{GraphError, VisualizationOutput};
use crate::tensor::Tensor;

use self::builder::BuildResult;
use self::callback::{DefaultCallback, EvolutionCallback};
use self::convergence::{ConvergenceConfig, TrainingBudget};
use self::error::EvolutionError;
use self::gene::{NetworkGenome, TaskMetric};
use self::mutation::{MutationError, MutationRegistry, SizeConstraints};
use self::task::{EvolutionTask, FitnessScore, SupervisedTask, auto_batch_size};

// ==================== EvolutionStatus ====================

/// 演化停止原因
///
/// `run()` 无论是否达标都返回 `Ok(EvolutionResult)`，
/// `status` 告诉用户"为什么停了"。
/// `Err` 仅用于真正的系统错误（Graph 构建失败等）。
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EvolutionStatus {
    /// primary >= target_metric，达标停止
    TargetReached,
    /// 到达最大代数限制（DefaultCallback.should_stop 触发）
    MaxGenerations,
    /// 自定义回调请求停止
    CallbackStopped,
    /// 所有变异操作均不可用（搜索空间耗尽）
    NoApplicableMutation,
    /// 全局 Pareto archive 连续多代未改进
    ParetoConverged,
}

// ==================== EvolutionResult ====================

/// 演化结果（搜索结果报告 + 推理句柄）
///
/// 封装了演化产出的网络，用户通过 `predict()` 进行推理，
/// 通过 `visualize()` 生成计算图可视化——无需接触 `Graph` / `Var` 等内部类型。
pub struct EvolutionResult {
    /// 内部构建结果（持有 input/output Var + Graph，支持推理和可视化）
    build: BuildResult,
    /// 最终适应度分数
    pub fitness: FitnessScore,
    /// 演化经历的总代数
    pub generations: usize,
    /// 人类可读的最终架构描述
    pub architecture_summary: String,
    /// 停止原因
    pub status: EvolutionStatus,
    /// 内部基因组（不暴露给用户，用于继续演化）
    #[allow(dead_code)]
    pub(crate) genome: NetworkGenome,
    /// Pareto 前沿摘要（公开的轻量描述，不含权重）
    pub pareto_front: Vec<ParetoSummary>,
    /// Pareto 前沿的完整基因组（含权重快照，支持 lazy rebuild）
    pareto_genomes: Vec<NetworkGenome>,
}

impl EvolutionResult {
    /// 推理：输入数据，返回模型预测
    ///
    /// 支持以下输入形状（自动添加 batch 维度）：
    /// - 平坦：`[input_dim]` → `[1, input_dim]`
    /// - 序列：`[seq_len, input_dim]` → `[1, seq_len, input_dim]`
    /// - 空间：`[C, H, W]` → `[1, C, H, W]`
    /// - 已含 batch 维度则直接使用
    ///
    /// 返回 `[batch, output_dim]`。
    ///
    /// # 示例
    /// ```ignore
    /// let result = Evolution::supervised(train, test, TaskMetric::Accuracy)
    ///     .with_target_metric(0.95)
    ///     .run()?;
    /// let predictions = result.predict(&new_data)?;
    /// ```
    pub fn predict(&self, input: &Tensor) -> Result<Tensor, EvolutionError> {
        self.build.graph.eval();

        let shaped = match input.dimension() {
            1 => input.reshape(&[1, input.size()]), // [feat] → [1, feat]
            2 if self.genome.seq_len.is_some() => {
                // [seq, feat] → [1, seq, feat]
                let s = input.shape();
                input.reshape(&[1, s[0], s[1]])
            }
            3 if self.genome.is_spatial() => {
                // [C, H, W] → [1, C, H, W]
                let s = input.shape();
                input.reshape(&[1, s[0], s[1], s[2]])
            }
            _ => input.clone(), // 已是 batch
        };

        self.build.input.set_value(&shaped)?;
        self.build.graph.forward(&self.build.output)?;

        self.build.output.value()?.ok_or_else(|| {
            EvolutionError::Graph(GraphError::ComputationError("推理时输出节点无值".into()))
        })
    }

    /// 可视化演化后的计算图（生成 .dot + .png）
    ///
    /// `base_path` 不含文件后缀，如 `"output/my_model"`。
    ///
    /// # 示例
    /// ```ignore
    /// let vis = result.visualize("output/evolution_result")?;
    /// println!("DOT: {}", vis.dot_path.display());
    /// ```
    pub fn visualize<P: AsRef<std::path::Path>>(
        &self,
        base_path: P,
    ) -> Result<VisualizationOutput, EvolutionError> {
        self.build
            .graph
            .visualize_snapshot(base_path)
            .map_err(EvolutionError::Graph)
    }

    /// 获取人类可读的架构描述
    pub fn architecture(&self) -> &str {
        &self.architecture_summary
    }

    /// 在满足 target 的 Pareto 成员中，找到 inference_cost 最小的索引
    pub fn smallest_meeting_target_index(&self, target: f32) -> Option<usize> {
        self.pareto_front
            .iter()
            .enumerate()
            .filter(|(_, s)| s.fitness.primary >= target)
            .min_by(|(_, a), (_, b)| {
                a.fitness
                    .inference_cost
                    .unwrap_or(f32::INFINITY)
                    .partial_cmp(&b.fitness.inference_cost.unwrap_or(f32::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// 按索引重建 Pareto 成员的完整 EvolutionResult（lazy rebuild）
    pub fn rebuild_pareto_member(&self, index: usize) -> Result<EvolutionResult, EvolutionError> {
        if index >= self.pareto_genomes.len() {
            return Err(EvolutionError::InvalidConfig(format!(
                "Pareto 成员索引 {} 超出范围（共 {} 个）",
                index,
                self.pareto_genomes.len()
            )));
        }
        let genome = &self.pareto_genomes[index];
        let mut rng = StdRng::from_entropy();
        let build = genome.build(&mut rng)?;
        genome.restore_weights(&build)?;
        build.graph.snapshot_once_from(&[&build.output]);
        Ok(EvolutionResult {
            build,
            fitness: self.pareto_front[index].fitness.clone(),
            generations: self.generations,
            architecture_summary: self.pareto_front[index].architecture_summary.clone(),
            status: self.status.clone(),
            genome: genome.clone(),
            pareto_front: self.pareto_front.clone(),
            pareto_genomes: self.pareto_genomes.clone(),
        })
    }
}

// ==================== ParetoSummary ====================

/// Pareto 前沿成员摘要（公开的轻量描述，不含权重）
#[derive(Clone, Debug)]
pub struct ParetoSummary {
    pub fitness: FitnessScore,
    pub architecture_summary: String,
}

// ==================== ComplexityMetric ====================

/// 复杂度度量方式（用于 inference_cost 计算）
///
/// 当前仅实现 ParamCount，预留未来扩展点：FLOPs、activation memory、真实 latency。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComplexityMetric {
    /// 参数总量
    ParamCount,
}

/// 根据复杂度度量计算 inference_cost
pub(crate) fn compute_inference_cost(
    genome: &NetworkGenome,
    metric: &ComplexityMetric,
) -> Result<f32, EvolutionError> {
    match metric {
        ComplexityMetric::ParamCount => genome
            .total_params()
            .map(|p| p as f32)
            .map_err(|e| {
                EvolutionError::Graph(GraphError::ComputationError(
                    format!("计算参数量失败: {e}"),
                ))
            }),
    }
}

// ==================== is_at_least_as_good ====================

/// 比较两个 FitnessScore：current 是否至少和 best 一样好
///
/// 比较规则由 FitnessScore 自身的 tiebreak_loss 驱动，不依赖 TaskMetric：
/// 1. primary 更高 → 接受
/// 2. primary 更低 → 拒绝
/// 3. primary 相等 → 看 tiebreak_loss（都有则越低越好，否则直接接受/中性漂移）
#[allow(dead_code)]
fn is_at_least_as_good(current: &FitnessScore, best: &FitnessScore) -> bool {
    if current.primary > best.primary {
        return true;
    }
    if current.primary < best.primary {
        return false;
    }
    match (current.tiebreak_loss, best.tiebreak_loss) {
        (Some(c), Some(b)) => c <= b,
        _ => true,
    }
}

// ==================== TaskSpec（延迟实例化）====================

/// 任务配置规范（构造器只存储原始配置，`run()` 时才实例化）
///
/// 每种学习范式对应一个变体，未来可扩展 RL / GAN / Transfer 等。
#[derive(Clone)]
enum TaskSpec {
    Supervised {
        train_data: (Vec<Tensor>, Vec<Tensor>),
        test_data: (Vec<Tensor>, Vec<Tensor>),
        metric: TaskMetric,
    },
}

impl TaskSpec {
    /// 该任务是否支持并行评估
    ///
    /// Supervised 任务基于 Tensor（Send+Sync），支持并行。
    /// 未来 RL / FFI 类任务可能需要返回 false。
    fn supports_parallel_evaluation(&self) -> bool {
        match self {
            TaskSpec::Supervised { .. } => true,
        }
    }
}

/// 实例化后的任务（`run()` 内部使用）
struct MaterializedTask {
    task: Box<dyn EvolutionTask>,
    input_dim: usize,
    output_dim: usize,
    /// 序列长度（None = 平坦输入，Some(n) = 序列输入）
    seq_len: Option<usize>,
    /// 空间输入尺寸 (H, W)（None = 非空间，input_dim 表示 in_channels）
    input_spatial: Option<(usize, usize)>,
    metric: TaskMetric,
    /// 训练样本数（用于 auto_constraints 和训练预算计算）
    n_train: usize,
}

/// 从 TaskSpec 实例化具体任务，提取维度信息并验证数据
fn materialize_task(spec: TaskSpec) -> Result<MaterializedTask, EvolutionError> {
    match spec {
        TaskSpec::Supervised {
            train_data,
            test_data,
            metric,
        } => {
            // 维度提取前的安全检查（SupervisedTask::new 做完整验证）
            if train_data.0.is_empty() {
                return Err(EvolutionError::InvalidData("训练输入不能为空".into()));
            }
            if train_data.1.is_empty() {
                return Err(EvolutionError::InvalidData("训练标签不能为空".into()));
            }
            // 检测输入数据维度：1D = 平坦，2D = 序列，3D = 空间
            let sample_ndim = train_data.0[0].dimension();
            let (input_dim, seq_len, input_spatial) = if sample_ndim == 3 {
                // 空间数据：每个样本 [C, H, W]
                let shape = train_data.0[0].shape();
                (shape[0], None, Some((shape[1], shape[2])))
            } else if sample_ndim == 2 {
                // 序列数据：每个样本 [seq_len_i, input_dim]
                let feat_dim = train_data.0[0].shape()[1];
                let max_seq = train_data
                    .0
                    .iter()
                    .chain(test_data.0.iter())
                    .map(|t| t.shape()[0])
                    .max()
                    .unwrap();
                (feat_dim, Some(max_seq), None)
            } else {
                (train_data.0[0].size(), None, None)
            };
            let output_dim = train_data.1[0].size();
            let n_train = train_data.0.len();
            let task = SupervisedTask::new(train_data, test_data, metric.clone())?;
            Ok(MaterializedTask {
                task: Box::new(task),
                input_dim,
                output_dim,
                seq_len,
                input_spatial,
                metric,
                n_train,
            })
        }
    }
}

// ==================== Evolution ====================

/// 演化主控结构体
///
/// 通过 `supervised()` 便捷构造器或手动组装创建，
/// `run()` 驱动完整的 genome-centric 演化主循环。
pub struct Evolution {
    task_spec: TaskSpec,
    target_metric: f32,
    eval_runs: usize,
    convergence_config: ConvergenceConfig,
    /// 自定义变异注册表（None = 根据任务指标自动选择默认注册表）
    mutation_registry: Option<MutationRegistry>,
    /// 用户显式指定的约束（None = 根据任务数据自动推导）
    constraints: Option<SizeConstraints>,
    seed: Option<u64>,
    /// 自定义回调（None 时使用 DefaultCallback）
    custom_callback: Option<Box<dyn EvolutionCallback>>,
    /// DefaultCallback 的最大代数（仅 custom_callback=None 时生效）
    max_generations: usize,
    /// DefaultCallback 的日志开关（仅 custom_callback=None 时生效）
    verbose: bool,
    /// 停滞耐心值：primary fitness 连续多少代未严格提升后，
    /// 强制选择结构变异（InsertLayer / RemoveLayer）以探索新拓扑
    stagnation_patience: usize,
    /// 训练 batch size（None = Task 自动策略，Some = 显式指定）
    batch_size: Option<usize>,
    /// 每代保留的种群大小（NSGA-II 环境选择后的幸存者数量）
    population_size: usize,
    /// 每代实际评估的新候选数量
    offspring_batch_size: usize,
    /// 并行评估线程数（None = auto = rayon::current_num_threads()）
    parallelism: Option<usize>,
    /// Pareto archive 连续未改进多少代后判定收敛（None = auto）
    pareto_patience: Option<usize>,
    /// 复杂度度量方式（用于 inference_cost 计算）
    complexity_metric: ComplexityMetric,
}

impl Evolution {
    // ==================== 构造器 ====================

    /// 监督学习便捷构造器
    ///
    /// 数据验证延迟到 `run()` 执行时进行，构造器本身不会失败。
    /// 使用 `MutationRegistry::default_registry` 注册 12 种默认变异。
    pub fn supervised(
        train_data: (Vec<Tensor>, Vec<Tensor>),
        test_data: (Vec<Tensor>, Vec<Tensor>),
        metric: TaskMetric,
    ) -> Self {
        Self {
            task_spec: TaskSpec::Supervised {
                train_data,
                test_data,
                metric,
            },
            target_metric: 1.0,
            eval_runs: 1,
            convergence_config: ConvergenceConfig::default(),
            mutation_registry: None,
            constraints: None,
            seed: None,
            custom_callback: None,
            max_generations: 100,
            verbose: true,
            stagnation_patience: 20,
            batch_size: None,
            population_size: rayon::current_num_threads().clamp(12, 32),
            offspring_batch_size: rayon::current_num_threads().max(12),
            parallelism: None,
            pareto_patience: None,
            complexity_metric: ComplexityMetric::ParamCount,
        }
    }

    // ==================== Builder methods ====================

    pub fn with_target_metric(mut self, target: f32) -> Self {
        self.target_metric = target;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_convergence(mut self, config: ConvergenceConfig) -> Self {
        self.convergence_config = config;
        self
    }

    pub fn with_constraints(mut self, constraints: SizeConstraints) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// 设置每代子代数量（已弃用，请使用 `with_offspring_batch_size`）
    #[deprecated(note = "请使用 with_offspring_batch_size")]
    pub fn with_offspring_count(mut self, n: usize) -> Self {
        assert!(n >= 1, "offspring_batch_size 必须 >= 1，当前值: {n}");
        self.offspring_batch_size = n;
        self
    }

    /// 设置每代保留的种群大小（NSGA-II 幸存者数量，默认 auto）
    pub fn with_population_size(mut self, n: usize) -> Self {
        assert!(n >= 1, "population_size 必须 >= 1，当前值: {n}");
        self.population_size = n;
        self
    }

    /// 设置每代实际评估的新候选数量（默认 = max(population_size, parallelism)）
    pub fn with_offspring_batch_size(mut self, n: usize) -> Self {
        assert!(n >= 1, "offspring_batch_size 必须 >= 1，当前值: {n}");
        self.offspring_batch_size = n;
        self
    }

    /// 设置并行评估线程数（None = auto）
    pub fn with_parallelism(mut self, n: usize) -> Self {
        assert!(n >= 1, "parallelism 必须 >= 1，当前值: {n}");
        self.parallelism = Some(n);
        self
    }

    /// 设置 Pareto archive 收敛耐心值（None = auto = max(20, population_size * 2)）
    pub fn with_pareto_patience(mut self, k: usize) -> Self {
        self.pareto_patience = Some(k);
        self
    }

    /// 设置复杂度度量方式（用于 inference_cost 计算）
    pub fn with_complexity_metric(mut self, metric: ComplexityMetric) -> Self {
        self.complexity_metric = metric;
        self
    }

    /// 设置自定义回调（替换 DefaultCallback）
    ///
    /// 设置后 `with_max_generations` / `with_verbose` 不再生效。
    pub fn with_callback(mut self, callback: impl EvolutionCallback + 'static) -> Self {
        self.custom_callback = Some(Box::new(callback));
        self
    }

    pub fn with_mutation_registry(mut self, registry: MutationRegistry) -> Self {
        self.mutation_registry = Some(registry);
        self
    }

    pub fn with_eval_runs(mut self, n: usize) -> Self {
        assert!(n >= 1, "eval_runs 必须 >= 1，当前值: {n}");
        self.eval_runs = n;
        self
    }

    /// 设置最大代数（仅 DefaultCallback 模式下生效）
    pub fn with_max_generations(mut self, n: usize) -> Self {
        self.max_generations = n;
        self
    }

    /// 设置日志详细模式（仅 DefaultCallback 模式下生效）
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// 设置停滞耐心值（primary fitness 连续多少代未提升后强制结构探索）
    pub fn with_stagnation_patience(mut self, patience: usize) -> Self {
        self.stagnation_patience = patience;
        self
    }

    /// 设置训练 batch size（覆盖 SupervisedTask 的自动策略）
    ///
    /// 仅对支持 batch size 配置的 Task 生效（如 SupervisedTask）。
    /// 自定义 Task 可自行忽略此设置。
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size 必须 > 0");
        self.batch_size = Some(batch_size);
        self
    }

    // ==================== 主循环 ====================

    /// 运行演化主循环（Pareto 种群搜索 + NSGA-II 选择）
    ///
    /// 首先实例化并验证任务数据，然后执行种群级别的 genome-centric 演化。
    /// 无论是否达标都返回 `Ok(EvolutionResult)`，
    /// `status` 字段标识停止原因。
    /// `Err` 用于数据验证失败或系统错误。
    pub fn run(self) -> Result<EvolutionResult, EvolutionError> {
        // 解构以支持部分移动
        let Evolution {
            task_spec,
            target_metric,
            eval_runs,
            convergence_config,
            mutation_registry,
            constraints,
            seed,
            custom_callback,
            max_generations,
            verbose,
            stagnation_patience,
            batch_size,
            population_size,
            offspring_batch_size,
            parallelism,
            pareto_patience,
            complexity_metric,
        } = self;

        // 延迟实例化：验证数据 + 构建任务 + 提取维度
        let prepared = materialize_task(task_spec.clone())?;
        let mut serial_task = prepared.task;
        serial_task.configure_batch_size(batch_size);

        let is_sequential = prepared.seq_len.is_some();
        let is_spatial = prepared.input_spatial.is_some();
        let user_registry = mutation_registry;

        // 自适应约束：用户未显式指定时自动推导
        let n_train = prepared.n_train;
        let constraints = constraints.unwrap_or_else(|| {
            SizeConstraints::auto(
                prepared.input_dim,
                prepared.output_dim,
                n_train,
                is_spatial,
                prepared.input_spatial,
            )
        });

        // 并行执行设置
        let auto_parallelism = rayon::current_num_threads();
        let _effective_parallelism = parallelism.unwrap_or(auto_parallelism);
        let effective_pareto_patience = pareto_patience
            .unwrap_or_else(|| (population_size * 2).max(20));
        let can_parallel =
            task_spec.supports_parallel_evaluation() && auto_parallelism > 1;

        // 两阶段训练预算
        let phase1_gens = (max_generations as f64 * 0.7).round() as usize;
        let user_convergence = convergence_config;

        // Phase 1 快速训练预算
        let effective_bs = batch_size.unwrap_or_else(|| auto_batch_size(n_train));
        let fast_epochs = (n_train / (effective_bs * 5)).max(3).min(10);
        let phase1_convergence = ConvergenceConfig {
            budget: TrainingBudget::FixedEpochs(fast_epochs),
            ..user_convergence.clone()
        };

        // 构建两阶段 mutation registry
        let (phase1_reg, phase2_reg) = if user_registry.is_some() {
            (None, None)
        } else {
            (
                Some(MutationRegistry::phase1_registry(
                    &prepared.metric,
                    is_sequential,
                    is_spatial,
                )),
                Some(MutationRegistry::phase2_registry(
                    &prepared.metric,
                    is_sequential,
                    is_spatial,
                )),
            )
        };
        let user_reg_val = user_registry;

        let using_default_callback = custom_callback.is_none();
        let mut callback: Box<dyn EvolutionCallback> = custom_callback
            .unwrap_or_else(|| Box::new(DefaultCallback::new(max_generations, verbose)));

        let minimal_genome = if prepared.seq_len.is_some() {
            let mut g =
                NetworkGenome::minimal_sequential(prepared.input_dim, prepared.output_dim);
            g.seq_len = prepared.seq_len;
            g
        } else if let Some(spatial) = prepared.input_spatial {
            NetworkGenome::minimal_spatial(prepared.input_dim, prepared.output_dim, spatial)
        } else {
            NetworkGenome::minimal(prepared.input_dim, prepared.output_dim)
        };

        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // 随机爆发初始化参数
        let burst_k = (constraints.max_layers / 2).max(2).min(8);

        // ====== 初始化种群 ======
        let init_reg = if let Some(ref user_reg) = user_reg_val {
            user_reg
        } else {
            phase1_reg.as_ref().unwrap()
        };
        let mut init_genomes = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            let mut genome = minimal_genome.clone();
            for _ in 0..burst_k {
                let _ = init_reg.apply_random(&mut genome, &constraints, &mut rng);
            }
            init_genomes.push(genome);
        }

        // 评估初始种群
        let seeds: Vec<u64> = (0..init_genomes.len())
            .map(|_| rng.r#gen::<u64>())
            .collect();
        let init_results = evaluate_batch(
            &task_spec,
            init_genomes,
            &phase1_convergence,
            eval_runs,
            batch_size,
            &complexity_metric,
            can_parallel,
            seeds,
        );

        let mut parents: Vec<(NetworkGenome, FitnessScore)> = init_results
            .into_iter()
            .map(|r| (r.genome, r.score))
            .collect();

        // 回退：全部失败则用 minimal genome
        if parents.is_empty() {
            let fallback_seed = rng.r#gen::<u64>();
            let fallback_results = evaluate_batch(
                &task_spec,
                vec![minimal_genome.clone()],
                &phase1_convergence,
                eval_runs,
                batch_size,
                &complexity_metric,
                false,
                vec![fallback_seed],
            );
            if let Some(r) = fallback_results.into_iter().next() {
                parents.push((r.genome, r.score));
            } else {
                return Err(EvolutionError::InvalidConfig(
                    "无法评估任何初始个体".into(),
                ));
            }
        }

        // 初始化全局 archive
        let mut archive: Vec<(NetworkGenome, FitnessScore)> = Vec::new();
        selection::update_archive(
            &mut archive,
            parents
                .iter()
                .map(|(g, s)| (g.clone(), s.clone()))
                .collect(),
        );
        let mut archive_scores_snapshot: Vec<FitnessScore> =
            archive.iter().map(|(_, s)| s.clone()).collect();
        let mut archive_stagnation: usize = 0;
        let mut best_primary: f32 = f32::NEG_INFINITY;
        let mut primary_stagnation: usize = 0;

        // ====== 主循环 ======
        for generation in 0.. {
            // 0. 回调检查是否终止
            if callback.should_stop(generation) {
                let status = if using_default_callback {
                    EvolutionStatus::MaxGenerations
                } else {
                    EvolutionStatus::CallbackStopped
                };
                let (repr_g, repr_s) = select_representative(&archive, target_metric);
                return build_population_result(
                    repr_g, repr_s, &archive, generation, status, &mut rng, &*serial_task,
                );
            }

            // 选择当前阶段的 mutation registry 和 convergence config
            let is_phase1 = generation < phase1_gens;
            let current_registry = if let Some(ref user_reg) = user_reg_val {
                user_reg
            } else if is_phase1 {
                phase1_reg.as_ref().unwrap()
            } else {
                phase2_reg.as_ref().unwrap()
            };
            let current_convergence = if is_phase1 {
                &phase1_convergence
            } else {
                &user_convergence
            };

            // 1. 从 parents 中按 rank / crowding 二元锦标赛采样，变异生成 offspring
            let force_structural = primary_stagnation >= stagnation_patience;
            let mut offspring_genomes = Vec::with_capacity(offspring_batch_size);
            let mut any_mutation_succeeded = false;

            let parent_scores: Vec<FitnessScore> =
                parents.iter().map(|(_, s)| s.clone()).collect();
            let parent_ranks = selection::pareto_rank(&parent_scores);
            let parent_distances = selection::crowding_distance(&parent_scores);

            for _ in 0..offspring_batch_size {
                let p1 = rng.gen_range(0..parents.len());
                let p2 = rng.gen_range(0..parents.len());
                let parent_idx = {
                    let r_cmp = parent_ranks[p1].cmp(&parent_ranks[p2]);
                    if r_cmp == std::cmp::Ordering::Less {
                        p1
                    } else if r_cmp == std::cmp::Ordering::Greater {
                        p2
                    } else if parent_distances[p1] >= parent_distances[p2] {
                        p1
                    } else {
                        p2
                    }
                };

                let mut child = parents[parent_idx].0.clone();
                let mutation_result = if force_structural {
                    current_registry
                        .apply_random_structural(&mut child, &constraints, &mut rng)
                } else {
                    current_registry.apply_random(&mut child, &constraints, &mut rng)
                };

                match mutation_result {
                    Ok(mutation_name) => {
                        any_mutation_succeeded = true;
                        callback.on_mutation(generation, &mutation_name, &child);
                        offspring_genomes.push(child);
                    }
                    Err(MutationError::InternalError(msg)) => {
                        return Err(EvolutionError::Graph(GraphError::ComputationError(
                            format!("变异内部错误: {msg}"),
                        )));
                    }
                    Err(_) => continue,
                }
            }

            if offspring_genomes.is_empty() {
                if !any_mutation_succeeded {
                    let (repr_g, repr_s) = select_representative(&archive, target_metric);
                    return build_population_result(
                        repr_g,
                        repr_s,
                        &archive,
                        generation,
                        EvolutionStatus::NoApplicableMutation,
                        &mut rng,
                        &*serial_task,
                    );
                }
                continue;
            }

            // 2. 评估 offspring（并行 / 串行）
            let eval_seeds: Vec<u64> = (0..offspring_genomes.len())
                .map(|_| rng.r#gen::<u64>())
                .collect();
            let eval_results = evaluate_batch(
                &task_spec,
                offspring_genomes,
                current_convergence,
                eval_runs,
                batch_size,
                &complexity_metric,
                can_parallel,
                eval_seeds,
            );

            let offspring_evaluated = eval_results.len();
            let offspring: Vec<(NetworkGenome, FitnessScore)> = eval_results
                .into_iter()
                .map(|r| (r.genome, r.score))
                .collect();

            if offspring.is_empty() {
                let archive_best = archive
                    .iter()
                    .max_by(|a, b| {
                        a.1.primary
                            .partial_cmp(&b.1.primary)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                callback.on_generation(
                    generation,
                    &archive_best.0,
                    f32::NAN,
                    &archive_best.1,
                );
                continue;
            }

            // 3. NSGA-II 选择：parents ∪ offspring → 保留 population_size
            let pool: Vec<(NetworkGenome, FitnessScore)> =
                parents.into_iter().chain(offspring).collect();

            // 更新 archive（用完整 pool，确保不丢失非支配解）
            selection::update_archive(
                &mut archive,
                pool.iter()
                    .map(|(g, s)| (g.clone(), s.clone()))
                    .collect(),
            );

            parents = selection::nsga2_select(pool, population_size);

            // 4. Archive 收敛跟踪
            let new_archive_scores: Vec<FitnessScore> =
                archive.iter().map(|(_, s)| s.clone()).collect();
            if selection::archive_changed(
                &archive_scores_snapshot,
                &new_archive_scores,
                1e-6,
            ) {
                archive_stagnation = 0;
                archive_scores_snapshot = new_archive_scores;
            } else {
                archive_stagnation += 1;
            }

            // 5. Best primary 跟踪 + on_new_best
            let current_best = archive
                .iter()
                .map(|(_, s)| s.primary)
                .fold(f32::NEG_INFINITY, f32::max);
            if current_best > best_primary {
                let best_member = archive
                    .iter()
                    .max_by(|a, b| {
                        a.1.primary
                            .partial_cmp(&b.1.primary)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                callback.on_new_best(generation, &best_member.0, &best_member.1);
                best_primary = current_best;
                primary_stagnation = 0;
            } else {
                primary_stagnation += 1;
            }

            if force_structural {
                primary_stagnation = 0;
            }

            // 6. 回调
            let archive_best = archive
                .iter()
                .max_by(|a, b| {
                    a.1.primary
                        .partial_cmp(&b.1.primary)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            callback.on_generation(
                generation,
                &archive_best.0,
                f32::NAN,
                &archive_best.1,
            );

            let best_cost = archive
                .iter()
                .map(|(_, s)| s.inference_cost.unwrap_or(f32::INFINITY))
                .fold(f32::INFINITY, f32::min);
            let front_scores: Vec<FitnessScore> =
                archive.iter().map(|(_, s)| s.clone()).collect();
            let front_size = selection::pareto_front_indices(&front_scores).len();
            callback.on_population_evaluated(
                generation,
                parents.len(),
                offspring_evaluated,
                archive.len(),
                front_size,
                best_primary,
                best_cost,
            );

            // 7. 达标检查：archive 中是否存在 primary >= target 的成员
            let target_member = archive
                .iter()
                .filter(|(_, s)| s.primary >= target_metric)
                .min_by(|a, b| {
                    a.1.inference_cost
                        .unwrap_or(f32::INFINITY)
                        .partial_cmp(&b.1.inference_cost.unwrap_or(f32::INFINITY))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

            if let Some((target_genome, target_score)) = target_member {
                let build = target_genome.build(&mut rng)?;
                target_genome.restore_weights(&build)?;
                snapshot_with_loss(&*serial_task, target_genome, &build);
                return Ok(EvolutionResult {
                    build,
                    fitness: target_score.clone(),
                    generations: generation,
                    architecture_summary: format!("{target_genome}"),
                    status: EvolutionStatus::TargetReached,
                    genome: target_genome.clone(),
                    pareto_front: build_pareto_summaries(&archive),
                    pareto_genomes: archive.into_iter().map(|(g, _)| g).collect(),
                });
            }

            // 8. Pareto 收敛检查
            if archive_stagnation >= effective_pareto_patience {
                let (repr_g, repr_s) = select_representative(&archive, target_metric);
                return build_population_result(
                    repr_g,
                    repr_s,
                    &archive,
                    generation,
                    EvolutionStatus::ParetoConverged,
                    &mut rng,
                    &*serial_task,
                );
            }
        }

        unreachable!("演化循环应由停止条件终止")
    }
}

// ==================== 独立辅助函数 ====================

/// 评估结果（内部使用，所有字段均为 Send）
struct EvalResult {
    genome: NetworkGenome,
    score: FitnessScore,
    #[allow(dead_code)]
    loss: f32,
}

/// 评估单个候选个体
fn eval_candidate(
    task: &dyn EvolutionTask,
    mut genome: NetworkGenome,
    convergence: &ConvergenceConfig,
    eval_runs: usize,
    complexity_metric: &ComplexityMetric,
    rng: &mut StdRng,
) -> Option<EvalResult> {
    let build = genome.build(rng).ok()?;
    let _ = genome.restore_weights(&build);
    let loss = task.train(&genome, &build, convergence, rng).ok()?;
    genome.capture_weights(&build).ok()?;
    let mut score = evaluate_conservative(task, &genome, &build, eval_runs, rng).ok()?;
    if let Ok(cost) = compute_inference_cost(&genome, complexity_metric) {
        score.inference_cost = Some(cost);
    }
    Some(EvalResult { genome, score, loss })
}

/// 批量评估候选个体（支持 rayon 并行）
///
/// 并行路径使用 `map_init` 确保每个 rayon 工作线程只 materialize 一次 task，
/// 避免对训练数据的冗余复制。
fn evaluate_batch(
    task_spec: &TaskSpec,
    offspring: Vec<NetworkGenome>,
    convergence: &ConvergenceConfig,
    eval_runs: usize,
    batch_size: Option<usize>,
    complexity_metric: &ComplexityMetric,
    parallel: bool,
    seeds: Vec<u64>,
) -> Vec<EvalResult> {
    assert_eq!(offspring.len(), seeds.len());

    if parallel && offspring.len() > 1 {
        use rayon::prelude::*;

        offspring
            .into_par_iter()
            .zip(seeds.into_par_iter())
            .map_init(
                || {
                    let spec = task_spec.clone();
                    let mut mat =
                        materialize_task(spec).expect("并行 task 实例化失败");
                    mat.task.configure_batch_size(batch_size);
                    mat
                },
                |local_task, (genome, seed)| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    eval_candidate(
                        &*local_task.task,
                        genome,
                        convergence,
                        eval_runs,
                        complexity_metric,
                        &mut rng,
                    )
                },
            )
            .flatten()
            .collect()
    } else {
        // 串行路径
        let spec = task_spec.clone();
        let mut mat = match materialize_task(spec) {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        mat.task.configure_batch_size(batch_size);

        offspring
            .into_iter()
            .zip(seeds)
            .filter_map(|(genome, seed)| {
                let mut rng = StdRng::seed_from_u64(seed);
                eval_candidate(
                    &*mat.task,
                    genome,
                    convergence,
                    eval_runs,
                    complexity_metric,
                    &mut rng,
                )
            })
            .collect()
    }
}

/// 多次评估取 primary 最低值（保守估计）
fn evaluate_conservative(
    task: &dyn EvolutionTask,
    genome: &NetworkGenome,
    build: &BuildResult,
    eval_runs: usize,
    rng: &mut StdRng,
) -> Result<FitnessScore, GraphError> {
    let mut best: Option<FitnessScore> = None;
    for _ in 0..eval_runs {
        let score = task.evaluate(genome, build, rng)?;
        best = Some(match best {
            None => score,
            Some(prev) => {
                if score.primary < prev.primary {
                    score
                } else {
                    prev
                }
            }
        });
    }
    Ok(best.unwrap())
}

/// 自动快照：优先包含 Loss + TargetInput，fallback 到仅 output
fn snapshot_with_loss(task: &dyn EvolutionTask, genome: &NetworkGenome, build: &BuildResult) {
    if let Some(vis_loss) = task.create_visualization_loss(genome, build) {
        build.graph.snapshot_once_from(&[&vis_loss]);
    } else {
        build.graph.snapshot_once_from(&[&build.output]);
    }
}

/// 从 archive 中选择代表个体（用于最终返回）
///
/// 优先选择满足 target 且 complexity 最小的成员；
/// 若无达标成员，选择 primary 最高的。
fn select_representative<'a>(
    archive: &'a [(NetworkGenome, FitnessScore)],
    target_metric: f32,
) -> (&'a NetworkGenome, &'a FitnessScore) {
    let target_member = archive
        .iter()
        .filter(|(_, s)| s.primary >= target_metric)
        .min_by(|a, b| {
            a.1.inference_cost
                .unwrap_or(f32::INFINITY)
                .partial_cmp(&b.1.inference_cost.unwrap_or(f32::INFINITY))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    if let Some((g, s)) = target_member {
        return (g, s);
    }
    let (g, s) = archive
        .iter()
        .max_by(|a, b| {
            a.1.primary
                .partial_cmp(&b.1.primary)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    (g, s)
}

/// 构建 Pareto 前沿摘要列表
fn build_pareto_summaries(archive: &[(NetworkGenome, FitnessScore)]) -> Vec<ParetoSummary> {
    archive
        .iter()
        .map(|(g, s)| ParetoSummary {
            fitness: s.clone(),
            architecture_summary: format!("{g}"),
        })
        .collect()
}

/// 从 archive 构建最终演化结果
fn build_population_result(
    genome: &NetworkGenome,
    score: &FitnessScore,
    archive: &[(NetworkGenome, FitnessScore)],
    generations: usize,
    status: EvolutionStatus,
    rng: &mut StdRng,
    task: &dyn EvolutionTask,
) -> Result<EvolutionResult, EvolutionError> {
    let build = genome.build(rng)?;
    genome.restore_weights(&build)?;
    snapshot_with_loss(task, genome, &build);

    Ok(EvolutionResult {
        build,
        fitness: score.clone(),
        generations,
        architecture_summary: format!("{genome}"),
        status,
        genome: genome.clone(),
        pareto_front: build_pareto_summaries(archive),
        pareto_genomes: archive.iter().map(|(g, _)| g.clone()).collect(),
    })
}

/// 从 best genome 重新构建最终结果（legacy helper）
#[allow(dead_code)]
fn build_final_result(
    genome: &NetworkGenome,
    best_score: Option<FitnessScore>,
    generations: usize,
    status: EvolutionStatus,
    rng: &mut StdRng,
    task: &dyn EvolutionTask,
) -> Result<EvolutionResult, EvolutionError> {
    let build = genome.build(rng)?;
    genome.restore_weights(&build)?;
    snapshot_with_loss(task, genome, &build);

    let fitness = best_score.unwrap_or(FitnessScore {
        primary: f32::NEG_INFINITY,
        inference_cost: None,
        tiebreak_loss: None,
    });

    Ok(EvolutionResult {
        build,
        fitness,
        generations,
        architecture_summary: format!("{genome}"),
        status,
        genome: genome.clone(),
        pareto_front: Vec::new(),
        pareto_genomes: Vec::new(),
    })
}

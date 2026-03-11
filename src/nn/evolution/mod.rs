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
pub mod task;

#[cfg(test)]
mod tests;

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
}

// ==================== is_at_least_as_good ====================

/// 比较两个 FitnessScore：current 是否至少和 best 一样好
///
/// 比较规则由 FitnessScore 自身的 tiebreak_loss 驱动，不依赖 TaskMetric：
/// 1. primary 更高 → 接受
/// 2. primary 更低 → 拒绝
/// 3. primary 相等 → 看 tiebreak_loss（都有则越低越好，否则直接接受/中性漂移）
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
enum TaskSpec {
    Supervised {
        train_data: (Vec<Tensor>, Vec<Tensor>),
        test_data: (Vec<Tensor>, Vec<Tensor>),
        metric: TaskMetric,
    },
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
    /// 每代生成的子代数量（(1+λ) 策略中的 λ）
    offspring_count: usize,
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
            offspring_count: 4,
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

    /// 设置每代子代数量（(1+λ) 中的 λ，默认 4）
    pub fn with_offspring_count(mut self, n: usize) -> Self {
        assert!(n >= 1, "offspring_count 必须 >= 1，当前值: {n}");
        self.offspring_count = n;
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

    /// 运行演化主循环
    ///
    /// 首先实例化并验证任务数据，然后执行 genome-centric 演化。
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
            offspring_count,
        } = self;

        // 延迟实例化：验证数据 + 构建任务 + 提取维度
        let prepared = materialize_task(task_spec)?;
        let mut task = prepared.task;
        task.configure_batch_size(batch_size);

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
                Some(MutationRegistry::phase1_registry(&prepared.metric, is_sequential, is_spatial)),
                Some(MutationRegistry::phase2_registry(&prepared.metric, is_sequential, is_spatial)),
            )
        };
        let user_reg_val = user_registry;

        let using_default_callback = custom_callback.is_none();
        let mut callback: Box<dyn EvolutionCallback> = custom_callback
            .unwrap_or_else(|| Box::new(DefaultCallback::new(max_generations, verbose)));

        let minimal_genome = if prepared.seq_len.is_some() {
            let mut g = NetworkGenome::minimal_sequential(prepared.input_dim, prepared.output_dim);
            g.seq_len = prepared.seq_len;
            g
        } else if let Some(spatial) = prepared.input_spatial {
            NetworkGenome::minimal_spatial(prepared.input_dim, prepared.output_dim, spatial)
        } else {
            NetworkGenome::minimal(prepared.input_dim, prepared.output_dim)
        };

        let mut best_genome: Option<NetworkGenome> = None;
        let mut best_score: Option<FitnessScore> = None;
        let mut stagnation: usize = 0;

        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // 随机爆发初始化参数
        let burst_k = (constraints.max_layers / 2).max(2).min(8);

        for generation in 0.. {
            // 0. 回调检查是否终止
            if callback.should_stop(generation) {
                let status = if using_default_callback {
                    EvolutionStatus::MaxGenerations
                } else {
                    EvolutionStatus::CallbackStopped
                };
                return build_final_result(
                    best_genome.as_ref().unwrap_or(&minimal_genome),
                    best_score,
                    generation,
                    status,
                    &mut rng,
                    &*task,
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

            // (1+λ) 搜索：生成 λ 个候选，评估后选最佳
            let λ = offspring_count;
            let mut gen_best_genome: Option<NetworkGenome> = None;
            let mut gen_best_score: Option<FitnessScore> = None;
            let mut gen_best_loss: f32 = f32::NAN;
            let mut gen_best_build: Option<BuildResult> = None;
            let mut any_mutation_succeeded = false;

            for _child_idx in 0..λ {
                // 生成候选基因组
                let mut child = if generation == 0 {
                    // Generation 0：随机爆发初始化——从 minimal genome K 次随机变异
                    let mut c = minimal_genome.clone();
                    for _ in 0..burst_k {
                        let _ = current_registry.apply_random(&mut c, &constraints, &mut rng);
                    }
                    c
                } else {
                    // Generation 1+：从 best genome 变异
                    let mut c = best_genome.as_ref().unwrap_or(&minimal_genome).clone();
                    let force_structural = stagnation >= stagnation_patience;
                    let mutation_result = if force_structural {
                        current_registry.apply_random_structural(&mut c, &constraints, &mut rng)
                    } else {
                        current_registry.apply_random(&mut c, &constraints, &mut rng)
                    };
                    match mutation_result {
                        Ok(mutation_name) => {
                            any_mutation_succeeded = true;
                            callback.on_mutation(generation, &mutation_name, &c);
                        }
                        Err(MutationError::InternalError(msg)) => {
                            return Err(EvolutionError::Graph(GraphError::ComputationError(
                                format!("变异内部错误: {msg}"),
                            )));
                        }
                        Err(_) => continue, // 该候选变异失败，尝试下一个
                    }
                    c
                };

                // build → restore_weights → train → capture_weights → evaluate
                let build = match child.build(&mut rng) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let _ = child.restore_weights(&build);
                let loss = match task.train(&child, &build, current_convergence, &mut rng) {
                    Ok(l) => l,
                    Err(_) => continue,
                };
                child.capture_weights(&build)?;
                let score = match evaluate_conservative(&*task, &child, &build, eval_runs, &mut rng) {
                    Ok(s) => s,
                    Err(_) => continue,
                };

                // 跟踪本代最佳候选
                let is_better = match &gen_best_score {
                    None => true,
                    Some(prev) => is_at_least_as_good(&score, prev),
                };
                if is_better {
                    gen_best_genome = Some(child);
                    gen_best_score = Some(score);
                    gen_best_loss = loss;
                    gen_best_build = Some(build);
                }

                if generation == 0 {
                    any_mutation_succeeded = true;
                }
            }

            // 所有候选都失败
            if gen_best_genome.is_none() {
                if !any_mutation_succeeded && generation > 0 {
                    return build_final_result(
                        best_genome.as_ref().unwrap_or(&minimal_genome),
                        best_score,
                        generation,
                        EvolutionStatus::NoApplicableMutation,
                        &mut rng,
                        &*task,
                    );
                }
                continue;
            }

            let gen_genome = gen_best_genome.unwrap();
            let gen_score = gen_best_score.unwrap();
            let gen_build = gen_best_build.unwrap();

            // 接受/回滚
            let accept = match &best_score {
                None => true,
                Some(best) => is_at_least_as_good(&gen_score, best),
            };

            if accept {
                let primary_improved = if let Some(best) = &best_score {
                    if gen_score.primary > best.primary {
                        callback.on_new_best(generation, &gen_genome, &gen_score);
                        true
                    } else {
                        false
                    }
                } else {
                    callback.on_new_best(generation, &gen_genome, &gen_score);
                    true
                };

                if primary_improved {
                    stagnation = 0;
                } else {
                    stagnation += 1;
                }

                best_score = Some(gen_score.clone());
                best_genome = Some(gen_genome.clone());
            } else {
                stagnation += 1;
            }

            // 停滞计数器重置（在强制结构探索后）
            if generation > 0 && stagnation >= stagnation_patience {
                stagnation = 0;
            }

            // 回调 on_generation
            callback.on_generation(generation, best_genome.as_ref().unwrap_or(&gen_genome), gen_best_loss, &gen_score);

            // 达标检查
            if gen_score.primary >= target_metric {
                snapshot_with_loss(&*task, &gen_genome, &gen_build);
                return Ok(EvolutionResult {
                    build: gen_build,
                    fitness: gen_score,
                    generations: generation,
                    architecture_summary: format!("{gen_genome}"),
                    status: EvolutionStatus::TargetReached,
                    genome: gen_genome,
                });
            }
        }

        unreachable!("演化循环应由 callback.should_stop 或达标检查终止")
    }
}

// ==================== 独立辅助函数 ====================

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

/// 从 best genome 重新构建最终结果
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
    // 自动快照计算图拓扑（供后续 visualize_snapshot 渲染）
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
    })
}

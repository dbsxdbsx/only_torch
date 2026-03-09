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
pub mod gene;
pub mod mutation;
pub mod error;
pub mod model_io;
pub mod task;

#[cfg(test)]
mod tests;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::nn::{GraphError, VisualizationOutput};
use crate::tensor::Tensor;

use self::builder::BuildResult;
use self::callback::{DefaultCallback, EvolutionCallback};
use self::convergence::ConvergenceConfig;
use self::error::EvolutionError;
use self::gene::{NetworkGenome, TaskMetric};
use self::mutation::{MutationError, MutationRegistry, SizeConstraints};
use self::task::{EvolutionTask, FitnessScore, SupervisedTask};

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
    /// 接受 `[batch, input_dim]` 或 `[input_dim]`（自动添加 batch 维度）。
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

        // 1D [input_dim] → 2D [1, input_dim]
        let input_2d = if input.dimension() == 1 {
            input.reshape(&[1, input.size()])
        } else {
            input.clone()
        };

        self.build.input.set_value(&input_2d)?;
        self.build.graph.forward(&self.build.output)?;

        self.build
            .output
            .value()?
            .ok_or_else(|| {
                EvolutionError::Graph(GraphError::ComputationError(
                    "推理时输出节点无值".into(),
                ))
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
    metric: TaskMetric,
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
            let input_dim = train_data.0[0].size();
            let output_dim = train_data.1[0].size();
            let task = SupervisedTask::new(train_data, test_data, metric.clone())?;
            Ok(MaterializedTask {
                task: Box::new(task),
                input_dim,
                output_dim,
                metric,
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
    constraints: SizeConstraints,
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
            constraints: SizeConstraints::default(),
            seed: None,
            custom_callback: None,
            max_generations: 100,
            verbose: true,
            stagnation_patience: 20,
            batch_size: None,
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
        self.constraints = constraints;
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
        } = self;

        // 延迟实例化：验证数据 + 构建任务 + 提取维度
        let prepared = materialize_task(task_spec)?;
        let mut task = prepared.task;
        task.configure_batch_size(batch_size);

        let mutation_registry = mutation_registry
            .unwrap_or_else(|| MutationRegistry::default_registry(&prepared.metric));

        let using_default_callback = custom_callback.is_none();
        let mut callback: Box<dyn EvolutionCallback> = custom_callback
            .unwrap_or_else(|| Box::new(DefaultCallback::new(max_generations, verbose)));

        let mut genome = NetworkGenome::minimal(prepared.input_dim, prepared.output_dim);
        let mut best_genome: Option<NetworkGenome> = None;
        let mut best_score: Option<FitnessScore> = None;
        let mut stagnation: usize = 0;

        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        for generation in 0.. {
            // 0. 回调检查是否终止
            if callback.should_stop(generation) {
                let status = if using_default_callback {
                    EvolutionStatus::MaxGenerations
                } else {
                    EvolutionStatus::CallbackStopped
                };
                return build_final_result(
                    best_genome.as_ref().unwrap_or(&genome),
                    best_score,
                    generation,
                    status,
                    &mut rng,
                    &*task,
                );
            }

            // 1. 从基因组构建图
            let build = genome.build(&mut rng)?;

            // 2. Lamarckian 权重继承
            genome.restore_weights(&build)?;

            // 3. 训练
            let loss = task.train(
                &genome,
                &build,
                &convergence_config,
                &mut rng,
            )?;

            // 4. 捕获权重
            genome.capture_weights(&build)?;

            // 5. 评估（N 次取 primary 最低值作为保守估计）
            let score = evaluate_conservative(
                &*task,
                &genome,
                &build,
                eval_runs,
                &mut rng,
            )?;

            // 6. 接受/回滚（在 on_generation 之前，确保 on_new_best 星标时序正确）
            let accept = match &best_score {
                None => true,
                Some(best) => is_at_least_as_good(&score, best),
            };

            if accept {
                let primary_improved = if let Some(best) = &best_score {
                    if score.primary > best.primary {
                        callback.on_new_best(generation, &genome, &score);
                        true
                    } else {
                        false
                    }
                } else {
                    callback.on_new_best(generation, &genome, &score);
                    true
                };

                // 停滞检测：仅 primary 严格提升时重置计数器
                if primary_improved {
                    stagnation = 0;
                } else {
                    stagnation += 1;
                }

                best_score = Some(score.clone());
                best_genome = Some(genome.clone());
            } else {
                stagnation += 1;
                genome = best_genome.as_ref().unwrap().clone();
            }

            // 7. 回调 on_generation（在 on_new_best 之后，星标已就绪）
            callback.on_generation(generation, &genome, loss, &score);

            // 8. 达标检查
            if score.primary >= target_metric {
                // 自动快照计算图拓扑（Var 还活着时捕获，供后续 visualize_snapshot 渲染）
                // 优先包含 Loss + TargetInput 以呈现完整管线
                snapshot_with_loss(&*task, &genome, &build);
                return Ok(EvolutionResult {
                    build,
                    fitness: score,
                    generations: generation,
                    architecture_summary: format!("{genome}"),
                    status: EvolutionStatus::TargetReached,
                    genome,
                });
            }

            // 9. 变异（停滞时强制结构探索）
            let force_structural = stagnation >= stagnation_patience;
            let mutation_result = if force_structural {
                stagnation = 0; // 重置，给新拓扑优化时间
                mutation_registry.apply_random_structural(
                    &mut genome,
                    &constraints,
                    &mut rng,
                )
            } else {
                mutation_registry.apply_random(
                    &mut genome,
                    &constraints,
                    &mut rng,
                )
            };
            match mutation_result {
                Ok(mutation_name) => {
                    callback.on_mutation(generation, &mutation_name, &genome);
                }
                Err(MutationError::NotApplicable(_)) => {
                    return build_final_result(
                        best_genome.as_ref().unwrap_or(&genome),
                        best_score,
                        generation + 1,
                        EvolutionStatus::NoApplicableMutation,
                        &mut rng,
                        &*task,
                    );
                }
                Err(MutationError::InternalError(msg)) => {
                    return Err(EvolutionError::Graph(GraphError::ComputationError(
                        format!("变异内部错误: {msg}"),
                    )));
                }
                Err(e) => {
                    return Err(EvolutionError::Graph(GraphError::ComputationError(
                        format!("变异失败: {e}"),
                    )));
                }
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
fn snapshot_with_loss(
    task: &dyn EvolutionTask,
    genome: &NetworkGenome,
    build: &BuildResult,
) {
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

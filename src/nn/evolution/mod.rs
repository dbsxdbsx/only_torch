/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化模块
 *
 * Genome-centric 节点级演化：
 * - gene.rs: 基因数据结构（NetworkGenome / NodeLevel 内部表示）
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

mod builder;
mod callback;
mod cell_migration;
mod convergence;
mod error;
mod fm_mutation;
mod fm_ops;
mod gene;
mod model_io;
mod mutation;
mod net2net;
mod node_expansion;
mod node_gene;
mod node_ops;
mod selection;
mod task;

#[cfg(test)]
mod tests;

use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::time::{Duration, Instant};

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::{GraphError, VisualizationOutput};
use crate::tensor::Tensor;

use self::builder::{BuildResult, InheritReport};
use self::callback::CandidateFamily;
pub use self::callback::{
    CandidateFamilyCounts, CandidatePrefilterSummary, DefaultCallback, EvaluationTimingSummary,
    EvolutionCallback,
};
pub use self::convergence::{ConvergenceConfig, TrainingBudget};
pub use self::error::EvolutionError;
pub(crate) use self::gene::NetworkGenome;
pub use self::gene::TaskMetric;
pub use self::mutation::{MutationError, MutationRegistry, SizeConstraints, SizeStrategy};
use self::node_ops::{NodeBlockKind, node_main_path};
pub use self::task::ProxyKind;
use self::task::{
    EvolutionTask, FitnessScore, MaterializedHead, SupervisedTask, TrainOutcome, auto_batch_size,
};
pub use self::task::{
    HeadMetricReport, HeadSpec, MetricReport, MetricValue, ReportMetric, SupervisedSpec,
};

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
    /// 演化时使用的 seed（用于 rebuild_pareto_member 的确定性重建）
    evolution_seed: Option<u64>,
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
        let idx = self.build.default_output_index();
        let output = self.build.outputs.get(idx).ok_or_else(|| {
            EvolutionError::Graph(GraphError::ComputationError("默认输出 head 不存在".into()))
        })?;
        self.predict_var(input, output)
    }

    /// 推理指定命名 head。
    pub fn predict_head(&self, name: &str, input: &Tensor) -> Result<Tensor, EvolutionError> {
        let output = self
            .build
            .output_by_name(name)
            .ok_or_else(|| EvolutionError::InvalidConfig(format!("找不到输出 head: {name}")))?;
        self.predict_var(input, output)
    }

    /// 推理多个命名 head，按传入名称顺序返回。
    pub fn predict_heads(
        &self,
        names: &[&str],
        input: &Tensor,
    ) -> Result<Vec<(String, Tensor)>, EvolutionError> {
        self.build.graph.eval();
        let shaped = self.shaped_input(input);
        self.build.input.set_value(&shaped)?;

        let mut outputs = Vec::with_capacity(names.len());
        for &name in names {
            let output = self
                .build
                .output_by_name(name)
                .ok_or_else(|| EvolutionError::InvalidConfig(format!("找不到输出 head: {name}")))?;
            self.build.graph.forward(output)?;
            let value = output.value()?.ok_or_else(|| {
                EvolutionError::Graph(GraphError::ComputationError("推理时输出节点无值".into()))
            })?;
            outputs.push((name.to_string(), value));
        }
        Ok(outputs)
    }

    fn predict_var(
        &self,
        input: &Tensor,
        output: &crate::nn::Var,
    ) -> Result<Tensor, EvolutionError> {
        self.build.graph.eval();
        let shaped = self.shaped_input(input);
        self.build.input.set_value(&shaped)?;
        self.build.graph.forward(output)?;

        output.value()?.ok_or_else(|| {
            EvolutionError::Graph(GraphError::ComputationError("推理时输出节点无值".into()))
        })
    }

    fn shaped_input(&self, input: &Tensor) -> Tensor {
        match input.dimension() {
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
        }
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
        let mut rng = match self.evolution_seed {
            Some(seed) => StdRng::seed_from_u64(seed ^ (index as u64)),
            None => StdRng::from_entropy(),
        };
        let build = genome.build(&mut rng)?;
        genome.restore_weights(&build)?;
        build.graph.snapshot_once_from(&build.output_refs());
        Ok(EvolutionResult {
            build,
            fitness: self.pareto_front[index].fitness.clone(),
            generations: self.generations,
            architecture_summary: self.pareto_front[index].architecture_summary.clone(),
            status: self.status.clone(),
            genome: genome.clone(),
            pareto_front: self.pareto_front.clone(),
            pareto_genomes: self.pareto_genomes.clone(),
            evolution_seed: self.evolution_seed,
        })
    }
}

/// 运行时任务副本（可克隆，供串行/并行评估复用）
#[derive(Clone)]
enum TaskRuntime {
    Supervised(SupervisedTask),
}

impl EvolutionTask for TaskRuntime {
    fn train(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
        convergence: &ConvergenceConfig,
        rng: &mut StdRng,
    ) -> Result<TrainOutcome, GraphError> {
        match self {
            TaskRuntime::Supervised(task) => task.train(genome, build, convergence, rng),
        }
    }

    fn evaluate(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
        rng: &mut StdRng,
    ) -> Result<FitnessScore, GraphError> {
        match self {
            TaskRuntime::Supervised(task) => task.evaluate(genome, build, rng),
        }
    }

    fn configure_batch_size(&mut self, batch_size: Option<usize>) {
        match self {
            TaskRuntime::Supervised(task) => task.configure_batch_size(batch_size),
        }
    }

    fn configure_proxy(&mut self, kind: Option<ProxyKind>) {
        match self {
            TaskRuntime::Supervised(task) => task.configure_proxy(kind),
        }
    }

    fn configure_report_metrics(&mut self, metrics: &[ReportMetric]) {
        match self {
            TaskRuntime::Supervised(task) => task.configure_report_metrics(metrics),
        }
    }

    fn create_visualization_loss(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
    ) -> Option<crate::nn::Var> {
        match self {
            TaskRuntime::Supervised(task) => task.create_visualization_loss(genome, build),
        }
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComplexityMetric {
    /// 参数总量
    ParamCount,
    /// 前向推理 FLOPs（对 Spatial 域尤其重要——同参数量的 Conv 和 FC 层 FLOPs 差异巨大）
    FLOPs,
}

/// 根据复杂度度量计算 inference_cost
pub(crate) fn compute_inference_cost(
    genome: &NetworkGenome,
    metric: &ComplexityMetric,
) -> Result<f32, EvolutionError> {
    match metric {
        ComplexityMetric::ParamCount => genome.total_params().map(|p| p as f32).map_err(|e| {
            EvolutionError::Graph(GraphError::ComputationError(format!("计算参数量失败: {e}")))
        }),
        ComplexityMetric::FLOPs => genome.total_flops().map(|f| f as f32).map_err(|e| {
            EvolutionError::Graph(GraphError::ComputationError(format!(
                "计算 FLOPs 失败: {e}"
            )))
        }),
    }
}

// ==================== is_at_least_as_good ====================

/// 比较两个 FitnessScore：current 是否至少和 best 一样好
///
/// 比较规则由 FitnessScore 自身的 tiebreak_loss 驱动，不依赖 TaskMetric：
/// 1. primary 更高 → 接受
/// 2. primary 更低 → 拒绝
/// 3. primary 相等 → 看 primary_proxy（都有则越高越好，F3 学习速度 tiebreak）
/// 4. 仍相等 → 看 tiebreak_loss（都有则越低越好，否则直接接受/中性漂移）
#[allow(dead_code)]
fn is_at_least_as_good(current: &FitnessScore, best: &FitnessScore) -> bool {
    if current.primary > best.primary {
        return true;
    }
    if current.primary < best.primary {
        return false;
    }
    // F3: primary 相等时，先比 proxy（越高越好）
    if let (Some(c), Some(b)) = (current.primary_proxy, best.primary_proxy) {
        if c > b {
            return true;
        }
        if c < b {
            return false;
        }
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
    Supervised { spec: SupervisedSpec },
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
    task: TaskRuntime,
    input_dim: usize,
    output_dim: usize,
    /// 序列长度（None = 平坦输入，Some(n) = 序列输入）
    seq_len: Option<usize>,
    /// 空间输入尺寸 (H, W)（None = 非空间，input_dim 表示 in_channels）
    input_spatial: Option<(usize, usize)>,
    metric: TaskMetric,
    heads: Vec<MaterializedHead>,
    /// 训练样本数（用于 auto_constraints 和训练预算计算）
    n_train: usize,
}

/// 从 TaskSpec 实例化具体任务，提取维度信息并验证数据
fn materialize_task(spec: TaskSpec) -> Result<MaterializedTask, EvolutionError> {
    match spec {
        TaskSpec::Supervised { spec } => {
            // 维度提取前的安全检查（SupervisedTask::new 做完整验证）
            if spec.train_inputs.is_empty() {
                return Err(EvolutionError::InvalidData("训练输入不能为空".into()));
            }
            if spec.heads.is_empty() {
                return Err(EvolutionError::InvalidData("训练 head 不能为空".into()));
            }
            // 检测输入数据维度：1D = 平坦，2D = 序列，3D = 空间
            let sample_ndim = spec.train_inputs[0].dimension();
            let (input_dim, seq_len, input_spatial) = if sample_ndim == 3 {
                // 空间数据：每个样本 [C, H, W]
                let shape = spec.train_inputs[0].shape();
                (shape[0], None, Some((shape[1], shape[2])))
            } else if sample_ndim == 2 {
                // 序列数据：每个样本 [seq_len_i, input_dim]
                let feat_dim = spec.train_inputs[0].shape()[1];
                let max_seq = spec
                    .train_inputs
                    .iter()
                    .chain(spec.test_inputs.iter())
                    .map(|t| t.shape()[0])
                    .max()
                    .unwrap();
                (feat_dim, Some(max_seq), None)
            } else {
                (spec.train_inputs[0].size(), None, None)
            };
            let task = SupervisedTask::from_spec(spec)?;
            let heads = task.head_metas();
            let metric = task.metric().clone();
            for head in &heads {
                if head.metric.is_segmentation() {
                    if input_spatial.is_none() {
                        return Err(EvolutionError::InvalidData(
                            "分割 head 要求输入为 [C,H,W]".into(),
                        ));
                    }
                }
            }
            let output_dim = heads.iter().map(|head| head.output_dim).sum();
            let n_train = task.train_len();
            Ok(MaterializedTask {
                task: TaskRuntime::Supervised(task),
                input_dim,
                output_dim,
                seq_len,
                input_spatial,
                metric,
                heads,
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
    /// 用户是否显式设置了 population_size
    population_size_explicit: bool,
    /// 每代实际评估的新候选数量
    offspring_batch_size: usize,
    /// 用户是否显式设置了 offspring_batch_size
    offspring_batch_size_explicit: bool,
    /// 并行评估线程数（None = auto = rayon::current_num_threads()）
    parallelism: Option<usize>,
    /// Pareto archive 连续未改进多少代后判定收敛（None = auto）
    pareto_patience: Option<usize>,
    /// 复杂度度量方式（用于 inference_cost 计算）
    complexity_metric: ComplexityMetric,
    /// 候选评估前的复杂度上限（None = 不限制）。
    ///
    /// 使用当前 `complexity_metric` 计算；空间任务默认 metric 为 FLOPs。
    max_inference_cost: Option<f32>,
    /// F3 学习速度代理（None = 关闭，Some = 启用对应 proxy）
    primary_proxy: Option<ProxyKind>,
    /// 附加报告指标（只影响日志和结果报告，不参与演化选择）
    report_metrics: Vec<ReportMetric>,
    /// F4 ASHA 多保真评估配置（None = 关闭）
    asha: Option<AshaConfig>,
    /// 初始种群随机爆发变异次数（None = 根据约束自动推导）
    initial_burst: Option<usize>,
    /// 初始候选 portfolio（None = 仅 minimal genome 随机爆发）
    initial_portfolio: Option<InitialPortfolioConfig>,
    /// 用户是否显式设置过 initial_portfolio
    initial_portfolio_explicit: bool,
    /// P5-lite 候选预筛配置（None = 不预筛）
    candidate_scoring: Option<CandidateScoringConfig>,
    /// 用户是否显式设置过 candidate_scoring
    candidate_scoring_explicit: bool,
    /// 接近目标时对 Pareto archive 前若干候选做最终复训/复评。
    final_refit: Option<FinalRefitConfig>,
    /// 用户是否显式设置过 final_refit
    final_refit_explicit: bool,
}

/// 接近目标时的最终复训配置。
///
/// 典型用途是把搜索阶段找到的 promising architecture 用更足的训练预算收尾，
/// 避免在 94% -> 95% 这类小差距上继续盲目扩展结构。
#[derive(Clone, Copy, Debug)]
pub struct FinalRefitConfig {
    /// 触发阈值：best primary >= target - trigger_margin 时启动。
    pub trigger_margin: f32,
    /// 每次从 archive 选择多少个 top 候选复训。
    pub top_k: usize,
    /// 复训的固定 epoch 数（增量训练，保留 Lamarckian 权重）。
    pub epochs: usize,
    /// best primary 至少提升多少才允许再次触发 refit。
    pub retrigger_delta: f32,
}

impl FinalRefitConfig {
    pub fn new(trigger_margin: f32, top_k: usize, epochs: usize) -> Self {
        Self {
            trigger_margin,
            top_k,
            epochs,
            retrigger_delta: 0.005,
        }
    }

    fn validated(&self) -> Self {
        Self {
            trigger_margin: self.trigger_margin.max(0.0),
            top_k: self.top_k.max(1),
            epochs: self.epochs.max(1),
            retrigger_delta: self.retrigger_delta.max(0.0),
        }
    }
}

/// 初始搜索空间 portfolio 配置。
///
/// 它不固定最终结构，只把若干高质量结构族放进初始种群，让后续演化和真实评估决定赢家。
#[derive(Clone, Copy, Debug)]
pub struct InitialPortfolioConfig {
    pub include_flat_mlp: bool,
    pub include_tiny_cnn: bool,
    pub include_lenet_tiny: bool,
    pub include_unet_lite: bool,
    pub flat_mlp_hidden: usize,
}

impl InitialPortfolioConfig {
    /// 低成本平坦 MLP 初始候选。
    ///
    /// 适合 MNIST 这类小图像分类任务：让展平 MLP 与 CNN 候选在同一套
    /// 训练 / 评估 / Pareto 选择流程中竞争。
    pub fn flat_mlp_only(hidden: usize) -> Self {
        Self {
            include_flat_mlp: true,
            include_tiny_cnn: false,
            include_lenet_tiny: false,
            include_unet_lite: false,
            flat_mlp_hidden: hidden.max(1),
        }
    }

    pub fn vision_classification() -> Self {
        Self {
            include_flat_mlp: true,
            include_tiny_cnn: true,
            include_lenet_tiny: true,
            include_unet_lite: false,
            flat_mlp_hidden: 128,
        }
    }

    /// Segmentation 初始候选族。
    ///
    /// `include_tiny_cnn` 表示最小 dense segmentation head，`include_lenet_tiny`
    /// 表示稍深的 dense conv head，`include_unet_lite` 表示 encoder-decoder + skip
    /// concat 的 U-Net-lite 起点。
    pub fn vision_segmentation() -> Self {
        Self {
            include_flat_mlp: false,
            include_tiny_cnn: true,
            include_lenet_tiny: true,
            include_unet_lite: true,
            flat_mlp_hidden: 1,
        }
    }

    fn validated(&self) -> Self {
        Self {
            include_flat_mlp: self.include_flat_mlp,
            include_tiny_cnn: self.include_tiny_cnn,
            include_lenet_tiny: self.include_lenet_tiny,
            include_unet_lite: self.include_unet_lite,
            flat_mlp_hidden: self.flat_mlp_hidden.max(1),
        }
    }
}

/// P5-lite 候选预筛配置。
///
/// 第一版不训练 surrogate 模型，而是用结构特征 + FLOPs 启发式给候选排序；
/// 真实 fitness 仍然只由完整训练/评估产生。
#[derive(Clone, Copy, Debug)]
pub struct CandidateScoringConfig {
    pub pool_multiplier: usize,
    pub keep_top_k: Option<usize>,
    /// P5-lite 预筛时每个结构族至少保留的候选数。
    ///
    /// 设为 0 时退化为纯 score 排序。默认保留 1 个，避免 FlatMLP 这类低 FLOPs
    /// 候选把 TinyCNN / LeNetLike 全部挤出完整训练评估。
    pub min_per_family: usize,
}

impl CandidateScoringConfig {
    pub fn p5_lite() -> Self {
        Self {
            pool_multiplier: 3,
            keep_top_k: None,
            min_per_family: 1,
        }
    }

    fn validated(&self) -> Self {
        Self {
            pool_multiplier: self.pool_multiplier.max(1),
            keep_top_k: self.keep_top_k.map(|n| n.max(1)),
            min_per_family: self.min_per_family,
        }
    }
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
        let spec = SupervisedSpec::new(train_data.0, test_data.0)
            .head_targets("output", train_data.1, test_data.1, metric)
            .primary_head("output");
        Self::supervised_task(spec)
    }

    /// 监督学习显式配置入口。
    ///
    /// 单头与多头任务共用同一 supervised 心智模型；旧的 `supervised(...)`
    /// 是该入口的便捷包装。
    pub fn supervised_task(spec: SupervisedSpec) -> Self {
        Self {
            task_spec: TaskSpec::Supervised { spec },
            target_metric: 1.0,
            eval_runs: 1,
            convergence_config: ConvergenceConfig::default(),
            mutation_registry: None,
            constraints: None,
            seed: None,
            custom_callback: None,
            max_generations: 100,
            verbose: false,
            stagnation_patience: 20,
            batch_size: None,
            population_size: rayon::current_num_threads().clamp(12, 32),
            population_size_explicit: false,
            offspring_batch_size: rayon::current_num_threads().max(12),
            offspring_batch_size_explicit: false,
            parallelism: None,
            pareto_patience: None,
            complexity_metric: ComplexityMetric::FLOPs,
            max_inference_cost: None,
            // F3/F4 默认开启：在所有演化任务（序列 / 图像 / 全连接）上稳定有益
            // - LossSlope：plateau 上给 NSGA-II 提供 tiebreak 信号；primary 有差异时零开销
            // - ASHA：阶梯式 Successive Halving 把 Phase 1 预算集中到有潜力的候选
            // 要关闭可调用 `.with_primary_proxy(None)` / `.with_asha(None)`
            primary_proxy: Some(ProxyKind::LossSlope),
            report_metrics: Vec::new(),
            asha: Some(AshaConfig::default()),
            initial_burst: None,
            initial_portfolio: None,
            initial_portfolio_explicit: false,
            candidate_scoring: None,
            candidate_scoring_explicit: false,
            final_refit: None,
            final_refit_explicit: false,
        }
    }

    // ==================== Builder methods ====================

    pub fn with_target_metric(mut self, target: f32) -> Self {
        self.target_metric = target;
        self
    }

    /// 在默认报告指标基础上追加附加评估指标。
    ///
    /// 报告指标只影响日志和 `FitnessScore::report`，不参与 primary fitness、
    /// target 判断或 NSGA-II 选择。
    pub fn with_report_metrics(mut self, metrics: impl IntoIterator<Item = ReportMetric>) -> Self {
        for metric in metrics {
            if !self.report_metrics.contains(&metric) {
                self.report_metrics.push(metric);
            }
        }
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
        self.offspring_batch_size_explicit = true;
        self
    }

    /// 设置每代保留的种群大小（NSGA-II 幸存者数量，默认 auto）
    pub fn with_population_size(mut self, n: usize) -> Self {
        assert!(n >= 1, "population_size 必须 >= 1，当前值: {n}");
        self.population_size = n;
        self.population_size_explicit = true;
        self
    }

    /// 设置每代实际评估的新候选数量（默认 = max(population_size, parallelism)）
    pub fn with_offspring_batch_size(mut self, n: usize) -> Self {
        assert!(n >= 1, "offspring_batch_size 必须 >= 1，当前值: {n}");
        self.offspring_batch_size = n;
        self.offspring_batch_size_explicit = true;
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

    /// 设置候选评估前的复杂度上限。
    ///
    /// 超过上限的 genome 会在训练前被过滤，避免少数超大候选拖慢整代搜索。
    /// 上限单位由当前 `complexity_metric` 决定，默认空间任务为 FLOPs。
    pub fn with_max_inference_cost(mut self, max_cost: f32) -> Self {
        assert!(
            max_cost.is_finite() && max_cost > 0.0,
            "max_cost 必须为正有限值"
        );
        self.max_inference_cost = Some(max_cost);
        self
    }

    /// 设置学习速度代理（F3）：在 plateau 上用 loss 下降速率打破 NSGA-II 平局。
    ///
    /// 默认启用 `ProxyKind::LossSlope`。传入 `None` 可关闭。
    ///
    /// 仅对支持 proxy 的 Task 生效（目前 `SupervisedTask`）。开启后会在
    /// 每次训练时额外记录 loss 轨迹并计算一次 proxy，开销可忽略。
    pub fn with_primary_proxy(mut self, kind: impl Into<Option<ProxyKind>>) -> Self {
        self.primary_proxy = kind.into();
        self
    }

    /// 设置 ASHA 多保真评估（F4）：Phase 1 从"均匀 FixedEpochs"改为阶梯式
    /// Successive Halving，将训练预算集中在有潜力的候选上。
    ///
    /// 默认启用 `AshaConfig::default()`（rung_epochs=[1,2,4], eta=3）。传入 `None` 可关闭。
    ///
    /// 只在 Phase 1 生效（Phase 2 仍用 user_convergence 完成最终训练）。
    pub fn with_asha(mut self, config: impl Into<Option<AshaConfig>>) -> Self {
        self.asha = config.into();
        self
    }

    /// 设置初始种群中每个个体基于 minimal genome 额外施加的随机变异次数。
    ///
    /// 较小的快速回归任务可设为 0..2，避免初始候选过重或过多无效；
    /// 常规搜索保持默认自动策略即可。
    pub fn with_initial_burst(mut self, n: usize) -> Self {
        self.initial_burst = Some(n);
        self
    }

    /// 设置初始结构 portfolio。
    ///
    /// 空间分类任务默认启用 `InitialPortfolioConfig::vision_classification()`；
    /// 显式传入 `None` 可关闭，传入配置可覆盖默认候选族。
    pub fn with_initial_portfolio(
        mut self,
        config: impl Into<Option<InitialPortfolioConfig>>,
    ) -> Self {
        self.initial_portfolio = config.into().map(|c| c.validated());
        self.initial_portfolio_explicit = true;
        self
    }

    /// 设置 P5-lite 候选预筛策略。
    ///
    /// 空间分类任务默认启用 P5-lite。该策略只影响候选进入完整训练评估前的
    /// 排序/截断，不改写最终 fitness；显式传入 `None` 可关闭。
    pub fn with_candidate_scoring(
        mut self,
        config: impl Into<Option<CandidateScoringConfig>>,
    ) -> Self {
        self.candidate_scoring = config.into().map(|c| c.validated());
        self.candidate_scoring_explicit = true;
        self
    }

    /// 设置接近目标时的最终复训策略。
    ///
    /// 空间分类任务默认开启；适合 MNIST 这类搜索已接近目标、但候选需要更足训练
    /// 才能冲线的任务。显式传入 `None` 可关闭。
    pub fn with_final_refit(mut self, config: impl Into<Option<FinalRefitConfig>>) -> Self {
        self.final_refit = config.into().map(|c| c.validated());
        self.final_refit_explicit = true;
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
            mut batch_size,
            mut population_size,
            population_size_explicit,
            mut offspring_batch_size,
            offspring_batch_size_explicit,
            parallelism,
            pareto_patience,
            complexity_metric,
            mut max_inference_cost,
            primary_proxy,
            report_metrics,
            asha,
            initial_burst,
            mut initial_portfolio,
            initial_portfolio_explicit,
            mut candidate_scoring,
            candidate_scoring_explicit,
            mut final_refit,
            final_refit_explicit,
        } = self;

        // 当指定 seed 时，自动固定 population_size/offspring_batch_size
        // 避免因不同机器线程数导致 RNG 消耗序列不同
        if seed.is_some() {
            if !population_size_explicit {
                population_size = 20;
            }
            if !offspring_batch_size_explicit {
                offspring_batch_size = 12;
            }
        }

        // 延迟实例化：验证数据 + 构建任务 + 提取维度
        let prepared = materialize_task(task_spec.clone())?;
        let mut base_task = prepared.task.clone();
        base_task.configure_batch_size(batch_size);
        base_task.configure_proxy(primary_proxy);
        base_task.configure_report_metrics(&report_metrics);
        let serial_task = base_task.clone();

        let is_sequential = prepared.seq_len.is_some();
        let is_spatial = prepared.input_spatial.is_some();
        let is_multi_head = prepared.heads.len() > 1;
        if is_multi_head && (is_sequential || is_spatial) {
            return Err(EvolutionError::InvalidConfig(
                "P3 第一阶段多头 supervised evolution 仅支持平坦共享输入；空间 / 序列多头后续扩展"
                    .into(),
            ));
        }
        let is_spatial_classification =
            is_spatial && !is_sequential && !prepared.metric.is_segmentation();
        let user_registry = mutation_registry;

        if is_spatial_classification {
            // 空间分类默认使用已验证的完整搜索策略；用户仍可通过对应 with_* 方法显式覆盖。
            if !population_size_explicit {
                population_size = 8;
            }
            if !offspring_batch_size_explicit {
                offspring_batch_size = 8;
            }
            if batch_size.is_none() {
                batch_size = Some(128);
            }
            if max_inference_cost.is_none() {
                max_inference_cost = Some(3_000_000.0);
            }
            if !initial_portfolio_explicit && initial_portfolio.is_none() {
                initial_portfolio = Some(InitialPortfolioConfig::vision_classification());
            }
            if !candidate_scoring_explicit && candidate_scoring.is_none() {
                candidate_scoring = Some(CandidateScoringConfig::p5_lite());
            }
            if !final_refit_explicit && final_refit.is_none() {
                final_refit = Some(FinalRefitConfig::new(0.02, 2, 12));
            }
        } else if is_spatial && prepared.metric.is_segmentation() {
            // Dense segmentation 仍处于审计阶段：默认保持保守搜索规模，但接入同源
            // portfolio / P5-lite 观测，便于和 MNIST 搜索矩阵对齐。
            if !initial_portfolio_explicit && initial_portfolio.is_none() {
                initial_portfolio = Some(InitialPortfolioConfig::vision_segmentation());
            }
            if !candidate_scoring_explicit && candidate_scoring.is_none() {
                candidate_scoring = Some(CandidateScoringConfig::p5_lite());
            }
        }

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
        let effective_parallelism = parallelism.unwrap_or(auto_parallelism).max(1);

        // BLAS 线程守卫：避免 rayon 线程池内每个线程再开 BLAS 子线程导致超订阅
        // MKL 已配置 seq 模式不受影响；纯 Rust 无 BLAS 时环境变量无人读取也不受影响
        if effective_parallelism > 1 {
            for var in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"] {
                if std::env::var(var).is_err() {
                    // SAFETY: 在线程池创建前调用，此时只有主线程在运行
                    unsafe { std::env::set_var(var, "1") };
                }
            }
        }

        let effective_pareto_patience =
            pareto_patience.unwrap_or_else(|| (population_size * 2).max(20));
        let can_parallel = task_spec.supports_parallel_evaluation() && effective_parallelism > 1;
        let parallel_pool = if can_parallel {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(effective_parallelism)
                    .build()
                    .map_err(|e| {
                        EvolutionError::InvalidConfig(format!("创建评估线程池失败: {e}"))
                    })?,
            )
        } else {
            None
        };

        // 两阶段训练预算
        let phase1_gens = ((max_generations as f64) * 0.4).ceil() as usize;
        let user_convergence = {
            let mut c = convergence_config;
            // 序列任务 loss 曲线常有短 plateau → 更长的 patience，避免 UntilConverged 过早停止
            if is_sequential && matches!(c.budget, TrainingBudget::UntilConverged) {
                c.patience = c.patience.max(10);
                c.max_epochs = c.max_epochs.max(200);
            }
            c
        };

        // Phase 1 快速训练预算
        //
        // 序列任务（RNN/GRU/LSTM）在悬崖型 landscape（如 parity）上
        // 权重需要更多 epoch 才能越过临界点，否则所有候选都停留在随机猜测区间，
        // fitness 信噪比过低，NSGA-II 选择压力失效。因此对序列任务把下限/上限
        // 都放宽约 1.5×～2×，作为 multi-fidelity 评估的过渡方案。
        let effective_bs = batch_size.unwrap_or_else(|| auto_batch_size(n_train));
        let (min_fast, max_fast) = if is_sequential { (6, 15) } else { (3, 10) };
        let fast_epochs = (n_train / (effective_bs * 5)).clamp(min_fast, max_fast);
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
            // 序列模式同样迁移到 NodeLevel，统一走节点级演化路径
            let mut g = NetworkGenome::minimal_sequential(prepared.input_dim, prepared.output_dim);
            g.seq_len = prepared.seq_len;
            let _ = g.migrate_to_node_level();
            g
        } else if let Some(spatial) = prepared.input_spatial {
            let mut g = if prepared.metric.is_segmentation() {
                NetworkGenome::minimal_spatial_segmentation(
                    prepared.input_dim,
                    prepared.output_dim,
                    spatial,
                )
            } else {
                NetworkGenome::minimal_spatial(prepared.input_dim, prepared.output_dim, spatial)
            };
            // 分割任务需要保持 dense H/W 输出，先停在 NodeLevel，避免当前 FM
            // 分解对连续 Conv2d 的重连假设影响输出形状协议。
            let _ = g.migrate_to_node_level();
            if !prepared.metric.is_segmentation() {
                g.migrate_to_fm_level();
            }
            g
        } else if is_multi_head {
            let head_defs: Vec<(String, usize, bool, bool)> = prepared
                .heads
                .iter()
                .map(|head| {
                    (
                        head.name.clone(),
                        head.output_dim,
                        head.inference,
                        head.primary,
                    )
                })
                .collect();
            NetworkGenome::minimal_multi_head_flat(prepared.input_dim, &head_defs)
        } else {
            let mut g = NetworkGenome::minimal(prepared.input_dim, prepared.output_dim);
            // 平坦模式迁移到 NodeLevel
            let _ = g.migrate_to_node_level();
            g
        };

        let mut rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // 随机爆发初始化参数。
        // 空间分类已有明确初始候选族，默认不再额外爆发随机重结构，避免初始评估被少数
        // 大候选拖慢；其他任务仍按约束自动生成多样化初始种群。
        let burst_k = initial_burst.unwrap_or_else(|| {
            if is_spatial_classification {
                0
            } else {
                (constraints.max_layers / 2).max(2).min(8)
            }
        });

        // ====== 初始化种群 ======
        let init_reg = if let Some(ref user_reg) = user_reg_val {
            user_reg
        } else {
            phase1_reg.as_ref().unwrap()
        };
        let initial_seeds = initial_seed_genomes(&minimal_genome, &prepared, initial_portfolio);
        let mut init_genomes = Vec::with_capacity(population_size);
        for i in 0..population_size {
            if i >= initial_seeds.len() && burst_k == 0 {
                break;
            }
            let mut genome = initial_seeds[i % initial_seeds.len()].clone();
            let effective_burst = if i < initial_seeds.len() { 0 } else { burst_k };
            for _ in 0..effective_burst {
                let _ = init_reg.apply_random(&mut genome, &constraints, &mut rng);
            }
            if within_inference_cost(&genome, &complexity_metric, max_inference_cost) {
                init_genomes.push(genome);
            }
        }

        // 评估初始种群
        let seeds: Vec<u64> = (0..init_genomes.len())
            .map(|_| rng.r#gen::<u64>())
            .collect();
        let init_batch = timed_evaluate_batch(
            &base_task,
            init_genomes,
            &phase1_convergence,
            eval_runs,
            &complexity_metric,
            parallel_pool.as_ref(),
            seeds,
        );
        callback.on_evaluation_timing(0, "init", &init_batch.timing);
        let init_results = init_batch.results;

        let mut parents: Vec<(NetworkGenome, FitnessScore)> = init_results
            .into_iter()
            .map(|r| (r.genome, r.score))
            .collect();

        // 回退：全部失败则用 minimal genome
        if parents.is_empty() {
            let fallback_seed = rng.r#gen::<u64>();
            let fallback_batch = timed_evaluate_batch(
                &base_task,
                vec![minimal_genome.clone()],
                &phase1_convergence,
                eval_runs,
                &complexity_metric,
                None,
                vec![fallback_seed],
            );
            callback.on_evaluation_timing(0, "fallback", &fallback_batch.timing);
            let fallback_results = fallback_batch.results;
            if let Some(r) = fallback_results.into_iter().next() {
                parents.push((r.genome, r.score));
            } else {
                return Err(EvolutionError::InvalidConfig("无法评估任何初始个体".into()));
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
        let (_, repr_score) = select_representative(&archive, target_metric);
        let mut representative_score_snapshot = repr_score.clone();
        let mut archive_stagnation: usize = 0;
        let mut best_primary: f32 = f32::NEG_INFINITY;
        let mut primary_stagnation: usize = 0;
        let mut phase2_unlocked = false;
        let mut last_refit_best: Option<f32> = None;

        if repr_score.primary >= target_metric {
            let (repr_g, repr_s) = select_representative(&archive, target_metric);
            callback.on_new_best(0, repr_g, repr_s);
            return build_population_result(
                repr_g,
                repr_s,
                &archive,
                0,
                EvolutionStatus::TargetReached,
                &mut rng,
                &serial_task,
                seed,
            );
        }

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
                    repr_g,
                    repr_s,
                    &archive,
                    generation,
                    status,
                    &mut rng,
                    &serial_task,
                    seed,
                );
            }

            // 选择当前阶段的 mutation registry 和 convergence config
            if primary_stagnation >= stagnation_patience {
                phase2_unlocked = true;
            }
            let is_phase1 = generation < phase1_gens && !phase2_unlocked;
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
            let candidate_pool_target =
                candidate_pool_target(offspring_batch_size, candidate_scoring);
            let mut offspring_genomes = Vec::with_capacity(candidate_pool_target);
            let mut any_mutation_succeeded = false;

            let parent_scores: Vec<FitnessScore> = parents.iter().map(|(_, s)| s.clone()).collect();
            let parent_ranks = selection::pareto_rank(&parent_scores);
            let parent_distances = selection::crowding_distance(&parent_scores);

            for _ in 0..candidate_pool_target {
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
                    current_registry.apply_random_structural(&mut child, &constraints, &mut rng)
                } else {
                    current_registry.apply_random(&mut child, &constraints, &mut rng)
                };

                match mutation_result {
                    Ok(mutation_name) => {
                        any_mutation_succeeded = true;
                        // 若变异引入新 Conv2d 层块，图像分类自动 FM 分解；
                        // dense segmentation 当前保留普通 NodeLevel，避免破坏 H/W 输出协议。
                        if !prepared.metric.is_segmentation() {
                            child.migrate_to_fm_level();
                        }
                        if !within_inference_cost(&child, &complexity_metric, max_inference_cost) {
                            continue;
                        }
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

            if let Some(scoring) = candidate_scoring {
                let (filtered, prefilter_summary) =
                    prefilter_candidates(offspring_genomes, scoring, offspring_batch_size);
                callback.on_candidate_prefilter(generation, &prefilter_summary);
                offspring_genomes = filtered;
            } else if offspring_genomes.len() > offspring_batch_size {
                offspring_genomes.truncate(offspring_batch_size);
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
                        &serial_task,
                        seed,
                    );
                }
                continue;
            }

            // 2. 评估 offspring（并行 / 串行）
            let eval_seeds: Vec<u64> = (0..offspring_genomes.len())
                .map(|_| rng.r#gen::<u64>())
                .collect();
            // F4: Phase 1 对非序列任务启用 ASHA 多保真评估；序列任务跳过 ASHA。
            // 原因：parity/RNN 等序列任务的 loss 在早期呈"悬崖型"下降，
            // 极短的 rung-0（1 epoch）无法区分好坏架构，导致结构变异全部被淘汰。
            // 用户可通过 `.with_asha(None)` 完全关闭 / 通过自定义 rung_epochs 手动启用。
            let (eval_results, eval_timing, eval_phase) =
                if is_phase1 && asha.is_some() && !is_sequential {
                    let batch = evaluate_batch_asha(
                        &base_task,
                        offspring_genomes,
                        asha.as_ref().unwrap(),
                        current_convergence,
                        eval_runs,
                        &complexity_metric,
                        parallel_pool.as_ref(),
                        eval_seeds,
                    );
                    (batch.results, batch.timing, "asha")
                } else {
                    let batch = timed_evaluate_batch(
                        &base_task,
                        offspring_genomes,
                        current_convergence,
                        eval_runs,
                        &complexity_metric,
                        parallel_pool.as_ref(),
                        eval_seeds,
                    );
                    (batch.results, batch.timing, "offspring")
                };
            callback.on_evaluation_timing(generation, eval_phase, &eval_timing);

            let offspring_evaluated = eval_results.len();
            // 汇总权重继承统计（仅 NodeLevel genome 有意义）
            let total_inherited: usize = eval_results
                .iter()
                .filter_map(|r| r.inherit_report.as_ref())
                .map(|rep| rep.inherited + rep.partially_inherited)
                .sum();
            let total_reinitialized: usize = eval_results
                .iter()
                .filter_map(|r| r.inherit_report.as_ref())
                .map(|rep| rep.reinitialized)
                .sum();
            let total_params_seen: usize = total_inherited + total_reinitialized;
            if total_params_seen > 0 {
                callback.on_inherit_stats(generation, total_inherited, total_reinitialized);
            }

            let offspring: Vec<(NetworkGenome, FitnessScore)> = eval_results
                .into_iter()
                .map(|r| (r.genome, r.score))
                .collect();

            if offspring.is_empty() {
                let (archive_best_g, archive_best_s) =
                    select_representative(&archive, target_metric);
                callback.on_generation(generation, archive_best_g, f32::NAN, archive_best_s);
                continue;
            }

            // 3. NSGA-II 选择：parents ∪ offspring → 保留 population_size
            let pool: Vec<(NetworkGenome, FitnessScore)> =
                parents.into_iter().chain(offspring).collect();

            // 更新 archive（用完整 pool，确保不丢失非支配解）
            selection::update_archive(
                &mut archive,
                pool.iter().map(|(g, s)| (g.clone(), s.clone())).collect(),
            );

            parents = selection::nsga2_select(pool, population_size);

            // 4. 代表成员收敛跟踪：避免 archive 细碎 trade-off 变化掩盖主目标平台期
            let (_, representative_score) = select_representative(&archive, target_metric);
            if fitness_changed(&representative_score_snapshot, representative_score, 1e-6) {
                archive_stagnation = 0;
                representative_score_snapshot = representative_score.clone();
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

            // 6. 回调
            let (archive_best_g, archive_best_s) = select_representative(&archive, target_metric);
            callback.on_generation(generation, archive_best_g, f32::NAN, archive_best_s);
            let front_scores: Vec<FitnessScore> = archive.iter().map(|(_, s)| s.clone()).collect();
            let front_size = selection::pareto_front_indices(&front_scores).len();
            callback.on_population_evaluated(
                generation,
                parents.len(),
                offspring_evaluated,
                archive.len(),
                front_size,
                archive_best_s.primary,
                archive_best_s.inference_cost.unwrap_or(f32::INFINITY),
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
                snapshot_with_loss(&serial_task, target_genome, &build);
                return Ok(EvolutionResult {
                    build,
                    fitness: target_score.clone(),
                    generations: generation,
                    architecture_summary: format!("{target_genome}"),
                    status: EvolutionStatus::TargetReached,
                    genome: target_genome.clone(),
                    pareto_front: build_pareto_summaries(&archive),
                    pareto_genomes: archive.into_iter().map(|(g, _)| g).collect(),
                    evolution_seed: seed,
                });
            }

            // 8. 接近目标时，对 archive top-k 做最终复训/复评。
            //
            // 这是 P5 surrogate 前的低风险收尾机制：不预测、不跳过评估，只把预算集中到
            // 已经接近目标的候选，避免继续盲目扩大结构。
            if let Some(ref refit_cfg) = final_refit {
                let should_refit = archive_best_s.primary + refit_cfg.trigger_margin
                    >= target_metric
                    && last_refit_best
                        .map(|last| archive_best_s.primary >= last + refit_cfg.retrigger_delta)
                        .unwrap_or(true);
                if should_refit {
                    last_refit_best = Some(archive_best_s.primary);
                    let refit_seed = rng.r#gen::<u64>();
                    let refit_batch = final_refit_archive(
                        &base_task,
                        &archive,
                        refit_cfg,
                        eval_runs,
                        &complexity_metric,
                        parallel_pool.as_ref(),
                        refit_seed,
                    );
                    callback.on_evaluation_timing(generation, "final_refit", &refit_batch.timing);

                    let refit_evaluated = refit_batch.results.len();
                    let refit_offspring: Vec<(NetworkGenome, FitnessScore)> = refit_batch
                        .results
                        .into_iter()
                        .map(|r| (r.genome, r.score))
                        .collect();
                    if !refit_offspring.is_empty() {
                        selection::update_archive(&mut archive, refit_offspring.clone());
                        let pool: Vec<(NetworkGenome, FitnessScore)> =
                            parents.into_iter().chain(refit_offspring).collect();
                        parents = selection::nsga2_select(pool, population_size);

                        let (_, refit_best_s) = select_representative(&archive, target_metric);
                        callback.on_population_evaluated(
                            generation,
                            parents.len(),
                            refit_evaluated,
                            archive.len(),
                            front_size,
                            refit_best_s.primary,
                            refit_best_s.inference_cost.unwrap_or(f32::INFINITY),
                        );

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
                            snapshot_with_loss(&serial_task, target_genome, &build);
                            return Ok(EvolutionResult {
                                build,
                                fitness: target_score.clone(),
                                generations: generation,
                                architecture_summary: format!("{target_genome}"),
                                status: EvolutionStatus::TargetReached,
                                genome: target_genome.clone(),
                                pareto_front: build_pareto_summaries(&archive),
                                pareto_genomes: archive.into_iter().map(|(g, _)| g).collect(),
                                evolution_seed: seed,
                            });
                        }
                    }
                }
            }

            // 9. Pareto 收敛检查
            if archive_stagnation >= effective_pareto_patience {
                let (repr_g, repr_s) = select_representative(&archive, target_metric);
                return build_population_result(
                    repr_g,
                    repr_s,
                    &archive,
                    generation,
                    EvolutionStatus::ParetoConverged,
                    &mut rng,
                    &serial_task,
                    seed,
                );
            }
        }

        unreachable!("演化循环应由停止条件终止")
    }
}

// ==================== 独立辅助函数 ====================

#[derive(Clone, Debug, Default)]
struct EvalTiming {
    total: Duration,
    build: Duration,
    restore: Duration,
    train: Duration,
    capture: Duration,
    evaluate: Duration,
    cost: Duration,
    train_setup: Duration,
    train_shuffle: Duration,
    train_batch_slice: Duration,
    train_set_value: Duration,
    train_zero_grad: Duration,
    train_backward: Duration,
    train_backward_forward: Duration,
    train_backward_propagate: Duration,
    train_step: Duration,
    train_grad_norm: Duration,
}

struct TimedEvalBatch {
    results: Vec<EvalResult>,
    timing: EvaluationTimingSummary,
}

/// 评估结果（内部使用，所有字段均为 Send）
struct EvalResult {
    genome: NetworkGenome,
    score: FitnessScore,
    #[allow(dead_code)]
    loss: f32,
    inherit_report: Option<InheritReport>,
    timing: EvalTiming,
}

fn summarize_eval_timings(results: &[EvalResult], wall: Duration) -> EvaluationTimingSummary {
    let mut summary = EvaluationTimingSummary {
        batches: 1,
        candidates: results.len(),
        wall_secs: wall.as_secs_f64(),
        ..Default::default()
    };

    for result in results {
        summary.candidate_secs += result.timing.total.as_secs_f64();
        summary.build_secs += result.timing.build.as_secs_f64();
        summary.restore_secs += result.timing.restore.as_secs_f64();
        summary.train_secs += result.timing.train.as_secs_f64();
        summary.capture_secs += result.timing.capture.as_secs_f64();
        summary.evaluate_secs += result.timing.evaluate.as_secs_f64();
        summary.cost_secs += result.timing.cost.as_secs_f64();
        summary.train_setup_secs += result.timing.train_setup.as_secs_f64();
        summary.train_shuffle_secs += result.timing.train_shuffle.as_secs_f64();
        summary.train_batch_slice_secs += result.timing.train_batch_slice.as_secs_f64();
        summary.train_set_value_secs += result.timing.train_set_value.as_secs_f64();
        summary.train_zero_grad_secs += result.timing.train_zero_grad.as_secs_f64();
        summary.train_backward_secs += result.timing.train_backward.as_secs_f64();
        summary.train_backward_forward_secs += result.timing.train_backward_forward.as_secs_f64();
        summary.train_backward_propagate_secs +=
            result.timing.train_backward_propagate.as_secs_f64();
        summary.train_step_secs += result.timing.train_step.as_secs_f64();
        summary.train_grad_norm_secs += result.timing.train_grad_norm.as_secs_f64();

        summary.primary_min = Some(
            summary
                .primary_min
                .map_or(result.score.primary, |v| v.min(result.score.primary)),
        );
        summary.primary_max = Some(
            summary
                .primary_max
                .map_or(result.score.primary, |v| v.max(result.score.primary)),
        );
        summary.primary_sum += result.score.primary;
        summary.primary_count += 1;
        if let Some(cost) = result.score.inference_cost {
            summary.cost_min = Some(summary.cost_min.map_or(cost, |v| v.min(cost)));
            summary.cost_max = Some(summary.cost_max.map_or(cost, |v| v.max(cost)));
            summary.cost_sum += cost;
            summary.cost_count += 1;
        }
        let family = CandidateFeatures::from_genome(&result.genome).family();
        summary.evaluated_families.observe(family);
        let replace_best = summary
            .best_family_primary
            .map_or(true, |best| result.score.primary > best);
        if replace_best {
            summary.best_family = Some(family.as_str());
            summary.best_family_primary = Some(result.score.primary);
        }
    }

    summary
}

fn within_inference_cost(
    genome: &NetworkGenome,
    metric: &ComplexityMetric,
    max_cost: Option<f32>,
) -> bool {
    let Some(max_cost) = max_cost else {
        return true;
    };
    compute_inference_cost(genome, metric)
        .map(|cost| cost <= max_cost)
        // 复杂度估算失败时交给 build/eval 路径报出真实错误，不在这里静默淘汰。
        .unwrap_or(true)
}

fn initial_seed_genomes(
    minimal_genome: &NetworkGenome,
    prepared: &MaterializedTask,
    portfolio: Option<InitialPortfolioConfig>,
) -> Vec<NetworkGenome> {
    let Some(config) = portfolio else {
        return vec![minimal_genome.clone()];
    };
    let Some(spatial) = prepared.input_spatial else {
        return vec![minimal_genome.clone()];
    };
    if prepared.seq_len.is_some() {
        return vec![minimal_genome.clone()];
    }
    if prepared.heads.len() > 1 {
        return vec![minimal_genome.clone()];
    }

    let mut seeds = Vec::new();
    if prepared.metric.is_segmentation() {
        if config.include_tiny_cnn {
            seeds.push(minimal_genome.clone());
        }
        if config.include_lenet_tiny {
            seeds.push(NetworkGenome::spatial_segmentation_tiny(
                prepared.input_dim,
                prepared.output_dim,
                spatial,
            ));
        }
        if config.include_unet_lite
            && spatial.0 >= 4
            && spatial.1 >= 4
            && spatial.0 % 2 == 0
            && spatial.1 % 2 == 0
        {
            seeds.push(NetworkGenome::spatial_segmentation_unet_lite(
                prepared.input_dim,
                prepared.output_dim,
                spatial,
            ));
        }
        if seeds.is_empty() {
            seeds.push(minimal_genome.clone());
        }
        return seeds;
    }

    if config.include_flat_mlp {
        seeds.push(NetworkGenome::spatial_flat_mlp(
            prepared.input_dim,
            prepared.output_dim,
            spatial,
            config.flat_mlp_hidden,
        ));
    }
    if config.include_tiny_cnn {
        seeds.push(minimal_genome.clone());
    }
    if config.include_lenet_tiny {
        seeds.push(NetworkGenome::spatial_lenet_tiny(
            prepared.input_dim,
            prepared.output_dim,
            spatial,
        ));
    }

    if seeds.is_empty() {
        seeds.push(minimal_genome.clone());
    }
    seeds
}

fn candidate_pool_target(
    offspring_batch_size: usize,
    scoring: Option<CandidateScoringConfig>,
) -> usize {
    scoring
        .map(|cfg| offspring_batch_size.saturating_mul(cfg.pool_multiplier.max(1)))
        .unwrap_or(offspring_batch_size)
        .max(offspring_batch_size)
}

fn prefilter_candidates(
    candidates: Vec<NetworkGenome>,
    config: CandidateScoringConfig,
    fallback_keep: usize,
) -> (Vec<NetworkGenome>, CandidatePrefilterSummary) {
    let keep = config.keep_top_k.unwrap_or(fallback_keep).max(1);
    let mut scored = Vec::with_capacity(candidates.len());
    let mut summary = CandidatePrefilterSummary::default();

    for genome in candidates {
        let features = CandidateFeatures::from_genome(&genome);
        let score = score_candidate_features(&features);
        let family = features.family();
        summary.observe_generated(score, features.flops, family);
        scored.push((score, family, genome));
    }

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let kept_scored = select_prefilter_survivors(scored, keep, config.min_per_family);
    let kept = kept_scored
        .into_iter()
        .map(|candidate| {
            summary.observe_kept(candidate.score, candidate.family);
            candidate.genome
        })
        .collect();
    (kept, summary)
}

#[derive(Clone, Debug)]
struct ScoredCandidate {
    score: f32,
    family: CandidateFamily,
    genome: NetworkGenome,
}

fn select_prefilter_survivors(
    scored: Vec<(f32, CandidateFamily, NetworkGenome)>,
    keep: usize,
    min_per_family: usize,
) -> Vec<ScoredCandidate> {
    let keep = keep.max(1);
    if min_per_family == 0 || scored.len() <= keep {
        return scored
            .into_iter()
            .take(keep)
            .map(|(score, family, genome)| ScoredCandidate {
                score,
                family,
                genome,
            })
            .collect();
    }

    let mut selected = Vec::with_capacity(keep);
    let mut selected_indices = vec![false; scored.len()];
    for family in CandidateFamily::all() {
        let mut taken = 0;
        for (idx, (score, candidate_family, genome)) in scored.iter().enumerate() {
            if selected.len() >= keep || taken >= min_per_family {
                break;
            }
            if *candidate_family != family || selected_indices[idx] {
                continue;
            }
            selected.push(ScoredCandidate {
                score: *score,
                family: *candidate_family,
                genome: genome.clone(),
            });
            selected_indices[idx] = true;
            taken += 1;
        }
    }

    for (idx, (score, family, genome)) in scored.into_iter().enumerate() {
        if selected.len() >= keep {
            break;
        }
        if selected_indices[idx] {
            continue;
        }
        selected.push(ScoredCandidate {
            score,
            family,
            genome,
        });
    }

    selected.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    selected
}

#[derive(Clone, Debug, Default)]
struct CandidateFeatures {
    depth: usize,
    conv2d_blocks: usize,
    deformable_conv2d_blocks: usize,
    conv_transpose2d_blocks: usize,
    pool2d_blocks: usize,
    linear_blocks: usize,
    has_flatten: bool,
    concat_nodes: usize,
    flops: Option<f32>,
}

impl CandidateFeatures {
    fn from_genome(genome: &NetworkGenome) -> Self {
        let blocks = node_main_path(genome);
        let mut features = CandidateFeatures {
            depth: blocks.len(),
            flops: genome.total_flops().ok().map(|v| v as f32),
            ..Default::default()
        };

        for block in blocks {
            match block.kind {
                NodeBlockKind::Conv2d { .. } => features.conv2d_blocks += 1,
                NodeBlockKind::DeformableConv2d { .. } => {
                    features.conv2d_blocks += 1;
                    features.deformable_conv2d_blocks += 1;
                }
                NodeBlockKind::Pool2d { .. } => features.pool2d_blocks += 1,
                NodeBlockKind::Linear { .. } => features.linear_blocks += 1,
                NodeBlockKind::Flatten => features.has_flatten = true,
                _ => {}
            }
        }
        for node in genome.nodes().iter().filter(|node| node.enabled) {
            match &node.node_type {
                NodeTypeDescriptor::ConvTranspose2d { .. } => features.conv_transpose2d_blocks += 1,
                NodeTypeDescriptor::Concat { .. } => features.concat_nodes += 1,
                _ => {}
            }
        }
        features
    }

    fn family(&self) -> CandidateFamily {
        if !self.has_flatten
            && self.linear_blocks == 0
            && self.conv2d_blocks >= 3
            && self.pool2d_blocks >= 1
            && self.conv_transpose2d_blocks >= 1
            && self.concat_nodes >= 1
        {
            CandidateFamily::EncoderDecoderSeg
        } else if !self.has_flatten && self.linear_blocks == 0 && self.conv2d_blocks >= 3 {
            CandidateFamily::DenseSegDeep
        } else if !self.has_flatten && self.linear_blocks == 0 && self.conv2d_blocks >= 1 {
            CandidateFamily::DenseSegHead
        } else if self.conv2d_blocks >= 2 && self.pool2d_blocks >= 2 && self.linear_blocks >= 2 {
            CandidateFamily::LenetLike
        } else if self.conv2d_blocks >= 1 && self.pool2d_blocks >= 1 && self.linear_blocks >= 1 {
            CandidateFamily::TinyCnn
        } else if self.conv2d_blocks == 0 && self.has_flatten && self.linear_blocks >= 2 {
            CandidateFamily::FlatMlp
        } else if self.conv2d_blocks >= 1 && self.has_flatten && self.linear_blocks >= 1 {
            CandidateFamily::Hybrid
        } else {
            CandidateFamily::Other
        }
    }
}

fn score_candidate_features(features: &CandidateFeatures) -> f32 {
    let mut score = 0.0;

    if features.conv2d_blocks == 0 && features.has_flatten && features.linear_blocks >= 2 {
        score += 0.45;
    }
    if features.conv2d_blocks >= 1 && features.pool2d_blocks >= 1 && features.linear_blocks >= 1 {
        score += 0.55;
    }
    if features.conv2d_blocks >= 2 && features.pool2d_blocks >= 2 && features.linear_blocks >= 2 {
        score += 0.35;
    }
    if !features.has_flatten && features.linear_blocks == 0 && features.conv2d_blocks >= 2 {
        score += 0.50;
    }
    if !features.has_flatten
        && features.linear_blocks == 0
        && features.deformable_conv2d_blocks >= 1
    {
        score += 0.18;
    }
    if !features.has_flatten && features.linear_blocks == 0 && features.conv2d_blocks >= 3 {
        score += 0.20;
    }
    if !features.has_flatten
        && features.linear_blocks == 0
        && features.pool2d_blocks >= 1
        && features.conv_transpose2d_blocks >= 1
        && features.concat_nodes >= 1
    {
        score += 0.45;
    }
    if features.has_flatten && features.linear_blocks >= 1 {
        score += 0.10;
    }

    let flops_m = features.flops.unwrap_or(0.0) / 1_000_000.0;
    score -= flops_m * 0.04;
    score -= (features.depth.saturating_sub(10) as f32) * 0.02;
    score
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
    let total_start = Instant::now();

    let build_start = Instant::now();
    let build = genome.build(rng).ok()?;
    let build_elapsed = build_start.elapsed();

    let restore_start = Instant::now();
    let inherit_report = genome.restore_weights(&build).ok();
    let restore_elapsed = restore_start.elapsed();

    let train_start = Instant::now();
    let outcome = task.train(&genome, &build, convergence, rng).ok()?;
    let train_elapsed = train_start.elapsed();

    let capture_start = Instant::now();
    genome.capture_weights(&build).ok()?;
    let capture_elapsed = capture_start.elapsed();

    let evaluate_start = Instant::now();
    let mut score = evaluate_conservative(task, &genome, &build, eval_runs, rng).ok()?;
    let evaluate_elapsed = evaluate_start.elapsed();

    let cost_start = Instant::now();
    if let Ok(cost) = compute_inference_cost(&genome, complexity_metric) {
        score.inference_cost = Some(cost);
    }
    let cost_elapsed = cost_start.elapsed();

    // F3: 合并训练阶段算出的 learning-speed proxy
    if outcome.proxy.is_some() {
        score.primary_proxy = outcome.proxy;
    }
    let timing = EvalTiming {
        total: total_start.elapsed(),
        build: build_elapsed,
        restore: restore_elapsed,
        train: train_elapsed,
        capture: capture_elapsed,
        evaluate: evaluate_elapsed,
        cost: cost_elapsed,
        train_setup: outcome.timing.setup,
        train_shuffle: outcome.timing.shuffle,
        train_batch_slice: outcome.timing.batch_slice,
        train_set_value: outcome.timing.set_value,
        train_zero_grad: outcome.timing.zero_grad,
        train_backward: outcome.timing.backward,
        train_backward_forward: outcome.timing.backward_forward,
        train_backward_propagate: outcome.timing.backward_propagate,
        train_step: outcome.timing.optimizer_step,
        train_grad_norm: outcome.timing.grad_norm,
    };
    Some(EvalResult {
        genome,
        score,
        loss: outcome.final_loss,
        inherit_report,
        timing,
    })
}

/// 批量评估候选个体（支持 rayon 并行）
///
/// 任务运行时在 `run()` 开始时只 materialize 一次；这里的并行路径只克隆轻量级的
/// `TaskRuntime`（监督学习任务内部用 Arc 共享已 stack 的 Tensor），避免每代每个
/// worker 重新堆叠训练/测试数据。
fn evaluate_batch(
    base_task: &TaskRuntime,
    offspring: Vec<NetworkGenome>,
    convergence: &ConvergenceConfig,
    eval_runs: usize,
    complexity_metric: &ComplexityMetric,
    parallel_pool: Option<&ThreadPool>,
    seeds: Vec<u64>,
) -> Vec<EvalResult> {
    assert_eq!(offspring.len(), seeds.len());
    if let Some(pool) = parallel_pool.filter(|_| offspring.len() > 1) {
        use rayon::prelude::*;
        pool.install(|| {
            offspring
                .into_par_iter()
                .zip(seeds.into_par_iter())
                .map_init(
                    || base_task.clone(),
                    |local_task, (genome, seed)| {
                        let mut rng = StdRng::seed_from_u64(seed);
                        eval_candidate(
                            local_task,
                            genome,
                            convergence,
                            eval_runs,
                            complexity_metric,
                            &mut rng,
                        )
                    },
                )
                // 先保序收集 Option，再串行 flatten，确保相同 seed 下结果顺序一致
                .collect::<Vec<_>>()
                .into_iter()
                .flatten()
                .collect()
        })
    } else {
        // 串行路径

        offspring
            .into_iter()
            .zip(seeds)
            .filter_map(|(genome, seed)| {
                let mut rng = StdRng::seed_from_u64(seed);
                eval_candidate(
                    base_task,
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

fn timed_evaluate_batch(
    base_task: &TaskRuntime,
    offspring: Vec<NetworkGenome>,
    convergence: &ConvergenceConfig,
    eval_runs: usize,
    complexity_metric: &ComplexityMetric,
    parallel_pool: Option<&ThreadPool>,
    seeds: Vec<u64>,
) -> TimedEvalBatch {
    let wall_start = Instant::now();
    let results = evaluate_batch(
        base_task,
        offspring,
        convergence,
        eval_runs,
        complexity_metric,
        parallel_pool,
        seeds,
    );
    let timing = summarize_eval_timings(&results, wall_start.elapsed());
    TimedEvalBatch { results, timing }
}

fn final_refit_archive(
    base_task: &TaskRuntime,
    archive: &[(NetworkGenome, FitnessScore)],
    config: &FinalRefitConfig,
    eval_runs: usize,
    complexity_metric: &ComplexityMetric,
    parallel_pool: Option<&ThreadPool>,
    seed: u64,
) -> TimedEvalBatch {
    if archive.is_empty() {
        return TimedEvalBatch {
            results: Vec::new(),
            timing: EvaluationTimingSummary::default(),
        };
    }

    let mut ranked: Vec<&(NetworkGenome, FitnessScore)> = archive.iter().collect();
    ranked.sort_by(|a, b| {
        b.1.primary
            .partial_cmp(&a.1.primary)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.1.inference_cost
                    .unwrap_or(f32::INFINITY)
                    .partial_cmp(&b.1.inference_cost.unwrap_or(f32::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| {
                a.1.tiebreak_loss
                    .unwrap_or(f32::INFINITY)
                    .partial_cmp(&b.1.tiebreak_loss.unwrap_or(f32::INFINITY))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let candidates: Vec<NetworkGenome> = ranked
        .into_iter()
        .take(config.top_k)
        .map(|(genome, _)| genome.clone())
        .collect();
    let seeds = derive_refit_seeds(seed, candidates.len());
    let convergence = ConvergenceConfig {
        budget: TrainingBudget::FixedEpochs(config.epochs),
        ..ConvergenceConfig::default()
    };

    timed_evaluate_batch(
        base_task,
        candidates,
        &convergence,
        eval_runs,
        complexity_metric,
        parallel_pool,
        seeds,
    )
}

fn derive_refit_seeds(seed: u64, len: usize) -> Vec<u64> {
    const REFIT_SEED_STEP: u64 = 0xD1B5_4A32_D192_ED03;
    (0..len)
        .map(|i| seed.wrapping_add((i as u64).wrapping_mul(REFIT_SEED_STEP)))
        .collect()
}

// ==================== F4: ASHA 多保真评估 ====================

/// ASHA（Asynchronous Successive Halving Algorithm）配置
///
/// 参考 Li et al. 2020。Phase 1 从"均匀 FixedEpochs"改为阶梯式评估：
/// - rung 0：所有候选训练 `rung_epochs[0]` 个 epoch，按 primary 排序
/// - rung k (k>=1)：保留上一 rung 的 top `1/eta`（至少 1 个），继续训练
///   `rung_epochs[k]` 个 epoch（Lamarckian 权重延续）；末轮所有幸存者返回
///
/// `rung_epochs` 是每轮**增量** epoch 数，不是累积值。
/// 总训练 cost ≈ Σ (survivors_at_rung[k] * rung_epochs[k])，可调参控制预算。
#[derive(Clone, Debug)]
pub struct AshaConfig {
    /// 每 rung 训练的 epoch 数（增量）
    pub rung_epochs: Vec<usize>,
    /// 每 rung 的淘汰比例：保留 top 1/eta
    pub eta: usize,
    /// 每个中间 rung 至少保留的候选数。
    ///
    /// 低保真早期指标噪声较大，保留至少 2 个候选可避免最后一轮只剩单一结构族。
    pub min_survivors: usize,
    /// 中间 rung 每个结构族尽量保留的候选数。
    ///
    /// ASHA 的早期低保真评估可能低估 CNN/LeNet 这类慢热结构。默认保留 1 个
    /// 非 top elite 的结构族代表，让后续 rung 有机会验证其真实潜力。
    pub min_per_family: usize,
}

impl Default for AshaConfig {
    fn default() -> Self {
        // 与原单轮 FixedEpochs(~3-10) 相当的总预算：1 + 2 + 4 = 7
        Self {
            rung_epochs: vec![1, 2, 4],
            eta: 3,
            min_survivors: 2,
            min_per_family: 1,
        }
    }
}

impl AshaConfig {
    /// 校验：`rung_epochs` 非空、eta >= 2
    fn validated(&self) -> Self {
        let mut cfg = self.clone();
        if cfg.rung_epochs.is_empty() {
            cfg.rung_epochs = vec![1];
        }
        if cfg.eta < 2 {
            cfg.eta = 2;
        }
        cfg.min_survivors = cfg.min_survivors.max(1);
        cfg
    }
}

/// 计算 rung `k` 存活下来的候选数：`ceil(prev / eta)`，至少保留 1 个。
pub(crate) fn asha_keep_count(prev: usize, eta: usize) -> usize {
    if prev == 0 {
        return 0;
    }
    let eta = eta.max(2);
    prev.div_ceil(eta).max(1)
}

/// 按 rung 对候选进行 Successive Halving 评估
///
/// 每一轮内部复用 `evaluate_batch`（保留并行 / 串行路径与 seed 语义）。
/// rung-to-rung 之间，幸存者通过 `capture_weights` / `restore_weights` 自动
/// 延续训练权重——`eval_candidate` 在 train 前会 restore、train 后会 capture。
fn evaluate_batch_asha(
    base_task: &TaskRuntime,
    candidates: Vec<NetworkGenome>,
    asha: &AshaConfig,
    base_convergence: &ConvergenceConfig,
    eval_runs: usize,
    complexity_metric: &ComplexityMetric,
    parallel_pool: Option<&ThreadPool>,
    seeds: Vec<u64>,
) -> TimedEvalBatch {
    assert_eq!(candidates.len(), seeds.len());
    let asha = asha.validated();
    if candidates.is_empty() {
        return TimedEvalBatch {
            results: Vec::new(),
            timing: EvaluationTimingSummary::default(),
        };
    }

    let mut survivors: Vec<(NetworkGenome, u64)> = candidates.into_iter().zip(seeds).collect();
    let mut aggregate_timing = EvaluationTimingSummary::default();

    let total_rungs = asha.rung_epochs.len();
    for (rung_idx, &epochs) in asha.rung_epochs.iter().enumerate() {
        let rung_conv = ConvergenceConfig {
            budget: TrainingBudget::FixedEpochs(epochs.max(1)),
            ..base_convergence.clone()
        };

        // 每 rung 用独立的 seed 派生（加上 rung_idx），避免多轮评估随机性完全重复
        let (genomes, orig_seeds): (Vec<_>, Vec<_>) = survivors.into_iter().unzip();
        let rung_seeds: Vec<u64> = orig_seeds
            .iter()
            .map(|&seed| derive_asha_rung_seed(seed, rung_idx))
            .collect();

        let rung_batch = timed_evaluate_batch(
            base_task,
            genomes,
            &rung_conv,
            eval_runs,
            complexity_metric,
            parallel_pool,
            rung_seeds,
        );
        aggregate_timing.merge(&rung_batch.timing);
        let rung_results = rung_batch.results;

        // 最后一轮：全部幸存者进入 NSGA-II
        if rung_idx + 1 == total_rungs {
            aggregate_timing.replace_distribution_with(&rung_batch.timing);
            return TimedEvalBatch {
                results: rung_results,
                timing: aggregate_timing,
            };
        }

        // 中间 rung：按 primary 降序保留 top 1/eta
        let keep = asha_keep_count(rung_results.len(), asha.eta)
            .max(asha.min_survivors.min(rung_results.len()));
        let mut indexed: Vec<(usize, EvalResult)> = rung_results.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| {
            b.1.score
                .primary
                .partial_cmp(&a.1.score.primary)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let indexed = select_asha_survivors(indexed, keep, asha.min_per_family);

        // 下一轮：继续使用原 seed（rung 内部已派生过）
        survivors = indexed
            .into_iter()
            .map(|(i, r)| (r.genome, orig_seeds[i]))
            .collect();
    }

    // 理论上不会到达：total_rungs >= 1 已确保循环内 return
    TimedEvalBatch {
        results: Vec::new(),
        timing: aggregate_timing,
    }
}

fn select_asha_survivors(
    indexed: Vec<(usize, EvalResult)>,
    keep: usize,
    min_per_family: usize,
) -> Vec<(usize, EvalResult)> {
    let keep = keep.max(1);
    if min_per_family == 0 || indexed.len() <= keep {
        return indexed.into_iter().take(keep).collect();
    }

    let mut remaining: Vec<Option<(usize, EvalResult)>> = indexed.into_iter().map(Some).collect();
    let mut selected = Vec::with_capacity(keep);
    let mut selected_families = CandidateFamilyCounts::default();

    take_best_remaining(&mut remaining, &mut selected, &mut selected_families);

    for family in CandidateFamily::all() {
        while selected.len() < keep
            && selected_family_count(&selected_families, family) < min_per_family
        {
            if !take_best_family_remaining(
                &mut remaining,
                family,
                &mut selected,
                &mut selected_families,
            ) {
                break;
            }
        }
    }

    while selected.len() < keep {
        if !take_best_remaining(&mut remaining, &mut selected, &mut selected_families) {
            break;
        }
    }

    selected
}

fn take_best_remaining(
    remaining: &mut [Option<(usize, EvalResult)>],
    selected: &mut Vec<(usize, EvalResult)>,
    selected_families: &mut CandidateFamilyCounts,
) -> bool {
    let Some(idx) = remaining.iter().position(Option::is_some) else {
        return false;
    };
    let (_, result) = remaining[idx].as_ref().expect("position 已确认候选存在");
    selected_families.observe(CandidateFeatures::from_genome(&result.genome).family());
    selected.push(remaining[idx].take().expect("position 已确认候选存在"));
    true
}

fn take_best_family_remaining(
    remaining: &mut [Option<(usize, EvalResult)>],
    family: CandidateFamily,
    selected: &mut Vec<(usize, EvalResult)>,
    selected_families: &mut CandidateFamilyCounts,
) -> bool {
    let Some(idx) = remaining.iter().position(|entry| {
        entry.as_ref().is_some_and(|(_, result)| {
            CandidateFeatures::from_genome(&result.genome).family() == family
        })
    }) else {
        return false;
    };
    selected_families.observe(family);
    selected.push(remaining[idx].take().expect("position 已确认候选存在"));
    true
}

fn selected_family_count(counts: &CandidateFamilyCounts, family: CandidateFamily) -> usize {
    counts.get_family(family)
}

fn derive_asha_rung_seed(seed: u64, rung_idx: usize) -> u64 {
    const ASHA_RUNG_SEED_STEP: u64 = 0x9E37_79B9_7F4A_7C15;
    seed.wrapping_add((rung_idx as u64).wrapping_mul(ASHA_RUNG_SEED_STEP))
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
        build.graph.snapshot_once_from(&build.output_refs());
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

/// 代表成员的 FitnessScore 是否发生了实质变化
fn fitness_changed(prev: &FitnessScore, next: &FitnessScore, tolerance: f32) -> bool {
    if (prev.primary - next.primary).abs() > tolerance {
        return true;
    }
    match (prev.inference_cost, next.inference_cost) {
        (Some(a), Some(b)) if (a - b).abs() > tolerance => return true,
        (Some(_), None) | (None, Some(_)) => return true,
        _ => {}
    }
    match (prev.tiebreak_loss, next.tiebreak_loss) {
        (Some(a), Some(b)) if (a - b).abs() > tolerance => true,
        (Some(_), None) | (None, Some(_)) => true,
        _ => false,
    }
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
    evolution_seed: Option<u64>,
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
        evolution_seed,
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
    evolution_seed: Option<u64>,
) -> Result<EvolutionResult, EvolutionError> {
    let build = genome.build(rng)?;
    genome.restore_weights(&build)?;
    snapshot_with_loss(task, genome, &build);

    let fitness = best_score.unwrap_or(FitnessScore {
        primary: f32::NEG_INFINITY,
        inference_cost: None,
        tiebreak_loss: None,
        primary_proxy: None,
        report: MetricReport::empty(),
        head_reports: Vec::new(),
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
        evolution_seed,
    })
}

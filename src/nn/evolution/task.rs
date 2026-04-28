/*
 * @Author       : 老董
 * @Date         : 2026-03-07
 * @Description  : 演化任务 trait + SupervisedTask 实现
 *
 * 核心接口 EvolutionTask 定义了训练+评估的统一协议，
 * SupervisedTask 实现监督学习场景的完整管道：
 * 数据喂入 → 训练到收敛 → 评估适应度。
 *
 * 三层价值：
 * - 全局机制层：EvolutionTask trait + FitnessScore（所有范式受益）
 * - 监督学习层：SupervisedTask 的训练/评估实现
 * - 正确性层：二值解码、阈值绑定（分类场景必需）
 */

use rand::Rng;
use rand::rngs::StdRng;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::metrics;
use crate::metrics::traits::IntoClassLabels;
use crate::nn::{Adam, GraphError, Optimizer, SGD, Var, VarLossOps};
use crate::tensor::Tensor;

use super::builder::BuildResult;
use super::convergence::{ConvergenceConfig, ConvergenceDetector, TrainingBudget};
use super::error::EvolutionError;
use super::gene::{LossType, NetworkGenome, OptimizerType, TaskMetric, compatible_losses};

// ==================== ReportMetric & MetricReport ====================

/// 演化评估报告中的附加指标。
///
/// `ReportMetric` 只影响日志与结果报告，不参与 primary fitness、target 判断或 NSGA-II 选择。
/// 手写模型仍可直接调用 `crate::metrics` 中的同名函数。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ReportMetric {
    Accuracy,
    Precision,
    Recall,
    F1,
    R2,
    MeanSquaredError,
    MeanAbsoluteError,
    RootMeanSquaredError,
    MultiLabelLooseAccuracy,
    MultiLabelStrictAccuracy,
    PixelAccuracy,
    BinaryIoU,
    Dice,
    MeanIoU,
}

impl ReportMetric {
    /// 稳定的机器可读名称，用于日志、序列化后的展示与用户查询。
    pub const fn name(&self) -> &'static str {
        match self {
            ReportMetric::Accuracy => "accuracy",
            ReportMetric::Precision => "precision",
            ReportMetric::Recall => "recall",
            ReportMetric::F1 => "f1",
            ReportMetric::R2 => "r2",
            ReportMetric::MeanSquaredError => "mse",
            ReportMetric::MeanAbsoluteError => "mae",
            ReportMetric::RootMeanSquaredError => "rmse",
            ReportMetric::MultiLabelLooseAccuracy => "multilabel_loose_accuracy",
            ReportMetric::MultiLabelStrictAccuracy => "multilabel_strict_accuracy",
            ReportMetric::PixelAccuracy => "pixel_accuracy",
            ReportMetric::BinaryIoU => "binary_iou",
            ReportMetric::Dice => "dice",
            ReportMetric::MeanIoU => "mean_iou",
        }
    }

    /// 该报告指标是否适用于当前任务主指标。
    pub fn is_compatible_with(&self, task_metric: &TaskMetric) -> bool {
        match task_metric {
            TaskMetric::Accuracy => matches!(
                self,
                ReportMetric::Accuracy
                    | ReportMetric::Precision
                    | ReportMetric::Recall
                    | ReportMetric::F1
            ),
            TaskMetric::R2 => matches!(
                self,
                ReportMetric::R2
                    | ReportMetric::MeanSquaredError
                    | ReportMetric::MeanAbsoluteError
                    | ReportMetric::RootMeanSquaredError
            ),
            TaskMetric::MultiLabelAccuracy => matches!(
                self,
                ReportMetric::MultiLabelLooseAccuracy | ReportMetric::MultiLabelStrictAccuracy
            ),
            TaskMetric::BinaryIoU => matches!(
                self,
                ReportMetric::PixelAccuracy | ReportMetric::BinaryIoU | ReportMetric::Dice
            ),
            TaskMetric::MeanIoU => {
                matches!(self, ReportMetric::PixelAccuracy | ReportMetric::MeanIoU)
            }
        }
    }

    pub(crate) fn defaults_for_task(task_metric: &TaskMetric) -> Vec<Self> {
        match task_metric {
            TaskMetric::Accuracy => vec![
                ReportMetric::Accuracy,
                ReportMetric::Precision,
                ReportMetric::Recall,
                ReportMetric::F1,
            ],
            TaskMetric::R2 => vec![
                ReportMetric::R2,
                ReportMetric::MeanSquaredError,
                ReportMetric::MeanAbsoluteError,
                ReportMetric::RootMeanSquaredError,
            ],
            TaskMetric::MultiLabelAccuracy => vec![
                ReportMetric::MultiLabelLooseAccuracy,
                ReportMetric::MultiLabelStrictAccuracy,
            ],
            TaskMetric::BinaryIoU => vec![
                ReportMetric::PixelAccuracy,
                ReportMetric::BinaryIoU,
                ReportMetric::Dice,
            ],
            TaskMetric::MeanIoU => vec![ReportMetric::PixelAccuracy, ReportMetric::MeanIoU],
        }
    }
}

impl fmt::Display for ReportMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// 单个报告指标值。
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MetricValue {
    pub metric: ReportMetric,
    pub value: f32,
    pub n_samples: usize,
}

impl MetricValue {
    pub const fn new(metric: ReportMetric, value: f32, n_samples: usize) -> Self {
        Self {
            metric,
            value,
            n_samples,
        }
    }

    pub const fn name(&self) -> &'static str {
        self.metric.name()
    }
}

/// 一次评估产生的附加指标报告。
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MetricReport {
    entries: Vec<MetricValue>,
}

impl MetricReport {
    pub fn new(entries: Vec<MetricValue>) -> Self {
        Self { entries }
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[MetricValue] {
        &self.entries
    }

    pub fn get(&self, metric: ReportMetric) -> Option<&MetricValue> {
        self.entries.iter().find(|entry| entry.metric == metric)
    }

    pub fn value(&self, metric: ReportMetric) -> Option<f32> {
        self.get(metric).map(|entry| entry.value)
    }

    /// 紧凑展示格式：`accuracy=0.750 f1=0.733`。
    pub fn format_compact(&self) -> String {
        self.entries
            .iter()
            .map(|entry| format!("{}={:.3}", entry.name(), entry.value))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// 单个输出 head 的评估报告。
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeadMetricReport {
    pub head_name: String,
    pub primary: f32,
    pub tiebreak_loss: Option<f32>,
    pub report: MetricReport,
}

fn push_report_metric(
    metrics: &mut Vec<ReportMetric>,
    task_metric: &TaskMetric,
    metric: ReportMetric,
) {
    if metric.is_compatible_with(task_metric) && !metrics.contains(&metric) {
        metrics.push(metric);
    }
}

// ==================== auto_batch_size ====================

/// 自动选择 batch size
///
/// 策略：
/// - n ≤ 128: full-batch（小数据集，mini-batch 梯度噪声有害）
/// - 128 < n ≤ 10000: batch_size = 64
/// - n > 10000: batch_size = 256
pub fn auto_batch_size(n_samples: usize) -> usize {
    if n_samples <= 128 {
        n_samples
    } else if n_samples <= 10000 {
        64
    } else {
        256
    }
}

// ==================== FitnessScore ====================

/// 适应度分数
///
/// `primary` 保持纯粹的用户指标值（Accuracy / R² / Reward），
/// 不融合任何 tiebreak 信息——日志、target_metric 比较、用户回调
/// 看到的都是干净的指标。
///
/// `tiebreak_loss` 用于离散指标（如 Accuracy）的同分比较：
/// 同 accuracy 时，test loss 更低的结构更优。
/// 接受/回滚逻辑通过字典序比较实现。
///
/// `primary_proxy` 是 F3 引入的"学习速度"代理（越高越好）：
/// 仅当用户显式启用（`Evolution::with_primary_proxy`）才会有值；
/// 当 primary 处于 plateau（同 rank 同 crowding distance）时，
/// NSGA-II 的 tiebreak 会优先比较 proxy，再回退到 tiebreak_loss。
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FitnessScore {
    /// 主目标（越高越好），如 Accuracy / R² / Reward
    pub primary: f32,
    /// 可选副目标（越低越好），如 FLOPs / 推理延迟
    pub inference_cost: Option<f32>,
    /// 离散指标的 tiebreaker（越低越好）：test set 上的 loss。
    /// 仅在 primary 相等时用于区分同指标的不同结构。
    /// 连续指标（R² 等）不需要 tiebreak，此字段为 None。
    pub tiebreak_loss: Option<f32>,
    /// F3 学习速度代理（越高越好）：在 plateau 上用于打破 NSGA-II 平局。
    /// 仅在启用 `ProxyKind` 时有值，默认为 `None` 以保持向后兼容。
    #[serde(default)]
    pub primary_proxy: Option<f32>,
    /// 附加评估指标报告。仅用于可观测性，不参与选择或收敛判断。
    #[serde(default)]
    pub report: MetricReport,
    /// 多头任务的逐 head 指标报告。单头旧任务默认为空，保持兼容。
    #[serde(default)]
    pub head_reports: Vec<HeadMetricReport>,
}

// ==================== ProxyKind & TrainOutcome（F3）====================

/// 学习速度代理的类型
///
/// 当前只支持 `LossSlope`（训练 loss 的下降速率），未来可扩展
/// `GradVar` / `EarlyAccuracy` 等。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProxyKind {
    /// Loss 斜率：`(l_start - l_end) / epochs`，越高越好（下降越快）。
    LossSlope,
}

/// 训练产出：最终 loss + 可选的学习速度 proxy
///
/// `evaluate()` 不感知训练动态；proxy 由 `train()` 在其内部记录 loss
/// 轨迹并计算好，然后通过 `TrainOutcome` 回传给主循环合并到
/// `FitnessScore::primary_proxy`。
#[derive(Clone, Debug)]
pub struct TrainOutcome {
    pub final_loss: f32,
    pub proxy: Option<f32>,
    pub(crate) timing: TrainTiming,
}

impl TrainOutcome {
    pub fn new(final_loss: f32) -> Self {
        Self {
            final_loss,
            proxy: None,
            timing: TrainTiming::default(),
        }
    }

    /// `final_loss.is_finite()` 的便捷投影，方便测试与调用点直接使用。
    pub fn is_finite(&self) -> bool {
        self.final_loss.is_finite()
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct TrainTiming {
    pub setup: Duration,
    pub shuffle: Duration,
    pub batch_slice: Duration,
    pub set_value: Duration,
    pub zero_grad: Duration,
    pub backward: Duration,
    pub backward_forward: Duration,
    pub backward_propagate: Duration,
    pub optimizer_step: Duration,
    pub grad_norm: Duration,
}

impl std::fmt::Display for TrainOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display 直接输出 final_loss 数值，便于测试里 `{loss}` 格式化。
        write!(f, "{}", self.final_loss)
    }
}

impl PartialEq<f32> for TrainOutcome {
    fn eq(&self, other: &f32) -> bool {
        self.final_loss == *other
    }
}

impl PartialOrd<f32> for TrainOutcome {
    fn partial_cmp(&self, other: &f32) -> Option<std::cmp::Ordering> {
        self.final_loss.partial_cmp(other)
    }
}

/// 从 loss 轨迹计算 `LossSlope` 代理：`(l_first_window - l_last_window) / epochs`。
///
/// - 轨迹长度 < 3 返回 None（信号不足）
/// - 窗口取 `max(1, n / 4)`：首尾各取一段均值，降低单点噪声影响
/// - 含 NaN/Inf 时返回 None
pub(crate) fn compute_loss_slope_proxy(curve: &[f32]) -> Option<f32> {
    if curve.len() < 3 {
        return None;
    }
    if curve.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let n = curve.len();
    let w = (n / 4).max(1);
    let head: f32 = curve[..w].iter().sum::<f32>() / w as f32;
    let tail: f32 = curve[n - w..].iter().sum::<f32>() / w as f32;
    // 以 epoch 数归一化，避免不同候选的训练长度差造成不公平比较
    Some((head - tail) / n as f32)
}

// ==================== EvolutionTask trait ====================

/// 演化任务 trait（所有学习范式的统一接口）
///
/// 接收 BuildResult（包含 input/output/layer_params），
/// 无需单独传递 Graph（Var 内部已持有图引用，Graph 由 BuildResult.graph 保活）。
///
/// rng 由 Evolution 主循环传入，用于训练/评估中的随机性（Dropout mask、
/// mini-batch 顺序等），确保 Evolution seed 能完整控制所有随机行为。
pub trait EvolutionTask {
    /// 训练网络，返回训练产出（最终 loss + 可选 proxy）
    fn train(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
        convergence: &ConvergenceConfig,
        rng: &mut StdRng,
    ) -> Result<TrainOutcome, GraphError>;

    /// 评估任务指标
    fn evaluate(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
        rng: &mut StdRng,
    ) -> Result<FitnessScore, GraphError>;

    /// 配置训练 batch size（默认空操作）
    ///
    /// `SupervisedTask` 使用此值覆盖自动策略；
    /// 自定义 Task（如 RL/半监督）忽略此方法，使用自身的采样策略。
    fn configure_batch_size(&mut self, _batch_size: Option<usize>) {}

    /// 配置学习速度代理（默认空操作）。
    ///
    /// `SupervisedTask` 启用后会在 `train()` 内记录 loss 轨迹并计算 proxy。
    fn configure_proxy(&mut self, _kind: Option<ProxyKind>) {}

    /// 追加报告指标（默认空操作）。
    ///
    /// 报告指标只用于日志和结果展示，不影响演化选择。
    fn configure_report_metrics(&mut self, _metrics: &[ReportMetric]) {}

    /// 创建仅用于可视化的 Loss 节点（默认返回 None，子类可覆盖）
    ///
    /// 返回的 Var 包含 TargetInput + Loss 节点，用于 snapshot 时
    /// 呈现完整的 "输入 → 网络 → Loss" 计算图（八角形 Loss + 橙色 Target）。
    /// 不需要设置实际数据——仅用于拓扑快照。
    fn create_visualization_loss(
        &self,
        _genome: &NetworkGenome,
        _build: &BuildResult,
    ) -> Option<Var> {
        None
    }
}

// ==================== SupervisedSpec / HeadSpec ====================

/// 监督学习任务配置：共享 inputs，一个或多个命名输出 head。
///
/// 第一阶段只支持每个 head 的 targets 与共享 inputs 样本一一对齐；
/// 不支持不同 head 使用不同 input dataset 或缺标注 masking。
#[derive(Clone)]
pub struct SupervisedSpec {
    pub(crate) train_inputs: Vec<Tensor>,
    pub(crate) test_inputs: Vec<Tensor>,
    pub(crate) heads: Vec<HeadSpec>,
    primary_head: Option<String>,
}

impl SupervisedSpec {
    pub fn new(train_inputs: Vec<Tensor>, test_inputs: Vec<Tensor>) -> Self {
        Self {
            train_inputs,
            test_inputs,
            heads: Vec::new(),
            primary_head: None,
        }
    }

    pub fn head_targets(
        mut self,
        name: impl Into<String>,
        train_targets: Vec<Tensor>,
        test_targets: Vec<Tensor>,
        metric: TaskMetric,
    ) -> Self {
        let is_first = self.heads.is_empty();
        self.heads.push(HeadSpec {
            name: name.into(),
            train_targets,
            test_targets,
            metric,
            loss_weight: 1.0,
            metric_weight: 1.0,
            inference: is_first,
            primary: is_first,
            loss_override: None,
        });
        self
    }

    pub fn primary_head(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        for head in &mut self.heads {
            let is_primary = head.name == name;
            head.primary = is_primary;
            if is_primary {
                head.inference = true;
            }
        }
        self.primary_head = Some(name);
        self
    }

    pub fn with_head_loss_weight(mut self, name: &str, weight: f32) -> Self {
        for head in &mut self.heads {
            if head.name == name {
                head.loss_weight = weight;
            }
        }
        self
    }

    pub fn with_head_metric_weight(mut self, name: &str, weight: f32) -> Self {
        for head in &mut self.heads {
            if head.name == name {
                head.metric_weight = weight;
            }
        }
        self
    }

    pub fn with_head_inference(mut self, name: &str, inference: bool) -> Self {
        for head in &mut self.heads {
            if head.name == name {
                head.inference = inference;
            }
        }
        self
    }
}

/// 单个 supervised 输出 head 的配置。
#[derive(Clone)]
pub struct HeadSpec {
    pub name: String,
    pub train_targets: Vec<Tensor>,
    pub test_targets: Vec<Tensor>,
    pub metric: TaskMetric,
    pub loss_weight: f32,
    pub metric_weight: f32,
    pub inference: bool,
    pub primary: bool,
    pub loss_override: Option<LossType>,
}

#[derive(Clone)]
pub(crate) struct MaterializedHead {
    pub name: String,
    pub metric: TaskMetric,
    pub output_dim: usize,
    pub loss_weight: f32,
    pub metric_weight: f32,
    pub inference: bool,
    pub primary: bool,
    pub loss_override: Option<LossType>,
}

#[derive(Clone)]
struct SupervisedHeadRuntime {
    meta: MaterializedHead,
    train_y: Arc<Tensor>,
    test_y: Arc<Tensor>,
    report_metrics: Vec<ReportMetric>,
}

// ==================== SupervisedTask ====================

/// 监督学习任务
///
/// 构造器接受 per-sample 的 `Vec<Tensor>`，内部立即 stack 成 batched Tensor。
/// 自动根据数据量选择 full-batch 或 mini-batch 训练策略。
///
/// 支持三种输入形态：
/// - 平坦数据：每个样本 `[input_dim]` → stack 为 `[n, input_dim]`
/// - 序列数据：每个样本 `[seq_len, input_dim]` → 变长零填充后 stack 为 `[n, max_seq, input_dim]`
/// - 空间数据：每个样本 `[C, H, W]` → stack 为 `[n, C, H, W]`
#[derive(Clone)]
pub struct SupervisedTask {
    train_x: Arc<Tensor>, // [n, input_dim] / [n, seq_len, input_dim] / [n, C, H, W]
    test_x: Arc<Tensor>,  // 同 train_x
    heads: Vec<SupervisedHeadRuntime>,
    batch_size: Option<usize>, // None = 自动策略，Some = 显式指定
    /// F3: 学习速度代理类型（None = 关闭，不记录 loss 轨迹）
    proxy_kind: Option<ProxyKind>,
}

impl SupervisedTask {
    /// 创建监督学习任务
    ///
    /// 输入/标签为空或数量不匹配时返回 `Err(EvolutionError::InvalidData)`。
    pub fn new(
        train_data: (Vec<Tensor>, Vec<Tensor>),
        test_data: (Vec<Tensor>, Vec<Tensor>),
        metric: TaskMetric,
    ) -> Result<Self, EvolutionError> {
        let spec = SupervisedSpec::new(train_data.0, test_data.0)
            .head_targets("output", train_data.1, test_data.1, metric)
            .primary_head("output");
        Self::from_spec(spec)
    }

    pub fn from_spec(spec: SupervisedSpec) -> Result<Self, EvolutionError> {
        if spec.train_inputs.is_empty() {
            return Err(EvolutionError::InvalidData("训练输入不能为空".into()));
        }
        if spec.test_inputs.is_empty() {
            return Err(EvolutionError::InvalidData("测试输入不能为空".into()));
        }
        if spec.heads.is_empty() {
            return Err(EvolutionError::InvalidData(
                "监督任务至少需要一个 head".into(),
            ));
        }

        // 根据样本维度分类处理
        let sample_ndim = spec.train_inputs[0].dimension();
        let (train_x_stacked, test_x_stacked) = if sample_ndim == 2 {
            // 序列数据 [seq_len, input_dim]：变长零填充至最大长度
            let max_seq = spec
                .train_inputs
                .iter()
                .chain(spec.test_inputs.iter())
                .map(|t| t.shape()[0])
                .max()
                .unwrap();
            // 零填充至 max_seq_len
            let pad_to_max = |tensors: &[Tensor]| -> Vec<Tensor> {
                tensors
                    .iter()
                    .map(|t| {
                        let s = t.shape()[0];
                        if s < max_seq {
                            t.pad(&[(0, max_seq - s), (0, 0)], 0.0)
                        } else {
                            t.clone()
                        }
                    })
                    .collect()
            };
            let padded_train = pad_to_max(&spec.train_inputs);
            let padded_test = pad_to_max(&spec.test_inputs);
            let tr_refs: Vec<&Tensor> = padded_train.iter().collect();
            let te_refs: Vec<&Tensor> = padded_test.iter().collect();
            (Tensor::stack(&tr_refs, 0), Tensor::stack(&te_refs, 0))
        } else {
            // 平坦数据 [input_dim] → [n, input_dim]
            // 空间数据 [C, H, W] → [n, C, H, W]
            // 两者均可通过 stack(dim=0) 统一处理
            let tr_refs: Vec<&Tensor> = spec.train_inputs.iter().collect();
            let te_refs: Vec<&Tensor> = spec.test_inputs.iter().collect();
            (Tensor::stack(&tr_refs, 0), Tensor::stack(&te_refs, 0))
        };

        let mut names = std::collections::HashSet::new();
        let mut heads = Vec::with_capacity(spec.heads.len());
        for mut head in spec.heads {
            if head.name.is_empty() {
                return Err(EvolutionError::InvalidData("head name 不能为空".into()));
            }
            if !names.insert(head.name.clone()) {
                return Err(EvolutionError::InvalidData(format!(
                    "重复的 head name: {}",
                    head.name
                )));
            }
            if head.train_targets.len() != spec.train_inputs.len() {
                return Err(EvolutionError::InvalidData(format!(
                    "head '{}' 训练 target 和输入数量不匹配：target={}, input={}",
                    head.name,
                    head.train_targets.len(),
                    spec.train_inputs.len()
                )));
            }
            if head.test_targets.len() != spec.test_inputs.len() {
                return Err(EvolutionError::InvalidData(format!(
                    "head '{}' 测试 target 和输入数量不匹配：target={}, input={}",
                    head.name,
                    head.test_targets.len(),
                    spec.test_inputs.len()
                )));
            }
            if !head.loss_weight.is_finite() || head.loss_weight < 0.0 {
                head.loss_weight = 1.0;
            }
            if !head.metric_weight.is_finite() || head.metric_weight < 0.0 {
                head.metric_weight = 1.0;
            }
            let output_dim = if head.metric.is_segmentation() {
                head.train_targets[0].shape()[0]
            } else {
                head.train_targets[0].size()
            };
            if let Some(loss) = &head.loss_override {
                if !compatible_losses(&head.metric, output_dim).contains(loss) {
                    return Err(EvolutionError::InvalidData(format!(
                        "head '{}' 的 loss_override {:?} 与 metric {:?} / output_dim {} 不兼容",
                        head.name, loss, head.metric, output_dim
                    )));
                }
            }
            let train_refs: Vec<&Tensor> = head.train_targets.iter().collect();
            let test_refs: Vec<&Tensor> = head.test_targets.iter().collect();
            let report_metrics = ReportMetric::defaults_for_task(&head.metric);
            heads.push(SupervisedHeadRuntime {
                meta: MaterializedHead {
                    name: head.name,
                    metric: head.metric,
                    output_dim,
                    loss_weight: head.loss_weight,
                    metric_weight: head.metric_weight,
                    inference: head.inference,
                    primary: head.primary,
                    loss_override: head.loss_override,
                },
                train_y: Arc::new(Tensor::stack(&train_refs, 0)),
                test_y: Arc::new(Tensor::stack(&test_refs, 0)),
                report_metrics,
            });
        }
        if !heads.iter().any(|head| head.meta.primary) {
            heads[0].meta.primary = true;
            heads[0].meta.inference = true;
        }

        Ok(Self {
            train_x: Arc::new(train_x_stacked),
            test_x: Arc::new(test_x_stacked),
            heads,
            batch_size: None,
            proxy_kind: None,
        })
    }

    /// 获取任务指标类型
    pub fn metric(&self) -> &TaskMetric {
        &self.primary_head().meta.metric
    }

    /// 显式设置 batch size（覆盖自动策略）
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size 必须 > 0");
        self.batch_size = Some(batch_size);
        self
    }

    /// 启用学习速度代理（F3）：训练时记录 loss 轨迹并计算 proxy。
    pub fn with_primary_proxy(mut self, kind: ProxyKind) -> Self {
        self.proxy_kind = Some(kind);
        self
    }

    /// 在默认报告指标基础上追加用户指定的报告指标。
    ///
    /// 不兼容当前任务类型的指标会被忽略，重复指标会自动去重。
    pub fn with_report_metrics(mut self, metrics: impl IntoIterator<Item = ReportMetric>) -> Self {
        self.configure_report_metrics(&metrics.into_iter().collect::<Vec<_>>());
        self
    }

    /// 获取当前报告指标配置。
    pub fn report_metrics(&self) -> &[ReportMetric] {
        &self.primary_head().report_metrics
    }

    pub(crate) fn head_metas(&self) -> Vec<MaterializedHead> {
        self.heads.iter().map(|head| head.meta.clone()).collect()
    }

    pub(crate) fn train_len(&self) -> usize {
        self.train_x.shape()[0]
    }

    fn primary_head(&self) -> &SupervisedHeadRuntime {
        self.heads
            .iter()
            .find(|head| head.meta.primary)
            .unwrap_or(&self.heads[0])
    }

    /// 解析实际使用的 batch size
    ///
    /// 优先级：genome 显式值 > 用户设置 > 自动策略
    fn effective_batch_size(&self, genome: &NetworkGenome, n_samples: usize) -> usize {
        genome
            .training_config
            .batch_size
            .or(self.batch_size)
            .unwrap_or_else(|| auto_batch_size(n_samples))
    }
}

impl EvolutionTask for SupervisedTask {
    fn train(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
        convergence: &ConvergenceConfig,
        rng: &mut StdRng,
    ) -> Result<TrainOutcome, GraphError> {
        assert!(
            genome.training_config.weight_decay == 0.0,
            "weight_decay 尚未支持，必须为 0.0"
        );

        // debug 模式下验证 loss_override 兼容性（正常演化路径由 MutateLossFunctionMutation 保障）
        for head in &self.heads {
            debug_assert!(
                genome
                    .training_config
                    .loss_override
                    .as_ref()
                    .map_or(true, |loss| {
                        compatible_losses(&head.meta.metric, head.meta.output_dim).contains(loss)
                    }),
                "loss_override {:?} 与 head '{}' 不兼容（metric={:?}, output_dim={}，兼容列表={:?}）",
                genome.training_config.loss_override,
                head.meta.name,
                head.meta.metric,
                head.meta.output_dim,
                compatible_losses(&head.meta.metric, head.meta.output_dim)
            );
        }

        let mut timing = TrainTiming::default();
        let setup_start = Instant::now();
        build.graph.train();

        let n_samples = self.train_x.shape()[0];
        let bs = self.effective_batch_size(genome, n_samples);

        // target / loss / optimizer 各创建一次，不在 epoch 内重建
        let target_vars: Vec<Var> = self
            .heads
            .iter()
            .map(|head| build.graph.target(head.train_y.shape()))
            .collect::<Result<Vec<_>, _>>()?;
        let loss_vars = self
            .heads
            .iter()
            .enumerate()
            .map(|(idx, head)| {
                let output = build.outputs.get(idx).ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "缺少 head '{}' 对应的输出节点",
                        head.meta.name
                    ))
                })?;
                let loss_type = head_loss_type(genome, &head.meta);
                create_weighted_loss(output, &target_vars[idx], &loss_type, head.meta.loss_weight)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let loss_var = aggregate_loss_vars(loss_vars)?;

        let params = build.all_parameters();
        let lr = genome.training_config.learning_rate;
        let mut optimizer: Box<dyn Optimizer> = match genome.training_config.optimizer_type {
            OptimizerType::Adam => Box::new(Adam::new(&build.graph, &params, lr)),
            OptimizerType::SGD => Box::new(SGD::new(&build.graph, &params, lr)),
        };

        let mut detector = ConvergenceDetector::new(convergence.clone());
        let mut final_loss = f32::NAN;
        // F3: 仅在启用 proxy 时才记录 loss 轨迹，避免不必要开销
        let mut loss_curve: Option<Vec<f32>> = self.proxy_kind.map(|_| Vec::new());
        let needs_grad_norm = !matches!(convergence.budget, TrainingBudget::FixedEpochs(_));
        timing.setup += setup_start.elapsed();

        if bs >= n_samples {
            // ====== Full-batch 路径 ======
            let set_start = Instant::now();
            build.input.set_value(&self.train_x)?;
            for (target, head) in target_vars.iter().zip(&self.heads) {
                target.set_value(&head.train_y)?;
            }
            timing.set_value += set_start.elapsed();

            for epoch in 0.. {
                let loss_val = train_one_batch(&mut *optimizer, &loss_var, &mut timing)?;
                let grad_norm = if needs_grad_norm {
                    let grad_start = Instant::now();
                    let norm = compute_grad_norm(&params)?;
                    timing.grad_norm += grad_start.elapsed();
                    norm
                } else {
                    0.0
                };
                final_loss = loss_val;
                if let Some(curve) = loss_curve.as_mut() {
                    curve.push(loss_val);
                }

                if detector.should_stop(epoch, loss_val, grad_norm).is_some() {
                    break;
                }
            }
        } else {
            // ====== Mini-batch 路径 ======
            for epoch in 0.. {
                let mut epoch_loss_sum = 0.0;
                let mut n_batches = 0;
                let mut offset = 0;
                let shuffle_seed: u64 = rng.r#gen();
                let shuffle_start = Instant::now();
                let mut shuffled_x = self.train_x.as_ref().clone();
                let mut shuffled_ys: Vec<Tensor> = self
                    .heads
                    .iter()
                    .map(|head| head.train_y.as_ref().clone())
                    .collect();
                shuffled_x.shuffle_mut_seeded(Some(0), shuffle_seed);
                for shuffled_y in &mut shuffled_ys {
                    shuffled_y.shuffle_mut_seeded(Some(0), shuffle_seed);
                }
                timing.shuffle += shuffle_start.elapsed();

                while offset < n_samples {
                    let end = (offset + bs).min(n_samples);
                    let slice_start = Instant::now();
                    let batch_x = shuffled_x.narrow(0, offset, end - offset);
                    let batch_ys: Vec<Tensor> = shuffled_ys
                        .iter()
                        .map(|y| y.narrow(0, offset, end - offset))
                        .collect();
                    timing.batch_slice += slice_start.elapsed();

                    let set_start = Instant::now();
                    build.input.set_value(&batch_x)?;
                    for (target, batch_y) in target_vars.iter().zip(&batch_ys) {
                        target.set_value(batch_y)?;
                    }
                    timing.set_value += set_start.elapsed();

                    let loss_val = train_one_batch(&mut *optimizer, &loss_var, &mut timing)?;
                    epoch_loss_sum += loss_val;
                    n_batches += 1;
                    offset = end;
                }

                let avg_loss = epoch_loss_sum / n_batches as f32;
                let grad_norm = if needs_grad_norm {
                    let grad_start = Instant::now();
                    let norm = compute_grad_norm(&params)?;
                    timing.grad_norm += grad_start.elapsed();
                    norm
                } else {
                    0.0
                };
                final_loss = avg_loss;
                if let Some(curve) = loss_curve.as_mut() {
                    curve.push(avg_loss);
                }

                if detector.should_stop(epoch, avg_loss, grad_norm).is_some() {
                    break;
                }
            }
        }

        let proxy = match (self.proxy_kind, loss_curve.as_ref()) {
            (Some(ProxyKind::LossSlope), Some(curve)) => compute_loss_slope_proxy(curve),
            _ => None,
        };
        Ok(TrainOutcome {
            final_loss,
            proxy,
            timing,
        })
    }

    fn configure_batch_size(&mut self, batch_size: Option<usize>) {
        self.batch_size = batch_size;
    }

    fn configure_proxy(&mut self, kind: Option<ProxyKind>) {
        self.proxy_kind = kind;
    }

    fn configure_report_metrics(&mut self, metrics: &[ReportMetric]) {
        for head in &mut self.heads {
            for &metric in metrics {
                push_report_metric(&mut head.report_metrics, &head.meta.metric, metric);
            }
        }
    }

    fn create_visualization_loss(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
    ) -> Option<Var> {
        let losses = self
            .heads
            .iter()
            .enumerate()
            .filter_map(|(idx, head)| {
                let output = build.outputs.get(idx)?;
                let output_shape = output.value_expected_shape();
                let target = build.graph.target(&output_shape).ok()?;
                let loss_type = head_loss_type(genome, &head.meta);
                create_weighted_loss(output, &target, &loss_type, head.meta.loss_weight).ok()
            })
            .collect::<Vec<_>>();
        aggregate_loss_vars(losses).ok()
    }

    fn evaluate(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
        _rng: &mut StdRng,
    ) -> Result<FitnessScore, GraphError> {
        build.graph.eval();

        build.input.set_value(&self.test_x)?;

        // 创建 target + loss 用于计算 tiebreak（eval 专用节点）
        let target_vars: Vec<Var> = self
            .heads
            .iter()
            .map(|head| build.graph.target(head.test_y.shape()))
            .collect::<Result<Vec<_>, _>>()?;
        for (target, head) in target_vars.iter().zip(&self.heads) {
            target.set_value(&head.test_y)?;
        }
        let loss_vars = self
            .heads
            .iter()
            .enumerate()
            .map(|(idx, head)| {
                let output = build.outputs.get(idx).ok_or_else(|| {
                    GraphError::ComputationError(format!(
                        "缺少 head '{}' 对应的输出节点",
                        head.meta.name
                    ))
                })?;
                let loss_type = head_loss_type(genome, &head.meta);
                create_weighted_loss(output, &target_vars[idx], &loss_type, head.meta.loss_weight)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let loss_var = aggregate_loss_vars(loss_vars)?;

        build.graph.forward(&loss_var)?;

        let test_loss_tensor = loss_var
            .value()?
            .ok_or_else(|| GraphError::ComputationError("评估时 loss 节点无值".into()))?;
        let test_loss_scalar = test_loss_tensor.to_vec()[0];

        let mut head_reports = Vec::with_capacity(self.heads.len());
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut primary_value = None;
        for (idx, head) in self.heads.iter().enumerate() {
            let output = build.outputs.get(idx).ok_or_else(|| {
                GraphError::ComputationError(format!(
                    "缺少 head '{}' 对应的输出节点",
                    head.meta.name
                ))
            })?;
            let predictions = output
                .value()?
                .ok_or_else(|| GraphError::ComputationError("评估时输出节点无值".into()))?;
            let loss_type = head_loss_type(genome, &head.meta);
            let head_primary = compute_primary_metric(
                &head.meta.metric,
                &predictions,
                &head.test_y,
                head.meta.output_dim,
                &loss_type,
            );
            let report = compute_metric_report(
                &head.meta.metric,
                &head.report_metrics,
                &predictions,
                &head.test_y,
                head.meta.output_dim,
                &loss_type,
            );
            if head.meta.primary {
                primary_value = Some(head_primary);
            }
            weighted_sum += head_primary * head.meta.metric_weight;
            weight_sum += head.meta.metric_weight;
            head_reports.push(HeadMetricReport {
                head_name: head.meta.name.clone(),
                primary: head_primary,
                tiebreak_loss: if head.meta.metric.is_discrete() {
                    Some(test_loss_scalar)
                } else {
                    None
                },
                report,
            });
        }
        let primary = primary_value.unwrap_or_else(|| {
            if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.0
            }
        });
        let report = head_reports
            .iter()
            .find(|entry| {
                self.heads
                    .iter()
                    .any(|head| head.meta.name == entry.head_name && head.meta.primary)
            })
            .map(|entry| entry.report.clone())
            .unwrap_or_default();

        let tiebreak_loss = if self.heads.iter().any(|head| head.meta.metric.is_discrete()) {
            Some(test_loss_scalar)
        } else {
            None
        };

        build.graph.train();

        Ok(FitnessScore {
            primary,
            inference_cost: None,
            tiebreak_loss,
            primary_proxy: None,
            report,
            head_reports,
        })
    }
}

// ==================== 辅助函数 ====================

fn create_loss_var(output: &Var, target: &Var, loss_type: &LossType) -> Result<Var, GraphError> {
    match loss_type {
        LossType::BCE => output.bce_loss(target),
        LossType::CrossEntropy => output.cross_entropy(target),
        LossType::MSE => output.mse_loss(target),
    }
}

fn head_loss_type(genome: &NetworkGenome, head: &MaterializedHead) -> LossType {
    head.loss_override
        .clone()
        .or_else(|| genome.training_config.loss_override.clone())
        .unwrap_or_else(|| {
            compatible_losses(&head.metric, head.output_dim)
                .into_iter()
                .next()
                .unwrap_or(LossType::MSE)
        })
}

fn create_weighted_loss(
    output: &Var,
    target: &Var,
    loss_type: &LossType,
    weight: f32,
) -> Result<Var, GraphError> {
    let loss = create_loss_var(output, target, loss_type)?;
    if (weight - 1.0).abs() <= f32::EPSILON {
        Ok(loss)
    } else {
        Ok(&loss * weight)
    }
}

fn aggregate_loss_vars(losses: Vec<Var>) -> Result<Var, GraphError> {
    let mut iter = losses.into_iter();
    let Some(mut total) = iter.next() else {
        return Err(GraphError::ComputationError(
            "多头监督任务至少需要一个 loss".into(),
        ));
    };
    for loss in iter {
        total = &total + &loss;
    }
    Ok(total)
}

fn train_one_batch(
    optimizer: &mut dyn Optimizer,
    loss_var: &Var,
    timing: &mut TrainTiming,
) -> Result<f32, GraphError> {
    let zero_start = Instant::now();
    optimizer.zero_grad()?;
    timing.zero_grad += zero_start.elapsed();

    let backward_start = Instant::now();
    let (loss_val, forward_elapsed, propagate_elapsed) = loss_var.backward_timed()?;
    timing.backward += backward_start.elapsed();
    timing.backward_forward += forward_elapsed;
    timing.backward_propagate += propagate_elapsed;

    let step_start = Instant::now();
    optimizer.step()?;
    timing.optimizer_step += step_start.elapsed();

    Ok(loss_val)
}

/// 按 TaskMetric 计算主指标值
pub(crate) fn compute_primary_metric(
    metric: &TaskMetric,
    predictions: &Tensor,
    labels: &Tensor,
    output_dim: usize,
    loss_type: &LossType,
) -> f32 {
    match metric {
        TaskMetric::Accuracy => {
            if output_dim == 1 {
                let pred_labels = binary_decode(predictions, loss_type);
                let true_labels = binary_decode(labels, &LossType::MSE);
                metrics::accuracy(&pred_labels, &true_labels).value()
            } else {
                metrics::accuracy(predictions, labels).value()
            }
        }
        TaskMetric::R2 => metrics::r2_score(predictions, labels).value(),
        TaskMetric::MultiLabelAccuracy => match loss_type {
            LossType::BCE => {
                // BCE 输出 logit：>= 0.0 等价于 sigmoid >= 0.5，判为正例。
                // 不能直接把 logit 传给 multilabel_loose_accuracy（它对 preds 和 labels
                // 使用同一阈值，0.0 会把标签 0.0 也判成正），需要先显式解码。
                let decoded_data: Vec<f32> = predictions
                    .to_vec()
                    .iter()
                    .map(|&v| if v >= 0.0 { 1.0 } else { 0.0 })
                    .collect();
                let decoded = Tensor::new(&decoded_data, predictions.shape());
                metrics::multilabel_loose_accuracy(&decoded, labels, 0.5).value()
            }
            _ => metrics::multilabel_loose_accuracy(predictions, labels, 0.5).value(),
        },
        TaskMetric::BinaryIoU => {
            let threshold = if matches!(loss_type, LossType::BCE) {
                0.0
            } else {
                0.5
            };
            metrics::binary_iou(predictions, labels, threshold).value()
        }
        TaskMetric::MeanIoU => metrics::mean_iou(predictions, labels).value(),
    }
}

/// 计算附加报告指标。
///
/// 报告指标不参与演化选择；这里复用 `metrics` 通用函数，确保手写模型和演化系统
/// 对同一指标的计算语义一致。
pub(crate) fn compute_metric_report(
    task_metric: &TaskMetric,
    report_metrics: &[ReportMetric],
    predictions: &Tensor,
    labels: &Tensor,
    output_dim: usize,
    loss_type: &LossType,
) -> MetricReport {
    let entries = report_metrics
        .iter()
        .filter(|metric| metric.is_compatible_with(task_metric))
        .filter_map(|&metric| {
            compute_report_metric(metric, predictions, labels, output_dim, loss_type)
        })
        .collect();
    MetricReport::new(entries)
}

fn compute_report_metric(
    metric: ReportMetric,
    predictions: &Tensor,
    labels: &Tensor,
    output_dim: usize,
    loss_type: &LossType,
) -> Option<MetricValue> {
    match metric {
        ReportMetric::Accuracy
        | ReportMetric::Precision
        | ReportMetric::Recall
        | ReportMetric::F1 => {
            let (pred_labels, true_labels) =
                classification_labels(predictions, labels, output_dim, loss_type);
            let value = match metric {
                ReportMetric::Accuracy => metrics::accuracy(&pred_labels, &true_labels),
                ReportMetric::Precision => metrics::precision(&pred_labels, &true_labels),
                ReportMetric::Recall => metrics::recall(&pred_labels, &true_labels),
                ReportMetric::F1 => metrics::f1_score(&pred_labels, &true_labels),
                _ => unreachable!(),
            };
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::R2 => {
            let value = metrics::r2_score(predictions, labels);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::MeanSquaredError => {
            let value = metrics::mean_squared_error(predictions, labels);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::MeanAbsoluteError => {
            let value = metrics::mean_absolute_error(predictions, labels);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::RootMeanSquaredError => {
            let value = metrics::root_mean_squared_error(predictions, labels);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::MultiLabelLooseAccuracy => {
            let decoded = multilabel_predictions(predictions, loss_type);
            let value = metrics::multilabel_loose_accuracy(&decoded, labels, 0.5);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::MultiLabelStrictAccuracy => {
            let decoded = multilabel_predictions(predictions, loss_type);
            let value = metrics::multilabel_strict_accuracy(&decoded, labels, 0.5);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::PixelAccuracy => {
            let value = if predictions.shape().len() == 4 && predictions.shape()[1] > 1 {
                metrics::semantic_pixel_accuracy(predictions, labels)
            } else {
                let threshold = if matches!(loss_type, LossType::BCE) {
                    0.0
                } else {
                    0.5
                };
                metrics::pixel_accuracy(predictions, labels, threshold)
            };
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::BinaryIoU => {
            let threshold = if matches!(loss_type, LossType::BCE) {
                0.0
            } else {
                0.5
            };
            let value = metrics::binary_iou(predictions, labels, threshold);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::Dice => {
            let threshold = if matches!(loss_type, LossType::BCE) {
                0.0
            } else {
                0.5
            };
            let value = metrics::dice_score(predictions, labels, threshold);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
        ReportMetric::MeanIoU => {
            let value = metrics::mean_iou(predictions, labels);
            Some(MetricValue::new(metric, value.value(), value.n_samples()))
        }
    }
}

fn classification_labels(
    predictions: &Tensor,
    labels: &Tensor,
    output_dim: usize,
    loss_type: &LossType,
) -> (Vec<usize>, Vec<usize>) {
    if output_dim == 1 {
        (
            binary_decode(predictions, loss_type),
            binary_decode(labels, &LossType::MSE),
        )
    } else {
        (predictions.to_class_labels(), labels.to_class_labels())
    }
}

fn multilabel_predictions(predictions: &Tensor, loss_type: &LossType) -> Tensor {
    if matches!(loss_type, LossType::BCE) {
        let decoded_data: Vec<f32> = predictions
            .to_vec()
            .iter()
            .map(|&v| if v >= 0.0 { 1.0 } else { 0.0 })
            .collect();
        Tensor::new(&decoded_data, predictions.shape())
    } else {
        predictions.clone()
    }
}

/// 二分类显式解码：[batch, 1] → Vec<usize>
///
/// 阈值与当前生效的 loss 绑定：
/// - BCE：输出是 logit，logit >= 0.0 → 类别 1（等价于 sigmoid >= 0.5）
/// - MSE/CrossEntropy：pred >= 0.5 → 类别 1
///
/// 标签侧统一用 MSE 规则（>= 0.5 → 类别 1），因为标签值域为 {0, 1}。
pub(crate) fn binary_decode(tensor: &Tensor, loss_type: &LossType) -> Vec<usize> {
    let threshold = match loss_type {
        LossType::BCE => 0.0,
        _ => 0.5,
    };
    tensor
        .to_vec()
        .iter()
        .map(|&v| if v >= threshold { 1 } else { 0 })
        .collect()
}

/// 计算所有可训练参数的梯度 L2 范数
///
/// `minimize()` 内部执行 zero_grad → backward → step，
/// step() 只更新参数值不清除梯度，因此 `minimize()` 返回后梯度仍可读取。
pub(crate) fn compute_grad_norm(params: &[Var]) -> Result<f32, GraphError> {
    let mut sum_sq = 0.0f32;
    for param in params {
        if let Some(grad) = param.grad()? {
            for &g in &grad.to_vec() {
                sum_sq += g * g;
            }
        }
    }
    Ok(sum_sq.sqrt())
}

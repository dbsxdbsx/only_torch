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

use rand::rngs::StdRng;

use crate::metrics;
use crate::nn::{Adam, GraphError, Optimizer, Var, VarLossOps, SGD};
use crate::tensor::Tensor;

use super::builder::BuildResult;
use super::convergence::{ConvergenceConfig, ConvergenceDetector};
use super::error::EvolutionError;
use super::gene::{compatible_losses, LossType, NetworkGenome, OptimizerType, TaskMetric};

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
#[derive(Clone, Debug)]
pub struct FitnessScore {
    /// 主目标（越高越好），如 Accuracy / R² / Reward
    pub primary: f32,
    /// 可选副目标（越低越好），如 FLOPs / 推理延迟
    pub inference_cost: Option<f32>,
    /// 离散指标的 tiebreaker（越低越好）：test set 上的 loss。
    /// 仅在 primary 相等时用于区分同指标的不同结构。
    /// 连续指标（R² 等）不需要 tiebreak，此字段为 None。
    pub tiebreak_loss: Option<f32>,
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
    /// 训练网络，返回最终 loss
    fn train(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
        convergence: &ConvergenceConfig,
        rng: &mut StdRng,
    ) -> Result<f32, GraphError>;

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

// ==================== SupervisedTask ====================

/// 监督学习任务
///
/// 构造器接受 per-sample 的 `Vec<Tensor>`，内部立即 stack 成 batched Tensor。
/// 自动根据数据量选择 full-batch 或 mini-batch 训练策略。
pub struct SupervisedTask {
    train_x: Tensor, // [n_train, input_dim]
    train_y: Tensor, // [n_train, output_dim]
    test_x: Tensor,  // [n_test, input_dim]
    test_y: Tensor,  // [n_test, output_dim]
    metric: TaskMetric,
    batch_size: Option<usize>, // None = 自动策略，Some = 显式指定
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
        if train_data.0.is_empty() {
            return Err(EvolutionError::InvalidData("训练输入不能为空".into()));
        }
        if train_data.0.len() != train_data.1.len() {
            return Err(EvolutionError::InvalidData(format!(
                "训练输入({})和标签({})数量不匹配",
                train_data.0.len(),
                train_data.1.len()
            )));
        }
        if test_data.0.is_empty() {
            return Err(EvolutionError::InvalidData("测试输入不能为空".into()));
        }
        if test_data.0.len() != test_data.1.len() {
            return Err(EvolutionError::InvalidData(format!(
                "测试输入({})和标签({})数量不匹配",
                test_data.0.len(),
                test_data.1.len()
            )));
        }

        let train_x_refs: Vec<&Tensor> = train_data.0.iter().collect();
        let train_y_refs: Vec<&Tensor> = train_data.1.iter().collect();
        let test_x_refs: Vec<&Tensor> = test_data.0.iter().collect();
        let test_y_refs: Vec<&Tensor> = test_data.1.iter().collect();

        Ok(Self {
            train_x: Tensor::stack(&train_x_refs, 0),
            train_y: Tensor::stack(&train_y_refs, 0),
            test_x: Tensor::stack(&test_x_refs, 0),
            test_y: Tensor::stack(&test_y_refs, 0),
            metric,
            batch_size: None,
        })
    }

    /// 获取任务指标类型
    pub fn metric(&self) -> &TaskMetric {
        &self.metric
    }

    /// 显式设置 batch size（覆盖自动策略）
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size 必须 > 0");
        self.batch_size = Some(batch_size);
        self
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
        _rng: &mut StdRng,
    ) -> Result<f32, GraphError> {
        assert!(
            genome.training_config.weight_decay == 0.0,
            "weight_decay 尚未支持，必须为 0.0"
        );

        // debug 模式下验证 loss_override 兼容性（正常演化路径由 MutateLossFunctionMutation 保障）
        debug_assert!(
            genome.training_config.loss_override.as_ref().map_or(true, |loss| {
                compatible_losses(&self.metric, genome.output_dim).contains(loss)
            }),
            "loss_override {:?} 与当前任务不兼容（metric={:?}, output_dim={}，兼容列表={:?}）",
            genome.training_config.loss_override,
            self.metric,
            genome.output_dim,
            compatible_losses(&self.metric, genome.output_dim)
        );

        build.graph.train();

        let n_samples = self.train_x.shape()[0];
        let bs = self.effective_batch_size(genome, n_samples);

        // target / loss / optimizer 各创建一次，不在 epoch 内重建
        let target = build.graph.target(self.train_y.shape())?;
        let loss_type = genome.effective_loss(&self.metric);
        let loss_var = create_loss_var(&build.output, &target, &loss_type)?;

        let params = build.all_parameters();
        let lr = genome.training_config.learning_rate;
        let mut optimizer: Box<dyn Optimizer> = match genome.training_config.optimizer_type {
            OptimizerType::Adam => Box::new(Adam::new(&build.graph, &params, lr)),
            OptimizerType::SGD => Box::new(SGD::new(&build.graph, &params, lr)),
        };

        let mut detector = ConvergenceDetector::new(convergence.clone());
        let mut final_loss = f32::NAN;

        if bs >= n_samples {
            // ====== Full-batch 路径 ======
            build.input.set_value(&self.train_x)?;
            target.set_value(&self.train_y)?;

            for epoch in 0.. {
                let loss_val = optimizer.minimize(&loss_var)?;
                let grad_norm = compute_grad_norm(&params)?;
                final_loss = loss_val;

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

                while offset < n_samples {
                    let end = (offset + bs).min(n_samples);
                    let batch_x = self.train_x.narrow(0, offset, end - offset);
                    let batch_y = self.train_y.narrow(0, offset, end - offset);

                    build.input.set_value(&batch_x)?;
                    target.set_value(&batch_y)?;

                    let loss_val = optimizer.minimize(&loss_var)?;
                    epoch_loss_sum += loss_val;
                    n_batches += 1;
                    offset = end;
                }

                let avg_loss = epoch_loss_sum / n_batches as f32;
                let grad_norm = compute_grad_norm(&params)?;
                final_loss = avg_loss;

                if detector.should_stop(epoch, avg_loss, grad_norm).is_some() {
                    break;
                }
            }
        }

        Ok(final_loss)
    }

    fn configure_batch_size(&mut self, batch_size: Option<usize>) {
        self.batch_size = batch_size;
    }

    fn create_visualization_loss(
        &self,
        genome: &NetworkGenome,
        build: &BuildResult,
    ) -> Option<Var> {
        let loss_type = genome.effective_loss(&self.metric);
        // 用 output 的预期形状创建 target（仅拓扑，无需实际数据）
        let output_shape = build.output.value_expected_shape();
        let target = build.graph.target(&output_shape).ok()?;
        create_loss_var(&build.output, &target, &loss_type).ok()
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
        let loss_type = genome.effective_loss(&self.metric);
        let target = build.graph.target(self.test_y.shape())?;
        target.set_value(&self.test_y)?;
        let loss_var = create_loss_var(&build.output, &target, &loss_type)?;

        build.graph.forward(&loss_var)?;

        let predictions = build
            .output
            .value()?
            .ok_or_else(|| GraphError::ComputationError("评估时输出节点无值".into()))?;
        let test_loss_tensor = loss_var
            .value()?
            .ok_or_else(|| GraphError::ComputationError("评估时 loss 节点无值".into()))?;
        let test_loss_scalar = test_loss_tensor.to_vec()[0];

        let primary = compute_primary_metric(
            &self.metric,
            &predictions,
            &self.test_y,
            genome.output_dim,
            &loss_type,
        );

        let tiebreak_loss = if self.metric.is_discrete() {
            Some(test_loss_scalar)
        } else {
            None
        };

        build.graph.train();

        Ok(FitnessScore {
            primary,
            inference_cost: None,
            tiebreak_loss,
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

/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化的基因数据结构
 *
 * 核心类型：NetworkGenome（网络基因组）以层为最小演化单位，
 * 通过 resolve_dimensions() 推导维度链，支持 skip edge 聚合。
 * 权重快照实现 Lamarckian 继承。
 */

use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fmt;

/// Input 节点的虚拟创新号（skip edge 可引用此值作为源）
pub const INPUT_INNOVATION: u64 = 0;

// ==================== 形状域 ====================

/// 形状域：描述张量在网络中的维度语义
///
/// 用于验证相邻层的域链合法性（如不允许 `Flat → Sequence` 回溯）。
/// 未来扩展 Conv2d 时只需添加 `Spatial` 变体。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShapeDomain {
    /// 2D 平坦数据 `[batch, features]`
    Flat,
    /// 3D 序列数据 `[batch, seq_len, features]`
    Sequence,
}

// ==================== 错误类型 ====================

/// 基因组操作错误
#[derive(Debug)]
pub enum GenomeError {
    /// 维度无效（如零维度）
    InvalidDimension(String),
    /// 维度不兼容（如 Add 聚合时输入维度不同）
    IncompatibleDimensions(String),
    /// 基因组为空（无启用的层）
    EmptyGenome(String),
    /// Skip edge 引用无效（源层创新号不存在）
    InvalidSkipEdge(String),
}

impl fmt::Display for GenomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GenomeError::InvalidDimension(msg) => write!(f, "维度无效: {msg}"),
            GenomeError::IncompatibleDimensions(msg) => write!(f, "维度不兼容: {msg}"),
            GenomeError::EmptyGenome(msg) => write!(f, "空基因组: {msg}"),
            GenomeError::InvalidSkipEdge(msg) => write!(f, "无效 skip edge: {msg}"),
        }
    }
}

impl std::error::Error for GenomeError {}

// ==================== 激活函数类型 ====================

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    ReLU,
    LeakyReLU { alpha: f32 },
    Tanh,
    Sigmoid,
    GELU,
    SiLU,
    Softplus,
    ReLU6,
    ELU { alpha: f32 },
    SELU,
    Mish,
    HardSwish,
    HardSigmoid,
}

impl fmt::Display for ActivationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ActivationType::ReLU => write!(f, "ReLU"),
            ActivationType::LeakyReLU { alpha } => write!(f, "LeakyReLU({alpha})"),
            ActivationType::Tanh => write!(f, "Tanh"),
            ActivationType::Sigmoid => write!(f, "Sigmoid"),
            ActivationType::GELU => write!(f, "GELU"),
            ActivationType::SiLU => write!(f, "SiLU"),
            ActivationType::Softplus => write!(f, "Softplus"),
            ActivationType::ReLU6 => write!(f, "ReLU6"),
            ActivationType::ELU { alpha } => write!(f, "ELU({alpha})"),
            ActivationType::SELU => write!(f, "SELU"),
            ActivationType::Mish => write!(f, "Mish"),
            ActivationType::HardSwish => write!(f, "HardSwish"),
            ActivationType::HardSigmoid => write!(f, "HardSigmoid"),
        }
    }
}

// ==================== 层配置 ====================

/// 层配置（纯计算层，不含聚合节点）
///
/// 聚合操作由 SkipEdge 携带策略，build() 时自动派生。
/// 只存输出侧参数，输入维度由 `resolve_dimensions()` 推导。
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerConfig {
    Linear { out_features: usize },
    Activation { activation_type: ActivationType },
    Rnn { hidden_size: usize },
    Lstm { hidden_size: usize },
    Gru { hidden_size: usize },
    Dropout { p: f32 },
}

impl fmt::Display for LayerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerConfig::Linear { out_features } => write!(f, "Linear({out_features})"),
            LayerConfig::Activation { activation_type } => write!(f, "{activation_type}"),
            LayerConfig::Rnn { hidden_size } => write!(f, "RNN({hidden_size})"),
            LayerConfig::Lstm { hidden_size } => write!(f, "LSTM({hidden_size})"),
            LayerConfig::Gru { hidden_size } => write!(f, "GRU({hidden_size})"),
            LayerConfig::Dropout { p } => write!(f, "Dropout({p})"),
        }
    }
}

// ==================== 聚合策略 ====================

/// 聚合策略（与 SkipEdge 绑定）
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AggregateStrategy {
    Add,
    Concat { dim: i32 },
    Mean,
    Max,
}

// ==================== 层基因 ====================

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerGene {
    pub innovation_number: u64,
    pub layer_config: LayerConfig,
    pub enabled: bool,
}

// ==================== 跳跃边 ====================

/// 跳跃边（携带聚合策略）
///
/// 聚合操作不作为独立层存在于 layers 中，
/// 而是在 build() 时根据 SkipEdge 信息自动在目标层输入处生成。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SkipEdge {
    pub innovation_number: u64,
    pub from_innovation: u64,
    pub to_innovation: u64,
    pub strategy: AggregateStrategy,
    pub enabled: bool,
}

// ==================== 训练配置 ====================

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    SGD,
    Adam,
}

/// 损失函数类型（由 TaskMetric 自动推断）
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LossType {
    BCE,
    CrossEntropy,
    MSE,
}

// ==================== 任务指标 ====================

/// 任务指标选择器（驱动 loss 推断 + 评估函数选择）
///
/// 与 metrics::Metric trait 的区别：
/// - TaskMetric 是任务类型的标识符（评估之前，用于推断 loss）
/// - metrics::Metric 是计算结果的统一接口（评估之后）
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TaskMetric {
    /// 分类准确率 → 自动推断 CrossEntropy (output_dim > 1) 或 BCE (output_dim == 1)
    Accuracy,
    /// 决定系数 → 自动推断 MSE
    R2,
    /// 多标签准确率 → 自动推断 BCE
    MultiLabelAccuracy,
}

impl TaskMetric {
    /// 是否为离散指标（需要 loss tiebreaker 辅助比较）
    pub fn is_discrete(&self) -> bool {
        matches!(self, TaskMetric::Accuracy | TaskMetric::MultiLabelAccuracy)
    }
}

/// 根据 TaskMetric + output_dim 返回兼容的 loss 列表
pub fn compatible_losses(metric: &TaskMetric, output_dim: usize) -> Vec<LossType> {
    match metric {
        TaskMetric::Accuracy if output_dim == 1 => vec![LossType::BCE, LossType::MSE],
        TaskMetric::Accuracy => vec![LossType::CrossEntropy],
        TaskMetric::R2 => vec![LossType::MSE],
        TaskMetric::MultiLabelAccuracy => vec![LossType::BCE],
    }
}

/// 训练配置（与 Genome 绑定，未来可参与演化）
///
/// 当前使用 Default，后续版本可加入超参数变异操作。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f32,
    pub batch_size: Option<usize>,
    pub weight_decay: f32,
    /// None = 自动推断（默认），Some = 显式指定
    pub loss_override: Option<LossType>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam,
            learning_rate: 0.01,
            batch_size: None,
            weight_decay: 0.0,
            loss_override: None,
        }
    }
}

// ==================== 维度推导结果 ====================

/// `resolve_dimensions` 的输出条目
#[derive(Debug)]
pub struct ResolvedDim {
    pub innovation_number: u64,
    pub in_dim: usize,
    pub out_dim: usize,
}

// ==================== 网络基因组 ====================

/// 网络基因组：完整的网络拓扑描述
///
/// layers 的最后一个启用层始终是输出头：`Linear(out_features=output_dim)`，受保护不可变异。
/// 最小结构下 layers 只有输出头一层。
///
/// 权重快照实现 Lamarckian 继承——clone 时权重一并复制，回滚无需依赖旧 Graph。
#[derive(Serialize, Deserialize)]
pub struct NetworkGenome {
    pub layers: Vec<LayerGene>,
    pub skip_edges: Vec<SkipEdge>,
    pub input_dim: usize,
    pub output_dim: usize,
    /// 序列长度（None = 平坦输入，Some(n) = 序列输入，每个时间步 input_dim 维特征）
    pub seq_len: Option<usize>,
    pub training_config: TrainingConfig,
    pub generated_by: String,
    pub(crate) next_innovation: u64,
    pub(crate) weight_snapshots: HashMap<u64, Vec<Tensor>>,
}

impl Clone for NetworkGenome {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            skip_edges: self.skip_edges.clone(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            seq_len: self.seq_len,
            training_config: self.training_config.clone(),
            generated_by: self.generated_by.clone(),
            next_innovation: self.next_innovation,
            weight_snapshots: self.weight_snapshots.clone(),
        }
    }
}

impl fmt::Debug for NetworkGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NetworkGenome")
            .field("layers", &self.layers)
            .field("skip_edges", &self.skip_edges)
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("seq_len", &self.seq_len)
            .field("training_config", &self.training_config)
            .field("generated_by", &self.generated_by)
            .field("next_innovation", &self.next_innovation)
            .field(
                "weight_snapshots_keys",
                &self.weight_snapshots.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl NetworkGenome {
    /// 从完整字段重建 NetworkGenome（反序列化用）
    ///
    /// 仅供 `model_io` 模块在加载 .otm 文件时调用。
    pub(crate) fn from_parts(
        layers: Vec<LayerGene>,
        skip_edges: Vec<SkipEdge>,
        input_dim: usize,
        output_dim: usize,
        seq_len: Option<usize>,
        training_config: TrainingConfig,
        generated_by: String,
        next_innovation: u64,
        weight_snapshots: HashMap<u64, Vec<Tensor>>,
    ) -> Self {
        Self {
            layers,
            skip_edges,
            input_dim,
            output_dim,
            seq_len,
            training_config,
            generated_by,
            next_innovation,
            weight_snapshots,
        }
    }

    /// 最小初始网络：layers = [Linear(out=output_dim)]（仅输出头，无隐藏层）
    ///
    /// # Panics
    /// `input_dim` 或 `output_dim` 为零时 panic。
    pub fn minimal(input_dim: usize, output_dim: usize) -> Self {
        assert!(input_dim > 0, "input_dim 不能为零");
        assert!(output_dim > 0, "output_dim 不能为零");

        let output_head = LayerGene {
            innovation_number: 1,
            layer_config: LayerConfig::Linear {
                out_features: output_dim,
            },
            enabled: true,
        };

        Self {
            layers: vec![output_head],
            skip_edges: Vec::new(),
            input_dim,
            output_dim,
            seq_len: None,
            training_config: TrainingConfig::default(),
            generated_by: "minimal".to_string(),
            next_innovation: 2, // 0 = INPUT, 1 = 输出头
            weight_snapshots: HashMap::new(),
        }
    }

    /// 最小序列网络：layers = [Rnn(hidden_size=output_dim), Linear(output_dim)]
    ///
    /// 初始 cell 类型为最简单的 Rnn，后续 MutateCellType 可升级为 LSTM/GRU，
    /// InsertLayer 可在序列域再插入更多 RNN 层。
    ///
    /// # Panics
    /// `input_dim` 或 `output_dim` 为零时 panic。
    pub fn minimal_sequential(input_dim: usize, output_dim: usize) -> Self {
        assert!(input_dim > 0, "input_dim 不能为零");
        assert!(output_dim > 0, "output_dim 不能为零");

        let rnn_layer = LayerGene {
            innovation_number: 1,
            layer_config: LayerConfig::Rnn {
                hidden_size: output_dim,
            },
            enabled: true,
        };

        let output_head = LayerGene {
            innovation_number: 2,
            layer_config: LayerConfig::Linear {
                out_features: output_dim,
            },
            enabled: true,
        };

        Self {
            layers: vec![rnn_layer, output_head],
            skip_edges: Vec::new(),
            input_dim,
            output_dim,
            seq_len: Some(0), // 占位，materialize_task 会设置实际值
            training_config: TrainingConfig::default(),
            generated_by: "minimal_sequential".to_string(),
            next_innovation: 3, // 0 = INPUT, 1 = Rnn, 2 = 输出头
            weight_snapshots: HashMap::new(),
        }
    }

    /// 获取下一个创新号（单调递增，不重复）
    pub fn next_innovation_number(&mut self) -> u64 {
        let id = self.next_innovation;
        self.next_innovation += 1;
        id
    }

    /// 推导每层的实际输入/输出维度，同时验证聚合节点的维度兼容性
    ///
    /// 从 `input_dim` 出发，沿 layers 顺序遍历，维护 `current_dim`：
    /// - Linear/Rnn/Lstm/Gru：由 `current_dim` 和配置参数决定
    /// - Activation/Dropout：维度透传
    /// - 含 skip edge 目标层：根据聚合策略计算有效输入维度
    ///
    /// 维度不兼容（如 Add 的输入维度不同）时返回 Err。
    pub fn resolve_dimensions(&self) -> Result<Vec<ResolvedDim>, GenomeError> {
        // 已计算的每层输出维度：innovation_number → out_dim
        let mut dim_map: HashMap<u64, usize> = HashMap::new();
        dim_map.insert(INPUT_INNOVATION, self.input_dim);

        let mut results = Vec::new();
        let mut current_dim = self.input_dim;

        for layer in self.layers.iter().filter(|l| l.enabled) {
            let effective_in_dim = self.compute_effective_input(
                layer.innovation_number,
                current_dim,
                &dim_map,
            )?;

            let out_dim = Self::compute_output_dim(&layer.layer_config, effective_in_dim);

            results.push(ResolvedDim {
                innovation_number: layer.innovation_number,
                in_dim: effective_in_dim,
                out_dim,
            });

            dim_map.insert(layer.innovation_number, out_dim);
            current_dim = out_dim;
        }

        if results.is_empty() {
            return Err(GenomeError::EmptyGenome(
                "没有启用的层（至少需要输出头）".into(),
            ));
        }

        Ok(results)
    }

    /// 当前总参数量（基于 `resolve_dimensions()` 推导维度后累加）
    pub fn total_params(&self) -> Result<usize, GenomeError> {
        let resolved = self.resolve_dimensions()?;
        let mut total = 0;

        for dim in &resolved {
            let layer = self
                .layers
                .iter()
                .find(|l| l.innovation_number == dim.innovation_number)
                .expect("resolve_dimensions 返回的创新号必须对应一个层");

            total += Self::compute_layer_params(&layer.layer_config, dim.in_dim, dim.out_dim);
        }

        Ok(total)
    }

    /// 当前层数（启用的层，含输出头）
    pub fn layer_count(&self) -> usize {
        self.layers.iter().filter(|l| l.enabled).count()
    }

    /// 权重快照是否为空
    pub fn has_weight_snapshots(&self) -> bool {
        !self.weight_snapshots.is_empty()
    }

    /// 获取权重快照引用（供 build/restore 使用）
    pub fn weight_snapshots(&self) -> &HashMap<u64, Vec<Tensor>> {
        &self.weight_snapshots
    }

    /// 设置权重快照（供 capture_weights 使用）
    pub fn set_weight_snapshots(&mut self, snapshots: HashMap<u64, Vec<Tensor>>) {
        self.weight_snapshots = snapshots;
    }

    /// 当前生效的 loss 函数（显式指定 > 自动推断）
    pub fn effective_loss(&self, metric: &TaskMetric) -> LossType {
        self.training_config
            .loss_override
            .clone()
            .unwrap_or_else(|| Self::infer_loss(metric, self.output_dim))
    }

    fn infer_loss(metric: &TaskMetric, output_dim: usize) -> LossType {
        match metric {
            TaskMetric::Accuracy if output_dim > 1 => LossType::CrossEntropy,
            TaskMetric::Accuracy => LossType::BCE,
            TaskMetric::R2 => LossType::MSE,
            TaskMetric::MultiLabelAccuracy => LossType::BCE,
        }
    }

    /// 验证域链合法性（序列模式专用）
    ///
    /// 遍历启用层，追踪 `current_domain`：
    /// - 合法转换：Seq→Seq（RNN return_seq）、Seq→Flat（RNN last hidden）、
    ///   Flat→Flat（Linear）、任意域→同域（Activation/Dropout）
    /// - 非法转换：Flat→Sequence（不允许回溯）、Sequence 直接到 Linear（跳过 RNN）
    /// - 终态必须为 Flat（输出头需要 2D 输入）
    ///
    /// 平坦模式（seq_len=None）直接返回 true。
    pub fn is_domain_valid(&self) -> bool {
        if self.seq_len.is_none() {
            return true;
        }

        let enabled: Vec<&LayerGene> = self.layers.iter().filter(|l| l.enabled).collect();
        if enabled.is_empty() {
            return false;
        }

        let mut current_domain = ShapeDomain::Sequence;

        for (i, layer) in enabled.iter().enumerate() {
            match &layer.layer_config {
                LayerConfig::Rnn { .. }
                | LayerConfig::Lstm { .. }
                | LayerConfig::Gru { .. } => {
                    if current_domain != ShapeDomain::Sequence {
                        return false; // Flat→Sequence 非法
                    }
                    // 判断输出域：看下一个实质层（跳过 Activation/Dropout）
                    let next_is_recurrent = enabled[i + 1..].iter().any(|l| {
                        match &l.layer_config {
                            LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => {
                                return false; // 继续查找
                            }
                            LayerConfig::Rnn { .. }
                            | LayerConfig::Lstm { .. }
                            | LayerConfig::Gru { .. } => true,
                            _ => false,
                        }
                    });
                    current_domain = if next_is_recurrent {
                        ShapeDomain::Sequence
                    } else {
                        ShapeDomain::Flat
                    };
                }
                LayerConfig::Linear { .. } => {
                    if current_domain != ShapeDomain::Flat {
                        return false; // Sequence 直接到 Linear 非法
                    }
                    // Linear 保持 Flat
                }
                LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => {
                    // 透传，保持当前域
                }
            }
        }

        current_domain == ShapeDomain::Flat
    }

    /// 判断给定层配置是否为循环层（Rnn/Lstm/Gru）
    pub fn is_recurrent(config: &LayerConfig) -> bool {
        matches!(
            config,
            LayerConfig::Rnn { .. } | LayerConfig::Lstm { .. } | LayerConfig::Gru { .. }
        )
    }

    /// 计算每个节点的输出形状域映射
    ///
    /// 返回 `innovation_number → ShapeDomain`，描述每个节点输出张量的域。
    /// 平坦模式（`seq_len=None`）下所有节点为 `Flat`。
    /// 用于 skip edge 域兼容性检查：跨域（Sequence↔Flat）的 skip edge 无法聚合。
    pub fn compute_domain_map(&self) -> HashMap<u64, ShapeDomain> {
        let mut map = HashMap::new();

        if self.seq_len.is_none() {
            map.insert(INPUT_INNOVATION, ShapeDomain::Flat);
            for layer in self.layers.iter().filter(|l| l.enabled) {
                map.insert(layer.innovation_number, ShapeDomain::Flat);
            }
            return map;
        }

        map.insert(INPUT_INNOVATION, ShapeDomain::Sequence);
        let enabled: Vec<&LayerGene> = self.layers.iter().filter(|l| l.enabled).collect();
        let mut current_domain = ShapeDomain::Sequence;

        for (i, layer) in enabled.iter().enumerate() {
            match &layer.layer_config {
                LayerConfig::Rnn { .. }
                | LayerConfig::Lstm { .. }
                | LayerConfig::Gru { .. } => {
                    // 找下一个实质层（跳过 Activation/Dropout）
                    let mut next_is_recurrent = false;
                    for next_layer in &enabled[i + 1..] {
                        match &next_layer.layer_config {
                            LayerConfig::Activation { .. }
                            | LayerConfig::Dropout { .. } => continue,
                            _ => {
                                next_is_recurrent =
                                    Self::is_recurrent(&next_layer.layer_config);
                                break;
                            }
                        }
                    }
                    current_domain = if next_is_recurrent {
                        ShapeDomain::Sequence
                    } else {
                        ShapeDomain::Flat
                    };
                }
                _ => { /* Linear/Activation/Dropout 保持当前域 */ }
            }
            map.insert(layer.innovation_number, current_domain);
        }

        map
    }

    // ==================== 内部方法 ====================

    /// 计算某层的有效输入维度（考虑 skip edge 聚合）
    fn compute_effective_input(
        &self,
        target_innovation: u64,
        main_path_dim: usize,
        dim_map: &HashMap<u64, usize>,
    ) -> Result<usize, GenomeError> {
        let incoming: Vec<&SkipEdge> = self
            .skip_edges
            .iter()
            .filter(|e| e.enabled && e.to_innovation == target_innovation)
            .collect();

        if incoming.is_empty() {
            return Ok(main_path_dim);
        }

        let mut all_dims = vec![main_path_dim];
        for skip in &incoming {
            let src_dim = dim_map.get(&skip.from_innovation).ok_or_else(|| {
                GenomeError::InvalidSkipEdge(format!(
                    "skip edge 源创新号 {} 在当前维度表中不存在（目标层 {}）",
                    skip.from_innovation, target_innovation
                ))
            })?;
            all_dims.push(*src_dim);
        }

        let strategy = &incoming[0].strategy;
        match strategy {
            AggregateStrategy::Add | AggregateStrategy::Mean | AggregateStrategy::Max => {
                if !all_dims.iter().all(|&d| d == all_dims[0]) {
                    return Err(GenomeError::IncompatibleDimensions(format!(
                        "{strategy:?} 聚合要求所有输入维度相同，实际为 {all_dims:?}"
                    )));
                }
                Ok(all_dims[0])
            }
            AggregateStrategy::Concat { .. } => Ok(all_dims.iter().sum()),
        }
    }

    /// 根据层类型和输入维度计算输出维度
    fn compute_output_dim(config: &LayerConfig, in_dim: usize) -> usize {
        match config {
            LayerConfig::Linear { out_features } => *out_features,
            LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => in_dim,
            LayerConfig::Rnn { hidden_size }
            | LayerConfig::Lstm { hidden_size }
            | LayerConfig::Gru { hidden_size } => *hidden_size,
        }
    }

    /// 计算单层参数量
    fn compute_layer_params(config: &LayerConfig, in_dim: usize, _out_dim: usize) -> usize {
        match config {
            LayerConfig::Linear { out_features } => {
                // W: [in_dim, out_features] + b: [out_features]
                in_dim * out_features + out_features
            }
            LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => 0,
            LayerConfig::Rnn { hidden_size } => {
                // W_ih + W_hh + b_h（实际 Rnn 层只有 1 个偏置）
                in_dim * hidden_size + hidden_size * hidden_size + hidden_size
            }
            LayerConfig::Lstm { hidden_size } => {
                // 4 gates × (W_ih + W_hh + bias)
                4 * (in_dim * hidden_size + hidden_size * hidden_size + hidden_size)
            }
            LayerConfig::Gru { hidden_size } => {
                // 3 gates × (W_ih + W_hh + bias)
                3 * (in_dim * hidden_size + hidden_size * hidden_size + hidden_size)
            }
        }
    }
}

// ==================== Display ====================

impl NetworkGenome {
    /// 主路径单行摘要（用于 DefaultCallback 日志等单行场景）
    ///
    /// 与 `Display` 的第一行内容相同，不含 skip edge 注解。
    /// 当存在重名层时，自动追加 `#N` 后缀做消歧。
    pub fn main_path_summary(&self) -> String {
        let names = self.build_display_names();
        let enabled: Vec<&LayerGene> = self.layers.iter().filter(|l| l.enabled).collect();
        let mut parts: Vec<&str> = Vec::with_capacity(enabled.len() + 1);
        parts.push(names[&INPUT_INNOVATION].as_str());
        for layer in &enabled {
            parts.push(names[&layer.innovation_number].as_str());
        }
        parts.join(" → ")
    }

    /// 构建 innovation_number → 显示名称 的映射
    ///
    /// 当同一显示名称出现多次时，自动追加 `#1`, `#2`, … 后缀做消歧。
    /// 用于主路径摘要及 skip edge 注解中引用层的人类可读名称。
    fn build_display_names(&self) -> HashMap<u64, String> {
        let enabled: Vec<&LayerGene> = self.layers.iter().filter(|l| l.enabled).collect();

        // 第一遍：收集 (innovation, raw_name)
        let mut entries: Vec<(u64, String)> = Vec::with_capacity(enabled.len() + 1);
        let input_label = if self.seq_len.is_some() {
            format!("Input(seq×{})", self.input_dim)
        } else {
            format!("Input({})", self.input_dim)
        };
        entries.push((INPUT_INNOVATION, input_label));
        for (i, layer) in enabled.iter().enumerate() {
            let is_last = i == enabled.len() - 1;
            let name = if is_last {
                format!("[{}]", layer.layer_config)
            } else {
                format!("{}", layer.layer_config)
            };
            entries.push((layer.innovation_number, name));
        }

        // 第二遍：统计每个名称出现次数
        let mut freq: HashMap<String, usize> = HashMap::new();
        for (_, name) in &entries {
            *freq.entry(name.clone()).or_insert(0) += 1;
        }

        // 第三遍：对重名追加 #N 后缀
        let mut seq: HashMap<String, usize> = HashMap::new();
        let mut names = HashMap::new();
        for (inn, name) in entries {
            let display = if freq[&name] > 1 {
                let n = seq.entry(name.clone()).or_insert(0);
                *n += 1;
                format!("{}#{}", name, n)
            } else {
                name
            };
            names.insert(inn, display);
        }

        names
    }
}

/// 输出人类可读的架构描述
///
/// 输出头用 `[]` 标注，`enabled=false` 的层和 skip edge 不出现。
/// 无 skip edge 时为单行，有 skip edge 时追加脚注式注解。
///
/// 无重名时：
/// ```text
/// Input(2) → Linear(4) → ReLU → [Linear(1)]
///   ├─ skip: Input(2) ──(Add)──→ ReLU
///   └─ skip: Linear(4) ──(Concat)──→ [Linear(1)]
/// ```
///
/// 存在重名层时自动追加 `#N` 消歧：
/// ```text
/// Input(2) → Linear(4)#1 → ReLU → Linear(4)#2 → [Linear(1)]
///   └─ skip: Linear(4)#1 ──(Add)──→ Linear(4)#2
/// ```
impl fmt::Display for NetworkGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // 主路径
        write!(f, "{}", self.main_path_summary())?;

        // Skip edge 注解
        let active_skips: Vec<&SkipEdge> = self
            .skip_edges
            .iter()
            .filter(|e| e.enabled)
            .collect();

        if !active_skips.is_empty() {
            let names = self.build_display_names();
            for (i, skip) in active_skips.iter().enumerate() {
                let from = names.get(&skip.from_innovation)
                    .map(|s| s.as_str()).unwrap_or("?");
                let to = names.get(&skip.to_innovation)
                    .map(|s| s.as_str()).unwrap_or("?");
                let strategy = match &skip.strategy {
                    AggregateStrategy::Add => "Add",
                    AggregateStrategy::Concat { .. } => "Concat",
                    AggregateStrategy::Mean => "Mean",
                    AggregateStrategy::Max => "Max",
                };
                let prefix = if i == active_skips.len() - 1 {
                    "└─"
                } else {
                    "├─"
                };
                write!(f, "\n  {prefix} skip: {from} ──({strategy})──→ {to}")?;
            }
        }

        Ok(())
    }
}

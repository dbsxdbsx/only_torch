/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化的基因数据结构
 *
 * 核心类型：NetworkGenome（网络基因组）
 * - GenomeRepr::LayerLevel: 层级基因表示（LayerGene + SkipEdge），阶段 9 起降级为用户 DSL 入口层
 * - GenomeRepr::NodeLevel:  节点级基因表示（NodeGene），阶段 2 起引入，阶段 8 起为全局唯一内核表示
 *
 * 当前阶段（阶段 8/9 过渡期）设计原则：
 * - 演化主循环 run() 对 Flat/Spatial/Sequential 三类任务均调用 migrate_to_node_level()
 * - NodeLevel 是唯一受支持的持久化、构图和 mutation 内核表示
 * - LayerLevel 仍保留在枚举中，作为阶段 9 前的用户入口 DSL 兼容层
 * - GenomeRepr::LayerLevel 将在阶段 9 正式完成后移除或降格为 DSL 转换器
 */

use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fmt;

use super::migration::migrate_network_genome;
use super::node_gene::{GenomeAnalysis, NodeGene};

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
    /// 4D 空间数据 `[batch, channels, H, W]`
    Spatial,
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

// ==================== 池化类型 ====================

/// 池化类型（Max / Avg）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolType {
    Max,
    Avg,
}

impl fmt::Display for PoolType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PoolType::Max => write!(f, "Max"),
            PoolType::Avg => write!(f, "Avg"),
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
    Linear {
        out_features: usize,
    },
    Activation {
        activation_type: ActivationType,
    },
    Rnn {
        hidden_size: usize,
    },
    Lstm {
        hidden_size: usize,
    },
    Gru {
        hidden_size: usize,
    },
    Dropout {
        p: f32,
    },
    /// 2D 卷积：stride=1, padding=kernel_size/2（same padding，不改变 H/W）
    Conv2d {
        out_channels: usize,
        kernel_size: usize,
    },
    /// 2D 池化（空间降维，不改变 channels）
    Pool2d {
        pool_type: PoolType,
        kernel_size: usize,
        stride: usize,
    },
    /// 展平：Spatial(C,H,W) → Flat(C*H*W)
    Flatten,
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
            LayerConfig::Conv2d {
                out_channels,
                kernel_size,
            } => {
                write!(f, "Conv2d({out_channels}, k={kernel_size})")
            }
            LayerConfig::Pool2d {
                pool_type,
                kernel_size,
                stride,
            } => {
                write!(f, "{pool_type}Pool({kernel_size}, s={stride})")
            }
            LayerConfig::Flatten => write!(f, "Flatten"),
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

// ==================== GenomeRepr ====================

/// 基因组内部表示：LayerLevel（旧）或 NodeLevel（新）
///
/// 这是阶段 3 的核心变更——将两种互斥的表示封装进枚举，
/// 消除双套并行字段的歧义。
/// 终态（阶段 6 完成后）只保留 NodeLevel 变体，LayerLevel 被移除。
#[derive(Serialize, Deserialize)]
pub(crate) enum GenomeRepr {
    /// 旧层级表示（阶段 4 前所有 mutation 操作的对象）
    LayerLevel {
        layers: Vec<LayerGene>,
        skip_edges: Vec<SkipEdge>,
        next_innovation: u64,
        /// 层粒度权重快照：layer_innovation → [W_tensor, b_tensor, ...]
        weight_snapshots: HashMap<u64, Vec<Tensor>>,
    },
    /// 新节点级表示（阶段 3 开始启用构图新路径）
    NodeLevel {
        nodes: Vec<NodeGene>,
        next_innovation: u64,
        /// 参数节点粒度快照：param_node_innovation → Tensor
        weight_snapshots: HashMap<u64, Tensor>,
    },
}

// ==================== 网络基因组 ====================

/// 网络基因组：完整的网络拓扶描述
///
/// 内部表示由 `GenomeRepr` 枚举区分：
/// - `LayerLevel`：原有层级表示，阶段 4 前所有 mutation 操作的对象
/// - `NodeLevel`：新节点级表示，阶段 3 起启用构图新路径
///
/// 对外访问通过 facade 方法（`layers()`/`nodes()`等），不直接操作 `repr` 字段。
#[derive(Serialize, Deserialize)]
pub struct NetworkGenome {
    // === 共享元数据（永远在此）===
    pub input_dim: usize,
    pub output_dim: usize,
    /// 序列长度（None = 平坦输入，Some(n) = 序列输入，每个时间步 input_dim 维特征）
    pub seq_len: Option<usize>,
    /// 空间输入尺寸 (H, W)（None = 非空间输入，Some = 空间输入，input_dim 表示 in_channels）
    pub input_spatial: Option<(usize, usize)>,
    pub training_config: TrainingConfig,
    pub generated_by: String,
    // === 内部表示层（LayerLevel 或 NodeLevel）===
    pub(crate) repr: GenomeRepr,
}

impl Clone for NetworkGenome {
    fn clone(&self) -> Self {
        let repr = match &self.repr {
            GenomeRepr::LayerLevel {
                layers,
                skip_edges,
                next_innovation,
                weight_snapshots,
            } => GenomeRepr::LayerLevel {
                layers: layers.clone(),
                skip_edges: skip_edges.clone(),
                next_innovation: *next_innovation,
                weight_snapshots: weight_snapshots.clone(),
            },
            GenomeRepr::NodeLevel {
                nodes,
                next_innovation,
                weight_snapshots,
            } => GenomeRepr::NodeLevel {
                nodes: nodes.clone(),
                next_innovation: *next_innovation,
                weight_snapshots: weight_snapshots.clone(),
            },
        };
        Self {
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            seq_len: self.seq_len,
            input_spatial: self.input_spatial,
            training_config: self.training_config.clone(),
            generated_by: self.generated_by.clone(),
            repr,
        }
    }
}

impl fmt::Debug for NetworkGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_node_level() {
            "NodeLevel"
        } else {
            "LayerLevel"
        };
        f.debug_struct("NetworkGenome")
            .field("kind", &kind)
            .field("input_dim", &self.input_dim)
            .field("output_dim", &self.output_dim)
            .field("seq_len", &self.seq_len)
            .field("input_spatial", &self.input_spatial)
            .field("training_config", &self.training_config)
            .field("generated_by", &self.generated_by)
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
        input_spatial: Option<(usize, usize)>,
        training_config: TrainingConfig,
        generated_by: String,
        next_innovation: u64,
        weight_snapshots: HashMap<u64, Vec<Tensor>>,
    ) -> Self {
        Self {
            input_dim,
            output_dim,
            seq_len,
            input_spatial,
            training_config,
            generated_by,
            repr: GenomeRepr::LayerLevel {
                layers,
                skip_edges,
                next_innovation,
                weight_snapshots,
            },
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
            input_dim,
            output_dim,
            seq_len: None,
            input_spatial: None,
            training_config: TrainingConfig::default(),
            generated_by: "minimal".to_string(),
            repr: GenomeRepr::LayerLevel {
                layers: vec![output_head],
                skip_edges: Vec::new(),
                next_innovation: 2, // 0 = INPUT, 1 = 输出头
                weight_snapshots: HashMap::new(),
            },
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
            input_dim,
            output_dim,
            seq_len: Some(0),
            input_spatial: None,
            training_config: TrainingConfig::default(),
            generated_by: "minimal_sequential".to_string(),
            repr: GenomeRepr::LayerLevel {
                layers: vec![rnn_layer, output_head],
                skip_edges: Vec::new(),
                next_innovation: 3,
                weight_snapshots: HashMap::new(),
            },
        }
    }

    /// 最小空间网络：layers = [Flatten, Linear(output_dim)]
    ///
    /// 从最简单的 Flatten+FC 结构出发，Conv2d/Pool2d 由演化自主发现。
    /// 对于小图像（如 MNIST 28×28），纯 FC 方案参数量更少、训练更快；
    /// 对于大图像，演化会通过 InsertLayer 在 Flatten 前插入 Conv2d/Pool2d 来降维。
    ///
    /// # Panics
    /// `input_channels` 或 `output_dim` 为零，或 `spatial` 的 H/W 为零时 panic。
    pub fn minimal_spatial(
        input_channels: usize,
        output_dim: usize,
        spatial: (usize, usize),
    ) -> Self {
        assert!(input_channels > 0, "input_channels 不能为零");
        assert!(output_dim > 0, "output_dim 不能为零");
        assert!(spatial.0 > 0 && spatial.1 > 0, "spatial (H, W) 不能为零");

        let flatten_layer = LayerGene {
            innovation_number: 1,
            layer_config: LayerConfig::Flatten,
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
            input_dim: input_channels,
            output_dim,
            seq_len: None,
            input_spatial: Some(spatial),
            training_config: TrainingConfig::default(),
            generated_by: "minimal_spatial".to_string(),
            repr: GenomeRepr::LayerLevel {
                layers: vec![flatten_layer, output_head],
                skip_edges: Vec::new(),
                next_innovation: 3,
                weight_snapshots: HashMap::new(),
            },
        }
    }

    /// 获取下一个创新号（单调递增，不重复）
    pub fn next_innovation_number(&mut self) -> u64 {
        match &mut self.repr {
            GenomeRepr::LayerLevel {
                next_innovation, ..
            }
            | GenomeRepr::NodeLevel {
                next_innovation, ..
            } => {
                let id = *next_innovation;
                *next_innovation += 1;
                id
            }
        }
    }

    // ==================== Facade 方法（外部访问入口）====================

    /// 返回层级基因列表（LayerLevel 专属）
    pub fn layers(&self) -> &[LayerGene] {
        match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => &[],
        }
    }

    /// 返回层级基因列表可变引用（LayerLevel 专属）
    pub fn layers_mut(&mut self) -> &mut Vec<LayerGene> {
        match &mut self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => panic!("layers_mut() 只支持 LayerLevel 基因组"),
        }
    }

    /// 返回跳跃边列表（LayerLevel 专属）
    pub fn skip_edges(&self) -> &[SkipEdge] {
        match &self.repr {
            GenomeRepr::LayerLevel { skip_edges, .. } => skip_edges,
            GenomeRepr::NodeLevel { .. } => &[],
        }
    }

    /// 返回跳跃边列表可变引用（LayerLevel 专属）
    pub fn skip_edges_mut(&mut self) -> &mut Vec<SkipEdge> {
        match &mut self.repr {
            GenomeRepr::LayerLevel { skip_edges, .. } => skip_edges,
            GenomeRepr::NodeLevel { .. } => panic!("skip_edges_mut() 只支持 LayerLevel 基因组"),
        }
    }

    /// 返回节点级基因列表（NodeLevel 专属）
    pub fn nodes(&self) -> &[NodeGene] {
        match &self.repr {
            GenomeRepr::LayerLevel { .. } => &[],
            GenomeRepr::NodeLevel { nodes, .. } => nodes,
        }
    }

    /// 返回节点级基因列表可变引用（NodeLevel 专属）
    pub fn nodes_mut(&mut self) -> &mut Vec<NodeGene> {
        match &mut self.repr {
            GenomeRepr::LayerLevel { .. } => panic!("nodes_mut() 只支持 NodeLevel 基因组"),
            GenomeRepr::NodeLevel { nodes, .. } => nodes,
        }
    }

    /// 当前是否为节点级表示
    pub fn is_node_level(&self) -> bool {
        matches!(self.repr, GenomeRepr::NodeLevel { .. })
    }

    /// 当前是否为层级表示
    pub fn is_layer_level(&self) -> bool {
        matches!(self.repr, GenomeRepr::LayerLevel { .. })
    }

    /// 返回当前创新号计数器的下一个将分配值（不消耗，供序列化使用）
    pub(crate) fn peek_next_innovation(&self) -> u64 {
        match &self.repr {
            GenomeRepr::LayerLevel {
                next_innovation, ..
            }
            | GenomeRepr::NodeLevel {
                next_innovation, ..
            } => *next_innovation,
        }
    }

    // ==================== 迁移与分析 ====================

    /// 将当前基因组迁移到节点级表示
    ///
    /// 若已是 NodeLevel 则无操作直接返回。
    /// 迁移后，`layers()` 和 `skip_edges()` 返回空切片，
    /// `nodes()` 返回展开后的节点列表。
    pub fn migrate_to_node_level(&mut self) -> Result<(), super::migration::MigrationError> {
        if self.is_node_level() {
            return Ok(());
        }
        let out = migrate_network_genome(self)?;
        self.repr = GenomeRepr::NodeLevel {
            nodes: out.nodes,
            next_innovation: out.next_innovation,
            weight_snapshots: HashMap::new(),
        };
        Ok(())
    }

    /// 对当前节点级基因组执行静态分析，返回不可变快照。
    ///
    /// `GenomeAnalysis` 是 mutation/builder/serializer 三方共同依赖的分析结果。
    /// 调用方式：`genome.analyze()`
    ///
    /// 注意：仅对 NodeLevel 基因组有意义。如需分析 LayerLevel，先调用 `migrate_to_node_level()`。
    pub fn analyze(&self) -> GenomeAnalysis {
        match &self.repr {
            GenomeRepr::NodeLevel { nodes, .. } => {
                let input_shape = if let Some((h, w)) = self.input_spatial {
                    vec![1, self.input_dim, h, w]
                } else if let Some(seq) = self.seq_len {
                    vec![1, seq, self.input_dim]
                } else {
                    vec![1, self.input_dim]
                };
                let input_domain = if self.input_spatial.is_some() {
                    super::gene::ShapeDomain::Spatial
                } else if self.seq_len.is_some() {
                    super::gene::ShapeDomain::Sequence
                } else {
                    super::gene::ShapeDomain::Flat
                };
                GenomeAnalysis::compute(nodes, INPUT_INNOVATION, input_shape, input_domain)
            }
            GenomeRepr::LayerLevel { .. } => {
                // LayerLevel 基因组暂不支持 analyze()，返回空分析
                GenomeAnalysis::compute(
                    &[],
                    INPUT_INNOVATION,
                    vec![1, self.input_dim],
                    ShapeDomain::Flat,
                )
            }
        }
    }

    // ==================== 权重快照（层级粒度，LayerLevel 专属）====================

    /// 权重快照是否为空
    pub fn has_weight_snapshots(&self) -> bool {
        match &self.repr {
            GenomeRepr::LayerLevel {
                weight_snapshots, ..
            } => !weight_snapshots.is_empty(),
            GenomeRepr::NodeLevel {
                weight_snapshots, ..
            } => !weight_snapshots.is_empty(),
        }
    }

    /// 获取层级权重快照引用（LayerLevel 专属）
    pub fn weight_snapshots(&self) -> &HashMap<u64, Vec<Tensor>> {
        match &self.repr {
            GenomeRepr::LayerLevel {
                weight_snapshots, ..
            } => weight_snapshots,
            GenomeRepr::NodeLevel { .. } => panic!("weight_snapshots() 只支持 LayerLevel 基因组"),
        }
    }

    /// 设置层级权重快照（LayerLevel 专属）
    pub fn set_weight_snapshots(&mut self, snapshots: HashMap<u64, Vec<Tensor>>) {
        match &mut self.repr {
            GenomeRepr::LayerLevel {
                weight_snapshots, ..
            } => *weight_snapshots = snapshots,
            GenomeRepr::NodeLevel { .. } => {
                panic!("set_weight_snapshots() 只支持 LayerLevel 基因组")
            }
        }
    }

    /// 移除指定层的权重快照（MutateCellType 等变异用，LayerLevel 专属）
    pub(crate) fn remove_layer_weight_snapshot(&mut self, inn: u64) {
        match &mut self.repr {
            GenomeRepr::LayerLevel {
                weight_snapshots, ..
            } => {
                weight_snapshots.remove(&inn);
            }
            GenomeRepr::NodeLevel { .. } => {} // NodeLevel 无需操作
        }
    }

    /// 获取节点级权重快照引用（NodeLevel 专属）
    pub fn node_weight_snapshots(&self) -> &HashMap<u64, Tensor> {
        match &self.repr {
            GenomeRepr::NodeLevel {
                weight_snapshots, ..
            } => weight_snapshots,
            GenomeRepr::LayerLevel { .. } => {
                panic!("node_weight_snapshots() 只支持 NodeLevel 基因组")
            }
        }
    }

    /// 推导每层的实际输入/输出维度，同时验证聚合节点的维度兼容性。
    /// 仅对 LayerLevel 基因组有效；NodeLevel 基因组应使用 `analyze()`。
    pub fn resolve_dimensions(&self) -> Result<Vec<ResolvedDim>, GenomeError> {
        let layers = match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => {
                return Err(GenomeError::EmptyGenome(
                    "resolve_dimensions 不支持 NodeLevel 基因组，请使用 analyze()".into(),
                ));
            }
        };
        let mut dim_map: HashMap<u64, usize> = HashMap::new();
        dim_map.insert(INPUT_INNOVATION, self.input_dim);

        let mut results = Vec::new();
        let mut current_dim = self.input_dim;
        let mut current_spatial = self.input_spatial;

        for layer in layers.iter().filter(|l| l.enabled) {
            let effective_in_dim =
                self.compute_effective_input(layer.innovation_number, current_dim, &dim_map)?;

            let out_dim =
                Self::compute_output_dim(&layer.layer_config, effective_in_dim, current_spatial);

            results.push(ResolvedDim {
                innovation_number: layer.innovation_number,
                in_dim: effective_in_dim,
                out_dim,
            });

            dim_map.insert(layer.innovation_number, out_dim);
            current_dim = out_dim;
            current_spatial = Self::compute_next_spatial(&layer.layer_config, current_spatial);
        }

        if results.is_empty() {
            return Err(GenomeError::EmptyGenome(
                "没有启用的层（至少需要输出头）".into(),
            ));
        }

        Ok(results)
    }

    /// 当前总参数量——支持 LayerLevel 和 NodeLevel
    pub fn total_params(&self) -> Result<usize, GenomeError> {
        let layers = match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => {
                // NodeLevel 通过 GenomeAnalysis 获取参数量
                return Ok(super::node_ops::node_param_count(self));
            }
        };
        let resolved = self.resolve_dimensions()?;
        let mut total = 0;
        for dim in &resolved {
            let layer = layers
                .iter()
                .find(|l| l.innovation_number == dim.innovation_number)
                .expect("resolve_dimensions 返回的创新号必须对应一个层");
            total += Self::compute_layer_params(&layer.layer_config, dim.in_dim, dim.out_dim);
        }
        Ok(total)
    }

    /// 当前"layer"数—— LayerLevel: 启用层数； NodeLevel: 主路块数
    pub fn layer_count(&self) -> usize {
        match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers.iter().filter(|l| l.enabled).count(),
            GenomeRepr::NodeLevel { .. } => super::node_ops::node_main_path(self).len(),
        }
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

    /// 验证域链合法性
    ///
    /// 遍历启用层，追踪 `current_domain`：
    /// - 空间模式：Spatial→Spatial（Conv2d/Pool2d）、Spatial→Flat（Flatten）、
    ///   Flat→Flat（Linear）；RNN 非法
    /// - 序列模式：Seq→Seq/Flat（RNN）、Flat→Flat（Linear）；空间层非法
    /// - 终态必须为 Flat
    ///
    /// 纯平坦模式直接返回 true。
    pub fn is_domain_valid(&self) -> bool {
        let layers = match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => return true, // NodeLevel 通过 analyze() 验证
        };
        // 空间模式
        if self.input_spatial.is_some() {
            let enabled: Vec<&LayerGene> = layers.iter().filter(|l| l.enabled).collect();
            if enabled.is_empty() {
                return false;
            }
            let mut current_domain = ShapeDomain::Spatial;
            for layer in &enabled {
                match &layer.layer_config {
                    LayerConfig::Conv2d { .. } | LayerConfig::Pool2d { .. } => {
                        if current_domain != ShapeDomain::Spatial {
                            return false;
                        }
                    }
                    LayerConfig::Flatten => {
                        if current_domain != ShapeDomain::Spatial {
                            return false;
                        }
                        current_domain = ShapeDomain::Flat;
                    }
                    LayerConfig::Linear { .. } => {
                        if current_domain != ShapeDomain::Flat {
                            return false;
                        }
                    }
                    LayerConfig::Rnn { .. }
                    | LayerConfig::Lstm { .. }
                    | LayerConfig::Gru { .. } => {
                        return false; // RNN 在空间模式下非法
                    }
                    LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => {}
                }
            }
            return current_domain == ShapeDomain::Flat;
        }

        // 纯平坦模式
        if self.seq_len.is_none() {
            return true;
        }

        // 序列模式
        let enabled: Vec<&LayerGene> = layers.iter().filter(|l| l.enabled).collect();
        if enabled.is_empty() {
            return false;
        }

        let mut current_domain = ShapeDomain::Sequence;

        for (i, layer) in enabled.iter().enumerate() {
            match &layer.layer_config {
                LayerConfig::Rnn { .. } | LayerConfig::Lstm { .. } | LayerConfig::Gru { .. } => {
                    if current_domain != ShapeDomain::Sequence {
                        return false;
                    }
                    let mut next_is_recurrent = false;
                    for next_layer in &enabled[i + 1..] {
                        match &next_layer.layer_config {
                            LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => {
                                continue;
                            }
                            _ => {
                                next_is_recurrent = Self::is_recurrent(&next_layer.layer_config);
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
                LayerConfig::Linear { .. } => {
                    if current_domain != ShapeDomain::Flat {
                        return false;
                    }
                }
                LayerConfig::Conv2d { .. } | LayerConfig::Pool2d { .. } | LayerConfig::Flatten => {
                    return false; // 空间层在序列模式下非法
                }
                LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => {}
            }
        }

        current_domain == ShapeDomain::Flat
    }

    /// 验证所有启用的 skip edge 在当前域映射下仍然合法
    ///
    /// 源和目标必须在同一域内；序列模式只允许 Flat 域 skip edge；
    /// 空间模式的 Spatial 域 skip edge 还需要 H/W 匹配。
    ///
    /// 纯平坦模式直接返回 true。
    pub fn validate_skip_edge_domains(&self) -> bool {
        let (layers, skip_edges) = match &self.repr {
            GenomeRepr::LayerLevel {
                layers, skip_edges, ..
            } => (layers, skip_edges),
            GenomeRepr::NodeLevel { .. } => return true,
        };
        if self.seq_len.is_none() && self.input_spatial.is_none() {
            return true;
        }

        let active_edges: Vec<_> = skip_edges.iter().filter(|e| e.enabled).collect();
        if active_edges.is_empty() {
            return true;
        }

        let domain_map = self.compute_domain_map();
        let spatial_map = if self.input_spatial.is_some() {
            Some(self.compute_spatial_map())
        } else {
            None
        };

        let enabled: Vec<&LayerGene> = layers.iter().filter(|l| l.enabled).collect();

        for edge in &active_edges {
            let from_domain = domain_map
                .get(&edge.from_innovation)
                .copied()
                .unwrap_or(ShapeDomain::Flat);

            let to_idx = enabled
                .iter()
                .position(|l| l.innovation_number == edge.to_innovation);
            let to_input_domain = match to_idx {
                Some(0) => *domain_map
                    .get(&INPUT_INNOVATION)
                    .unwrap_or(&ShapeDomain::Flat),
                Some(idx) => {
                    let pred_inn = enabled[idx - 1].innovation_number;
                    domain_map
                        .get(&pred_inn)
                        .copied()
                        .unwrap_or(ShapeDomain::Flat)
                }
                None => continue,
            };

            // 源和目标必须在同一域
            if from_domain != to_input_domain {
                return false;
            }

            // 序列模式：skip edge 只能在 Flat 域
            if self.seq_len.is_some() && from_domain != ShapeDomain::Flat {
                return false;
            }

            // 空间模式：Spatial 域内 skip edge 需要 H/W 匹配
            if from_domain == ShapeDomain::Spatial {
                if let Some(ref smap) = spatial_map {
                    let from_sp = smap.get(&edge.from_innovation).copied().flatten();
                    let to_sp = match to_idx {
                        Some(0) => smap.get(&INPUT_INNOVATION).copied().flatten(),
                        Some(idx) => {
                            let pred_inn = enabled[idx - 1].innovation_number;
                            smap.get(&pred_inn).copied().flatten()
                        }
                        None => continue,
                    };
                    if from_sp != to_sp {
                        return false;
                    }
                }
            }
        }

        true
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
    /// 返回 `innovation_number → ShapeDomain`。
    /// 用于 skip edge 域兼容性检查。
    pub fn compute_domain_map(&self) -> HashMap<u64, ShapeDomain> {
        let layers = match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => return HashMap::new(),
        };
        let mut map = HashMap::new();

        // 空间模式
        if self.input_spatial.is_some() {
            map.insert(INPUT_INNOVATION, ShapeDomain::Spatial);
            let enabled: Vec<&LayerGene> = layers.iter().filter(|l| l.enabled).collect();
            let mut current_domain = ShapeDomain::Spatial;
            for layer in &enabled {
                match &layer.layer_config {
                    LayerConfig::Conv2d { .. } | LayerConfig::Pool2d { .. } => {}
                    LayerConfig::Flatten => {
                        current_domain = ShapeDomain::Flat;
                    }
                    _ => {}
                }
                map.insert(layer.innovation_number, current_domain);
            }
            return map;
        }

        // 平坦模式
        if self.seq_len.is_none() {
            map.insert(INPUT_INNOVATION, ShapeDomain::Flat);
            for layer in layers.iter().filter(|l| l.enabled) {
                map.insert(layer.innovation_number, ShapeDomain::Flat);
            }
            return map;
        }

        // 序列模式
        map.insert(INPUT_INNOVATION, ShapeDomain::Sequence);
        let enabled: Vec<&LayerGene> = layers.iter().filter(|l| l.enabled).collect();
        let mut current_domain = ShapeDomain::Sequence;

        for (i, layer) in enabled.iter().enumerate() {
            match &layer.layer_config {
                LayerConfig::Rnn { .. } | LayerConfig::Lstm { .. } | LayerConfig::Gru { .. } => {
                    let mut next_is_recurrent = false;
                    for next_layer in &enabled[i + 1..] {
                        match &next_layer.layer_config {
                            LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => {
                                continue;
                            }
                            _ => {
                                next_is_recurrent = Self::is_recurrent(&next_layer.layer_config);
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
                _ => {}
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
        let skip_edges = match &self.repr {
            GenomeRepr::LayerLevel { skip_edges, .. } => skip_edges,
            GenomeRepr::NodeLevel { .. } => return Ok(main_path_dim),
        };
        let incoming: Vec<&SkipEdge> = skip_edges
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
    fn compute_output_dim(
        config: &LayerConfig,
        in_dim: usize,
        spatial: Option<(usize, usize)>,
    ) -> usize {
        match config {
            LayerConfig::Linear { out_features } => *out_features,
            LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => in_dim,
            LayerConfig::Rnn { hidden_size }
            | LayerConfig::Lstm { hidden_size }
            | LayerConfig::Gru { hidden_size } => *hidden_size,
            LayerConfig::Conv2d { out_channels, .. } => *out_channels,
            LayerConfig::Pool2d { .. } => in_dim, // channels 不变
            LayerConfig::Flatten => match spatial {
                Some((h, w)) => in_dim * h * w,
                None => in_dim,
            },
        }
    }

    /// 计算单层参数量
    fn compute_layer_params(config: &LayerConfig, in_dim: usize, _out_dim: usize) -> usize {
        match config {
            LayerConfig::Linear { out_features } => in_dim * out_features + out_features,
            LayerConfig::Activation { .. }
            | LayerConfig::Dropout { .. }
            | LayerConfig::Pool2d { .. }
            | LayerConfig::Flatten => 0,
            LayerConfig::Rnn { hidden_size } => {
                in_dim * hidden_size + hidden_size * hidden_size + hidden_size
            }
            LayerConfig::Lstm { hidden_size } => {
                4 * (in_dim * hidden_size + hidden_size * hidden_size + hidden_size)
            }
            LayerConfig::Gru { hidden_size } => {
                3 * (in_dim * hidden_size + hidden_size * hidden_size + hidden_size)
            }
            LayerConfig::Conv2d {
                out_channels,
                kernel_size,
            } => {
                // W: [out_ch, in_ch, k, k] + b: [out_ch]
                out_channels * in_dim * kernel_size * kernel_size + out_channels
            }
        }
    }

    /// 计算经过一个层后的空间尺寸变化
    fn compute_next_spatial(
        config: &LayerConfig,
        spatial: Option<(usize, usize)>,
    ) -> Option<(usize, usize)> {
        match config {
            LayerConfig::Conv2d { .. } => spatial, // same padding, H/W 不变
            LayerConfig::Pool2d {
                kernel_size,
                stride,
                ..
            } => spatial.map(|(h, w)| {
                let new_h = if h >= *kernel_size {
                    (h - kernel_size) / stride + 1
                } else {
                    1
                };
                let new_w = if w >= *kernel_size {
                    (w - kernel_size) / stride + 1
                } else {
                    1
                };
                (new_h, new_w)
            }),
            LayerConfig::Flatten => None,
            _ => spatial,
        }
    }

    /// 计算每个节点的输出空间尺寸映射
    pub(crate) fn compute_spatial_map(&self) -> HashMap<u64, Option<(usize, usize)>> {
        let layers = match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => return HashMap::new(),
        };
        let mut map = HashMap::new();
        let mut current = self.input_spatial;
        map.insert(INPUT_INNOVATION, current);
        for layer in layers.iter().filter(|l| l.enabled) {
            current = Self::compute_next_spatial(&layer.layer_config, current);
            map.insert(layer.innovation_number, current);
        }
        map
    }

    /// 是否为空间模式
    pub fn is_spatial(&self) -> bool {
        self.input_spatial.is_some()
    }
}

// ==================== Display ====================

impl NetworkGenome {
    /// 主路径单行摘要（用于 DefaultCallback 日志等单行场景）
    ///
    /// 与 `Display` 的第一行内容相同，不含 skip edge 注解。
    /// 当存在重名层时，自动追加 `#N` 后缀做消歧。
    pub fn main_path_summary(&self) -> String {
        match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => {
                let names = self.build_display_names();
                let enabled: Vec<&LayerGene> = layers.iter().filter(|l| l.enabled).collect();
                let mut parts: Vec<&str> = Vec::with_capacity(enabled.len() + 1);
                parts.push(names[&INPUT_INNOVATION].as_str());
                for layer in &enabled {
                    parts.push(names[&layer.innovation_number].as_str());
                }
                parts.join(" → ")
            }
            GenomeRepr::NodeLevel { nodes, .. } => {
                if nodes.is_empty() {
                    return "nodes=0 active=0 params=0".to_string();
                }
                self.analyze().summary()
            }
        }
    }

    /// 构建 innovation_number → 显示名称 的映射
    ///
    /// 当同一显示名称出现多次时，自动追加 `#1`, `#2`, … 后缀做消歧。
    /// 用于主路径摘要及 skip edge 注解中引用层的人类可读名称。
    fn build_display_names(&self) -> HashMap<u64, String> {
        let layers = match &self.repr {
            GenomeRepr::LayerLevel { layers, .. } => layers,
            GenomeRepr::NodeLevel { .. } => return HashMap::new(),
        };
        let enabled: Vec<&LayerGene> = layers.iter().filter(|l| l.enabled).collect();

        // 第一遍：收集 (innovation, raw_name)
        let mut entries: Vec<(u64, String)> = Vec::with_capacity(enabled.len() + 1);
        let input_label = if self.seq_len.is_some() {
            format!("Input(seq×{})", self.input_dim)
        } else if let Some((h, w)) = self.input_spatial {
            format!("Input({}@{}×{})", self.input_dim, h, w)
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

        // Skip edge 注解（仅 LayerLevel 有注解）
        let active_skips: Vec<&SkipEdge> = match &self.repr {
            GenomeRepr::LayerLevel { skip_edges, .. } => {
                skip_edges.iter().filter(|e| e.enabled).collect()
            }
            GenomeRepr::NodeLevel { .. } => Vec::new(),
        };

        if !active_skips.is_empty() {
            let names = self.build_display_names();
            for (i, skip) in active_skips.iter().enumerate() {
                let from = names
                    .get(&skip.from_innovation)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
                let to = names
                    .get(&skip.to_innovation)
                    .map(|s| s.as_str())
                    .unwrap_or("?");
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

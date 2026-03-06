/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化的基因数据结构
 *
 * 核心类型：NetworkGenome（网络基因组）以层为最小演化单位，
 * 通过 resolve_dimensions() 推导维度链，支持 skip edge 聚合。
 * 权重快照实现 Lamarckian 继承。
 */

use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fmt;

/// Input 节点的虚拟创新号（skip edge 可引用此值作为源）
pub const INPUT_INNOVATION: u64 = 0;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationType {
    ReLU,
    LeakyReLU { alpha: f32 },
    Tanh,
    Sigmoid,
    GELU,
    SiLU,
    Softplus,
    ReLU6,
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
        }
    }
}

// ==================== 层配置 ====================

/// 层配置（纯计算层，不含聚合节点）
///
/// 聚合操作由 SkipEdge 携带策略，build() 时自动派生。
/// 只存输出侧参数，输入维度由 `resolve_dimensions()` 推导。
#[derive(Clone, Debug, PartialEq)]
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

/// 聚合策略（Phase 7B 引入，与 SkipEdge 绑定）
#[derive(Clone, Debug, PartialEq)]
pub enum AggregateStrategy {
    Add,
    Concat { dim: i32 },
    Mean,
    Max,
}

// ==================== 层基因 ====================

#[derive(Clone, Debug)]
pub struct LayerGene {
    pub innovation_number: u64,
    pub layer_config: LayerConfig,
    pub enabled: bool,
}

// ==================== 跳跃边 ====================

/// 跳跃边（携带聚合策略，Phase 7B 引入）
///
/// 聚合操作不作为独立层存在于 layers 中，
/// 而是在 build() 时根据 SkipEdge 信息自动在目标层输入处生成。
#[derive(Clone, Debug)]
pub struct SkipEdge {
    pub innovation_number: u64,
    pub from_innovation: u64,
    pub to_innovation: u64,
    pub strategy: AggregateStrategy,
    pub enabled: bool,
}

// ==================== 训练配置 ====================

#[derive(Clone, Debug, PartialEq)]
pub enum OptimizerType {
    SGD,
    Adam,
}

/// 损失函数类型（Phase 7A 由 TaskMetric 自动推断）
#[derive(Clone, Debug, PartialEq)]
pub enum LossType {
    BCE,
    CrossEntropy,
    MSE,
}

/// 训练配置（与 Genome 绑定，未来可参与演化）
///
/// Phase 7A 使用 Default，后续版本可加入超参数变异操作。
#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f32,
    pub batch_size: Option<usize>,
    pub weight_decay: f32,
    /// None = 自动推断（Phase 7A 默认），Some = 显式指定
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
pub struct NetworkGenome {
    pub layers: Vec<LayerGene>,
    pub skip_edges: Vec<SkipEdge>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub training_config: TrainingConfig,
    pub generated_by: String,
    next_innovation: u64,
    weight_snapshots: HashMap<u64, Vec<Tensor>>,
}

impl Clone for NetworkGenome {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
            skip_edges: self.skip_edges.clone(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
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
            training_config: TrainingConfig::default(),
            generated_by: "minimal".to_string(),
            next_innovation: 2, // 0 = INPUT, 1 = 输出头
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

/// 输出人类可读的架构描述（一行 ASCII）
///
/// 输出头用 `[]` 标注，`enabled=false` 的层不出现。
/// 示例：`"Input(2) → Linear(4) → ReLU → [Linear(1)]"`
impl fmt::Display for NetworkGenome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Input({})", self.input_dim)?;

        let enabled: Vec<&LayerGene> = self.layers.iter().filter(|l| l.enabled).collect();

        for (i, layer) in enabled.iter().enumerate() {
            let is_last = i == enabled.len() - 1;
            if is_last {
                write!(f, " → [{}]", layer.layer_config)?;
            } else {
                write!(f, " → {}", layer.layer_config)?;
            }
        }

        Ok(())
    }
}

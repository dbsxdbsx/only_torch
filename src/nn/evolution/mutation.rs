/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化的变异操作
 *
 * Mutation trait + MutationRegistry + 12 种变异操作：
 * - 7 种结构/参数变异
 * - 3 种 SkipEdge 变异
 * - 2 种训练超参数变异
 *
 * 每种变异通过 is_applicable() 自检合法性，apply() 执行变异。
 * MutationRegistry 按权重随机选择可用变异并执行。
 */

use super::gene::{
    ActivationType, AggregateStrategy, GenomeRepr, INPUT_INNOVATION, LayerConfig, LayerGene,
    LossType, NetworkGenome, OptimizerType, PoolType, ShapeDomain, SkipEdge, TaskMetric,
    compatible_losses,
};
use super::migration::{
    activation_to_node_type, expand_activation, expand_dropout, expand_gru, expand_lstm,
    expand_rnn,
};
use super::node_ops::{
    NodeBlock, NodeBlockKind, add_skip_connection, commit_counter, create_insert_nodes,
    find_connectable_pairs, find_removable_skip_connections, insert_after, is_activation_node,
    is_dropout_node, is_skip_projection_block, make_counter, node_main_path,
    node_output_shape_at, node_param_count, node_spatial_at, remove_block,
    remove_skip_connection, repair_param_input_dims, resize_conv2d_out, resize_linear_out,
    resize_recurrent_out, sync_computation_shapes,
};
use super::net2net::apply_widen_to_snapshots;
use super::cell_migration::{migrate_cell_weights, CellKind};
use crate::tensor::Tensor;
use super::node_gene::{NodeGene, RecurrentEdge};
use crate::nn::descriptor::NodeTypeDescriptor;
use rand::Rng;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use std::fmt;

// ==================== 错误类型 ====================

/// 变异操作错误
#[derive(Debug)]
pub enum MutationError {
    /// 变异不适用于当前基因组
    NotApplicable(String),
    /// 变异结果违反约束
    ConstraintViolation(String),
    /// 内部错误（不应发生）
    InternalError(String),
}

impl fmt::Display for MutationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MutationError::NotApplicable(msg) => write!(f, "变异不适用: {msg}"),
            MutationError::ConstraintViolation(msg) => write!(f, "违反约束: {msg}"),
            MutationError::InternalError(msg) => write!(f, "内部错误: {msg}"),
        }
    }
}

impl std::error::Error for MutationError {}

// ==================== 规模约束 ====================

/// 大小增长策略
#[derive(Clone, Debug)]
pub enum SizeStrategy {
    /// 自由增长（+1 / x2）
    Free,
    /// 对齐到指定倍数（如 8 for AVX2）
    AlignTo(usize),
}

/// 网络规模约束
#[derive(Clone, Debug)]
pub struct SizeConstraints {
    pub max_layers: usize,
    pub max_hidden_size: usize,
    pub max_total_params: usize,
    pub min_hidden_size: usize,
    pub size_strategy: SizeStrategy,
}

impl Default for SizeConstraints {
    fn default() -> Self {
        Self {
            max_layers: 10,
            max_hidden_size: 64,
            max_total_params: 10000,
            min_hidden_size: 1,
            size_strategy: SizeStrategy::Free,
        }
    }
}

impl SizeConstraints {
    /// 数据驱动的自适应约束（用户未显式传入 `with_constraints()` 时自动使用）
    ///
    /// 根据任务的输入/输出维度、训练样本数、是否空间数据，自动推导合理的搜索空间约束。
    /// 确保搜索空间足够容纳有效网络（如 MNIST 的 784→128→10 MLP）。
    pub fn auto(
        input_dim: usize,
        output_dim: usize,
        n_train: usize,
        is_spatial: bool,
        spatial_hw: Option<(usize, usize)>,
    ) -> Self {
        // flatten_dim：空间 = C*H*W，非空间 = input_dim
        let flatten_dim = if let Some((h, w)) = spatial_hw {
            input_dim * h * w
        } else {
            input_dim
        };

        // max_total_params
        let base_params = if is_spatial {
            // 参考基线："Conv(ch→32,k=3) + Conv(32→64,k=3) + 2×Pool + FC(pooled→64) + FC(64→out)"
            // 假设至少 2 次 stride-2 pool 降低空间分辨率，FC 隐藏层 64 即可
            let conv_base = 32 * input_dim * 9 + 64 * 32 * 9; // ~20K conv params
            let spatial_after_pool = spatial_hw
                .map(|(h, w)| (h / 4) * (w / 4))
                .unwrap_or(49);
            let fc_base = 64 * spatial_after_pool * 64 + 64 * output_dim;
            (conv_base + fc_base).max(50_000)
        } else {
            flatten_dim * 128 + 128 * output_dim
        };
        let data_factor = (n_train as f64 / 1000.0).sqrt().clamp(1.0, 10.0);
        let max_total_params = ((base_params as f64 * data_factor) as usize).max(50_000);

        // max_hidden_size（空间模型中是 channels 上限）
        let max_hidden_size = if is_spatial {
            // CNN channels 通常 8..256，给予更大范围
            256
        } else {
            (input_dim / 2).max(128).min(512)
        };

        // max_layers（空间模型需要更多层：Conv+BN+Pool+Flatten+FC...）
        let max_layers = if is_spatial { 20 } else { 10 };

        // min_hidden_size
        let min_hidden_size = if flatten_dim > 100 { 16 } else { 1 };

        // size_strategy
        let size_strategy = if is_spatial || flatten_dim > 100 {
            SizeStrategy::AlignTo(8)
        } else {
            SizeStrategy::Free
        };

        Self {
            max_layers,
            max_hidden_size,
            max_total_params,
            min_hidden_size,
            size_strategy,
        }
    }
}

// ==================== Mutation trait ====================

/// 变异操作 trait
pub trait Mutation: Send + Sync {
    fn name(&self) -> &str;
    fn apply(
        &self,
        genome: &mut NetworkGenome,
        constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError>;
    fn is_applicable(&self, genome: &NetworkGenome, constraints: &SizeConstraints) -> bool;
    /// 是否为结构变异（改变网络拓扑）
    ///
    /// 结构变异（InsertLayer、RemoveLayer）改变层的数量或类型，
    /// 参数变异（Grow/Shrink/MutateParam 等）只调整现有结构的参数。
    /// 停滞检测使用此标记强制结构探索。
    fn is_structural(&self) -> bool {
        false
    }
}

// ==================== MutationRegistry ====================

/// 变异注册表：管理所有变异操作及其权重
pub struct MutationRegistry {
    entries: Vec<(f32, Box<dyn Mutation>)>,
}

impl MutationRegistry {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn register(&mut self, weight: f32, mutation: impl Mutation + 'static) {
        self.entries.push((weight, Box::new(mutation)));
    }

    /// 按权重随机选择一个可用变异并执行，返回变异名称
    ///
    /// `is_applicable` 做轻量预筛选，`apply` 做权威判定。
    /// 若 `apply` 失败（如 params 超标），自动排除该变异并从剩余候选中重试，
    /// 直到成功或所有候选耗尽。`InternalError` 立即传播（表示 bug）。
    pub fn apply_random(
        &self,
        genome: &mut NetworkGenome,
        constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<String, MutationError> {
        let mut candidates: Vec<(f32, &dyn Mutation)> = self
            .entries
            .iter()
            .filter(|(_, m)| m.is_applicable(genome, constraints))
            .map(|(w, m)| (*w, m.as_ref()))
            .collect();

        while !candidates.is_empty() {
            let total_weight: f32 = candidates.iter().map(|(w, _)| w).sum();
            let mut pick = rng.gen_range(0.0..total_weight);
            let mut selected_idx = candidates.len() - 1;
            for (i, (w, _)) in candidates.iter().enumerate() {
                pick -= w;
                if pick <= 0.0 {
                    selected_idx = i;
                    break;
                }
            }

            let (_, selected) = candidates[selected_idx];
            let name = selected.name().to_string();

            match selected.apply(genome, constraints, rng) {
                Ok(()) => {
                    genome.generated_by = name.clone();
                    return Ok(name);
                }
                Err(MutationError::InternalError(msg)) => {
                    return Err(MutationError::InternalError(msg));
                }
                Err(_) => {
                    candidates.remove(selected_idx);
                }
            }
        }

        Err(MutationError::NotApplicable("没有可用的变异操作".into()))
    }

    /// 默认注册表（向后兼容，等价于 `phase1_registry`）
    ///
    /// `is_sequential`: 序列模式时额外注册 MutateCellType。
    /// `is_spatial`: 空间模式时额外注册 MutateKernelSize。
    pub fn default_registry(metric: &TaskMetric, is_sequential: bool, is_spatial: bool) -> Self {
        Self::phase1_registry(metric, is_sequential, is_spatial)
    }

    /// Phase 1 注册表：拓扑搜索（偏向结构探索）
    pub fn phase1_registry(metric: &TaskMetric, is_sequential: bool, is_spatial: bool) -> Self {
        let mut reg = Self::new();
        // 结构变异：模板块插入 + 原子节点插入
        reg.register(0.20, InsertLayerMutation::default());
        reg.register(0.10, InsertAtomicNodeMutation::default());
        reg.register(0.08, RemoveLayerMutation);
        reg.register(0.04, ReplaceLayerTypeMutation::default());
        reg.register(0.12, GrowHiddenSizeMutation);
        reg.register(0.08, ShrinkHiddenSizeMutation);
        reg.register(0.05, MutateLayerParamMutation);
        reg.register(
            0.02,
            MutateLossFunctionMutation {
                task_metric: metric.clone(),
            },
        );
        // SkipEdge 变异（当前仅 LayerLevel 可用；NodeLevel 会因 layers()/skip_edges() 为空而自动失效）
        reg.register(0.08, AddSkipEdgeMutation);
        reg.register(0.05, RemoveSkipEdgeMutation);
        reg.register(0.03, MutateAggregateStrategyMutation);
        // 训练超参数变异（Phase 1 低权重）
        reg.register(0.05, MutateLearningRateMutation);
        reg.register(0.02, MutateOptimizerMutation);
        // NodeLevel 跨层连接变异（非序列 NodeLevel 时自动生效；LayerLevel 因 is_applicable=false 静默跳过）
        reg.register(0.06, AddConnectionMutation);
        reg.register(0.04, RemoveConnectionMutation);
        // 序列模式专属
        if is_sequential {
            reg.register(0.10, MutateCellTypeMutation);
            reg.register(0.08, AddRecurrentEdgeMutation);
            reg.register(0.04, RemoveRecurrentEdgeMutation);
        }
        // 空间模式专属
        if is_spatial {
            reg.register(0.10, MutateKernelSizeMutation);
            reg.register(0.06, MutateStrideMutation);
            // FM 级别变异：结构探索偏重拓扑（Add/Split），参数调整适度降权
            use super::fm_mutation::*;
            reg.register(0.12, AddFeatureMapMutation);
            reg.register(0.04, RemoveFeatureMapMutation);
            reg.register(0.10, AddFMEdgeMutation);
            reg.register(0.04, RemoveFMEdgeMutation);
            reg.register(0.06, SplitFMEdgeMutation);
            reg.register(0.02, ChangeFMEdgeTypeMutation);
            reg.register(0.02, MutateFMEdgeKernelSizeMutation);
            reg.register(0.02, MutateFMEdgeStrideMutation);
            reg.register(0.02, MutateFMEdgeDilationMutation);
            reg.register(0.02, ChangeFeatureMapSizeMutation);
        }
        reg
    }

    /// Phase 2 注册表：精炼（偏向超参数调优）
    pub fn phase2_registry(metric: &TaskMetric, is_sequential: bool, is_spatial: bool) -> Self {
        let mut reg = Self::new();
        // 结构变异：精炼阶段原子节点插入权重提升
        reg.register(0.06, InsertLayerMutation::default());
        reg.register(0.08, InsertAtomicNodeMutation::default());
        reg.register(0.08, RemoveLayerMutation);
        reg.register(0.08, ReplaceLayerTypeMutation::default());
        reg.register(0.10, GrowHiddenSizeMutation);
        reg.register(0.10, ShrinkHiddenSizeMutation);
        reg.register(0.05, MutateLayerParamMutation);
        reg.register(
            0.02,
            MutateLossFunctionMutation {
                task_metric: metric.clone(),
            },
        );
        // SkipEdge 变异（当前仅 LayerLevel 可用；NodeLevel 会因 layers()/skip_edges() 为空而自动失效）
        reg.register(0.06, AddSkipEdgeMutation);
        reg.register(0.05, RemoveSkipEdgeMutation);
        reg.register(0.03, MutateAggregateStrategyMutation);
        // 训练超参数变异（Phase 2 权重上调）
        reg.register(0.15, MutateLearningRateMutation);
        reg.register(0.08, MutateOptimizerMutation);
        // NodeLevel 跨层连接变异（Phase 2 权重略低，专注精炼阶段）
        reg.register(0.04, AddConnectionMutation);
        reg.register(0.04, RemoveConnectionMutation);
        // 序列模式专属
        if is_sequential {
            reg.register(0.10, MutateCellTypeMutation);
            reg.register(0.06, AddRecurrentEdgeMutation);
            reg.register(0.04, RemoveRecurrentEdgeMutation);
        }
        // 空间模式专属
        if is_spatial {
            reg.register(0.10, MutateKernelSizeMutation);
            reg.register(0.06, MutateStrideMutation);
            // FM 级别变异（Phase 2 偏向参数调整，结构探索适度保留）
            use super::fm_mutation::*;
            reg.register(0.06, AddFeatureMapMutation);
            reg.register(0.04, RemoveFeatureMapMutation);
            reg.register(0.06, AddFMEdgeMutation);
            reg.register(0.04, RemoveFMEdgeMutation);
            reg.register(0.04, SplitFMEdgeMutation);
            reg.register(0.02, ChangeFMEdgeTypeMutation);
            reg.register(0.04, MutateFMEdgeKernelSizeMutation);
            reg.register(0.02, MutateFMEdgeStrideMutation);
            reg.register(0.02, MutateFMEdgeDilationMutation);
            reg.register(0.02, ChangeFeatureMapSizeMutation);
        }
        reg
    }

    /// 只从结构变异中随机选择并执行（停滞检测触发时使用）
    ///
    /// 逻辑与 `apply_random` 相同，但候选池限定为 `is_structural() == true` 的变异。
    /// 如果没有可用的结构变异，回退到完整 registry（`apply_random`）。
    pub fn apply_random_structural(
        &self,
        genome: &mut NetworkGenome,
        constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<String, MutationError> {
        let mut candidates: Vec<(f32, &dyn Mutation)> = self
            .entries
            .iter()
            .filter(|(_, m)| m.is_structural() && m.is_applicable(genome, constraints))
            .map(|(w, m)| (*w, m.as_ref()))
            .collect();

        // 没有可用的结构变异时回退到完整 registry
        if candidates.is_empty() {
            return self.apply_random(genome, constraints, rng);
        }

        while !candidates.is_empty() {
            let total_weight: f32 = candidates.iter().map(|(w, _)| w).sum();
            let mut pick = rng.gen_range(0.0..total_weight);
            let mut selected_idx = candidates.len() - 1;
            for (i, (w, _)) in candidates.iter().enumerate() {
                pick -= w;
                if pick <= 0.0 {
                    selected_idx = i;
                    break;
                }
            }

            let (_, selected) = candidates[selected_idx];
            let name = selected.name().to_string();

            match selected.apply(genome, constraints, rng) {
                Ok(()) => {
                    genome.generated_by = name.clone();
                    return Ok(name);
                }
                Err(MutationError::InternalError(msg)) => {
                    return Err(MutationError::InternalError(msg));
                }
                Err(_) => {
                    candidates.remove(selected_idx);
                }
            }
        }

        // 所有结构变异都失败了，回退
        self.apply_random(genome, constraints, rng)
    }

    /// 已注册的变异数量
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ==================== 辅助函数 ====================

/// 筛选非输出头的 enabled 层索引（输出头 = layers 中最后一个 enabled 层）
fn hidden_layer_indices(genome: &NetworkGenome) -> Vec<usize> {
    let last_enabled_idx = genome.layers().iter().rposition(|l| l.enabled);
    genome
        .layers()
        .iter()
        .enumerate()
        .filter(|(i, l)| l.enabled && Some(*i) != last_enabled_idx)
        .map(|(i, _)| i)
        .collect()
}

/// 所有默认激活函数类型
fn default_activations() -> Vec<ActivationType> {
    vec![
        ActivationType::ReLU,
        ActivationType::LeakyReLU { alpha: 0.01 },
        ActivationType::Tanh,
        ActivationType::Sigmoid,
        ActivationType::GELU,
        ActivationType::SiLU,
        ActivationType::Softplus,
        ActivationType::ReLU6,
        ActivationType::ELU { alpha: 1.0 },
        ActivationType::SELU,
        ActivationType::Mish,
        ActivationType::HardSwish,
        ActivationType::HardSigmoid,
    ]
}

/// 判断层配置是否包含可变异的连续参数（供 MutateLayerParamMutation 使用）
fn is_parameterized_layer(config: &LayerConfig) -> bool {
    matches!(
        config,
        LayerConfig::Activation {
            activation_type: ActivationType::LeakyReLU { .. } | ActivationType::ELU { .. }
        } | LayerConfig::Dropout { .. }
    )
}

/// 获取可调整尺寸的层的当前大小（Linear out_features 或 RNN hidden_size）
fn get_resizable_size(config: &LayerConfig) -> Option<usize> {
    match config {
        LayerConfig::Linear { out_features } => Some(*out_features),
        LayerConfig::Rnn { hidden_size }
        | LayerConfig::Lstm { hidden_size }
        | LayerConfig::Gru { hidden_size } => Some(*hidden_size),
        LayerConfig::Conv2d { out_channels, .. } => Some(*out_channels),
        _ => None,
    }
}

/// 设置可调整尺寸层的新大小
fn set_resizable_size(config: &mut LayerConfig, new_size: usize) {
    match config {
        LayerConfig::Linear { out_features } => *out_features = new_size,
        LayerConfig::Rnn { hidden_size }
        | LayerConfig::Lstm { hidden_size }
        | LayerConfig::Gru { hidden_size } => *hidden_size = new_size,
        LayerConfig::Conv2d { out_channels, .. } => *out_channels = new_size,
        _ => {}
    }
}

/// 检查指定位置（在 enabled 层序列中）的相邻层是否有 Activation
fn has_adjacent_activation(genome: &NetworkGenome, insert_pos: usize) -> bool {
    let enabled: Vec<&LayerGene> = genome.layers().iter().filter(|l| l.enabled).collect();
    let before = if insert_pos > 0 {
        matches!(
            enabled.get(insert_pos - 1).map(|l| &l.layer_config),
            Some(LayerConfig::Activation { .. })
        )
    } else {
        false
    };
    let after = matches!(
        enabled.get(insert_pos).map(|l| &l.layer_config),
        Some(LayerConfig::Activation { .. })
    );
    before || after
}

/// 将采样尺寸约束到当前 SizeStrategy
fn sample_size_in_range(
    min: usize,
    max: usize,
    strategy: &SizeStrategy,
    rng: &mut StdRng,
) -> usize {
    assert!(
        min <= max,
        "sample_size_in_range: min({min}) 不能大于 max({max})"
    );
    match strategy {
        SizeStrategy::Free => rng.gen_range(min..=max),
        SizeStrategy::AlignTo(align) => {
            let start = min.div_ceil(*align) * align;
            if start > max {
                max
            } else {
                let end = max / align * align;
                let candidates: Vec<usize> = (start..=end).step_by(*align).collect();
                candidates.choose(rng).copied().unwrap_or(max)
            }
        }
    }
}

/// 获取插入位置的主路径输入维度
fn insert_input_dim(genome: &NetworkGenome, insert_pos: usize) -> usize {
    if insert_pos == 0 {
        return genome.input_dim;
    }
    let enabled: Vec<&LayerGene> = genome.layers().iter().filter(|l| l.enabled).collect();
    let pred_inn = enabled[insert_pos - 1].innovation_number;
    genome
        .resolve_dimensions()
        .ok()
        .and_then(|dims| {
            dims.into_iter()
                .find(|d| d.innovation_number == pred_inn)
                .map(|d| d.out_dim)
        })
        .unwrap_or(genome.input_dim)
}

/// 获取插入位置的主路径输入空间尺寸（空间模式专用）
fn insert_input_spatial(genome: &NetworkGenome, insert_pos: usize) -> Option<(usize, usize)> {
    let spatial_map = genome.compute_spatial_map();
    if insert_pos == 0 {
        return spatial_map.get(&INPUT_INNOVATION).copied().flatten();
    }
    let enabled: Vec<&LayerGene> = genome.layers().iter().filter(|l| l.enabled).collect();
    let pred_inn = enabled[insert_pos - 1].innovation_number;
    spatial_map.get(&pred_inn).copied().flatten()
}

/// 根据 SizeStrategy 计算增长后的值
///
/// Free 模式：40% +step（至少 25%）、40% ×1.5、20% ×2
/// AlignTo 模式：跳到下一个对齐值
fn grow_size(current: usize, max: usize, strategy: &SizeStrategy, rng: &mut StdRng) -> usize {
    let new_size = match strategy {
        SizeStrategy::Free => {
            let roll: f32 = rng.r#gen();
            if roll < 0.4 {
                // +step（增长至少 25%）
                let step = (current / 4).max(1);
                current + step
            } else if roll < 0.8 {
                // ×1.5
                (current as f64 * 1.5).ceil() as usize
            } else {
                // ×2
                current.saturating_mul(2)
            }
        }
        SizeStrategy::AlignTo(align) => {
            let next = ((current / align) + 1) * align;
            next
        }
    };
    new_size.min(max)
}

/// 根据 SizeStrategy 计算缩小后的值
///
/// Free 模式：40% -step（缩小至少 20%）、40% ×0.67、20% ÷2
/// AlignTo 模式：退到上一个对齐值
fn shrink_size(current: usize, min: usize, strategy: &SizeStrategy, rng: &mut StdRng) -> usize {
    let new_size = match strategy {
        SizeStrategy::Free => {
            let roll: f32 = rng.r#gen();
            if roll < 0.4 {
                // -step（缩小至少 20%）
                let step = (current / 5).max(1);
                current.saturating_sub(step)
            } else if roll < 0.8 {
                // ×0.67
                (current as f64 * 0.67).floor() as usize
            } else {
                // ÷2
                current / 2
            }
        }
        SizeStrategy::AlignTo(align) => {
            let prev = current.saturating_sub(1) / align * align;
            if prev == 0 { *align } else { prev }
        }
    };
    new_size.max(min)
}

// ==================== InsertLayerMutation ====================

pub struct InsertLayerMutation {
    available_activations: Vec<ActivationType>,
}

impl InsertLayerMutation {
    pub fn new(available_activations: Vec<ActivationType>) -> Self {
        Self {
            available_activations,
        }
    }
}

impl Default for InsertLayerMutation {
    fn default() -> Self {
        Self::new(default_activations())
    }
}

impl Mutation for InsertLayerMutation {
    fn name(&self) -> &str {
        "InsertLayer"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            // NodeLevel（含序列模式）：create_insert_nodes 支持 Rnn/Lstm/Gru 块
            return genome.layer_count() < constraints.max_layers;
        }
        genome.layer_count() < constraints.max_layers
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        // NodeLevel 分派
        if genome.is_node_level() {
            return node_level_insert_apply(self, genome, constraints, rng);
        }

        let enabled_count = genome.layer_count();
        if enabled_count >= constraints.max_layers {
            return Err(MutationError::ConstraintViolation("已达 max_layers".into()));
        }

        // 插入位置：在 enabled 层序列的 [0, output_head] 之间（含输出头正前方）
        // 但操作的是 layers vec 的实际索引
        let enabled_indices: Vec<usize> = genome
            .layers()
            .iter()
            .enumerate()
            .filter(|(_, l)| l.enabled)
            .map(|(i, _)| i)
            .collect();

        // 输出头在 enabled 序列中的最后一个
        let output_head_vec_idx = *enabled_indices.last().unwrap();

        // 插入到 layers vec 中 [0, output_head_vec_idx] 的某个位置
        // insert(output_head_vec_idx) 会把新层插在输出头正前方，输出头后移
        let insert_vec_idx = rng.gen_range(0..=output_head_vec_idx);

        // 判断在 enabled 序列中此位置的逻辑索引
        let logical_pos = enabled_indices
            .iter()
            .position(|&i| i >= insert_vec_idx)
            .unwrap_or(enabled_indices.len() - 1);

        // 决定插入层类型
        let adjacent_act = has_adjacent_activation(genome, logical_pos);
        let can_insert_activation = !adjacent_act && !self.available_activations.is_empty();
        let is_sequential = genome.seq_len.is_some();
        let is_spatial = genome.input_spatial.is_some();

        // 空间模式：判断插入点所在域
        let insert_domain = if is_spatial {
            let domain_map = genome.compute_domain_map();
            let enabled_layers: Vec<&LayerGene> =
                genome.layers().iter().filter(|l| l.enabled).collect();
            if logical_pos == 0 {
                *domain_map
                    .get(&INPUT_INNOVATION)
                    .unwrap_or(&ShapeDomain::Flat)
            } else {
                let pred_inn = enabled_layers[logical_pos - 1].innovation_number;
                *domain_map.get(&pred_inn).unwrap_or(&ShapeDomain::Flat)
            }
        } else {
            ShapeDomain::Flat
        };

        let insert_input_dim = insert_input_dim(genome, logical_pos);
        let insert_input_spatial = if insert_domain == ShapeDomain::Spatial {
            insert_input_spatial(genome, logical_pos)
        } else {
            None
        };
        let can_insert_pool = insert_input_spatial
            .map(|(h, w)| h >= 2 && w >= 2)
            .unwrap_or(false);

        let new_config = if can_insert_activation && rng.gen_bool(0.3) {
            let act = self.available_activations.choose(rng).unwrap();
            LayerConfig::Activation {
                activation_type: *act,
            }
        } else if insert_domain == ShapeDomain::Spatial {
            // 空间域：Conv2d 或 Pool2d
            if can_insert_pool && rng.gen_bool(0.15) {
                let pool_type = if rng.gen_bool(0.5) {
                    PoolType::Max
                } else {
                    PoolType::Avg
                };
                LayerConfig::Pool2d {
                    pool_type,
                    kernel_size: 2,
                    stride: 2,
                }
            } else {
                let effective_min = constraints.min_hidden_size.max(8);
                let out_ch_cap = insert_input_dim
                    .saturating_mul(16)
                    .max(64)
                    .min(constraints.max_hidden_size)
                    .max(effective_min);
                let out_ch = sample_size_in_range(
                    effective_min,
                    out_ch_cap,
                    &constraints.size_strategy,
                    rng,
                );
                let k = *[1usize, 3, 5, 7].choose(rng).unwrap();
                LayerConfig::Conv2d {
                    out_channels: out_ch,
                    kernel_size: k,
                }
            }
        } else if is_sequential && rng.gen_bool(0.5) {
            let effective_min = constraints.min_hidden_size.max(8);
            let size_cap = insert_input_dim
                .max(effective_min * 2)
                .min(constraints.max_hidden_size)
                .max(effective_min);
            let size =
                sample_size_in_range(effective_min, size_cap, &constraints.size_strategy, rng);
            match rng.gen_range(0..3) {
                0 => LayerConfig::Rnn { hidden_size: size },
                1 => LayerConfig::Lstm { hidden_size: size },
                _ => LayerConfig::Gru { hidden_size: size },
            }
        } else {
            let effective_min = constraints.min_hidden_size.max(8);
            let size_cap = insert_input_dim
                .min(256)
                .max(effective_min * 2)
                .min(constraints.max_hidden_size)
                .max(effective_min);
            let size =
                sample_size_in_range(effective_min, size_cap, &constraints.size_strategy, rng);
            LayerConfig::Linear { out_features: size }
        };

        let inn = genome.next_innovation_number();
        genome.layers_mut().insert(
            insert_vec_idx,
            LayerGene {
                innovation_number: inn,
                layer_config: new_config,
                enabled: true,
            },
        );

        // 插入新层后验证维度兼容性 + 域链合法性 + 已有 skip edge 域兼容性
        if genome.resolve_dimensions().is_err()
            || !genome.is_domain_valid()
            || !genome.validate_skip_edge_domains()
        {
            genome.layers_mut().remove(insert_vec_idx);
            return Err(MutationError::ConstraintViolation(
                "插入层后维度、域链或 skip edge 域不兼容".into(),
            ));
        }

        Ok(())
    }
}

// ==================== RemoveLayerMutation ====================

pub struct RemoveLayerMutation;

impl Mutation for RemoveLayerMutation {
    fn name(&self) -> &str {
        "RemoveLayer"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            let blocks = node_main_path(genome);
            // 需要存在非末尾、且有具体 block_id 的块才可移除
            return blocks.len() > 1
                && blocks[..blocks.len() - 1]
                    .iter()
                    .any(|b| b.block_id.is_some());
        }
        !hidden_layer_indices(genome).is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        if genome.is_node_level() {
            return node_level_remove_apply(genome, rng);
        }
        let candidates = hidden_layer_indices(genome);
        if candidates.is_empty() {
            return Err(MutationError::NotApplicable("没有可移除的隐藏层".into()));
        }
        let &idx = candidates.choose(rng).unwrap();
        let removed_gene = genome.layers_mut().remove(idx);
        let removed_inn = removed_gene.innovation_number;

        // 清理引用已删除层的 skip edges（保留副本用于回滚）
        let removed_edges: Vec<_> = genome
            .skip_edges()
            .iter()
            .filter(|e| e.from_innovation == removed_inn || e.to_innovation == removed_inn)
            .cloned()
            .collect();
        genome
            .skip_edges_mut()
            .retain(|e| e.from_innovation != removed_inn && e.to_innovation != removed_inn);

        // 验证删除后维度兼容性 + 域链合法性 + 已有 skip edge 域兼容性
        if genome.resolve_dimensions().is_err()
            || !genome.is_domain_valid()
            || !genome.validate_skip_edge_domains()
        {
            genome.layers_mut().insert(idx, removed_gene);
            genome.skip_edges_mut().extend(removed_edges);
            return Err(MutationError::ConstraintViolation(
                "删除层后维度、域链或 skip edge 域不兼容".into(),
            ));
        }

        Ok(())
    }
}

// ==================== ReplaceLayerTypeMutation ====================

pub struct ReplaceLayerTypeMutation {
    available_activations: Vec<ActivationType>,
}

impl ReplaceLayerTypeMutation {
    pub fn new(available_activations: Vec<ActivationType>) -> Self {
        Self {
            available_activations,
        }
    }
}

impl Default for ReplaceLayerTypeMutation {
    fn default() -> Self {
        Self::new(default_activations())
    }
}

impl Mutation for ReplaceLayerTypeMutation {
    fn name(&self) -> &str {
        "ReplaceLayerType"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            return !self.available_activations.is_empty()
                && node_main_path(genome)
                    .iter()
                    .any(|b| b.kind.is_activation());
        }
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| {
            matches!(
                genome.layers()[i].layer_config,
                LayerConfig::Activation { .. }
            )
        })
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        if genome.is_node_level() {
            return node_level_replace_activation_apply(self, genome, rng);
        }
        let hidden = hidden_layer_indices(genome);
        let act_indices: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| {
                matches!(
                    genome.layers()[i].layer_config,
                    LayerConfig::Activation { .. }
                )
            })
            .collect();

        if act_indices.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可替换的 Activation 层".into(),
            ));
        }

        let &idx = act_indices.choose(rng).unwrap();
        let current = &genome.layers()[idx].layer_config;

        let alternatives: Vec<&ActivationType> = self
            .available_activations
            .iter()
            .filter(|a| {
                LayerConfig::Activation {
                    activation_type: **a,
                } != *current
            })
            .collect();

        let &&new_act = alternatives
            .choose(rng)
            .ok_or_else(|| MutationError::NotApplicable("没有可选的替代激活函数".into()))?;

        genome.layers_mut()[idx].layer_config = LayerConfig::Activation {
            activation_type: new_act,
        };

        Ok(())
    }
}

// ==================== GrowHiddenSizeMutation ====================

pub struct GrowHiddenSizeMutation;

impl Mutation for GrowHiddenSizeMutation {
    fn name(&self) -> &str {
        "GrowHiddenSize"
    }

    fn is_applicable(&self, genome: &NetworkGenome, constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            return node_main_path(genome).iter().any(|b| {
                b.kind.is_resizable()
                    && b.kind
                        .current_size()
                        .map(|s| s < constraints.max_hidden_size)
                        .unwrap_or(false)
            });
        }
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| {
            get_resizable_size(&genome.layers()[i].layer_config)
                .map(|s| s < constraints.max_hidden_size)
                .unwrap_or(false)
        })
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        if genome.is_node_level() {
            return node_level_grow_apply(genome, constraints, rng, true);
        }
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| {
                get_resizable_size(&genome.layers()[i].layer_config)
                    .map(|s| s < constraints.max_hidden_size)
                    .unwrap_or(false)
            })
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable("没有可增长的层".into()));
        }

        let &idx = candidates.choose(rng).unwrap();
        let old_config = genome.layers()[idx].layer_config.clone();
        let old_size = get_resizable_size(&old_config).unwrap();

        let new_size = grow_size(
            old_size,
            constraints.max_hidden_size,
            &constraints.size_strategy,
            rng,
        );

        debug_assert!(new_size > old_size);

        set_resizable_size(&mut genome.layers_mut()[idx].layer_config, new_size);
        match genome.total_params() {
            Ok(params) if params > constraints.max_total_params => {
                genome.layers_mut()[idx].layer_config = old_config;
                Err(MutationError::ConstraintViolation(format!(
                    "增长后 total_params={params} 超过上限 {}",
                    constraints.max_total_params
                )))
            }
            Err(_) => {
                genome.layers_mut()[idx].layer_config = old_config;
                Err(MutationError::ConstraintViolation(
                    "增长后维度不兼容（skip edge 约束）".into(),
                ))
            }
            _ => {
                // 成功：尝试 Net2Net 扩宽快照，失败时 builder 会随机初始化
                widen_layer_snapshots(genome, idx, old_size, new_size, rng);
                Ok(())
            }
        }
    }
}

// ==================== ShrinkHiddenSizeMutation ====================

pub struct ShrinkHiddenSizeMutation;

impl Mutation for ShrinkHiddenSizeMutation {
    fn name(&self) -> &str {
        "ShrinkHiddenSize"
    }

    fn is_applicable(&self, genome: &NetworkGenome, constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            return node_main_path(genome).iter().any(|b| {
                b.kind.is_resizable()
                    && b.kind
                        .current_size()
                        .map(|s| s > constraints.min_hidden_size)
                        .unwrap_or(false)
            });
        }
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| {
            get_resizable_size(&genome.layers()[i].layer_config)
                .map(|s| s > constraints.min_hidden_size)
                .unwrap_or(false)
        })
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        if genome.is_node_level() {
            return node_level_grow_apply(genome, constraints, rng, false);
        }
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| {
                get_resizable_size(&genome.layers()[i].layer_config)
                    .map(|s| s > constraints.min_hidden_size)
                    .unwrap_or(false)
            })
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable("没有可缩小的层".into()));
        }

        let &idx = candidates.choose(rng).unwrap();
        let old_config = genome.layers()[idx].layer_config.clone();
        let old_size = get_resizable_size(&old_config).unwrap();
        let new_size = shrink_size(
            old_size,
            constraints.min_hidden_size,
            &constraints.size_strategy,
            rng,
        );

        set_resizable_size(&mut genome.layers_mut()[idx].layer_config, new_size);

        if genome.resolve_dimensions().is_err() {
            genome.layers_mut()[idx].layer_config = old_config;
            return Err(MutationError::ConstraintViolation(
                "缩小后维度不兼容（skip edge 约束）".into(),
            ));
        }

        Ok(())
    }
}

// ==================== MutateLayerParamMutation ====================

pub struct MutateLayerParamMutation;

impl Mutation for MutateLayerParamMutation {
    fn name(&self) -> &str {
        "MutateLayerParam"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            return genome
                .nodes()
                .iter()
                .any(|n| n.enabled && matches!(n.node_type, NodeTypeDescriptor::Dropout { .. }));
        }
        let hidden = hidden_layer_indices(genome);
        hidden
            .iter()
            .any(|&i| is_parameterized_layer(&genome.layers()[i].layer_config))
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        if genome.is_node_level() {
            return node_level_mutate_param_apply(genome, rng);
        }
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| is_parameterized_layer(&genome.layers()[i].layer_config))
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable("没有可参数化的层".into()));
        }

        let &idx = candidates.choose(rng).unwrap();
        match &mut genome.layers_mut()[idx].layer_config {
            LayerConfig::Activation {
                activation_type: ActivationType::LeakyReLU { alpha },
            } => {
                // 负半轴斜率，典型值 0.01〜0.3
                *alpha = rng.gen_range(0.001..=0.5);
            }
            LayerConfig::Activation {
                activation_type: ActivationType::ELU { alpha },
            } => {
                // 负半轴饱和值，典型值 1.0（PyTorch 默认）
                *alpha = rng.gen_range(0.1..=2.0);
            }
            LayerConfig::Dropout { p } => {
                *p = rng.gen_range(0.05..=0.8);
            }
            _ => {}
        }

        Ok(())
    }
}

// ==================== MutateLossFunctionMutation ====================

pub struct MutateLossFunctionMutation {
    pub task_metric: TaskMetric,
}

impl Mutation for MutateLossFunctionMutation {
    fn name(&self) -> &str {
        "MutateLossFunction"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        compatible_losses(&self.task_metric, genome.output_dim).len() >= 2
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let candidates = compatible_losses(&self.task_metric, genome.output_dim);
        let current = genome.effective_loss(&self.task_metric);

        let alternatives: Vec<&LossType> = candidates.iter().filter(|l| **l != current).collect();

        let new_loss = alternatives
            .choose(rng)
            .ok_or_else(|| MutationError::NotApplicable("没有可替换的 loss 函数".into()))?;

        genome.training_config.loss_override = Some((*new_loss).clone());
        Ok(())
    }
}

// ==================== MutateLearningRateMutation ====================

/// 学习率离散台阶（覆盖 5 个数量级，共 13 个台阶）
///
/// 离散 ladder 而非连续 log-uniform 的理由：
/// 1. 单 genome 局部变异 + 接受/回滚下，离散台阶避免产生大量"几乎一样"的值
/// 2. 回滚后再次变异能稳定地访问到上次的好值
/// 3. verbose 日志更可读（lr: 1e-2 → 5e-3）
/// 4. 测试断言更确定性
pub(crate) const LR_LADDER: &[f32] = &[
    1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1,
];

/// Adam 有效学习率区间
pub(crate) const ADAM_LR_BAND: (f32, f32) = (1e-4, 1e-2);

/// SGD 有效学习率区间
pub(crate) const SGD_LR_BAND: (f32, f32) = (5e-3, 1e-1);

/// 将任意 lr 值 snap 到 ladder 中最近的台阶索引（log 空间距离）
pub(crate) fn snap_to_nearest_index(value: f32, ladder: &[f32]) -> usize {
    let log_value = value.ln();
    ladder
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (a.ln() - log_value).abs();
            let db = (b.ln() - log_value).abs();
            da.partial_cmp(&db).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap()
}

/// 若 lr 在 band 内则保持不变，否则 snap 到 band 内最近的 ladder 台阶值
pub(crate) fn snap_to_nearest_in_band(lr: f32, band: (f32, f32), ladder: &[f32]) -> f32 {
    if lr >= band.0 && lr <= band.1 {
        return lr;
    }
    let log_lr = lr.ln();
    ladder
        .iter()
        .filter(|&&v| v >= band.0 && v <= band.1)
        .min_by(|a, b| {
            let da = (a.ln() - log_lr).abs();
            let db = (b.ln() - log_lr).abs();
            da.partial_cmp(&db).unwrap()
        })
        .copied()
        .unwrap_or(if lr < band.0 { band.0 } else { band.1 })
}

pub struct MutateLearningRateMutation;

impl Mutation for MutateLearningRateMutation {
    fn name(&self) -> &str {
        "MutateLearningRate"
    }

    fn is_applicable(&self, _genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        true
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let current_lr = genome.training_config.learning_rate;
        let current_idx = snap_to_nearest_index(current_lr, LR_LADDER);

        // 80% 移动 1 步，20% 移动 2 步
        let steps: i32 = if rng.gen_bool(0.8) { 1 } else { 2 };
        // 上下等概率
        let direction: i32 = if rng.gen_bool(0.5) { 1 } else { -1 };

        let new_idx = (current_idx as i32 + steps * direction)
            .clamp(0, (LR_LADDER.len() - 1) as i32) as usize;

        genome.training_config.learning_rate = LR_LADDER[new_idx];
        Ok(())
    }
}

// ==================== MutateOptimizerMutation ====================

pub struct MutateOptimizerMutation;

impl Mutation for MutateOptimizerMutation {
    fn name(&self) -> &str {
        "MutateOptimizer"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        // 仅输出头时不切换（结构太简单，优化器选择无意义）
        genome.layer_count() > 1
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        _rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let (new_optimizer, target_band) = match genome.training_config.optimizer_type {
            OptimizerType::Adam => (OptimizerType::SGD, SGD_LR_BAND),
            OptimizerType::SGD => (OptimizerType::Adam, ADAM_LR_BAND),
        };

        genome.training_config.optimizer_type = new_optimizer;
        genome.training_config.learning_rate =
            snap_to_nearest_in_band(genome.training_config.learning_rate, target_band, LR_LADDER);
        Ok(())
    }
}

// ==================== MutateCellTypeMutation ====================

/// 循环层类型切换（Rnn ↔ Lstm ↔ Gru）
///
/// 保持 hidden_size 不变，仅切换 cell 类型。
/// 权重快照失效后由 builder 重新初始化。
pub struct MutateCellTypeMutation;

impl Mutation for MutateCellTypeMutation {
    fn name(&self) -> &str {
        "MutateCellType"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            return node_main_path(genome).iter().any(|b| b.kind.is_recurrent());
        }
        let hidden = hidden_layer_indices(genome);
        hidden
            .iter()
            .any(|&i| NetworkGenome::is_recurrent(&genome.layers()[i].layer_config))
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        if genome.is_node_level() {
            return node_level_mutate_cell_type_apply(genome, rng);
        }
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| NetworkGenome::is_recurrent(&genome.layers()[i].layer_config))
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable("没有可切换的循环层".into()));
        }

        let &idx = candidates.choose(rng).unwrap();
        let hidden_size = get_resizable_size(&genome.layers()[idx].layer_config).unwrap();

        // 排除当前类型，从剩余两种中随机选择
        let new_config = match &genome.layers()[idx].layer_config {
            LayerConfig::Rnn { .. } => {
                if rng.gen_bool(0.5) {
                    LayerConfig::Lstm { hidden_size }
                } else {
                    LayerConfig::Gru { hidden_size }
                }
            }
            LayerConfig::Lstm { .. } => {
                if rng.gen_bool(0.5) {
                    LayerConfig::Rnn { hidden_size }
                } else {
                    LayerConfig::Gru { hidden_size }
                }
            }
            LayerConfig::Gru { .. } => {
                if rng.gen_bool(0.5) {
                    LayerConfig::Rnn { hidden_size }
                } else {
                    LayerConfig::Lstm { hidden_size }
                }
            }
            _ => unreachable!(),
        };

        // 尝试 informed initialization：将旧 cell 权重迁移到新类型，失败则清空快照走随机初始化
        let inn = genome.layers()[idx].innovation_number;
        let old_kind = layer_cell_kind(&genome.layers()[idx].layer_config);
        let new_kind_opt = layer_cell_kind(&new_config);
        let hidden = get_resizable_size(&genome.layers()[idx].layer_config).unwrap_or(0);
        if let (Some(ok), Some(nk)) = (old_kind, new_kind_opt) {
            let migrated = genome
                .weight_snapshots()
                .get(&inn)
                .cloned()
                .and_then(|snap| migrate_layer_cell_weights_vec(&snap, ok, nk, hidden));
            genome.remove_layer_weight_snapshot(inn);
            if let Some(new_snap) = migrated {
                if let GenomeRepr::LayerLevel { weight_snapshots, .. } = &mut genome.repr {
                    weight_snapshots.insert(inn, new_snap);
                }
            }
        } else {
            genome.remove_layer_weight_snapshot(inn);
        }

        let old_config = genome.layers()[idx].layer_config.clone();
        genome.layers_mut()[idx].layer_config = new_config;

        // 切换 cell 类型不会改变域链（仍然是循环层），但保险起见仍检查 skip edge 域兼容性
        if !genome.validate_skip_edge_domains() {
            genome.layers_mut()[idx].layer_config = old_config;
            return Err(MutationError::ConstraintViolation(
                "切换 cell 类型后 skip edge 域不兼容".into(),
            ));
        }

        Ok(())
    }
}

// ==================== MutateCellType NodeLevel 辅助 ====================

/// NodeLevel 循环单元类型切换：替换 Cell* 节点（及其参数节点）为新类型
///
/// 流程：
/// 1. 找到含有 CellRnn/CellLstm/CellGru 的块
/// 2. 随机选择一个
/// 3. 移除旧的参数节点 + Cell 节点
/// 4. 用新类型的 expand_rnn/lstm/gru 重新插入
fn node_level_mutate_cell_type_apply(
    genome: &mut NetworkGenome,
    rng: &mut StdRng,
) -> Result<(), MutationError> {
    use NodeTypeDescriptor as NT;

    let blocks = node_main_path(genome);
    let recurrent_blocks: Vec<NodeBlock> = blocks
        .into_iter()
        .filter(|b| b.kind.is_recurrent())
        .collect();

    if recurrent_blocks.is_empty() {
        return Err(MutationError::NotApplicable(
            "没有可切换类型的循环单元块".into(),
        ));
    }

    let block = recurrent_blocks.choose(rng).unwrap().clone();

    // 从 Cell 节点读取元数据
    let cell_node = genome
        .nodes()
        .iter()
        .find(|n| {
            block.node_ids.contains(&n.innovation_number)
                && matches!(
                    n.node_type,
                    NT::CellRnn { .. } | NT::CellLstm { .. } | NT::CellGru { .. }
                )
        })
        .cloned()
        .ok_or_else(|| MutationError::InternalError("找不到 Cell* 节点".into()))?;

    let (hidden_size, return_sequences, seq_len) = match &cell_node.node_type {
        NT::CellRnn {
            hidden_size,
            return_sequences,
            seq_len,
            ..
        } => (*hidden_size, *return_sequences, *seq_len),
        NT::CellLstm {
            hidden_size,
            return_sequences,
            seq_len,
            ..
        } => (*hidden_size, *return_sequences, *seq_len),
        NT::CellGru {
            hidden_size,
            return_sequences,
            seq_len,
            ..
        } => (*hidden_size, *return_sequences, *seq_len),
        _ => unreachable!(),
    };

    // 确定输入维度（进入 Cell 节点的 input 父节点的形状最后一维）
    // 若父节点是虚拟输入 INPUT_INNOVATION，则直接使用 genome.input_dim。
    let input_node_id = cell_node.parents[0];
    let in_dim = if input_node_id == INPUT_INNOVATION {
        genome.input_dim
    } else {
        genome
            .nodes()
            .iter()
            .find(|n| n.innovation_number == input_node_id)
            .and_then(|n| n.output_shape.last().copied())
            .ok_or_else(|| MutationError::InternalError("找不到循环单元的输入节点".into()))?
    };

    // 选择新的 cell 类型（排除当前，0=Rnn 1=Lstm 2=Gru）
    let new_type_idx: u8 = match &cell_node.node_type {
        NT::CellRnn { .. } => {
            if rng.gen_bool(0.5) {
                1
            } else {
                2
            }
        }
        NT::CellLstm { .. } => {
            if rng.gen_bool(0.5) {
                0
            } else {
                2
            }
        }
        NT::CellGru { .. } => {
            if rng.gen_bool(0.5) {
                0
            } else {
                1
            }
        }
        _ => unreachable!(),
    };

    // 块的 block_id（新节点复用同 block_id）
    let block_bid = block.block_id.unwrap_or(0);

    let old_output_id = block.output_id;
    let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();

    // ===== F2: 迁移前保存旧 cell 的参数快照（按 expand_* 的参数顺序）=====
    //
    // cell_node.parents 布局为 [input_id, param_id_0, param_id_1, ...]，
    // 跳过首元素后剩余 id 顺序与 expand_* 返回参数节点的顺序一致。
    let old_cell_kind = match &cell_node.node_type {
        NT::CellRnn { .. } => Some(CellKind::Rnn),
        NT::CellLstm { .. } => Some(CellKind::Lstm),
        NT::CellGru { .. } => Some(CellKind::Gru),
        _ => None,
    };
    let old_param_snapshots: Vec<Option<Tensor>> = if genome.is_node_level() {
        cell_node
            .parents
            .iter()
            .skip(1)
            .map(|pid| genome.node_weight_snapshots().get(pid).cloned())
            .collect()
    } else {
        Vec::new()
    };

    // 删除旧 block 中的全部节点
    genome
        .nodes_mut()
        .retain(|n| !bid_set.contains(&n.innovation_number));

    // 同步移除旧快照中属于被删节点的条目，避免孤儿快照
    if genome.is_node_level() {
        let snaps = genome.node_weight_snapshots_mut();
        snaps.retain(|k, _| !bid_set.contains(k));
    }

    // 生成新 block
    let mut counter = make_counter(genome);
    let new_nodes = match new_type_idx {
        0 => expand_rnn(
            input_node_id,
            in_dim,
            hidden_size,
            return_sequences,
            seq_len,
            block_bid,
            &mut counter,
        ),
        1 => expand_lstm(
            input_node_id,
            in_dim,
            hidden_size,
            return_sequences,
            seq_len,
            block_bid,
            &mut counter,
        ),
        _ => expand_gru(
            input_node_id,
            in_dim,
            hidden_size,
            return_sequences,
            seq_len,
            block_bid,
            &mut counter,
        ),
    };

    let new_output_id = new_nodes.last().map(|n| n.innovation_number).unwrap();

    // ===== F2: 为新参数节点构造迁移快照 =====
    let new_cell_kind = match new_type_idx {
        0 => CellKind::Rnn,
        1 => CellKind::Lstm,
        _ => CellKind::Gru,
    };
    // new_nodes 除了最后一个是 Cell* 之外，前 N 个都是 Parameter 节点，
    // 顺序与 expand_* 的参数顺序一致。
    let new_param_ids: Vec<u64> = new_nodes
        .iter()
        .take(new_nodes.len().saturating_sub(1))
        .filter(|n| n.is_parameter())
        .map(|n| n.innovation_number)
        .collect();

    genome.nodes_mut().extend(new_nodes);
    commit_counter(genome, &counter);

    // 将后续节点中引用 old_output_id 的父节点全部替换为 new_output_id
    for node in genome.nodes_mut().iter_mut() {
        for pid in node.parents.iter_mut() {
            if *pid == old_output_id {
                *pid = new_output_id;
            }
        }
    }

    // 写入迁移快照（仅 NodeLevel + 新旧快照都齐全时才生效）
    if let (true, Some(old_kind)) = (genome.is_node_level(), old_cell_kind) {
        if let Some(migrated) = migrate_cell_weights(
            old_kind,
            &old_param_snapshots,
            new_cell_kind,
            &new_param_ids,
            hidden_size,
        ) {
            let snaps = genome.node_weight_snapshots_mut();
            for (id, t) in migrated {
                snaps.insert(id, t);
            }
        }
    }

    // 重新推导计算节点形状
    sync_computation_shapes(genome);
    Ok(())
}

// ==================== MutateKernelSizeMutation ====================
pub struct MutateKernelSizeMutation;

const KERNEL_SIZES: &[usize] = &[1, 3, 5, 7];

impl Mutation for MutateKernelSizeMutation {
    fn name(&self) -> &str {
        "MutateKernelSize"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if genome.is_node_level() {
            return node_main_path(genome).iter().any(|b| b.kind.is_conv2d());
        }
        let hidden = hidden_layer_indices(genome);
        hidden
            .iter()
            .any(|&i| matches!(genome.layers()[i].layer_config, LayerConfig::Conv2d { .. }))
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        if genome.is_node_level() {
            return node_level_mutate_kernel_size_apply(genome, rng);
        }
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| matches!(genome.layers()[i].layer_config, LayerConfig::Conv2d { .. }))
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有 Conv2d 层可变异 kernel_size".into(),
            ));
        }

        let &idx = candidates.choose(rng).unwrap();
        if let LayerConfig::Conv2d {
            kernel_size: ref current_k,
            ..
        } = genome.layers()[idx].layer_config
        {
            let alternatives: Vec<usize> = KERNEL_SIZES
                .iter()
                .copied()
                .filter(|&k| k != *current_k)
                .collect();
            if let Some(&new_k) = alternatives.choose(rng) {
                if let LayerConfig::Conv2d {
                    ref mut kernel_size,
                    ..
                } = genome.layers_mut()[idx].layer_config
                {
                    *kernel_size = new_k;
                }
            }
        }
        Ok(())
    }
}

// ==================== AddSkipEdgeMutation ====================

/// 随机可选的聚合策略列表
fn all_aggregate_strategies() -> Vec<AggregateStrategy> {
    vec![
        AggregateStrategy::Add,
        AggregateStrategy::Concat { dim: 1 },
        AggregateStrategy::Mean,
        AggregateStrategy::Max,
    ]
}

pub struct AddSkipEdgeMutation;

impl AddSkipEdgeMutation {
    /// 收集所有可行的 (from, to, strategy) 候选
    ///
    /// 预过滤逻辑：
    /// 1. DAG 前向约束：from 在层序列中的位置 < to
    /// 2. 不重复：已存在的 (from, to) 对被排除
    /// 3. 目标层 group 约束：已有 skip edge 的目标层只许沿用已有策略
    /// 4. 域约束：序列模型中只允许 Flat 域内的 skip edge
    ///    记忆单元（RNN/LSTM/GRU）作为原子单元，不允许 skip edge
    ///    跨越或穿透 Sequence 域（避免 3D/2D 形状不兼容和 concat dim 语义混乱）
    /// 5. 维度兼容性：trial resolve_dimensions 验证
    fn feasible_candidates(genome: &NetworkGenome) -> Vec<(u64, u64, AggregateStrategy)> {
        let enabled: Vec<u64> = genome
            .layers()
            .iter()
            .filter(|l| l.enabled)
            .map(|l| l.innovation_number)
            .collect();

        let existing: std::collections::HashSet<(u64, u64)> = genome
            .skip_edges()
            .iter()
            .filter(|e| e.enabled)
            .map(|e| (e.from_innovation, e.to_innovation))
            .collect();

        // 域映射：序列模型中只允许 Flat 域内的 skip edge。
        // Sequence 域内的 skip edge 有 concat dim 语义问题（dim=1 对 3D 是 seq_len
        // 而非 features），且记忆单元应作为原子单元不被 skip 穿透。
        let domain_map = genome.compute_domain_map();
        // 每层的“输入域”：即主路径在聚合点处的域
        //   = 前一层的输出域（或 Input 域）
        let input_domain_at: std::collections::HashMap<u64, ShapeDomain> = {
            let mut map = std::collections::HashMap::new();
            let input_domain = *domain_map.get(&INPUT_INNOVATION).unwrap();
            let mut prev_domain = input_domain;
            for &inn in &enabled {
                map.insert(inn, prev_domain);
                prev_domain = *domain_map.get(&inn).unwrap();
            }
            map
        };

        let mut candidates = Vec::new();

        // 空间模式需要 spatial_map 做 H/W 兼容检查
        let spatial_map = if genome.is_spatial() {
            Some(genome.compute_spatial_map())
        } else {
            None
        };

        for (to_idx, &to_inn) in enabled.iter().enumerate() {
            let to_domain = input_domain_at.get(&to_inn).copied().unwrap();
            // Sequence 域不允许 skip edge
            if to_domain == ShapeDomain::Sequence {
                continue;
            }

            // 确定该目标层允许的策略
            let group_strategy = genome
                .skip_edges()
                .iter()
                .find(|e| e.enabled && e.to_innovation == to_inn)
                .map(|e| e.strategy.clone());
            let strategies = match group_strategy {
                Some(s) => vec![s],
                None => all_aggregate_strategies(),
            };

            // 直接前驱的创新号：主路径已经将该层的输出送到 to，
            // skip edge 会携带完全相同的张量，对所有聚合策略都是退化的
            // （Add=×2 缩放、Mean/Max=恒等、Concat=冗余翻倍）
            let immediate_pred = if to_idx > 0 {
                Some(enabled[to_idx - 1])
            } else {
                None // to 是第一层，直接前驱为 INPUT
            };

            // 空间域 skip edge 目标的输入 spatial（用于 H/W 兼容检查）
            let to_input_spatial = if to_domain == ShapeDomain::Spatial {
                if to_idx == 0 {
                    spatial_map
                        .as_ref()
                        .and_then(|m| m.get(&INPUT_INNOVATION).copied().flatten())
                } else {
                    spatial_map
                        .as_ref()
                        .and_then(|m| m.get(&enabled[to_idx - 1]).copied().flatten())
                }
            } else {
                None
            };

            // 收集所有前向 from，要求同域 + 空间域 H/W 匹配
            let mut froms = Vec::new();
            if !existing.contains(&(INPUT_INNOVATION, to_inn)) && immediate_pred.is_some() {
                let from_domain = *domain_map.get(&INPUT_INNOVATION).unwrap();
                if from_domain == to_domain {
                    let sp_ok = if to_domain == ShapeDomain::Spatial {
                        spatial_map
                            .as_ref()
                            .and_then(|m| m.get(&INPUT_INNOVATION).copied().flatten())
                            == to_input_spatial
                    } else {
                        true
                    };
                    if sp_ok {
                        froms.push(INPUT_INNOVATION);
                    }
                }
            }
            for &from_inn in &enabled[..to_idx] {
                if Some(from_inn) == immediate_pred {
                    continue;
                }
                if !existing.contains(&(from_inn, to_inn)) {
                    let from_domain = *domain_map.get(&from_inn).unwrap();
                    if from_domain == to_domain {
                        let sp_ok = if to_domain == ShapeDomain::Spatial {
                            spatial_map
                                .as_ref()
                                .and_then(|m| m.get(&from_inn).copied().flatten())
                                == to_input_spatial
                        } else {
                            true
                        };
                        if sp_ok {
                            froms.push(from_inn);
                        }
                    }
                }
            }

            // 对每个 (from, strategy) 组合做 trial 验证
            for &from in &froms {
                for strategy in &strategies {
                    let mut trial = genome.clone();
                    trial.skip_edges_mut().push(SkipEdge {
                        innovation_number: u64::MAX, // placeholder
                        from_innovation: from,
                        to_innovation: to_inn,
                        strategy: strategy.clone(),
                        enabled: true,
                    });
                    if trial.resolve_dimensions().is_ok() {
                        candidates.push((from, to_inn, strategy.clone()));
                    }
                }
            }
        }

        candidates
    }
}

impl Mutation for AddSkipEdgeMutation {
    fn name(&self) -> &str {
        "AddSkipEdge"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        // 快速预筛：至少有 1 个 enabled 层才可能有候选对
        genome.layers().iter().any(|l| l.enabled)
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let candidates = Self::feasible_candidates(genome);
        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可行的 skip edge 候选".into(),
            ));
        }

        let &(from, to, ref strategy) = candidates.choose(rng).unwrap();
        let inn = genome.next_innovation_number();
        genome.skip_edges_mut().push(SkipEdge {
            innovation_number: inn,
            from_innovation: from,
            to_innovation: to,
            strategy: strategy.clone(),
            enabled: true,
        });
        Ok(())
    }
}

// ==================== RemoveSkipEdgeMutation ====================

pub struct RemoveSkipEdgeMutation;

impl Mutation for RemoveSkipEdgeMutation {
    fn name(&self) -> &str {
        "RemoveSkipEdge"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        genome.skip_edges().iter().any(|e| e.enabled)
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let enabled_indices: Vec<usize> = genome
            .skip_edges()
            .iter()
            .enumerate()
            .filter(|(_, e)| e.enabled)
            .map(|(i, _)| i)
            .collect();

        if enabled_indices.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可移除的 skip edge".into(),
            ));
        }

        let &idx = enabled_indices.choose(rng).unwrap();
        genome.skip_edges_mut().remove(idx);
        Ok(())
    }
}

// ==================== MutateAggregateStrategyMutation ====================

pub struct MutateAggregateStrategyMutation;

impl Mutation for MutateAggregateStrategyMutation {
    fn name(&self) -> &str {
        "MutateAggregateStrategy"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        genome.skip_edges().iter().any(|e| e.enabled)
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        // 收集所有不同的 target group
        let mut target_inns: Vec<u64> = genome
            .skip_edges()
            .iter()
            .filter(|e| e.enabled)
            .map(|e| e.to_innovation)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        if target_inns.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有 skip edge target group".into(),
            ));
        }

        // 打乱顺序，依次尝试每个 target group
        target_inns.shuffle(rng);

        for target in &target_inns {
            let current_strategy = genome
                .skip_edges()
                .iter()
                .find(|e| e.enabled && e.to_innovation == *target)
                .map(|e| e.strategy.clone())
                .unwrap();

            // 计算可行的替代策略（排除当前 + trial 验证）
            let feasible: Vec<AggregateStrategy> = all_aggregate_strategies()
                .into_iter()
                .filter(|s| s != &current_strategy)
                .filter(|s| {
                    let mut trial = genome.clone();
                    for edge in trial.skip_edges_mut().iter_mut() {
                        if edge.enabled && edge.to_innovation == *target {
                            edge.strategy = s.clone();
                        }
                    }
                    trial.resolve_dimensions().is_ok()
                })
                .collect();

            if let Some(new_strategy) = feasible.choose(rng) {
                let new_strategy = new_strategy.clone();
                for edge in genome.skip_edges_mut().iter_mut() {
                    if edge.enabled && edge.to_innovation == *target {
                        edge.strategy = new_strategy.clone();
                    }
                }
                return Ok(());
            }
        }

        Err(MutationError::NotApplicable(
            "所有 target group 均无可行替代策略".into(),
        ))
    }
}

// ==================== NodeLevel 变异分派实现 ====================

/// 内部辅助：将 NodeLevel 基因组的 next_innovation 前进 by 步
fn advance_node_counter(genome: &mut NetworkGenome, by: u64) {
    use super::gene::GenomeRepr;
    if let GenomeRepr::NodeLevel {
        next_innovation, ..
    } = &mut genome.repr
    {
        *next_innovation += by;
    }
}

/// 内部辅助：将 NodeLevel 基因组的 next_innovation 重置为 to
fn reset_node_counter(genome: &mut NetworkGenome, to: u64) {
    use super::gene::GenomeRepr;
    if let GenomeRepr::NodeLevel {
        next_innovation, ..
    } = &mut genome.repr
    {
        *next_innovation = to;
    }
}

/// InsertLayer 的 NodeLevel 实现
///
/// 在主路径中某个非末尾块的输出节点之后插入新节点组，并检查参数量约束。
/// 若参数量超标，回滚插入操作。
fn node_level_insert_apply(
    m: &InsertLayerMutation,
    genome: &mut NetworkGenome,
    constraints: &SizeConstraints,
    rng: &mut StdRng,
) -> Result<(), MutationError> {
    let blocks = node_main_path(genome);
    if blocks.is_empty() {
        return Err(MutationError::NotApplicable("基因组没有可用块".into()));
    }

    // 只允许在非末尾块的 output_id 之后插入，避免在输出头后面形成悬空节点
    // 若当前仅剩输出头，则允许在输入后插入
    let insert_candidates: Vec<u64> = if blocks.len() > 1 {
        blocks[..blocks.len() - 1]
            .iter()
            .map(|b| b.output_id)
            .collect()
    } else {
        vec![INPUT_INNOVATION]
    };
    let after_id = *insert_candidates
        .choose(rng)
        .ok_or_else(|| MutationError::NotApplicable("没有可用的插入点".into()))?;

    let adjacent_act = is_activation_node(genome, after_id);
    let start_inn = genome.peek_next_innovation();

    let new_nodes = create_insert_nodes(
        genome,
        after_id,
        constraints,
        rng,
        &m.available_activations,
        adjacent_act,
    )
    .ok_or_else(|| MutationError::NotApplicable("无法生成插入节点".into()))?;

    let n = new_nodes.len() as u64;
    insert_after(genome, after_id, new_nodes).map_err(|e| MutationError::InternalError(e))?;
    advance_node_counter(genome, n);
    repair_param_input_dims(genome);

    let analysis = genome.analyze();
    if !analysis.is_valid {
        let new_node_ids: std::collections::HashSet<u64> = (start_inn..start_inn + n).collect();
        let new_output_id = start_inn + n - 1;
        for node in genome.nodes_mut().iter_mut() {
            for pid in node.parents.iter_mut() {
                if *pid == new_output_id {
                    *pid = after_id;
                }
            }
        }
        genome
            .nodes_mut()
            .retain(|nd| !new_node_ids.contains(&nd.innovation_number));
        reset_node_counter(genome, start_inn);
        repair_param_input_dims(genome);
        return Err(MutationError::ConstraintViolation(
            "插入后图不合法（域或形状不兼容）".into(),
        ));
    }

    // 总参数量约束检查（repair 后才能得到准确的维度）
    let params = node_param_count(genome);
    if params > constraints.max_total_params {
        // 回滚：撤销 insert_after 的效果
        let new_node_ids: std::collections::HashSet<u64> = (start_inn..start_inn + n).collect();
        let new_output_id = start_inn + n - 1;
        for node in genome.nodes_mut().iter_mut() {
            for pid in node.parents.iter_mut() {
                if *pid == new_output_id {
                    *pid = after_id;
                }
            }
        }
        genome
            .nodes_mut()
            .retain(|nd| !new_node_ids.contains(&nd.innovation_number));
        reset_node_counter(genome, start_inn);
        // 回滚后再次修复形状（恢复上游 W 的原始输入维度）
        repair_param_input_dims(genome);
        return Err(MutationError::ConstraintViolation(format!(
            "插入后 total_params={params} 超过上限 {}",
            constraints.max_total_params
        )));
    }

    Ok(())
}

/// RemoveLayer 的 NodeLevel 实现
///
/// 移除主路径中一个非末尾的、有具体 block_id 的块，并重新连线。
fn node_level_remove_apply(
    genome: &mut NetworkGenome,
    rng: &mut StdRng,
) -> Result<(), MutationError> {
    let old_genome = genome.clone();
    let blocks = node_main_path(genome);
    if blocks.len() <= 1 {
        return Err(MutationError::NotApplicable("只剩输出头，无法移除".into()));
    }

    // 只移除非末尾、有具体 block_id 的块（排除 None = 独立激活节点组）
    let removable: Vec<&NodeBlock> = blocks[..blocks.len() - 1]
        .iter()
        .filter(|b| b.block_id.is_some())
        .collect();

    if removable.is_empty() {
        return Err(MutationError::NotApplicable("没有可移除的中间块".into()));
    }

    let block = (*removable.choose(rng).unwrap()).clone();
    if genome.seq_len.is_some() && block.kind.is_recurrent() {
        let recurrent_count = blocks.iter().filter(|b| b.kind.is_recurrent()).count();
        if recurrent_count <= 1 {
            return Err(MutationError::NotApplicable(
                "序列图至少需要保留一个循环块".into(),
            ));
        }
    }
    remove_block(genome, &block);
    repair_param_input_dims(genome);

    let analysis = genome.analyze();
    if !analysis.is_valid {
        *genome = old_genome;
        return Err(MutationError::ConstraintViolation(
            "删除后图不合法（域或形状不兼容）".into(),
        ));
    }
    Ok(())
}

/// GrowHiddenSize / ShrinkHiddenSize 的 NodeLevel 共用实现
///
/// `is_grow=true` → 增大，`is_grow=false` → 缩小。
/// 增大时检查 max_total_params 并可回滚。
fn is_fm_edge_block(genome: &NetworkGenome, block: &NodeBlock) -> bool {
    let nodes = genome.nodes();
    block.node_ids.iter().any(|&nid| {
        nodes
            .iter()
            .any(|n| n.innovation_number == nid && n.fm_id.is_some())
    }) || block.node_ids.iter().any(|&nid| {
        nodes.iter().any(|n| {
            n.innovation_number == nid
                && n.is_parameter()
                && n.output_shape.len() == 4
                && n.output_shape[0] == 1
                && n.output_shape[1] == 1
        })
    })
}

fn node_level_grow_apply(
    genome: &mut NetworkGenome,
    constraints: &SizeConstraints,
    rng: &mut StdRng,
    is_grow: bool,
) -> Result<(), MutationError> {
    let blocks = node_main_path(genome);
    let candidates: Vec<&NodeBlock> = blocks
        .iter()
        .filter(|b| {
            // 跳跃投影块不参与 Grow/Shrink（由 repair_skip_connections 自动维护）
            if is_skip_projection_block(genome, b) {
                return false;
            }
            // FM edge 块不参与 Grow/Shrink（由 FM 级别变异处理）
            if is_fm_edge_block(genome, b) {
                return false;
            }
            if is_grow {
                b.kind.is_resizable()
                    && b.kind
                        .current_size()
                        .map(|s| s < constraints.max_hidden_size)
                        .unwrap_or(false)
            } else {
                b.kind.is_resizable()
                    && b.kind
                        .current_size()
                        .map(|s| s > constraints.min_hidden_size)
                        .unwrap_or(false)
            }
        })
        .collect();

    if candidates.is_empty() {
        return Err(MutationError::NotApplicable(
            if is_grow {
                "没有可增长的块"
            } else {
                "没有可缩小的块"
            }
            .into(),
        ));
    }

    let block = (*candidates.choose(rng).unwrap()).clone();
    let current_size = block.kind.current_size().unwrap();

    let new_size = if is_grow {
        grow_size(
            current_size,
            constraints.max_hidden_size,
            &constraints.size_strategy,
            rng,
        )
    } else {
        shrink_size(
            current_size,
            constraints.min_hidden_size,
            &constraints.size_strategy,
            rng,
        )
    };

    // 执行 resize
    match &block.kind {
        NodeBlockKind::Linear { .. } => resize_linear_out(genome, &block, new_size)
            .map_err(|e| MutationError::InternalError(e))?,
        NodeBlockKind::Conv2d { .. } => resize_conv2d_out(genome, &block, new_size)
            .map_err(|e| MutationError::InternalError(e))?,
        NodeBlockKind::Rnn { .. } | NodeBlockKind::Lstm { .. } | NodeBlockKind::Gru { .. } => {
            resize_recurrent_out(genome, &block, new_size)
                .map_err(|e| MutationError::InternalError(e))?
        }
        _ => return Err(MutationError::NotApplicable("不可调整大小的块类型".into())),
    }

    // 增大时检查参数量约束，超标则回滚
    if is_grow {
        let params = node_param_count(genome);
        if params > constraints.max_total_params {
            // 回滚：resize 回原始大小
            let blocks_after = node_main_path(genome);
            if let Some(updated) = blocks_after.iter().find(|b| b.block_id == block.block_id) {
                let _ = match &block.kind {
                    NodeBlockKind::Linear { .. } => {
                        resize_linear_out(genome, updated, current_size)
                    }
                    NodeBlockKind::Conv2d { .. } => {
                        resize_conv2d_out(genome, updated, current_size)
                    }
                    NodeBlockKind::Rnn { .. }
                    | NodeBlockKind::Lstm { .. }
                    | NodeBlockKind::Gru { .. } => {
                        resize_recurrent_out(genome, updated, current_size)
                    }
                    _ => Ok(()),
                };
            }
            return Err(MutationError::ConstraintViolation(format!(
                "增长后 total_params={params} 超过上限 {}",
                constraints.max_total_params
            )));
        }

        // 参数预算检查通过后，对快照应用 Net2Net 函数保持扩宽
        // （仅 NodeLevel 基因组有效；失败时让后续 try_partial_inherit 走朴素回退）
        if genome.is_node_level() {
            let _ = apply_widen_to_snapshots(genome, &block, current_size, new_size, rng);
        }
    }

    Ok(())
}

/// MutateLayerParam 的 NodeLevel 实现
///
/// 变异 Dropout 节点的丢弃率 `p`。
fn node_level_mutate_param_apply(
    genome: &mut NetworkGenome,
    rng: &mut StdRng,
) -> Result<(), MutationError> {
    let dropout_ids: Vec<u64> = genome
        .nodes()
        .iter()
        .filter(|n| n.enabled && matches!(n.node_type, NodeTypeDescriptor::Dropout { .. }))
        .map(|n| n.innovation_number)
        .collect();

    if dropout_ids.is_empty() {
        return Err(MutationError::NotApplicable(
            "没有 Dropout 节点可变异".into(),
        ));
    }

    let &target_id = dropout_ids.choose(rng).unwrap();
    for node in genome.nodes_mut().iter_mut() {
        if node.innovation_number == target_id {
            if let NodeTypeDescriptor::Dropout { ref mut p } = node.node_type {
                *p = rng.gen_range(0.05..=0.8);
            }
            break;
        }
    }
    Ok(())
}

/// ReplaceLayerType 的 NodeLevel 实现
///
/// 随机选择一个激活节点，将其替换为另一种激活函数。
fn node_level_replace_activation_apply(
    m: &ReplaceLayerTypeMutation,
    genome: &mut NetworkGenome,
    rng: &mut StdRng,
) -> Result<(), MutationError> {
    let blocks = node_main_path(genome);
    let act_node_ids: Vec<u64> = blocks
        .iter()
        .filter(|b| b.kind.is_activation() && !b.node_ids.is_empty())
        .map(|b| b.node_ids[0])
        .collect();

    if act_node_ids.is_empty() || m.available_activations.is_empty() {
        return Err(MutationError::NotApplicable(
            "没有激活节点可替换或无替代激活函数".into(),
        ));
    }

    let &node_id = act_node_ids.choose(rng).unwrap();
    let new_act = m.available_activations.choose(rng).unwrap();
    let new_nt = activation_to_node_type(new_act);

    for node in genome.nodes_mut().iter_mut() {
        if node.innovation_number == node_id {
            node.node_type = new_nt;
            break;
        }
    }
    Ok(())
}

/// MutateKernelSize 的 NodeLevel 实现
///
/// 在主路径 Conv2d 块中随机切换 kernel_size，同步更新 padding 和空间形状。
fn node_level_mutate_kernel_size_apply(
    genome: &mut NetworkGenome,
    rng: &mut StdRng,
) -> Result<(), MutationError> {
    let blocks = node_main_path(genome);
    let conv_blocks: Vec<NodeBlock> = blocks.into_iter().filter(|b| b.kind.is_conv2d()).collect();

    if conv_blocks.is_empty() {
        return Err(MutationError::NotApplicable(
            "没有 Conv2d 块可变异 kernel_size".into(),
        ));
    }

    let block = conv_blocks.choose(rng).unwrap().clone();
    let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();

    // 通过 Conv2d 节点的父节点关系精确定位 kernel 参数
    let kernel_id: Option<u64> = genome
        .nodes()
        .iter()
        .find(|n| {
            bid_set.contains(&n.innovation_number)
                && matches!(n.node_type, NodeTypeDescriptor::Conv2d { .. })
        })
        .and_then(|conv| {
            conv.parents.iter().find(|&&pid| {
                bid_set.contains(&pid)
                    && genome
                        .nodes()
                        .iter()
                        .any(|n| n.innovation_number == pid && n.is_parameter())
            })
        })
        .copied();

    let current_k = kernel_id
        .and_then(|kid| {
            genome
                .nodes()
                .iter()
                .find(|n| n.innovation_number == kid)
                .map(|n| n.output_shape[2])
        })
        .unwrap_or(3);

    let alternatives: Vec<usize> = KERNEL_SIZES
        .iter()
        .copied()
        .filter(|&k| k != current_k)
        .collect();
    let new_k = *alternatives
        .choose(rng)
        .ok_or_else(|| MutationError::NotApplicable("没有替代的 kernel_size".into()))?;
    let new_padding = new_k / 2;

    // 只更新 kernel 参数节点形状 [out_ch, in_ch, k, k]
    if let Some(kid) = kernel_id {
        for node in genome.nodes_mut().iter_mut() {
            if node.innovation_number == kid && node.output_shape.len() == 4 {
                node.output_shape[2] = new_k;
                node.output_shape[3] = new_k;
                break;
            }
        }
    }

    // 更新 Conv2d op 节点的 padding
    for node in genome.nodes_mut().iter_mut() {
        if bid_set.contains(&node.innovation_number) {
            if let NodeTypeDescriptor::Conv2d {
                ref mut padding, ..
            } = node.node_type
            {
                *padding = (new_padding, new_padding);
            }
        }
    }

    sync_computation_shapes(genome);
    Ok(())
}

// ==================== MutateStrideMutation ====================

/// 变异 Conv2d 的 stride（在 (1,1) 和 (2,2) 之间切换）
///
/// stride=2 允许卷积层自身进行空间降维，不完全依赖 Pool2d。
pub struct MutateStrideMutation;

impl Mutation for MutateStrideMutation {
    fn name(&self) -> &str {
        "MutateStride"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !genome.is_node_level() {
            return false;
        }
        node_main_path(genome).iter().any(|b| b.kind.is_conv2d())
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let blocks = node_main_path(genome);
        let conv_blocks: Vec<NodeBlock> =
            blocks.into_iter().filter(|b| b.kind.is_conv2d()).collect();

        if conv_blocks.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有 Conv2d 块可变异 stride".into(),
            ));
        }

        let block = conv_blocks.choose(rng).unwrap().clone();
        let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();

        // 读取当前 stride 并切换
        let current_stride = genome
            .nodes()
            .iter()
            .find_map(|n| {
                if bid_set.contains(&n.innovation_number) {
                    if let NodeTypeDescriptor::Conv2d { stride, .. } = &n.node_type {
                        return Some(*stride);
                    }
                }
                None
            })
            .unwrap_or((1, 1));

        let new_stride = if current_stride == (1, 1) {
            // stride=1 → stride=2：需要空间尺寸 >= 2
            let spatial = node_spatial_at(genome, block.input_id);
            if spatial.map(|(h, w)| h >= 2 && w >= 2).unwrap_or(false) {
                (2, 2)
            } else {
                return Err(MutationError::NotApplicable(
                    "空间尺寸过小，无法使用 stride=2".into(),
                ));
            }
        } else {
            (1, 1) // 恢复为 stride=1
        };

        // 更新 Conv2d op 节点的 stride
        for node in genome.nodes_mut().iter_mut() {
            if bid_set.contains(&node.innovation_number) {
                if let NodeTypeDescriptor::Conv2d {
                    ref mut stride, ..
                } = node.node_type
                {
                    *stride = new_stride;
                }
            }
        }

        sync_computation_shapes(genome);
        repair_param_input_dims(genome);
        Ok(())
    }
}

// ==================== AddConnectionMutation / RemoveConnectionMutation ====================

/// 为 NodeLevel 基因组添加一条跨层跳跃连接（Add 聚合，可选投影）
///
/// 在主路径的任意两个非直接相邻节点之间加入前向 Add 连接。
/// 形状不兼容时自动插入投影块（Flat: Linear；Spatial: 1×1 Conv2d）。
/// 仅适用于 NodeLevel 且非序列模式的基因组。
pub struct AddConnectionMutation;

impl Mutation for AddConnectionMutation {
    fn name(&self) -> &str {
        "AddConnection"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        genome.is_node_level()
            && genome.seq_len.is_none()
            && !find_connectable_pairs(genome).is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let pairs = find_connectable_pairs(genome);
        if pairs.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可添加的跳跃连接候选对".into(),
            ));
        }

        let pair = pairs.choose(rng).unwrap().clone();
        let mut counter = make_counter(genome);

        add_skip_connection(genome, &pair, &mut counter)
            .map_err(|e| MutationError::InternalError(e))?;

        commit_counter(genome, &counter);
        Ok(())
    }
}

/// 移除 NodeLevel 基因组中一条已有的跳跃连接
///
/// 删除由 `AddConnectionMutation` 添加的 Add 聚合节点，
/// 并通过 `cleanup_orphan_nodes` 自动清理孤立的投影块（若有）。
pub struct RemoveConnectionMutation;

impl Mutation for RemoveConnectionMutation {
    fn name(&self) -> &str {
        "RemoveConnection"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        genome.is_node_level() && !find_removable_skip_connections(genome).is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let candidates = find_removable_skip_connections(genome);
        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可移除的跳跃聚合节点".into(),
            ));
        }

        let &agg_id = candidates.choose(rng).unwrap();
        remove_skip_connection(genome, agg_id).map_err(|e| MutationError::InternalError(e))
    }
}

// ==================== InsertAtomicNodeMutation ====================

/// 在主路径的两个块之间插入**单个**激活函数节点（NEAT "Add Node" 的等价操作）
///
/// 与 InsertLayerMutation 的区别：
/// - InsertLayer 插入完整模板块（多节点，共享 block_id）
/// - InsertAtomicNode 只插入一个激活函数节点（block_id = None，零参数）
///
/// 合法性保障：
/// - 不在两个相邻的激活函数之间插入（避免连续激活）
/// - 不在输出头之后插入
/// - 通过 analyze() 验证形状合法性，失败则回滚
pub struct InsertAtomicNodeMutation {
    available_activations: Vec<ActivationType>,
}

impl Default for InsertAtomicNodeMutation {
    fn default() -> Self {
        Self {
            available_activations: default_activations(),
        }
    }
}

impl Mutation for InsertAtomicNodeMutation {
    fn name(&self) -> &str {
        "InsertAtomicNode"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !genome.is_node_level() || self.available_activations.is_empty() {
            return false;
        }
        let blocks = node_main_path(genome);
        if blocks.is_empty() {
            return false;
        }
        // 至少存在一个插入点：非末尾块的 output_id（或 INPUT），且该点不在两个激活之间
        let candidates = atomic_insert_candidates(&blocks, genome);
        !candidates.is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let blocks = node_main_path(genome);
        let candidates = atomic_insert_candidates(&blocks, genome);
        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有合法的原子节点插入点".into(),
            ));
        }

        let &after_id = candidates
            .choose(rng)
            .ok_or_else(|| MutationError::NotApplicable("没有可用的插入点".into()))?;

        let output_shape = node_output_shape_at(genome, after_id);
        let start_inn = genome.peek_next_innovation();

        // 15% 概率插入 Dropout（需确认前后不是 Dropout）
        let insert_dropout = rng.gen_bool(0.15)
            && !is_dropout_node(genome, after_id)
            && !is_next_dropout(genome, after_id);

        let new_nodes = if insert_dropout {
            let p = *[0.1f32, 0.2, 0.3, 0.5].choose(rng).unwrap();
            expand_dropout(after_id, output_shape, p, &mut make_counter(genome))
        } else {
            let act = self
                .available_activations
                .choose(rng)
                .ok_or_else(|| MutationError::NotApplicable("没有可用的激活函数".into()))?;
            expand_activation(after_id, output_shape, act, &mut make_counter(genome))
        };

        let n = new_nodes.len() as u64; // 始终为 1
        insert_after(genome, after_id, new_nodes)
            .map_err(|e| MutationError::InternalError(e))?;
        advance_node_counter(genome, n);

        let analysis = genome.analyze();
        if !analysis.is_valid {
            // 回滚
            let new_output_id = start_inn;
            for node in genome.nodes_mut().iter_mut() {
                for pid in node.parents.iter_mut() {
                    if *pid == new_output_id {
                        *pid = after_id;
                    }
                }
            }
            genome
                .nodes_mut()
                .retain(|nd| nd.innovation_number != start_inn);
            reset_node_counter(genome, start_inn);
            return Err(MutationError::ConstraintViolation(
                "原子节点插入后图不合法".into(),
            ));
        }

        Ok(())
    }
}

/// 收集原子节点的合法插入点
///
/// 返回可在其后插入激活函数的节点 ID 列表。
/// 排除规则：
/// - 不在输出头之后插入（取 blocks[..len-1]）
/// - 不在两个激活函数之间插入（避免连续激活）
fn atomic_insert_candidates(blocks: &[NodeBlock], genome: &NetworkGenome) -> Vec<u64> {
    if blocks.is_empty() {
        return Vec::new();
    }

    let mut candidates = Vec::new();

    // 只在非末尾块的 output_id 之后插入（与 InsertLayer 相同的输出头保护）
    // 若仅剩输出头，允许在 INPUT 之后插入
    let insert_points: Vec<(u64, usize)> = if blocks.len() > 1 {
        blocks[..blocks.len() - 1]
            .iter()
            .enumerate()
            .map(|(i, b)| (b.output_id, i))
            .collect()
    } else {
        vec![(INPUT_INNOVATION, usize::MAX)]
    };

    for (after_id, block_idx) in insert_points {
        let before_is_act = is_activation_node(genome, after_id);
        // 查看下一个块是否也是激活函数
        let next_is_act = if block_idx < blocks.len().saturating_sub(1) {
            blocks
                .get(block_idx + 1)
                .map(|b| b.kind.is_activation())
                .unwrap_or(false)
        } else if block_idx == usize::MAX {
            // INPUT 后面第一个块
            blocks.first().map(|b| b.kind.is_activation()).unwrap_or(false)
        } else {
            false
        };

        // 如果前后都是激活函数，跳过（避免三连激活）
        if before_is_act && next_is_act {
            continue;
        }

        candidates.push(after_id);
    }

    candidates
}

/// 检查 after_id 后面的下一个块是否为 Dropout
fn is_next_dropout(genome: &NetworkGenome, after_id: u64) -> bool {
    let blocks = node_main_path(genome);
    if let Some(idx) = blocks.iter().position(|b| b.output_id == after_id) {
        if let Some(next) = blocks.get(idx + 1) {
            return matches!(next.kind, NodeBlockKind::Dropout { .. });
        }
    }
    // INPUT 后的第一个块
    if after_id == INPUT_INNOVATION {
        return blocks
            .first()
            .map(|b| matches!(b.kind, NodeBlockKind::Dropout { .. }))
            .unwrap_or(false);
    }
    false
}

// ==================== AddRecurrentEdgeMutation ====================

/// 在序列基因组中添加一条循环边（EXAMM 风格 recurrent connection）
///
/// 操作步骤：
/// 1. 随机选取源节点 `source` 和目标节点 `target`（均为主路径上的非叶节点）
/// 2. 创建一个 `Parameter` 节点，持有权重矩阵 `[target_dim, source_dim]`
/// 3. 在 `target.recurrent_parents` 中添加 `(source_id, weight_param_id)`
///
/// 合法性保障：
/// - 仅序列模式（`seq_len.is_some()`）可用
/// - 不在已有 cell-based 循环节点的基因组中使用（范式互斥）
/// - 不允许重复的 `(source, target)` 循环边
/// - 目标不能是叶节点（Parameter/Input/State）
pub struct AddRecurrentEdgeMutation;

impl Mutation for AddRecurrentEdgeMutation {
    fn name(&self) -> &str {
        "AddRecurrentEdge"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !genome.is_node_level() || genome.seq_len.is_none() {
            return false;
        }
        // 范式互斥：不能已有 cell-based 循环节点
        let has_cell = genome.nodes().iter().any(|n| {
            n.enabled
                && matches!(
                    n.node_type,
                    NodeTypeDescriptor::CellRnn { .. }
                        | NodeTypeDescriptor::CellLstm { .. }
                        | NodeTypeDescriptor::CellGru { .. }
                )
        });
        if has_cell {
            return false;
        }
        // 至少需要 2 个非叶节点（作为源和目标）
        let computation_nodes: Vec<_> = genome
            .nodes()
            .iter()
            .filter(|n| n.enabled && !n.is_leaf())
            .collect();
        computation_nodes.len() >= 2
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let computation_ids: Vec<u64> = genome
            .nodes()
            .iter()
            .filter(|n| n.enabled && !n.is_leaf())
            .map(|n| n.innovation_number)
            .collect();

        if computation_ids.len() < 2 {
            return Err(MutationError::NotApplicable(
                "计算节点不足，无法添加循环边".into(),
            ));
        }

        // 随机选择源和目标（允许不同或相同节点用于自环）
        let &source_id = computation_ids
            .choose(rng)
            .ok_or_else(|| MutationError::NotApplicable("无可用源节点".into()))?;

        // 目标排除已有同一 source 的循环边
        let target_candidates: Vec<u64> = computation_ids
            .iter()
            .filter(|&&tid| {
                let node = genome
                    .nodes()
                    .iter()
                    .find(|n| n.innovation_number == tid)
                    .unwrap();
                !node
                    .recurrent_parents
                    .iter()
                    .any(|e| e.source_id == source_id)
            })
            .copied()
            .collect();

        if target_candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "所有候选目标已与该源存在循环边".into(),
            ));
        }

        let &target_id = target_candidates
            .choose(rng)
            .ok_or_else(|| MutationError::NotApplicable("无可用目标节点".into()))?;

        // 获取源和目标的特征维度（取输出形状最后一维）
        let source_shape = node_output_shape_at(genome, source_id);
        let target_shape = node_output_shape_at(genome, target_id);
        let source_dim = *source_shape.last().unwrap_or(&1);
        let target_dim = *target_shape.last().unwrap_or(&1);

        // 创建权重参数节点 [target_dim, source_dim]
        let start_inn = genome.peek_next_innovation();
        let weight_param = NodeGene::new(
            start_inn,
            NodeTypeDescriptor::Parameter,
            vec![target_dim, source_dim],
            vec![],
            None,
        );
        let weight_param_id = weight_param.innovation_number;

        genome.nodes_mut().push(weight_param);
        advance_node_counter(genome, 1);

        // 在目标节点添加循环边引用
        let target_node = genome
            .nodes_mut()
            .iter_mut()
            .find(|n| n.innovation_number == target_id)
            .ok_or_else(|| MutationError::InternalError("目标节点不存在".into()))?;

        target_node.recurrent_parents.push(RecurrentEdge {
            source_id,
            weight_param_id,
        });

        // 验证合法性
        let analysis = genome.analyze();
        if !analysis.is_valid {
            // 回滚：移除权重参数和循环边
            let target_node = genome
                .nodes_mut()
                .iter_mut()
                .find(|n| n.innovation_number == target_id)
                .unwrap();
            target_node
                .recurrent_parents
                .retain(|e| e.weight_param_id != weight_param_id);
            genome
                .nodes_mut()
                .retain(|n| n.innovation_number != weight_param_id);
            reset_node_counter(genome, start_inn);

            return Err(MutationError::ConstraintViolation(
                "添加循环边后图不合法".into(),
            ));
        }

        Ok(())
    }
}

// ==================== RemoveRecurrentEdgeMutation ====================

/// 从序列基因组中移除一条循环边及其关联的权重参数节点
pub struct RemoveRecurrentEdgeMutation;

impl Mutation for RemoveRecurrentEdgeMutation {
    fn name(&self) -> &str {
        "RemoveRecurrentEdge"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !genome.is_node_level() {
            return false;
        }
        genome
            .nodes()
            .iter()
            .any(|n| n.enabled && !n.recurrent_parents.is_empty())
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        // 收集所有 (target_id, edge_index) 对
        let mut candidates: Vec<(u64, usize)> = Vec::new();
        for node in genome.nodes().iter() {
            if node.enabled && !node.recurrent_parents.is_empty() {
                for (i, _) in node.recurrent_parents.iter().enumerate() {
                    candidates.push((node.innovation_number, i));
                }
            }
        }

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable("没有可移除的循环边".into()));
        }

        let &(target_id, edge_idx) = candidates
            .choose(rng)
            .ok_or_else(|| MutationError::NotApplicable("无法选择循环边".into()))?;

        // 获取要删除的权重参数 ID
        let weight_param_id = genome
            .nodes()
            .iter()
            .find(|n| n.innovation_number == target_id)
            .and_then(|n| n.recurrent_parents.get(edge_idx))
            .map(|e| e.weight_param_id)
            .ok_or_else(|| MutationError::InternalError("循环边索引越界".into()))?;

        // 删除循环边
        let target_node = genome
            .nodes_mut()
            .iter_mut()
            .find(|n| n.innovation_number == target_id)
            .ok_or_else(|| MutationError::InternalError("目标节点不存在".into()))?;
        target_node.recurrent_parents.remove(edge_idx);

        // 检查权重参数是否被其他循环边引用
        let still_referenced = genome.nodes().iter().any(|n| {
            n.recurrent_parents
                .iter()
                .any(|e| e.weight_param_id == weight_param_id)
        });

        if !still_referenced {
            // 删除孤立权重参数节点
            genome
                .nodes_mut()
                .retain(|n| n.innovation_number != weight_param_id);

            // 清理权重快照
            if let super::gene::GenomeRepr::NodeLevel {
                weight_snapshots, ..
            } = &mut genome.repr
            {
                weight_snapshots.remove(&weight_param_id);
            }
        }

        Ok(())
    }
}

// ==================== LayerLevel Net2Net 扩宽辅助 ====================

/// LayerLevel 路径的 Net2Net 扩宽：对层级快照就地更新，使权重与新 hidden_size 兼容。
///
/// 对 `target_idx` 所指向层的快照做 owner 扩宽（输出维度），
/// 并对下一个有参数的层做 consumer 缩放（输入维度）。
/// 若快照不存在或形状不匹配，直接跳过（builder 会随机初始化）。
fn widen_layer_snapshots(
    genome: &mut NetworkGenome,
    target_idx: usize,
    old_size: usize,
    new_size: usize,
    rng: &mut StdRng,
) {
    use super::net2net::{counts_of, widening_mapping};
    let mapping = widening_mapping(old_size, new_size, rng);
    let counts = counts_of(&mapping, old_size);

    let owner_inn = genome.layers()[target_idx].innovation_number;
    let owner_config = genome.layers()[target_idx].layer_config.clone();

    // 找下游第一个有输入参数的层
    let layers_len = genome.layers().len();
    let consumer: Option<(u64, LayerConfig)> = {
        let mut found = None;
        for i in (target_idx + 1)..layers_len {
            let l = &genome.layers()[i];
            if l.enabled && layer_has_input_params(&l.layer_config) {
                found = Some((l.innovation_number, l.layer_config.clone()));
                break;
            }
        }
        found
    };

    // 提前克隆快照（避免借用冲突）
    let owner_snap_opt: Option<Vec<Tensor>> =
        genome.weight_snapshots().get(&owner_inn).cloned();
    let consumer_snap_opt: Option<(u64, Vec<Tensor>)> = consumer.as_ref().and_then(|(inn, _)| {
        genome.weight_snapshots().get(inn).map(|s| (*inn, s.clone()))
    });

    // 扩宽 owner 快照
    if let Some(mut snap) = owner_snap_opt {
        if layer_widen_owner_snap(&mut snap, &owner_config, &mapping, &counts) {
            if let GenomeRepr::LayerLevel {
                weight_snapshots, ..
            } = &mut genome.repr
            {
                weight_snapshots.insert(owner_inn, snap);
            }
        }
    }

    // 扩宽 consumer 快照（输入维度）
    if let Some((consumer_inn, consumer_config)) = consumer {
        if let Some((_, mut csnap)) = consumer_snap_opt {
            if layer_widen_consumer_snap(&mut csnap, &consumer_config, &mapping, &counts) {
                if let GenomeRepr::LayerLevel {
                    weight_snapshots, ..
                } = &mut genome.repr
                {
                    weight_snapshots.insert(consumer_inn, csnap);
                }
            }
        }
    }
}

/// owner 快照就地扩宽：输出维度从 old_size 扩展到 new_size（faithful copy）。
/// 失败时返回 false，快照保持不变。
fn layer_widen_owner_snap(
    snap: &mut Vec<Tensor>,
    config: &LayerConfig,
    mapping: &[usize],
    counts: &[usize],
) -> bool {
    use super::net2net::{gather_along_axis, gather_along_axis_scaled};
    match config {
        LayerConfig::Linear { .. } => {
            // [W[in, old], b[1, old]]
            if snap.len() < 2 {
                return false;
            }
            snap[0] = gather_along_axis(&snap[0], 1, mapping); // W axis=1（输出列）
            snap[1] = gather_along_axis(&snap[1], 1, mapping); // b axis=1
            true
        }
        LayerConfig::Rnn { .. } | LayerConfig::Lstm { .. } | LayerConfig::Gru { .. } => {
            // 每门 3 个参数: (W_ih[in,old], W_hh[old,old], b[1,old])
            let n = snap.len();
            if n % 3 != 0 {
                return false;
            }
            let mut new_snap = Vec::with_capacity(n);
            for i in (0..n).step_by(3) {
                // W_ih: 输出列扩展
                let w_ih = gather_along_axis(&snap[i], 1, mapping);
                // W_hh: 先按行缩放（输入侧来自旧输出），再按列扩展（输出侧）
                let whh_mid = gather_along_axis_scaled(&snap[i + 1], 0, mapping, counts);
                let w_hh = gather_along_axis(&whh_mid, 1, mapping);
                // b: 输出列扩展
                let b = gather_along_axis(&snap[i + 2], 1, mapping);
                new_snap.push(w_ih);
                new_snap.push(w_hh);
                new_snap.push(b);
            }
            *snap = new_snap;
            true
        }
        _ => false,
    }
}

/// consumer 快照就地更新：输入维度从 old_size 扩展（按 counts 缩放）。
/// 失败时返回 false，快照保持不变。
fn layer_widen_consumer_snap(
    snap: &mut Vec<Tensor>,
    config: &LayerConfig,
    mapping: &[usize],
    counts: &[usize],
) -> bool {
    use super::net2net::gather_along_axis_scaled;
    match config {
        LayerConfig::Linear { .. } => {
            // [W[old, out], b[...]] — 只扩 W axis=0（输入行）
            if snap.is_empty() {
                return false;
            }
            snap[0] = gather_along_axis_scaled(&snap[0], 0, mapping, counts);
            true
        }
        LayerConfig::Rnn { .. } | LayerConfig::Lstm { .. } | LayerConfig::Gru { .. } => {
            // 每门 3 参数: W_ih[old,h] axis=0 缩放，W_hh 和 b 不变
            let n = snap.len();
            if n % 3 != 0 {
                return false;
            }
            for i in (0..n).step_by(3) {
                snap[i] = gather_along_axis_scaled(&snap[i], 0, mapping, counts);
            }
            true
        }
        _ => false,
    }
}

/// 判断层是否有输入侧参数（需要处理 consumer 扩宽）
fn layer_has_input_params(cfg: &LayerConfig) -> bool {
    matches!(
        cfg,
        LayerConfig::Linear { .. }
            | LayerConfig::Rnn { .. }
            | LayerConfig::Lstm { .. }
            | LayerConfig::Gru { .. }
    )
}

// ==================== LayerLevel cell 迁移辅助 ====================

/// 将 LayerLevel 层级快照（Vec<Tensor>）从旧 cell 类型迁移到新类型。
/// 失败返回 None（调用方应清空快照，让 builder 走随机初始化）。
fn migrate_layer_cell_weights_vec(
    old_snap: &[Tensor],
    old_kind: CellKind,
    new_kind: CellKind,
    hidden: usize,
) -> Option<Vec<Tensor>> {
    if old_snap.len() != old_kind.param_count() {
        return None;
    }
    // 用连续整数作为虚拟 id
    let dummy_ids: Vec<u64> = (0..new_kind.param_count() as u64).collect();
    let old_snaps_opt: Vec<Option<Tensor>> =
        old_snap.iter().map(|t| Some(t.clone())).collect();
    let migrated = migrate_cell_weights(old_kind, &old_snaps_opt, new_kind, &dummy_ids, hidden)?;
    // 按 id 顺序重组为 Vec<Tensor>
    dummy_ids.iter().map(|k| migrated.get(k).cloned()).collect()
}

/// 从 LayerConfig 提取 CellKind（非循环层返回 None）
fn layer_cell_kind(config: &LayerConfig) -> Option<CellKind> {
    match config {
        LayerConfig::Rnn { .. } => Some(CellKind::Rnn),
        LayerConfig::Lstm { .. } => Some(CellKind::Lstm),
        LayerConfig::Gru { .. } => Some(CellKind::Gru),
        _ => None,
    }
}

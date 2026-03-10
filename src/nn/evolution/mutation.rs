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
    compatible_losses, ActivationType, AggregateStrategy, LayerConfig, LayerGene, LossType,
    NetworkGenome, OptimizerType, PoolType, ShapeDomain, SkipEdge, TaskMetric, INPUT_INNOVATION,
};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::Rng;
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

        Err(MutationError::NotApplicable(
            "没有可用的变异操作".into(),
        ))
    }

    /// 默认注册表
    ///
    /// `is_sequential`: 序列模式时额外注册 MutateCellType。
    /// `is_spatial`: 空间模式时额外注册 MutateKernelSize。
    pub fn default_registry(metric: &TaskMetric, is_sequential: bool, is_spatial: bool) -> Self {
        let mut reg = Self::new();
        reg.register(0.15, InsertLayerMutation::default());
        reg.register(0.15, RemoveLayerMutation);
        reg.register(0.10, ReplaceLayerTypeMutation::default());
        reg.register(0.24, GrowHiddenSizeMutation);
        reg.register(0.29, ShrinkHiddenSizeMutation);
        reg.register(0.05, MutateLayerParamMutation);
        reg.register(
            0.02,
            MutateLossFunctionMutation {
                task_metric: metric.clone(),
            },
        );
        // SkipEdge 变异
        reg.register(0.08, AddSkipEdgeMutation);
        reg.register(0.05, RemoveSkipEdgeMutation);
        reg.register(0.03, MutateAggregateStrategyMutation);
        // 训练超参数变异
        reg.register(0.05, MutateLearningRateMutation);
        reg.register(0.02, MutateOptimizerMutation);
        // 序列模式专属
        if is_sequential {
            reg.register(0.10, MutateCellTypeMutation);
        }
        // 空间模式专属
        if is_spatial {
            reg.register(0.10, MutateKernelSizeMutation);
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
    let last_enabled_idx = genome
        .layers
        .iter()
        .rposition(|l| l.enabled);
    genome
        .layers
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
    let enabled: Vec<&LayerGene> = genome.layers.iter().filter(|l| l.enabled).collect();
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

/// 根据 SizeStrategy 计算增长后的值
fn grow_size(current: usize, max: usize, strategy: &SizeStrategy, rng: &mut StdRng) -> usize {
    let new_size = match strategy {
        SizeStrategy::Free => {
            if rng.gen_bool(0.5) {
                current + 1
            } else {
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
fn shrink_size(current: usize, min: usize, strategy: &SizeStrategy, rng: &mut StdRng) -> usize {
    let new_size = match strategy {
        SizeStrategy::Free => {
            if rng.gen_bool(0.5) {
                current.saturating_sub(1)
            } else {
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
        genome.layer_count() < constraints.max_layers
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let enabled_count = genome.layer_count();
        if enabled_count >= constraints.max_layers {
            return Err(MutationError::ConstraintViolation("已达 max_layers".into()));
        }

        // 插入位置：在 enabled 层序列的 [0, output_head] 之间（含输出头正前方）
        // 但操作的是 layers vec 的实际索引
        let enabled_indices: Vec<usize> = genome
            .layers
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
                genome.layers.iter().filter(|l| l.enabled).collect();
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

        let new_config = if can_insert_activation && rng.gen_bool(0.3) {
            let act = self.available_activations.choose(rng).unwrap();
            LayerConfig::Activation {
                activation_type: *act,
            }
        } else if insert_domain == ShapeDomain::Spatial {
            // 空间域：Conv2d 或 Pool2d
            if rng.gen_bool(0.2) {
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
                let small_cap = genome
                    .input_dim
                    .min(constraints.max_hidden_size)
                    .max(constraints.min_hidden_size);
                let out_ch = rng.gen_range(constraints.min_hidden_size..=small_cap);
                let k = *[1usize, 3, 5, 7].choose(rng).unwrap();
                LayerConfig::Conv2d {
                    out_channels: out_ch,
                    kernel_size: k,
                }
            }
        } else if is_sequential && rng.gen_bool(0.5) {
            let small_cap = genome
                .input_dim
                .min(constraints.max_hidden_size)
                .max(constraints.min_hidden_size);
            let size = rng.gen_range(constraints.min_hidden_size..=small_cap);
            match rng.gen_range(0..3) {
                0 => LayerConfig::Rnn { hidden_size: size },
                1 => LayerConfig::Lstm { hidden_size: size },
                _ => LayerConfig::Gru { hidden_size: size },
            }
        } else {
            let small_cap = genome
                .input_dim
                .min(constraints.max_hidden_size)
                .max(constraints.min_hidden_size);
            let size = rng.gen_range(constraints.min_hidden_size..=small_cap);
            LayerConfig::Linear {
                out_features: size,
            }
        };

        let inn = genome.next_innovation_number();
        genome.layers.insert(
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
            genome.layers.remove(insert_vec_idx);
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
        !hidden_layer_indices(genome).is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let candidates = hidden_layer_indices(genome);
        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可移除的隐藏层".into(),
            ));
        }
        let &idx = candidates.choose(rng).unwrap();
        let removed_gene = genome.layers.remove(idx);
        let removed_inn = removed_gene.innovation_number;

        // 清理引用已删除层的 skip edges（保留副本用于回滚）
        let removed_edges: Vec<_> = genome
            .skip_edges
            .iter()
            .filter(|e| e.from_innovation == removed_inn || e.to_innovation == removed_inn)
            .cloned()
            .collect();
        genome.skip_edges.retain(|e| {
            e.from_innovation != removed_inn && e.to_innovation != removed_inn
        });

        // 验证删除后维度兼容性 + 域链合法性 + 已有 skip edge 域兼容性
        if genome.resolve_dimensions().is_err()
            || !genome.is_domain_valid()
            || !genome.validate_skip_edge_domains()
        {
            genome.layers.insert(idx, removed_gene);
            genome.skip_edges.extend(removed_edges);
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
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| {
            matches!(
                genome.layers[i].layer_config,
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
        let hidden = hidden_layer_indices(genome);
        let act_indices: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| {
                matches!(
                    genome.layers[i].layer_config,
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
        let current = &genome.layers[idx].layer_config;

        let alternatives: Vec<&ActivationType> = self
            .available_activations
            .iter()
            .filter(|a| {
                LayerConfig::Activation {
                    activation_type: **a,
                } != *current
            })
            .collect();

        let &&new_act = alternatives.choose(rng).ok_or_else(|| {
            MutationError::NotApplicable("没有可选的替代激活函数".into())
        })?;

        genome.layers[idx].layer_config = LayerConfig::Activation {
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
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| {
            get_resizable_size(&genome.layers[i].layer_config)
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
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| {
                get_resizable_size(&genome.layers[i].layer_config)
                    .map(|s| s < constraints.max_hidden_size)
                    .unwrap_or(false)
            })
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可增长的层".into(),
            ));
        }

        let &idx = candidates.choose(rng).unwrap();
        let old_config = genome.layers[idx].layer_config.clone();
        let old_size = get_resizable_size(&old_config).unwrap();

        let new_size = grow_size(
            old_size,
            constraints.max_hidden_size,
            &constraints.size_strategy,
            rng,
        );

        debug_assert!(new_size > old_size);

        set_resizable_size(&mut genome.layers[idx].layer_config, new_size);
        match genome.total_params() {
            Ok(params) if params > constraints.max_total_params => {
                genome.layers[idx].layer_config = old_config;
                Err(MutationError::ConstraintViolation(format!(
                    "增长后 total_params={params} 超过上限 {}",
                    constraints.max_total_params
                )))
            }
            Err(_) => {
                genome.layers[idx].layer_config = old_config;
                Err(MutationError::ConstraintViolation(
                    "增长后维度不兼容（skip edge 约束）".into(),
                ))
            }
            _ => Ok(()),
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
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| {
            get_resizable_size(&genome.layers[i].layer_config)
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
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| {
                get_resizable_size(&genome.layers[i].layer_config)
                    .map(|s| s > constraints.min_hidden_size)
                    .unwrap_or(false)
            })
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可缩小的层".into(),
            ));
        }

        let &idx = candidates.choose(rng).unwrap();
        let old_config = genome.layers[idx].layer_config.clone();
        let old_size = get_resizable_size(&old_config).unwrap();
        let new_size = shrink_size(
            old_size,
            constraints.min_hidden_size,
            &constraints.size_strategy,
            rng,
        );

        set_resizable_size(&mut genome.layers[idx].layer_config, new_size);

        if genome.resolve_dimensions().is_err() {
            genome.layers[idx].layer_config = old_config;
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
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| is_parameterized_layer(&genome.layers[i].layer_config))
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| is_parameterized_layer(&genome.layers[i].layer_config))
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可参数化的层".into(),
            ));
        }

        let &idx = candidates.choose(rng).unwrap();
        match &mut genome.layers[idx].layer_config {
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

        let new_loss = alternatives.choose(rng).ok_or_else(|| {
            MutationError::NotApplicable("没有可替换的 loss 函数".into())
        })?;

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
    1e-5, 2e-5, 5e-5,
    1e-4, 2e-4, 5e-4,
    1e-3, 2e-3, 5e-3,
    1e-2, 2e-2, 5e-2,
    1e-1,
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
        genome.training_config.learning_rate = snap_to_nearest_in_band(
            genome.training_config.learning_rate,
            target_band,
            LR_LADDER,
        );
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
        let hidden = hidden_layer_indices(genome);
        hidden.iter().any(|&i| {
            NetworkGenome::is_recurrent(&genome.layers[i].layer_config)
        })
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| NetworkGenome::is_recurrent(&genome.layers[i].layer_config))
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可切换的循环层".into(),
            ));
        }

        let &idx = candidates.choose(rng).unwrap();
        let hidden_size = get_resizable_size(&genome.layers[idx].layer_config).unwrap();

        // 排除当前类型，从剩余两种中随机选择
        let new_config = match &genome.layers[idx].layer_config {
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

        // 切换后旧权重快照失效（参数数量不同），移除该层快照
        let inn = genome.layers[idx].innovation_number;
        genome.weight_snapshots.remove(&inn);

        let old_config = genome.layers[idx].layer_config.clone();
        genome.layers[idx].layer_config = new_config;

        // 切换 cell 类型不会改变域链（仍然是循环层），但保险起见仍检查 skip edge 域兼容性
        if !genome.validate_skip_edge_domains() {
            genome.layers[idx].layer_config = old_config;
            return Err(MutationError::ConstraintViolation(
                "切换 cell 类型后 skip edge 域不兼容".into(),
            ));
        }

        Ok(())
    }
}

// ==================== MutateKernelSizeMutation ====================

/// Conv2d kernel_size 变异（在 {1, 3, 5, 7} 中切换）
pub struct MutateKernelSizeMutation;

const KERNEL_SIZES: &[usize] = &[1, 3, 5, 7];

impl Mutation for MutateKernelSizeMutation {
    fn name(&self) -> &str {
        "MutateKernelSize"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        let hidden = hidden_layer_indices(genome);
        hidden
            .iter()
            .any(|&i| matches!(genome.layers[i].layer_config, LayerConfig::Conv2d { .. }))
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let hidden = hidden_layer_indices(genome);
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| matches!(genome.layers[i].layer_config, LayerConfig::Conv2d { .. }))
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
        } = genome.layers[idx].layer_config
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
                } = genome.layers[idx].layer_config
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
    fn feasible_candidates(
        genome: &NetworkGenome,
    ) -> Vec<(u64, u64, AggregateStrategy)> {
        let enabled: Vec<u64> = genome
            .layers
            .iter()
            .filter(|l| l.enabled)
            .map(|l| l.innovation_number)
            .collect();

        let existing: std::collections::HashSet<(u64, u64)> = genome
            .skip_edges
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
                .skip_edges
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
            if !existing.contains(&(INPUT_INNOVATION, to_inn))
                && immediate_pred.is_some()
            {
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
                    trial.skip_edges.push(SkipEdge {
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
        genome.layers.iter().any(|l| l.enabled)
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
        genome.skip_edges.push(SkipEdge {
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
        genome.skip_edges.iter().any(|e| e.enabled)
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let enabled_indices: Vec<usize> = genome
            .skip_edges
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
        genome.skip_edges.remove(idx);
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
        genome.skip_edges.iter().any(|e| e.enabled)
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        // 收集所有不同的 target group
        let mut target_inns: Vec<u64> = genome
            .skip_edges
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
                .skip_edges
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
                    for edge in &mut trial.skip_edges {
                        if edge.enabled && edge.to_innovation == *target {
                            edge.strategy = s.clone();
                        }
                    }
                    trial.resolve_dimensions().is_ok()
                })
                .collect();

            if let Some(new_strategy) = feasible.choose(rng) {
                let new_strategy = new_strategy.clone();
                for edge in &mut genome.skip_edges {
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

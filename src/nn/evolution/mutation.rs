/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : 神经架构演化的变异操作
 *
 * Mutation trait + MutationRegistry + 7 种 Phase 7A 变异操作。
 * 每种变异通过 is_applicable() 自检合法性，apply() 执行变异。
 * MutationRegistry 按权重随机选择可用变异并执行。
 */

use super::gene::{
    compatible_losses, ActivationType, LayerConfig, LayerGene, LossType, NetworkGenome, TaskMetric,
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

    /// Phase 7A 默认注册表（7 种变异）
    pub fn default_registry(metric: &TaskMetric) -> Self {
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

/// 所有 Phase 7A 默认激活函数类型
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
    ]
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

        // 决定插入层类型：如果相邻有 Activation 则只能插 Linear
        let adjacent_act = has_adjacent_activation(genome, logical_pos);
        let can_insert_activation = !adjacent_act && !self.available_activations.is_empty();

        let new_config = if can_insert_activation && rng.gen_bool(0.5) {
            let act = self.available_activations.choose(rng).unwrap();
            LayerConfig::Activation {
                activation_type: *act,
            }
        } else {
            let size = rng.gen_range(constraints.min_hidden_size..=constraints.max_hidden_size);
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
        genome.layers.remove(idx);
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
            if let LayerConfig::Linear { out_features } = genome.layers[i].layer_config {
                out_features < constraints.max_hidden_size
            } else {
                false
            }
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
                if let LayerConfig::Linear { out_features } = genome.layers[i].layer_config {
                    out_features < constraints.max_hidden_size
                } else {
                    false
                }
            })
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可增长的 Linear 层".into(),
            ));
        }

        let &idx = candidates.choose(rng).unwrap();
        let old_size = match genome.layers[idx].layer_config {
            LayerConfig::Linear { out_features } => out_features,
            _ => unreachable!(),
        };

        let new_size = grow_size(
            old_size,
            constraints.max_hidden_size,
            &constraints.size_strategy,
            rng,
        );

        // new_size > old_size 由前置条件保证：
        // candidates 筛选确保 old_size < max_hidden_size，
        // grow_size 的两条路径（+1 / x2）对正整数输入必然增长。
        debug_assert!(new_size > old_size);

        // 设置新值，检查约束，不满足则回滚
        genome.layers[idx].layer_config = LayerConfig::Linear {
            out_features: new_size,
        };
        match genome.total_params() {
            Ok(params) if params > constraints.max_total_params => {
                genome.layers[idx].layer_config = LayerConfig::Linear {
                    out_features: old_size,
                };
                Err(MutationError::ConstraintViolation(format!(
                    "增长后 total_params={params} 超过上限 {}",
                    constraints.max_total_params
                )))
            }
            Err(_) => {
                genome.layers[idx].layer_config = LayerConfig::Linear {
                    out_features: old_size,
                };
                Err(MutationError::InternalError(
                    "增长后维度推导失败".into(),
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
            if let LayerConfig::Linear { out_features } = genome.layers[i].layer_config {
                out_features > constraints.min_hidden_size
            } else {
                false
            }
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
                if let LayerConfig::Linear { out_features } = genome.layers[i].layer_config {
                    out_features > constraints.min_hidden_size
                } else {
                    false
                }
            })
            .collect();

        if candidates.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可缩小的 Linear 层".into(),
            ));
        }

        let &idx = candidates.choose(rng).unwrap();
        if let LayerConfig::Linear { ref mut out_features } = genome.layers[idx].layer_config {
            *out_features = shrink_size(
                *out_features,
                constraints.min_hidden_size,
                &constraints.size_strategy,
                rng,
            );
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
        hidden.iter().any(|&i| {
            matches!(
                genome.layers[i].layer_config,
                LayerConfig::Activation {
                    activation_type: ActivationType::LeakyReLU { .. }
                } | LayerConfig::Dropout { .. }
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
        let candidates: Vec<usize> = hidden
            .into_iter()
            .filter(|&i| {
                matches!(
                    genome.layers[i].layer_config,
                    LayerConfig::Activation {
                        activation_type: ActivationType::LeakyReLU { .. }
                    } | LayerConfig::Dropout { .. }
                )
            })
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
                *alpha = rng.gen_range(0.001..=0.5);
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

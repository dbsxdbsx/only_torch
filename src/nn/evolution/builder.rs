/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : Genome → Graph 转换 + Lamarckian 权重继承
 *
 * build() 按 resolve_dimensions() 顺序逐层创建计算图，
 * capture_weights() / restore_weights() 实现跨代权重复用。
 */

use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::Rng;

use crate::nn::layer::{AvgPool2d, Conv2d, Gru, Lstm, MaxPool2d, Rnn};
use crate::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps, VarShapeOps};
use crate::tensor::Tensor;

use super::gene::{
    ActivationType, AggregateStrategy, LayerConfig, NetworkGenome, PoolType, SkipEdge,
    INPUT_INNOVATION,
};

// ==================== BuildResult ====================

/// build() 的完整返回值
///
/// 将构建过程中产生的所有信息一次性交付给调用者，
/// 避免事后从 Graph 中重新收集。
pub struct BuildResult {
    /// 输入节点（用于设置训练/测试数据）
    pub input: Var,
    /// 输出节点（用于获取预测结果）
    pub output: Var,
    /// innovation_number → 该层的参数变量列表（如 Linear 有 [W, b]）
    ///
    /// capture_weights() / restore_weights() 按 innovation_number 匹配参数。
    /// Optimizer 所需的扁平参数列表通过 all_parameters() 派生。
    pub layer_params: HashMap<u64, Vec<Var>>,
    /// 内部引用的 Graph（保持 Graph 存活，防止 Var 中的 Weak 失效）
    pub graph: Graph,
}

impl BuildResult {
    /// 所有可训练参数的扁平列表（用于创建 Optimizer）
    ///
    /// 从 layer_params 派生，确保与 capture/restore 使用同一数据源。
    /// 按 innovation_number 排序以保证确定性顺序（HashMap 迭代顺序不确定）。
    pub fn all_parameters(&self) -> Vec<Var> {
        let mut keys: Vec<_> = self.layer_params.keys().copied().collect();
        keys.sort_unstable();
        keys.iter()
            .flat_map(|k| self.layer_params[k].iter().cloned())
            .collect()
    }
}

// ==================== InheritReport ====================

/// 权重继承报告
pub struct InheritReport {
    /// 成功继承的参数张量数
    pub inherited: usize,
    /// 保留初始化值的参数张量数（新层或形状变化）
    pub reinitialized: usize,
}

// ==================== 内部辅助 ====================

fn apply_activation(var: &Var, act_type: &ActivationType) -> Var {
    match act_type {
        ActivationType::ReLU => var.relu(),
        ActivationType::LeakyReLU { alpha } => var.leaky_relu(*alpha),
        ActivationType::Tanh => var.tanh(),
        ActivationType::Sigmoid => var.sigmoid(),
        ActivationType::GELU => var.gelu(),
        ActivationType::SiLU => var.silu(),
        ActivationType::Softplus => var.softplus(),
        ActivationType::ReLU6 => var.relu6(),
        ActivationType::ELU { alpha } => var.elu(*alpha),
        ActivationType::SELU => var.selu(),
        ActivationType::Mish => var.mish(),
        ActivationType::HardSwish => var.hard_swish(),
        ActivationType::HardSigmoid => var.hard_sigmoid(),
    }
}

/// 将 main path 输出与 incoming skip edges 聚合
///
/// 约束：同一目标层的所有 skip edges 必须使用相同的 AggregateStrategy。
/// 聚合语义：将 main + 所有 skip 源按策略合并，作为目标层的输入。
fn apply_aggregation(
    main: &Var,
    incoming: &[&SkipEdge],
    var_map: &HashMap<u64, Var>,
) -> Result<Var, GraphError> {
    // 收集所有 skip 源 Var
    let skip_vars: Vec<&Var> = incoming
        .iter()
        .map(|e| {
            var_map.get(&e.from_innovation).unwrap_or_else(|| {
                panic!(
                    "skip edge 源 innovation={} 未在 var_map 中找到",
                    e.from_innovation
                )
            })
        })
        .collect();

    let strategy = &incoming[0].strategy;

    match strategy {
        AggregateStrategy::Add => {
            let mut result = main.clone();
            for sv in &skip_vars {
                result = result.try_add(sv)?;
            }
            Ok(result)
        }
        AggregateStrategy::Concat { dim } => {
            let mut all_vars: Vec<&Var> = vec![main];
            all_vars.extend(skip_vars);
            Var::concat(&all_vars, *dim as usize)
        }
        AggregateStrategy::Mean => {
            let mut result = main.clone();
            for sv in &skip_vars {
                result = result.try_add(sv)?;
            }
            let n = (1 + skip_vars.len()) as f32;
            Ok(result / n)
        }
        AggregateStrategy::Max => {
            let mut result = main.clone();
            for sv in &skip_vars {
                result = result.maximum(sv)?;
            }
            Ok(result)
        }
    }
}

// ==================== NetworkGenome 构建与权重管理 ====================

impl NetworkGenome {
    /// 从基因组构建计算图
    ///
    /// 内部调用 resolve_dimensions() 推导维度链，逐层创建 Layer 并收集参数。
    /// 遇到 skip_edge 的目标层时，自动在其输入处生成聚合操作。
    ///
    /// rng 用于派生 Graph seed，确保参数初始化受 Evolution seed 控制。
    pub fn build(&self, rng: &mut StdRng) -> Result<BuildResult, GraphError> {
        let resolved = self
            .resolve_dimensions()
            .map_err(|e| GraphError::ComputationError(e.to_string()))?;
        let spatial_map = self.compute_spatial_map();

        let graph_seed: u64 = rng.r#gen();
        let graph = Graph::new_with_seed(graph_seed).with_model_name("EvolutionNet");

        let input = if let Some(seq_len) = self.seq_len {
            let var = graph.input_shape(&[1, seq_len, self.input_dim], Some("evo_input"))?;
            // RNN 层的 validate_input 需要读取输入值来确定 seq_len，
            // 因此在 build 时设置占位零值（训练时会被覆盖）
            var.set_value(&Tensor::zeros(&[1, seq_len, self.input_dim]))?;
            var
        } else if let Some((h, w)) = self.input_spatial {
            graph.input_shape(&[1, self.input_dim, h, w], Some("evo_input"))?
        } else {
            graph.input_shape(&[1, self.input_dim], Some("evo_input"))?
        };
        let mut current = input.clone();
        let mut layer_params: HashMap<u64, Vec<Var>> = HashMap::new();

        // innovation_number → Var 映射，供 skip edge 查找源层输出
        let mut var_map: HashMap<u64, Var> = HashMap::new();
        var_map.insert(INPUT_INNOVATION, input.clone());

        for dim in &resolved {
            let layer = self
                .layers
                .iter()
                .find(|l| l.innovation_number == dim.innovation_number && l.enabled)
                .expect("resolve_dimensions 返回的创新号必须对应一个启用的层");

            // 聚合：检查是否有 incoming skip edges 指向当前层
            let incoming: Vec<_> = self
                .skip_edges
                .iter()
                .filter(|e| e.enabled && e.to_innovation == layer.innovation_number)
                .collect();

            if !incoming.is_empty() {
                current =
                    apply_aggregation(&current, &incoming, &var_map)?;
            }

            match &layer.layer_config {
                LayerConfig::Linear { out_features } => {
                    let name = format!("evo_L{}", layer.innovation_number);
                    let linear =
                        Linear::new(&graph, dim.in_dim, *out_features, true, &name)?;
                    current = linear.forward(&current);

                    let mut params = vec![linear.weights().clone()];
                    if let Some(bias) = linear.bias() {
                        params.push(bias.clone());
                    }
                    layer_params.insert(layer.innovation_number, params);
                }
                LayerConfig::Activation { activation_type } => {
                    current = apply_activation(&current, activation_type);
                }
                LayerConfig::Rnn { hidden_size } => {
                    let name = format!("evo_rnn{}", layer.innovation_number);
                    let return_seq = self.needs_return_sequences(
                        layer.innovation_number,
                        &resolved,
                    );
                    let rnn = Rnn::new(&graph, dim.in_dim, *hidden_size, &name)?;
                    current = if return_seq {
                        rnn.forward_seq(&current)?
                    } else {
                        rnn.forward(&current)?
                    };
                    layer_params.insert(layer.innovation_number, rnn.parameters());
                }
                LayerConfig::Lstm { hidden_size } => {
                    let name = format!("evo_lstm{}", layer.innovation_number);
                    let return_seq = self.needs_return_sequences(
                        layer.innovation_number,
                        &resolved,
                    );
                    let lstm = Lstm::new(&graph, dim.in_dim, *hidden_size, &name)?;
                    current = if return_seq {
                        lstm.forward_seq(&current)?
                    } else {
                        lstm.forward(&current)?
                    };
                    layer_params.insert(layer.innovation_number, lstm.parameters());
                }
                LayerConfig::Gru { hidden_size } => {
                    let name = format!("evo_gru{}", layer.innovation_number);
                    let return_seq = self.needs_return_sequences(
                        layer.innovation_number,
                        &resolved,
                    );
                    let gru = Gru::new(&graph, dim.in_dim, *hidden_size, &name)?;
                    current = if return_seq {
                        gru.forward_seq(&current)?
                    } else {
                        gru.forward(&current)?
                    };
                    layer_params.insert(layer.innovation_number, gru.parameters());
                }
                LayerConfig::Conv2d {
                    out_channels,
                    kernel_size,
                } => {
                    let name = format!("evo_conv{}", layer.innovation_number);
                    let k = *kernel_size;
                    let padding = k / 2; // same padding
                    let conv = Conv2d::new(
                        &graph,
                        dim.in_dim,
                        *out_channels,
                        (k, k),
                        (1, 1),
                        (padding, padding),
                        true,
                        &name,
                    )?;
                    current = conv.forward(&current);
                    layer_params.insert(layer.innovation_number, conv.parameters());
                }
                LayerConfig::Pool2d {
                    pool_type,
                    kernel_size,
                    stride,
                } => {
                    // 找前驱层的输出空间作为本层输入空间；若 kernel 超出则跳过池化（identity pass-through）
                    let input_spatial = {
                        let pos = resolved.iter().position(|d| d.innovation_number == layer.innovation_number);
                        match pos {
                            Some(0) => self.input_spatial,
                            Some(p) => spatial_map.get(&resolved[p - 1].innovation_number).copied().flatten(),
                            None => None,
                        }
                    };
                    let can_pool = input_spatial
                        .map(|(h, w)| h >= *kernel_size && w >= *kernel_size)
                        .unwrap_or(false);

                    if can_pool {
                        let name = format!("evo_pool{}", layer.innovation_number);
                        let k = *kernel_size;
                        let s = *stride;
                        current = match pool_type {
                            PoolType::Max => {
                                MaxPool2d::new(&graph, (k, k), Some((s, s)), &name)
                                    .forward(&current)
                            }
                            PoolType::Avg => {
                                AvgPool2d::new(&graph, (k, k), Some((s, s)), &name)
                                    .forward(&current)
                            }
                        };
                    }
                    // else: identity pass-through（Pool2d 无可学习参数，跳过不影响梯度）
                }
                LayerConfig::Flatten => {
                    current = current.flatten()?;
                    // Flatten 无可学习参数
                }
                LayerConfig::Dropout { .. } => {
                    return Err(GraphError::ComputationError(format!(
                        "build() 尚未支持 {} 层类型（层 innovation={}）",
                        layer.layer_config, layer.innovation_number
                    )));
                }
            }

            // 记录当前层的输出，供后续 skip edge 作为源
            var_map.insert(layer.innovation_number, current.clone());
        }

        Ok(BuildResult {
            input,
            output: current,
            layer_params,
            graph,
        })
    }

    /// 将当前计算图的权重捕获到 Genome 的 weight_snapshots 中
    ///
    /// 训练完成后调用。按 innovation_number 索引，
    /// 每层的参数张量列表保存为 Vec<Tensor>（如 Linear 保存 [W, b]）。
    ///
    /// 注意：此方法是全量替换（非 merge），不在当前 build 中的层（如 disabled）
    /// 的旧快照会被丢弃。引入 disable/enable 语义时需评估是否改为 merge。
    pub fn capture_weights(&mut self, build: &BuildResult) -> Result<(), GraphError> {
        let mut snapshots: HashMap<u64, Vec<Tensor>> = HashMap::new();
        for (&inn, params) in &build.layer_params {
            let mut tensors = Vec::new();
            for param in params {
                let tensor = param.value()?.ok_or_else(|| {
                    GraphError::ComputationError(format!("层 {inn} 的参数无值"))
                })?;
                tensors.push(tensor);
            }
            snapshots.insert(inn, tensors);
        }
        self.set_weight_snapshots(snapshots);
        Ok(())
    }

    /// 判断指定 RNN 层是否需要 return_sequences
    ///
    /// 在 resolved 中找到当前层后，跳过 Activation/Dropout，
    /// 若下一个实质层也是循环层则返回 true。
    fn needs_return_sequences(
        &self,
        current_innovation: u64,
        resolved: &[super::gene::ResolvedDim],
    ) -> bool {
        // 找到 current_innovation 在 resolved 中的位置
        let pos = resolved
            .iter()
            .position(|d| d.innovation_number == current_innovation);
        let pos = match pos {
            Some(p) => p,
            None => return false,
        };

        // 向后扫描，跳过 Activation/Dropout
        for dim in &resolved[pos + 1..] {
            let layer = self
                .layers
                .iter()
                .find(|l| l.innovation_number == dim.innovation_number && l.enabled);
            if let Some(layer) = layer {
                match &layer.layer_config {
                    LayerConfig::Activation { .. } | LayerConfig::Dropout { .. } => continue,
                    _ => return NetworkGenome::is_recurrent(&layer.layer_config),
                }
            }
        }
        false
    }

    /// 从 weight_snapshots 恢复权重到当前计算图
    ///
    /// build() 之后、训练之前调用。
    /// 按 innovation_number 匹配：相同创新号且形状相同的参数直接复制，
    /// 形状不匹配或无快照的参数保留初始化值。
    pub fn restore_weights(&self, build: &BuildResult) -> Result<InheritReport, GraphError> {
        let mut inherited = 0usize;
        let mut reinitialized = 0usize;

        for (&inn, params) in &build.layer_params {
            if let Some(snapshots) = self.weight_snapshots().get(&inn) {
                for (i, param) in params.iter().enumerate() {
                    if let Some(snapshot) = snapshots.get(i) {
                        let current_val = param.value()?;
                        let shapes_match = current_val
                            .as_ref()
                            .map(|t| t.shape() == snapshot.shape())
                            .unwrap_or(false);

                        if shapes_match {
                            param.set_value(snapshot)?;
                            inherited += 1;
                        } else {
                            reinitialized += 1;
                        }
                    } else {
                        reinitialized += 1;
                    }
                }
            } else {
                reinitialized += params.len();
            }
        }

        Ok(InheritReport {
            inherited,
            reinitialized,
        })
    }
}

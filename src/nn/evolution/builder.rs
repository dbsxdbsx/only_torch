/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : Genome → Graph 转换 + Lamarckian 权重继承
 *
 * build() 按 resolve_dimensions() 顺序逐层创建计算图，
 * capture_weights() / restore_weights() 实现跨代权重复用。
 */

use std::collections::HashMap;

use rand::Rng;
use rand::rngs::StdRng;

use crate::nn::layer::{AvgPool2d, Conv2d, Gru, Lstm, MaxPool2d, Rnn};
use crate::nn::{Graph, GraphError, Linear, Module, Var, VarActivationOps, VarShapeOps};
use crate::tensor::Tensor;

use super::gene::{
    ActivationType, AggregateStrategy, INPUT_INNOVATION, LayerConfig, NetworkGenome, PoolType,
    SkipEdge,
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
#[derive(Debug)]
pub struct InheritReport {
    /// 完整形状匹配，直接复用旧权重的参数张量数
    pub inherited: usize,
    /// 形状兼容（仅一轴扩缩），保留重叠区域的参数张量数
    pub partially_inherited: usize,
    /// 保留初始化值的参数张量数（新层或两轴均变化）
    pub reinitialized: usize,
}

// ==================== 权重部分继承辅助 ====================

/// 沿单轴进行部分权重合并
///
/// - Grow（new_size > old_size）：拼接旧值 + current 的随机新列
/// - Shrink（new_size < old_size）：截取旧值前 new_size 个
fn partial_along_axis(snapshot: &Tensor, current: &Tensor, axis: usize) -> Option<Tensor> {
    let old_size = snapshot.shape()[axis];
    let new_size = current.shape()[axis];
    if new_size > old_size {
        let inherited_part = snapshot.narrow(axis, 0, old_size);
        let new_part = current.narrow(axis, old_size, new_size - old_size);
        Some(Tensor::concat(&[&inherited_part, &new_part], axis))
    } else if new_size < old_size {
        Some(snapshot.narrow(axis, 0, new_size))
    } else {
        None
    }
}

/// 尝试对形状不完全匹配的参数节点进行部分权重继承
///
/// 适用场景：Grow/Shrink 操作后，某一维度扩大或缩小，另一维度不变。
/// 重叠区域保留旧权重，新增区域保持 `current`（随机初始化）的值。
///
/// 返回 `Some(merged)` 若部分继承可行，`None` 若形状完全不兼容。
fn try_partial_inherit(snapshot: &Tensor, current: &Tensor) -> Option<Tensor> {
    let old_shape = snapshot.shape();
    let new_shape = current.shape();

    if old_shape.len() != new_shape.len() || old_shape.is_empty() || old_shape.len() > 2 {
        return None;
    }

    if old_shape.len() == 1 {
        let old_size = old_shape[0];
        let new_size = new_shape[0];
        if old_size == new_size {
            return None; // 完全一致，不该走这里
        }
        return partial_along_axis(snapshot, current, 0);
    }

    // 2D：恰好一轴变化
    let row_same = old_shape[0] == new_shape[0];
    let col_same = old_shape[1] == new_shape[1];
    match (row_same, col_same) {
        (true, false) => partial_along_axis(snapshot, current, 1),
        (false, true) => partial_along_axis(snapshot, current, 0),
        _ => None, // 两轴都变或都未变（后者不应走到这里）
    }
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

use super::gene::{GenomeRepr, ShapeDomain};
use super::node_gene::GenomeAnalysis;
use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor as NTD};

impl NetworkGenome {
    /// 将当前基因组转换为 `GraphDescriptor`
    ///
    /// - LayerLevel 基因组：自动内部迁移到节点级，再转换（不修改 self）
    /// - NodeLevel 基因组：直接转换
    ///
    /// 返回的 `GraphDescriptor` 可直接传入 `Graph::from_descriptor()` 构建计算图。
    pub fn to_graph_descriptor(&self) -> Result<GraphDescriptor, super::migration::MigrationError> {
        let nodes = match &self.repr {
            GenomeRepr::NodeLevel { nodes, .. } => std::borrow::Cow::Borrowed(nodes.as_slice()),
            GenomeRepr::LayerLevel { .. } => {
                let out = super::migration::migrate_network_genome(self)?;
                std::borrow::Cow::Owned(out.nodes)
            }
        };

        // 计算输入形状和域
        let input_shape: Vec<usize> = if let Some((h, w)) = self.input_spatial {
            vec![1, self.input_dim, h, w]
        } else if let Some(seq) = self.seq_len {
            vec![1, seq, self.input_dim]
        } else {
            vec![1, self.input_dim]
        };
        let input_domain = if self.input_spatial.is_some() {
            ShapeDomain::Spatial
        } else if self.seq_len.is_some() {
            ShapeDomain::Sequence
        } else {
            ShapeDomain::Flat
        };

        // 用 GenomeAnalysis 获取拓扑序（Graph::from_descriptor_seeded 要求父节点先于子节点）
        let analysis =
            GenomeAnalysis::compute(&nodes, INPUT_INNOVATION, input_shape.clone(), input_domain);
        let node_lookup: std::collections::HashMap<u64, &super::node_gene::NodeGene> = nodes
            .iter()
            .filter(|n| n.enabled)
            .map(|n| (n.innovation_number, n))
            .collect();

        let mut desc = GraphDescriptor::new("EvolutionNet");

        // 先添加虚拟输入节点
        let dynamic_input: Vec<Option<usize>> = std::iter::once(None)
            .chain(input_shape[1..].iter().map(|&d| Some(d)))
            .collect();
        desc.add_node(NodeDescriptor::new(
            INPUT_INNOVATION,
            "evo_input",
            NTD::BasicInput,
            input_shape,
            Some(dynamic_input),
            vec![],
        ));

        // 按拓扑序添加所有启用的 NodeGene（父节点必须在子节点之前）
        for &id in &analysis.topo_order {
            if let Some(node) = node_lookup.get(&id) {
                let dynamic = node.output_shape.first().map(|_| {
                    let mut d: Vec<Option<usize>> =
                        node.output_shape.iter().map(|&x| Some(x)).collect();
                    if !d.is_empty() {
                        d[0] = None;
                    } // batch 维动态
                    d
                });
                desc.add_node(NodeDescriptor::new(
                    node.innovation_number,
                    &format!("evo_{}", node.innovation_number),
                    node.node_type.clone(),
                    node.output_shape.clone(),
                    dynamic,
                    node.parents.clone(),
                ));
            }
        }

        Ok(desc)
    }

    /// 从基因组构建计算图
    ///
    /// - NodeLevel 基因组（当前唯一支持格式）：`to_graph_descriptor()` + `Graph::from_descriptor()`
    /// - LayerLevel 基因组（遗留兼容路径，仅用于尚未节点化的入口）：逐层构图
    ///
    /// rng 用于派生 Graph seed，确保参数初始化受 Evolution seed 控制。
    pub fn build(&self, rng: &mut StdRng) -> Result<BuildResult, GraphError> {
        if self.is_node_level() {
            return self.build_from_nodes(rng);
        }
        self.build_layer_level(rng)
    }

    /// NodeLevel 基因组的构图路径
    fn build_from_nodes(&self, rng: &mut StdRng) -> Result<BuildResult, GraphError> {
        let desc = self
            .to_graph_descriptor()
            .map_err(|e| GraphError::ComputationError(e.to_string()))?;

        let graph_seed: u64 = rng.r#gen();
        let rebuild =
            Graph::from_descriptor_seeded(&desc, graph_seed).map_err(|e| GraphError::from(e))?;

        let input = rebuild
            .inputs
            .first()
            .map(|(_, v)| v.clone())
            .ok_or_else(|| {
                GraphError::ComputationError("NodeLevel 基因组构图后无输入节点".into())
            })?;
        let output = rebuild.outputs.first().cloned().ok_or_else(|| {
            GraphError::ComputationError("NodeLevel 基因组构图后无输出节点".into())
        })?;

        // 收集参数节点：param_innovation → [Var]
        let nodes = self.nodes();
        let layer_params: HashMap<u64, Vec<Var>> = nodes
            .iter()
            .filter(|n| n.enabled && n.is_parameter())
            .filter_map(|n| {
                rebuild
                    .node_map
                    .get(&n.innovation_number)
                    .cloned()
                    .map(|v| (n.innovation_number, vec![v]))
            })
            .collect();

        // 回填 NodeGroupTag：将 NodeGene 的 block_id 映射为可视化 Cluster 标签
        backfill_node_group_tags(self, &rebuild.node_map);

        Ok(BuildResult {
            input,
            output,
            layer_params,
            graph: rebuild.graph,
        })
    }

    /// LayerLevel 基因组的遗留构图路径
    ///
    /// 遗留逐层构图路径；仅应在用户 DSL（`from_flat`/`from_spatial`/`from_sequential`）
    /// 仍产生 LayerLevel 基因组时使用。新代码应优先使用 NodeLevel 与 `build_from_nodes`。
    fn build_layer_level(&self, rng: &mut StdRng) -> Result<BuildResult, GraphError> {
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
                .layers()
                .iter()
                .find(|l| l.innovation_number == dim.innovation_number && l.enabled)
                .expect("resolve_dimensions 返回的创新号必须对应一个启用的层");

            // 聚合：检查是否有 incoming skip edges 指向当前层
            let incoming: Vec<_> = self
                .skip_edges()
                .iter()
                .filter(|e| e.enabled && e.to_innovation == layer.innovation_number)
                .collect();

            if !incoming.is_empty() {
                current = apply_aggregation(&current, &incoming, &var_map)?;
            }

            match &layer.layer_config {
                LayerConfig::Linear { out_features } => {
                    let name = format!("evo_L{}", layer.innovation_number);
                    let linear = Linear::new(&graph, dim.in_dim, *out_features, true, &name)?;
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
                    let return_seq =
                        self.needs_return_sequences(layer.innovation_number, &resolved);
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
                    let return_seq =
                        self.needs_return_sequences(layer.innovation_number, &resolved);
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
                    let return_seq =
                        self.needs_return_sequences(layer.innovation_number, &resolved);
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
                        let pos = resolved
                            .iter()
                            .position(|d| d.innovation_number == layer.innovation_number);
                        match pos {
                            Some(0) => self.input_spatial,
                            Some(p) => spatial_map
                                .get(&resolved[p - 1].innovation_number)
                                .copied()
                                .flatten(),
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
                            PoolType::Max => MaxPool2d::new(&graph, (k, k), Some((s, s)), &name)
                                .forward(&current),
                            PoolType::Avg => AvgPool2d::new(&graph, (k, k), Some((s, s)), &name)
                                .forward(&current),
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
    /// - LayerLevel：按层创新号索引，每层所有参数张量存为 `Vec<Tensor>`
    /// - NodeLevel：按参数节点创新号索引，每个 Parameter 节点存一个 `Tensor`
    pub fn capture_weights(&mut self, build: &BuildResult) -> Result<(), GraphError> {
        if self.is_node_level() {
            // NodeLevel：单参数节点粒度快照
            let mut node_snaps: HashMap<u64, Tensor> = HashMap::new();
            for (&inn, params) in &build.layer_params {
                if let Some(param) = params.first() {
                    let tensor = param.value()?.ok_or_else(|| {
                        GraphError::ComputationError(format!("Parameter 节点 {inn} 无值"))
                    })?;
                    node_snaps.insert(inn, tensor);
                }
            }
            match &mut self.repr {
                super::gene::GenomeRepr::NodeLevel {
                    weight_snapshots, ..
                } => {
                    *weight_snapshots = node_snaps;
                }
                _ => unreachable!(),
            }
            return Ok(());
        }

        // LayerLevel：原有层级粒度快照
        let mut snapshots: HashMap<u64, Vec<Tensor>> = HashMap::new();
        for (&inn, params) in &build.layer_params {
            let mut tensors = Vec::new();
            for param in params {
                let tensor = param
                    .value()?
                    .ok_or_else(|| GraphError::ComputationError(format!("层 {inn} 的参数无值")))?;
                tensors.push(tensor);
            }
            snapshots.insert(inn, tensors);
        }
        self.set_weight_snapshots(snapshots);
        Ok(())
    }

    /// 从 `GraphDescriptor` 创建 NodeLevel `NetworkGenome`
    ///
    /// 支持两种来源：
    /// - 手写训练后通过 `Var::vars_to_graph_descriptor()` 得到的描述符
    /// - `NetworkGenome::to_graph_descriptor()` 的逆向链路（用于序列化往返验证）
    ///
    /// # 设计
    /// - `BasicInput` 节点被视为虚拟输入（不进入 `nodes` 列表），其 id 重映射为 `INPUT_INNOVATION=0`
    /// - `TargetInput` 节点被跳过（不参与演化）
    /// - 所有其他节点（含 `Parameter`）转换为 `NodeGene`（`block_id=None`，`enabled=true`）
    /// - 输出维度从无子节点的末端节点形状推导
    ///
    /// # 错误
    /// 若描述符没有 `BasicInput` 节点、没有可用节点、或输入形状无法识别，返回 `Err`。
    pub fn from_graph_descriptor(
        desc: &GraphDescriptor,
    ) -> Result<Self, super::migration::MigrationError> {
        use super::migration::MigrationError;

        // 找 BasicInput 节点（虚拟输入）
        let input_nd = desc
            .nodes
            .iter()
            .find(|n| matches!(n.node_type, NTD::BasicInput))
            .ok_or_else(|| {
                MigrationError::DimensionError("GraphDescriptor 中没有 BasicInput 节点".into())
            })?;

        let original_input_id = input_nd.id;
        let input_shape = &input_nd.output_shape;

        // 从输入形状推导模式：[batch, features] / [batch,seq,feat] / [batch,C,H,W]
        let (input_dim, seq_len, input_spatial) = match input_shape.len() {
            2 => (input_shape[1], None, None),
            3 => (input_shape[2], Some(input_shape[1]), None),
            4 => (input_shape[1], None, Some((input_shape[2], input_shape[3]))),
            _ => {
                return Err(MigrationError::DimensionError(format!(
                    "不支持的输入形状 {:?}（期望 2D/3D/4D）",
                    input_shape
                )));
            }
        };

        // 将原始 input_id 重映射为 INPUT_INNOVATION=0，使 genome.analyze() 能正常工作
        let remap_id = |id: u64| -> u64 {
            if id == original_input_id {
                INPUT_INNOVATION
            } else {
                id
            }
        };

        // 转换所有非输入节点为 NodeGene
        let mut nodes: Vec<super::node_gene::NodeGene> = Vec::new();
        let mut max_id: u64 = 0;

        for nd in &desc.nodes {
            // 跳过虚拟输入节点（BasicInput 是外部数据源，不入节点列表）
            if matches!(nd.node_type, NTD::BasicInput | NTD::TargetInput) {
                continue;
            }
            let remapped_id = remap_id(nd.id);
            let remapped_parents: Vec<u64> = nd.parents.iter().map(|&p| remap_id(p)).collect();

            nodes.push(super::node_gene::NodeGene::new(
                remapped_id,
                nd.node_type.clone(),
                nd.output_shape.clone(),
                remapped_parents,
                None, // 手写模型没有 block_id 语义，统一设为 None
            ));
            if remapped_id > max_id {
                max_id = remapped_id;
            }
        }

        if nodes.is_empty() {
            return Err(MigrationError::DimensionError(
                "GraphDescriptor 中没有可转换的计算节点".into(),
            ));
        }

        // 从无子节点的末端节点推导 output_dim
        let child_ids: std::collections::HashSet<u64> = nodes
            .iter()
            .flat_map(|n| n.parents.iter().copied())
            .collect();
        let output_shape = nodes
            .iter()
            .filter(|n| !child_ids.contains(&n.innovation_number))
            .last()
            .or_else(|| nodes.last())
            .map(|n| &n.output_shape)
            .ok_or_else(|| MigrationError::DimensionError("无法确定输出节点".into()))?;
        let output_dim = match output_shape.len() {
            n if n >= 2 => output_shape[output_shape.len() - 1],
            1 => output_shape[0],
            _ => return Err(MigrationError::DimensionError("输出节点形状为空".into())),
        };

        Ok(Self {
            input_dim,
            output_dim,
            seq_len,
            input_spatial,
            training_config: super::gene::TrainingConfig::default(),
            generated_by: "from_graph_descriptor".to_string(),
            repr: super::gene::GenomeRepr::NodeLevel {
                nodes,
                next_innovation: max_id + 1,
                weight_snapshots: std::collections::HashMap::new(),
            },
        })
    }

    /// 从 .onnx 文件构建 NetworkGenome（用于后续演化或推理）
    ///
    /// 权重不会保留在 genome 中（ONNX 无 weight_snapshots 语义），
    /// 如需带权重推理，请使用 `Graph::from_onnx()`。
    ///
    /// # 示例
    /// ```ignore
    /// let genome = NetworkGenome::from_onnx("model.onnx")?;
    /// println!("输入维度: {}, 输出维度: {}", genome.input_dim, genome.output_dim);
    /// ```
    pub fn from_onnx<P: AsRef<std::path::Path>>(
        path: P,
    ) -> Result<Self, super::migration::MigrationError> {
        let import_result = crate::nn::graph::onnx_import::load_onnx(path).map_err(|e| {
            super::migration::MigrationError::DimensionError(format!("ONNX 导入失败: {e}"))
        })?;
        Self::from_graph_descriptor(&import_result.descriptor)
    }

    /// 从内存中的 .onnx 字节流构建 NetworkGenome
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self, super::migration::MigrationError> {
        let import_result =
            crate::nn::graph::onnx_import::load_onnx_from_bytes(bytes).map_err(|e| {
                super::migration::MigrationError::DimensionError(format!("ONNX 导入失败: {e}"))
            })?;
        Self::from_graph_descriptor(&import_result.descriptor)
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
                .layers()
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
    /// - LayerLevel：按层创新号 + 张量索引匹配
    /// - NodeLevel：按参数节点创新号匹配；
    ///   - 形状完全相同 → 全量继承（`inherited`）
    ///   - 仅一轴扩缩 → 部分继承，重叠区域保留旧值（`partially_inherited`）
    ///   - 两轴均变化或无快照 → 保留随机初始化（`reinitialized`）
    pub fn restore_weights(&self, build: &BuildResult) -> Result<InheritReport, GraphError> {
        let mut inherited = 0usize;
        let mut partially_inherited = 0usize;
        let mut reinitialized = 0usize;

        if self.is_node_level() {
            // NodeLevel：单参数节点粒度恢复
            let node_snaps = self.node_weight_snapshots();
            for (&inn, params) in &build.layer_params {
                if let Some(snapshot) = node_snaps.get(&inn) {
                    if let Some(param) = params.first() {
                        let current_val = param.value()?;
                        let shapes_match = current_val
                            .as_ref()
                            .map(|t| t.shape() == snapshot.shape())
                            .unwrap_or(false);
                        if shapes_match {
                            param.set_value(snapshot)?;
                            inherited += 1;
                        } else if let Some(ref current_tensor) = current_val {
                            // 尝试部分继承：保留重叠区域，新增区域保持随机初始化
                            if let Some(merged) = try_partial_inherit(snapshot, current_tensor) {
                                param.set_value(&merged)?;
                                partially_inherited += 1;
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
            return Ok(InheritReport {
                inherited,
                partially_inherited,
                reinitialized,
            });
        }

        // LayerLevel：原有层级粒度恢复
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
            partially_inherited,
            reinitialized,
        })
    }
}

// ==================== NodeGroupTag 回填 ====================

/// NodeLevel 构图后回填 NodeGroupTag，确保可视化能完整显示层级 Cluster
///
/// 背景：`build_layer_level()` 路径通过 RAII `NodeGroupContext` 自动为节点打标签；
/// `build_from_nodes()` 路径经 descriptor rebuild 无上下文，节点标签为空。
/// 此函数利用 `NodeGene::block_id` 在构图完成后补填。
///
/// 对每个 `block_id != None`、类型有意义的块，将同块所有节点（含 Parameter）
/// 打上相同的 `NodeGroupTag`，确保可视化渲染时归入同一 Cluster。
fn backfill_node_group_tags(genome: &NetworkGenome, node_map: &HashMap<u64, Var>) {
    use super::node_ops::{NodeBlockKind, node_main_path};
    use crate::nn::graph::{GroupStyle, NodeGroupTag};

    for block in node_main_path(genome) {
        let Some(bid) = block.block_id else { continue };

        let (group_type, style): (&str, GroupStyle) = match &block.kind {
            NodeBlockKind::Linear { .. } => ("Linear", GroupStyle::Layer),
            NodeBlockKind::Conv2d { .. } => ("Conv2d", GroupStyle::Layer),
            NodeBlockKind::Pool2d { .. } => ("Pool2d", GroupStyle::Layer),
            NodeBlockKind::Flatten => ("Flatten", GroupStyle::Layer),
            NodeBlockKind::Dropout { .. } => ("Dropout", GroupStyle::Layer),
            NodeBlockKind::Activation { .. } => ("Activation", GroupStyle::Layer),
            NodeBlockKind::Rnn { .. } => ("RNN", GroupStyle::Recurrent),
            NodeBlockKind::Lstm { .. } => ("LSTM", GroupStyle::Recurrent),
            NodeBlockKind::Gru { .. } => ("GRU", GroupStyle::Recurrent),
            NodeBlockKind::SkipAgg | NodeBlockKind::Unknown => continue,
        };

        // 描述：取输出节点的 output_shape → "[?, a, b]"
        let description = genome
            .nodes()
            .iter()
            .find(|n| n.innovation_number == block.output_id)
            .map(|n| {
                let shape: Vec<String> = n
                    .output_shape
                    .iter()
                    .enumerate()
                    .map(|(i, &d)| {
                        if i == 0 {
                            "?".to_string()
                        } else {
                            d.to_string()
                        }
                    })
                    .collect();
                format!("[{}]", shape.join(", "))
            });

        let tag = NodeGroupTag {
            group_type: group_type.to_string(),
            instance_id: bid as usize,
            display_name: Some(group_type.to_string()),
            description,
            style,
            hidden: false,
        };

        for &nid in &block.node_ids {
            if let Some(var) = node_map.get(&nid) {
                var.node().set_node_group_tag(Some(tag.clone()));
            }
        }
    }
}

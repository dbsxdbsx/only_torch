/*
 * @Author       : 老董
 * @Date         : 2026-03-25
 * @Description  : 阶段 4：NodeLevel 基因组的分析与变异辅助
 *
 * 核心类型：
 * - NodeBlock     : 描述 NodeLevel 基因组中一个"模板块"的结构信息
 * - NodeBlockKind : 模板块的语义类型（对应原 LayerConfig 的角色）
 *
 * 核心函数：
 * - node_main_path()        : 返回拓扑有序的模板块列表
 * - insert_after()          : 在指定节点后插入新节点并修复父连接
 * - remove_block()          : 删除一个模板块并重新连线
 * - grow_linear_out()       : 增大 Linear 块的输出维度（含级联）
 * - sync_computation_shapes(): 基于参数节点形状重新推导所有计算节点形状
 */

use std::collections::{HashMap, HashSet, VecDeque};

use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::{
    ActivationType, GenomeRepr, INPUT_INNOVATION, NetworkGenome, PoolType, ShapeDomain,
};
use crate::nn::evolution::migration::{
    InnovationCounter, expand_activation, expand_conv2d, expand_linear, expand_pool2d,
};
use crate::nn::evolution::mutation::SizeConstraints;
use crate::nn::evolution::node_gene::{GenomeAnalysis, NodeGene};

// ==================== 数据类型 ====================

/// NodeLevel 基因组中一个有序"模板块"的描述
///
/// 对应原 LayerLevel 中一个 LayerGene + 可能的聚合操作。
/// 单个节点（激活函数等）也作为独立块。
#[derive(Debug, Clone)]
pub struct NodeBlock {
    /// 模板组标识（None 表示独立单节点，如激活函数）
    pub block_id: Option<u64>,
    /// 该块内所有节点的创新号（按拓扑序）
    pub node_ids: Vec<u64>,
    /// 该块的外部输入节点创新号
    pub input_id: u64,
    /// 该块的输出节点创新号（下一个块的前驱）
    pub output_id: u64,
    /// 块的语义类型
    pub kind: NodeBlockKind,
}

/// 模板块的语义类型
#[derive(Debug, Clone)]
pub enum NodeBlockKind {
    Linear {
        out_features: usize,
    },
    Activation {
        act_type: ActivationType,
    },
    Conv2d {
        out_channels: usize,
        kernel_size: usize,
    },
    Pool2d {
        pool_type: PoolType,
        kernel_size: usize,
        stride: usize,
    },
    Flatten,
    Dropout {
        p: f32,
    },
    /// 跳跃连接聚合节点（Add/Concat/Maximum）
    SkipAgg,
    Unknown,
}

impl NodeBlockKind {
    pub fn is_activation(&self) -> bool {
        matches!(self, NodeBlockKind::Activation { .. })
    }
    pub fn is_linear(&self) -> bool {
        matches!(self, NodeBlockKind::Linear { .. })
    }
    pub fn is_conv2d(&self) -> bool {
        matches!(self, NodeBlockKind::Conv2d { .. })
    }
    pub fn is_resizable(&self) -> bool {
        self.is_linear() || self.is_conv2d()
    }
    pub fn current_size(&self) -> Option<usize> {
        match self {
            NodeBlockKind::Linear { out_features } => Some(*out_features),
            NodeBlockKind::Conv2d { out_channels, .. } => Some(*out_channels),
            _ => None,
        }
    }
}

// ==================== 主路径分析 ====================

/// 返回 NodeLevel 基因组的主路径块列表（按拓扑序）
///
/// 每个"块"对应原来的一个 LayerConfig 展开结果。
/// 单节点（激活、池化等）block_id = None，多节点模板有具体 block_id。
pub fn node_main_path(genome: &NetworkGenome) -> Vec<NodeBlock> {
    let nodes = genome.nodes();
    if nodes.is_empty() {
        return vec![];
    }

    let input_shape = genome_input_shape(genome);
    let input_domain = genome_input_domain(genome);
    let analysis = GenomeAnalysis::compute(nodes, INPUT_INNOVATION, input_shape, input_domain);
    let topo_order = &analysis.topo_order;

    let node_map: HashMap<u64, &NodeGene> = nodes
        .iter()
        .filter(|n| n.enabled)
        .map(|n| (n.innovation_number, n))
        .collect();
    let all_ids: HashSet<u64> = node_map.keys().copied().collect();

    // 构建 children 映射
    let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
    for node in nodes.iter().filter(|n| n.enabled) {
        for &pid in &node.parents {
            children
                .entry(pid)
                .or_default()
                .push(node.innovation_number);
        }
    }

    // 按拓扑序构建有序块列表：
    // - block_id=Some(bid) 的节点：同 bid 的全部节点合并为一个模板块（首次遇到时创建）
    // - block_id=None 的节点：每个计算节点单独成一个独立块（避免多个激活节点合并成 Unknown）
    //
    // 重要：只用计算节点（非参数节点）确定块顺序，参数节点（W/b）在拓扑序中可能提前出现
    let mut ordered_node_groups: Vec<(Option<u64>, Vec<u64>)> = Vec::new(); // (block_id, node_ids)
    let mut seen_bids: HashSet<u64> = HashSet::new();
    // 预先收集各 template block 的全部节点（按拓扑序）
    let mut template_nodes: HashMap<u64, Vec<u64>> = HashMap::new();
    for &id in topo_order {
        if let Some(node) = node_map.get(&id) {
            if let Some(bid) = node.block_id {
                template_nodes.entry(bid).or_default().push(id);
            }
        }
    }
    for &id in topo_order {
        if let Some(node) = node_map.get(&id) {
            if node.is_parameter() {
                continue;
            }
            match node.block_id {
                None => {
                    // 每个独立计算节点单独成块
                    ordered_node_groups.push((None, vec![id]));
                }
                Some(bid) => {
                    if seen_bids.insert(bid) {
                        let ids = template_nodes.get(&bid).cloned().unwrap_or_default();
                        ordered_node_groups.push((Some(bid), ids));
                    }
                }
            }
        }
    }

    ordered_node_groups
        .into_iter()
        .map(|(bid, ids)| {
            let bid_set: HashSet<u64> = ids.iter().copied().collect();

            // input_id：该块中第一个非叶节点的外部父节点
            let mut input_id = INPUT_INNOVATION;
            'find: for &nid in &ids {
                if let Some(node) = node_map.get(&nid) {
                    if !node.is_leaf() {
                        for &pid in &node.parents {
                            if !bid_set.contains(&pid) {
                                input_id = pid;
                                break 'find;
                            }
                        }
                    }
                }
            }

            // output_id：块内没有块内子节点的那个节点（= 块的出口）
            let output_id = ids
                .iter()
                .copied()
                .find(|&nid| {
                    let has_in_block_child = children
                        .get(&nid)
                        .map(|cs| cs.iter().any(|&c| bid_set.contains(&c)))
                        .unwrap_or(false);
                    !has_in_block_child && all_ids.contains(&nid)
                })
                .or_else(|| ids.last().copied())
                .unwrap_or(INPUT_INNOVATION);

            let kind = infer_block_kind(&ids, &node_map);

            NodeBlock {
                block_id: bid,
                node_ids: ids,
                input_id,
                output_id,
                kind,
            }
        })
        .collect()
}

/// 从节点列表推断块的语义类型
fn infer_block_kind(node_ids: &[u64], node_map: &HashMap<u64, &NodeGene>) -> NodeBlockKind {
    use NodeTypeDescriptor as NT;

    if node_ids.len() == 1 {
        let node = match node_map.get(&node_ids[0]) {
            Some(n) => n,
            None => return NodeBlockKind::Unknown,
        };
        return match &node.node_type {
            NT::ReLU => NodeBlockKind::Activation {
                act_type: ActivationType::ReLU,
            },
            NT::Tanh => NodeBlockKind::Activation {
                act_type: ActivationType::Tanh,
            },
            NT::Sigmoid => NodeBlockKind::Activation {
                act_type: ActivationType::Sigmoid,
            },
            NT::Gelu => NodeBlockKind::Activation {
                act_type: ActivationType::GELU,
            },
            NT::Swish => NodeBlockKind::Activation {
                act_type: ActivationType::SiLU,
            },
            NT::SoftPlus => NodeBlockKind::Activation {
                act_type: ActivationType::Softplus,
            },
            NT::ReLU6 => NodeBlockKind::Activation {
                act_type: ActivationType::ReLU6,
            },
            NT::Selu => NodeBlockKind::Activation {
                act_type: ActivationType::SELU,
            },
            NT::Mish => NodeBlockKind::Activation {
                act_type: ActivationType::Mish,
            },
            NT::HardSwish => NodeBlockKind::Activation {
                act_type: ActivationType::HardSwish,
            },
            NT::HardSigmoid => NodeBlockKind::Activation {
                act_type: ActivationType::HardSigmoid,
            },
            NT::LeakyReLU { alpha } => NodeBlockKind::Activation {
                act_type: ActivationType::LeakyReLU { alpha: *alpha },
            },
            NT::Elu { alpha } => NodeBlockKind::Activation {
                act_type: ActivationType::ELU { alpha: *alpha },
            },
            NT::Dropout { p } => NodeBlockKind::Dropout { p: *p },
            NT::Flatten { .. } => NodeBlockKind::Flatten,
            NT::MaxPool2d {
                kernel_size,
                stride,
            } => NodeBlockKind::Pool2d {
                pool_type: PoolType::Max,
                kernel_size: kernel_size.0,
                stride: stride.0,
            },
            NT::AvgPool2d {
                kernel_size,
                stride,
            } => NodeBlockKind::Pool2d {
                pool_type: PoolType::Avg,
                kernel_size: kernel_size.0,
                stride: stride.0,
            },
            NT::Add | NT::Maximum => NodeBlockKind::SkipAgg,
            NT::Concat { .. } => NodeBlockKind::SkipAgg,
            _ => NodeBlockKind::Unknown,
        };
    }

    // 多节点块：通过包含的计算节点类型判断
    let has_matmul = node_ids.iter().any(|&id| {
        node_map
            .get(&id)
            .map(|n| matches!(n.node_type, NT::MatMul))
            .unwrap_or(false)
    });
    let has_conv = node_ids.iter().any(|&id| {
        node_map
            .get(&id)
            .map(|n| matches!(n.node_type, NT::Conv2d { .. }))
            .unwrap_or(false)
    });

    if has_matmul {
        // 通过 MatMul 节点的父节点关系精确定位 W 参数
        // （不能用 shape[0] > 1 启发式——当 in_features=1 时 W 和 bias 形状相同）
        let bid_set: HashSet<u64> = node_ids.iter().copied().collect();
        for &id in node_ids {
            if let Some(n) = node_map.get(&id) {
                if matches!(n.node_type, NT::MatMul) {
                    for &pid in &n.parents {
                        if bid_set.contains(&pid) {
                            if let Some(p) = node_map.get(&pid) {
                                if p.is_parameter() && p.output_shape.len() == 2 {
                                    return NodeBlockKind::Linear {
                                        out_features: p.output_shape[1],
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if has_conv {
        for &id in node_ids {
            if let Some(n) = node_map.get(&id) {
                if n.is_parameter() && n.output_shape.len() == 4 {
                    return NodeBlockKind::Conv2d {
                        out_channels: n.output_shape[0],
                        kernel_size: n.output_shape[2],
                    };
                }
            }
        }
    }

    NodeBlockKind::Unknown
}

// ==================== 工具函数 ====================

/// 返回 genome 的输入形状（含 batch 维 =1）
pub fn genome_input_shape(genome: &NetworkGenome) -> Vec<usize> {
    if let Some((h, w)) = genome.input_spatial {
        vec![1, genome.input_dim, h, w]
    } else if let Some(seq) = genome.seq_len {
        vec![1, seq, genome.input_dim]
    } else {
        vec![1, genome.input_dim]
    }
}

/// 返回 genome 的输入域
pub fn genome_input_domain(genome: &NetworkGenome) -> ShapeDomain {
    if genome.input_spatial.is_some() {
        ShapeDomain::Spatial
    } else if genome.seq_len.is_some() {
        ShapeDomain::Sequence
    } else {
        ShapeDomain::Flat
    }
}

/// 创建从 genome 当前创新号起始的计数器
pub fn make_counter(genome: &NetworkGenome) -> InnovationCounter {
    InnovationCounter::new(genome.peek_next_innovation())
}

/// 将计数器状态写回 genome 的 next_innovation
pub fn commit_counter(genome: &mut NetworkGenome, counter: &InnovationCounter) {
    match &mut genome.repr {
        GenomeRepr::NodeLevel {
            next_innovation, ..
        } => *next_innovation = counter.peek(),
        _ => {}
    }
}

/// 下一个可用的 block_id（当前最大 + 1）
pub fn next_block_id(genome: &NetworkGenome) -> u64 {
    genome
        .nodes()
        .iter()
        .filter_map(|n| n.block_id)
        .max()
        .map(|m| m + 1)
        .unwrap_or(0)
}

/// 获取指定节点的输出特征维度（2D: [1, D] → D，4D: [1, C, H, W] → C）
pub fn node_out_dim_at(genome: &NetworkGenome, node_id: u64) -> usize {
    if node_id == INPUT_INNOVATION {
        return genome.input_dim;
    }
    let analysis = GenomeAnalysis::compute(
        genome.nodes(),
        INPUT_INNOVATION,
        genome_input_shape(genome),
        genome_input_domain(genome),
    );
    analysis
        .output_shapes
        .get(&node_id)
        .and_then(|s| s.get(1).copied())
        .unwrap_or(genome.input_dim)
}

/// 获取指定节点处的空间尺寸（4D 张量才有）
pub fn node_spatial_at(genome: &NetworkGenome, node_id: u64) -> Option<(usize, usize)> {
    if node_id == INPUT_INNOVATION {
        return genome.input_spatial;
    }
    let analysis = GenomeAnalysis::compute(
        genome.nodes(),
        INPUT_INNOVATION,
        genome_input_shape(genome),
        genome_input_domain(genome),
    );
    analysis.output_shapes.get(&node_id).and_then(|s| {
        if s.len() == 4 {
            Some((s[2], s[3]))
        } else {
            None
        }
    })
}

/// 获取指定节点处的域
pub fn node_domain_at(genome: &NetworkGenome, node_id: u64) -> ShapeDomain {
    if node_id == INPUT_INNOVATION {
        return genome_input_domain(genome);
    }
    let analysis = GenomeAnalysis::compute(
        genome.nodes(),
        INPUT_INNOVATION,
        genome_input_shape(genome),
        genome_input_domain(genome),
    );
    analysis.domain_of(node_id).unwrap_or(ShapeDomain::Flat)
}

/// 指定节点是否为激活函数节点
pub fn is_activation_node(genome: &NetworkGenome, node_id: u64) -> bool {
    genome
        .nodes()
        .iter()
        .find(|n| n.innovation_number == node_id)
        .map(|n| {
            matches!(
                n.node_type,
                NodeTypeDescriptor::ReLU
                    | NodeTypeDescriptor::Tanh
                    | NodeTypeDescriptor::Sigmoid
                    | NodeTypeDescriptor::Gelu
                    | NodeTypeDescriptor::Swish
                    | NodeTypeDescriptor::Selu
                    | NodeTypeDescriptor::Mish
                    | NodeTypeDescriptor::HardSwish
                    | NodeTypeDescriptor::HardSigmoid
                    | NodeTypeDescriptor::ReLU6
                    | NodeTypeDescriptor::SoftPlus
                    | NodeTypeDescriptor::LeakyReLU { .. }
                    | NodeTypeDescriptor::Elu { .. }
                    | NodeTypeDescriptor::HardTanh { .. }
            )
        })
        .unwrap_or(false)
}

/// 重新推导所有计算节点的形状（基于 Parameter 节点的权威形状）
pub fn sync_computation_shapes(genome: &mut NetworkGenome) {
    let input_shape = genome_input_shape(genome);
    let input_domain = genome_input_domain(genome);
    let analysis =
        GenomeAnalysis::compute(genome.nodes(), INPUT_INNOVATION, input_shape, input_domain);
    for node in genome.nodes_mut().iter_mut() {
        if !node.is_leaf() {
            if let Some(s) = analysis.shape_of(node.innovation_number) {
                node.output_shape = s.clone();
            }
        }
    }
}

/// 修复参数维度的核心逻辑（不触发跳跃连接修复，避免循环调用）
///
/// 沿拓扑序遍历所有块，将每个 Linear/Conv2d 块的参数节点输入维度
/// 更新为其实际前驱节点（`block.input_id`）的输出维度。
///
/// **关键设计**：使用 `dim_map: HashMap<u64, usize>` 记录每个节点的输出特征维度，
/// 而非单一 `prev_out` 线性传递——在含跳跃连接的 DAG 中，旁路块的输入来自更早的
/// 节点（如 INPUT），不是拓扑序中前一个块的输出，单一 prev_out 会导致错误级联。
fn repair_param_input_dims_inner(genome: &mut NetworkGenome) {
    let blocks = node_main_path(genome);

    // 维护每个节点输出的特征维度（node_id → feature_dim）
    let mut dim_map: HashMap<u64, usize> = HashMap::new();
    dim_map.insert(INPUT_INNOVATION, genome.input_dim);

    for block in &blocks {
        // 跳过 skip 投影块——投影块由 repair_skip_connections 自动维护
        if is_skip_projection_block(genome, block) {
            continue;
        }

        // 从 dim_map 查找该块实际输入节点的输出维度
        let prev_out = dim_map.get(&block.input_id).copied().unwrap_or_else(|| {
            // 回退：从节点自身的 output_shape 推断
            genome
                .nodes()
                .iter()
                .find(|n| n.innovation_number == block.input_id)
                .and_then(|n| n.output_shape.get(1).copied())
                .unwrap_or(genome.input_dim)
        });

        let bid_set: HashSet<u64> = block.node_ids.iter().copied().collect();
        match &block.kind {
            NodeBlockKind::Linear { out_features } => {
                let out = *out_features;
                // 通过 MatMul 的父节点关系精确定位 W（避免 in_features=1 时 W/bias 混淆）
                let w_id: Option<u64> = {
                    let nodes = genome.nodes();
                    nodes
                        .iter()
                        .find(|n| {
                            bid_set.contains(&n.innovation_number)
                                && matches!(n.node_type, NodeTypeDescriptor::MatMul)
                        })
                        .and_then(|mm| {
                            mm.parents.iter().find(|&&pid| {
                                bid_set.contains(&pid)
                                    && nodes
                                        .iter()
                                        .any(|n| n.innovation_number == pid && n.is_parameter())
                            })
                        })
                        .copied()
                };
                if let Some(wid) = w_id {
                    for node in genome.nodes_mut().iter_mut() {
                        if node.innovation_number == wid && node.output_shape.len() == 2 {
                            // W: [old_in, out] → [prev_out, out]
                            node.output_shape[0] = prev_out;
                            break;
                        }
                    }
                }
                dim_map.insert(block.output_id, out);
            }
            NodeBlockKind::Conv2d { out_channels, .. } => {
                let out = *out_channels;
                for node in genome.nodes_mut().iter_mut() {
                    if bid_set.contains(&node.innovation_number) && node.is_parameter() {
                        if node.output_shape.len() == 4 {
                            // kernel: [out_ch, old_in_ch, k, k] → [out_ch, prev_out, k, k]
                            node.output_shape[1] = prev_out;
                        }
                    }
                }
                dim_map.insert(block.output_id, out);
            }
            NodeBlockKind::Flatten => {
                // Flatten 后刷新形状，重新获取平坦维度
                sync_computation_shapes(genome);
                let analysis = GenomeAnalysis::compute(
                    genome.nodes(),
                    INPUT_INNOVATION,
                    genome_input_shape(genome),
                    genome_input_domain(genome),
                );
                if let Some(flat_dim) = block
                    .node_ids
                    .last()
                    .and_then(|&id| analysis.shape_of(id))
                    .and_then(|s| s.get(1).copied())
                {
                    dim_map.insert(block.output_id, flat_dim);
                }
            }
            // Pool2d、Activation、Dropout、SkipAgg 不改变通道/特征维度，透传 prev_out
            _ => {
                dim_map.insert(block.output_id, prev_out);
            }
        }
    }

    sync_computation_shapes(genome);
}

/// 修复所有参数节点的输入维度（插入/删除块后调用）
///
/// 先修复主路径参数维度，再修复跳跃连接形状一致性。
pub fn repair_param_input_dims(genome: &mut NetworkGenome) {
    repair_param_input_dims_inner(genome);
    repair_skip_connections(genome);
}

/// 获取 NodeLevel genome 的总参数量
pub fn node_param_count(genome: &NetworkGenome) -> usize {
    let input_shape = genome_input_shape(genome);
    let input_domain = genome_input_domain(genome);
    GenomeAnalysis::compute(genome.nodes(), INPUT_INNOVATION, input_shape, input_domain).param_count
}

// ==================== 插入操作 ====================

/// 在 `after_id` 后插入新节点组，并将主路径上引用 `after_id` 的下一个节点改为引用新输出
///
/// 返回新块的输出节点创新号。
pub fn insert_after(
    genome: &mut NetworkGenome,
    after_id: u64,
    new_nodes: Vec<NodeGene>,
) -> Result<u64, String> {
    if new_nodes.is_empty() {
        return Err("没有要插入的节点".into());
    }

    let new_node_ids: HashSet<u64> = new_nodes.iter().map(|n| n.innovation_number).collect();
    let new_output_id = new_nodes.last().unwrap().innovation_number;

    // 找到所有在主路径上、以 after_id 为父节点的节点
    let dependents: Vec<u64> = genome
        .nodes()
        .iter()
        .filter(|n| {
            n.enabled
                && !new_node_ids.contains(&n.innovation_number)
                && n.parents.contains(&after_id)
        })
        .map(|n| n.innovation_number)
        .collect();

    // 插入新节点
    genome.nodes_mut().extend(new_nodes);

    // 重新连线：将第一个依赖者的 after_id 父节点改为 new_output_id
    if let Some(&target_id) = dependents.first() {
        for node in genome.nodes_mut().iter_mut() {
            if node.innovation_number == target_id {
                for pid in node.parents.iter_mut() {
                    if *pid == after_id {
                        *pid = new_output_id;
                        break;
                    }
                }
                break;
            }
        }
    }

    Ok(new_output_id)
}

// ==================== 删除操作 ====================

/// 删除 block 内所有节点，并将引用 block.output_id 的下游节点改为引用 block.input_id
pub fn remove_block(genome: &mut NetworkGenome, block: &NodeBlock) {
    let bid_set: HashSet<u64> = block.node_ids.iter().copied().collect();
    let input_id = block.input_id;
    let output_id = block.output_id;

    // 重新连线：将引用 output_id 的下游节点改为引用 input_id
    for node in genome.nodes_mut().iter_mut() {
        if !bid_set.contains(&node.innovation_number) {
            for pid in node.parents.iter_mut() {
                if *pid == output_id {
                    *pid = input_id;
                }
            }
        }
    }

    // 删除块节点
    genome
        .nodes_mut()
        .retain(|n| !bid_set.contains(&n.innovation_number));

    // 清理被删参数节点的陈旧快照，防止 weight_snapshots 随演化代数无限膨胀。
    // 孤立快照不影响正确性（restore_weights 按 build.layer_params 匹配），
    // 但会导致序列化体积随代数增大，且阶段 6 格式迁移时需一并清理，趁早修复成本最低。
    if let GenomeRepr::NodeLevel {
        weight_snapshots, ..
    } = &mut genome.repr
    {
        for &id in &bid_set {
            weight_snapshots.remove(&id);
        }
    }
}

// ==================== 调整尺寸操作 ====================

/// 将 Linear 块的输出维度从旧值调整为 `new_out`（含级联到下一个 Linear 块）
pub fn resize_linear_out(
    genome: &mut NetworkGenome,
    block: &NodeBlock,
    new_out: usize,
) -> Result<(), String> {
    let _old_out = match &block.kind {
        NodeBlockKind::Linear { out_features } => *out_features,
        _ => return Err("不是 Linear 块".into()),
    };
    let bid_set: HashSet<u64> = block.node_ids.iter().copied().collect();

    // 更新本块的所有 2D 参数形状（W 和 b 都需要更新 dim[1] = new_out）
    for node in genome.nodes_mut().iter_mut() {
        if bid_set.contains(&node.innovation_number) && node.is_parameter() {
            if node.output_shape.len() == 2 {
                // W: [in, old_out] → [in, new_out]
                // b: [1, old_out] → [1, new_out]
                node.output_shape[1] = new_out;
            }
        }
    }

    // 用 repair_param_input_dims 替代手动级联——它会：
    // 1. 沿完整主路径修复所有 Linear/Conv2d 块的 W 输入维度（正确穿越 SkipAgg 节点）
    // 2. 调用 sync_computation_shapes 重新推导计算节点形状
    // 3. 调用 repair_skip_connections 修复跳跃连接形状一致性
    repair_param_input_dims(genome);
    Ok(())
}

/// 将 Conv2d 块的输出通道从旧值调整为 `new_ch`（含级联到下一个可调整块）
pub fn resize_conv2d_out(
    genome: &mut NetworkGenome,
    block: &NodeBlock,
    new_ch: usize,
) -> Result<(), String> {
    let old_ch = match &block.kind {
        NodeBlockKind::Conv2d { out_channels, .. } => *out_channels,
        _ => return Err("不是 Conv2d 块".into()),
    };
    let bid_set: HashSet<u64> = block.node_ids.iter().copied().collect();

    // 更新 kernel/bias 参数形状
    for node in genome.nodes_mut().iter_mut() {
        if bid_set.contains(&node.innovation_number) && node.is_parameter() {
            if node.output_shape.len() == 4 && node.output_shape[0] == old_ch {
                if node.output_shape[2] == 1 && node.output_shape[3] == 1 {
                    // bias: [1, old_ch, 1, 1] → [1, new_ch, 1, 1]
                    node.output_shape[1] = new_ch;
                } else {
                    // kernel: [old_ch, in_ch, kH, kW] → [new_ch, in_ch, kH, kW]
                    node.output_shape[0] = new_ch;
                }
            }
        }
    }

    // 用 repair_param_input_dims 替代手动级联——正确穿越 SkipAgg 等中间节点
    repair_param_input_dims(genome);
    Ok(())
}

// ==================== 变异决策辅助 ====================

/// 在 `after_id` 处创建新层节点（类型由域和约束决定），返回节点列表和使用的计数器
pub fn create_insert_nodes(
    genome: &NetworkGenome,
    after_id: u64,
    constraints: &SizeConstraints,
    rng: &mut StdRng,
    available_activations: &[ActivationType],
    adjacent_is_activation: bool,
) -> Option<Vec<NodeGene>> {
    let domain = node_domain_at(genome, after_id);
    let in_dim = node_out_dim_at(genome, after_id);
    let mut counter = make_counter(genome);
    let block_id = next_block_id(genome);

    let can_insert_act = !available_activations.is_empty() && !adjacent_is_activation;
    let is_spatial = domain == ShapeDomain::Spatial;
    let is_sequential = genome.seq_len.is_some();

    // 30% 概率插入激活（如果合法）
    if can_insert_act && rng.gen_bool(0.3) {
        let act = available_activations.choose(rng)?;
        return Some(expand_activation(
            after_id,
            vec![1, in_dim],
            act,
            &mut counter,
        ));
    }

    if is_spatial {
        let input_spatial = node_spatial_at(genome, after_id);
        if let Some(spatial) = input_spatial {
            let can_pool = spatial.0 >= 2 && spatial.1 >= 2;
            if can_pool && rng.gen_bool(0.15) {
                let pool_type = if rng.gen_bool(0.5) {
                    PoolType::Max
                } else {
                    PoolType::Avg
                };
                return Some(expand_pool2d(
                    after_id,
                    pool_type,
                    2,
                    2,
                    spatial,
                    in_dim,
                    &mut counter,
                ));
            }
            // 插入 Conv2d
            let effective_min = constraints.min_hidden_size.max(8);
            let out_ch_cap = (in_dim * 16)
                .max(64)
                .min(constraints.max_hidden_size)
                .max(effective_min);
            let out_ch = sample_size_in_range_simple(
                effective_min,
                out_ch_cap,
                &constraints.size_strategy,
                rng,
            );
            let k = *[1usize, 3, 5, 7].choose(rng)?;
            return Some(expand_conv2d(
                after_id,
                in_dim,
                out_ch,
                k,
                spatial,
                block_id,
                &mut counter,
            ));
        }
    }

    if is_sequential {
        // 序列域：这里简化，返回 None（让 is_applicable 处过滤掉）
        return None;
    }

    // 默认：插入 Linear
    let effective_min = constraints.min_hidden_size.max(8);
    let size_cap = in_dim
        .min(256)
        .max(effective_min * 2)
        .min(constraints.max_hidden_size)
        .max(effective_min);
    let size =
        sample_size_in_range_simple(effective_min, size_cap, &constraints.size_strategy, rng);
    Some(expand_linear(
        after_id,
        in_dim,
        size,
        block_id,
        &mut counter,
    ))
}

/// 简化版 sample_size_in_range（避免循环依赖）
fn sample_size_in_range_simple(
    min: usize,
    max: usize,
    strategy: &crate::nn::evolution::mutation::SizeStrategy,
    rng: &mut StdRng,
) -> usize {
    use crate::nn::evolution::mutation::SizeStrategy;
    let min = min.min(max);
    match strategy {
        SizeStrategy::Free => rng.gen_range(min..=max),
        SizeStrategy::AlignTo(align) => {
            let start = min.div_ceil(*align) * align;
            if start > max {
                return max;
            }
            let end = max / align * align;
            let candidates: Vec<usize> = (start..=end).step_by(*align).collect();
            candidates.choose(rng).copied().unwrap_or(max)
        }
    }
}

// ==================== 阶段 7：跨层连接变异辅助 ====================

/// 一个可合法添加跳跃连接的候选对
#[derive(Debug, Clone)]
pub struct ConnectablePair {
    /// 源节点输出创新号
    pub from_id: u64,
    /// 目标块入口节点创新号（其主路径父节点将被替换为新 Add 聚合节点）
    pub target_entry_id: u64,
    /// 源节点输出形状
    pub from_shape: Vec<usize>,
    /// 目标块主路径输入形状（投影目标形状）
    pub to_shape: Vec<usize>,
    /// 连接所在的形状域
    pub domain: ShapeDomain,
}

/// 找目标块的入口节点：块内第一个非叶节点，且其父节点列表中包含 `block.input_id`
fn find_entry_node(block: &NodeBlock, genome: &NetworkGenome) -> Option<u64> {
    for &nid in &block.node_ids {
        if let Some(node) = genome.nodes().iter().find(|n| n.innovation_number == nid) {
            if !node.is_leaf() && node.parents.contains(&block.input_id) {
                return Some(nid);
            }
        }
    }
    None
}

/// 判断一个块是否是跳跃投影块（其输出节点仅被 SkipAgg 节点引用）
///
/// 跳跃投影块由 `add_skip_connection` 插入，用于在形状不兼容时将跳跃源投影到目标形状。
/// 这类块不应参与 Grow/Shrink/Remove 变异，以避免投影与聚合节点形状失配。
pub fn is_skip_projection_block(genome: &NetworkGenome, block: &NodeBlock) -> bool {
    let output_id = block.output_id;
    let param_ids: HashSet<u64> = genome
        .nodes()
        .iter()
        .filter(|n| n.is_parameter())
        .map(|n| n.innovation_number)
        .collect();
    // 找所有以 output_id 为父节点的下游节点
    let downstream: Vec<&NodeGene> = genome
        .nodes()
        .iter()
        .filter(|n| n.parents.contains(&output_id))
        .collect();
    // 必须有下游，且所有下游都是 block_id=None 的 Add/Maximum（SkipAgg）
    if downstream.is_empty() {
        return false;
    }
    if !downstream.iter().all(|n| {
        n.block_id.is_none()
            && matches!(
                n.node_type,
                NodeTypeDescriptor::Add | NodeTypeDescriptor::Maximum
            )
    }) {
        return false;
    }
    // 关键区分：投影块的输出必须是 SkipAgg 的跳跃分支（第二个非参数父节点），
    // 而不是主分支（第一个非参数父节点）。主路径块的输出是 SkipAgg 的第一个非参数父节点。
    downstream.iter().all(|agg| {
        let non_param_parents: Vec<u64> = agg
            .parents
            .iter()
            .filter(|&&p| !param_ids.contains(&p))
            .copied()
            .collect();
        // output_id 不能是第一个非参数父节点（那是主路径输入）
        non_param_parents.len() >= 2 && non_param_parents[0] != output_id
    })
}

/// 查找所有可合法添加跳跃连接的候选对
///
/// 仅适用于 NodeLevel 且非序列模式的基因组。
/// 候选对满足以下所有条件：
/// 1. `from_id` 在拓扑序上严格早于目标块（保证 DAG 有效性）
/// 2. `from_id` 不是目标块的直接前驱（避免冗余主路径连接）
/// 3. `from_id` 不已是目标入口节点的直接非参数父节点（避免重复连边）
/// 4. 两者所在域相同（Flat 或 Spatial，不跨域连接）
/// 5. 形状相同（可直接 Add）或可投影（Flat: Linear；Spatial: 同 H/W 不同 channels 的 1×1 Conv2d）
pub fn find_connectable_pairs(genome: &NetworkGenome) -> Vec<ConnectablePair> {
    if !genome.is_node_level() || genome.seq_len.is_some() {
        return vec![];
    }

    let nodes = genome.nodes();
    if nodes.is_empty() {
        return vec![];
    }

    let input_shape = genome_input_shape(genome);
    let input_domain = genome_input_domain(genome);
    let analysis = GenomeAnalysis::compute(nodes, INPUT_INNOVATION, input_shape, input_domain);

    if !analysis.is_valid {
        return vec![];
    }

    let param_ids: HashSet<u64> = nodes
        .iter()
        .filter(|n| n.is_parameter())
        .map(|n| n.innovation_number)
        .collect();

    let blocks = node_main_path(genome);
    if blocks.len() < 2 {
        return vec![];
    }

    let mut pairs = Vec::new();

    for tgt_idx in 1..blocks.len() {
        let tgt = &blocks[tgt_idx];

        // 找目标块的入口节点
        let entry_id = match find_entry_node(tgt, genome) {
            Some(id) => id,
            None => continue,
        };

        // 目标块主路径输入形状（即 tgt.input_id 的输出形状）
        let to_shape = match analysis.shape_of(tgt.input_id) {
            Some(s) => s.clone(),
            None => continue,
        };
        let to_domain = match analysis.domain_of(tgt.input_id) {
            Some(d) => d,
            None => continue,
        };

        // 序列域不支持阶段 7 跳跃连接（等 RNN 节点级化后再开放）
        if to_domain == ShapeDomain::Sequence {
            continue;
        }

        // 入口节点当前的直接非参数父节点集合（用于避免重复添加同一来源）
        let entry_non_param_parents: HashSet<u64> = nodes
            .iter()
            .find(|n| n.innovation_number == entry_id)
            .map(|n| {
                n.parents
                    .iter()
                    .filter(|&&p| !param_ids.contains(&p))
                    .copied()
                    .collect::<HashSet<u64>>()
            })
            .unwrap_or_default();

        // 直接前驱块的输出（= tgt.input_id，是主路径的直接前驱）
        let immediate_pred_id = blocks[tgt_idx - 1].output_id;

        // 候选源：INPUT_INNOVATION + 所有位置在直接前驱之前的块输出
        let mut sources: Vec<u64> = Vec::new();

        // INPUT_INNOVATION（当直接前驱不是 INPUT 时才加入）
        if immediate_pred_id != INPUT_INNOVATION {
            sources.push(INPUT_INNOVATION);
        }

        // 索引 0..tgt_idx-1 的块输出（全部非直接前驱）
        for src_idx in 0..tgt_idx.saturating_sub(1) {
            let src_out = blocks[src_idx].output_id;
            if src_out != immediate_pred_id {
                sources.push(src_out);
            }
        }

        for from_id in sources {
            // 已是入口节点的直接非参数父节点 → 跳过（避免重复连接）
            if entry_non_param_parents.contains(&from_id) {
                continue;
            }

            let from_shape = match analysis.shape_of(from_id) {
                Some(s) => s.clone(),
                None => continue,
            };
            let from_domain = match analysis.domain_of(from_id) {
                Some(d) => d,
                None => continue,
            };

            // 域必须相同
            if from_domain != to_domain {
                continue;
            }

            let shapes_match = from_shape == to_shape;
            let can_project = !shapes_match
                && match to_domain {
                    ShapeDomain::Flat => {
                        // Flat: [1, F] → [1, T]，任意维度均可 Linear 投影
                        from_shape.len() == 2 && to_shape.len() == 2
                    }
                    ShapeDomain::Spatial => {
                        // Spatial: H/W 相同但 channels 不同，可用 1×1 Conv2d 投影
                        from_shape.len() == 4
                            && to_shape.len() == 4
                            && from_shape[2] == to_shape[2]
                            && from_shape[3] == to_shape[3]
                    }
                    ShapeDomain::Sequence => false,
                };

            if shapes_match || can_project {
                pairs.push(ConnectablePair {
                    from_id,
                    target_entry_id: entry_id,
                    from_shape,
                    to_shape: to_shape.clone(),
                    domain: to_domain,
                });
            }
        }
    }

    pairs
}

/// 在基因组中插入一条跳跃连接：`pair.from_id` → 目标块入口节点的主路径输入
///
/// 若形状不兼容，先插入投影块（Flat 域: Linear；Spatial 域: 1×1 Conv2d），
/// 再插入 Add 聚合节点。返回新 Add 聚合节点的创新号。
///
/// **调用方须在返回后调用 `commit_counter(genome, &counter)` 以同步计数器。**
pub fn add_skip_connection(
    genome: &mut NetworkGenome,
    pair: &ConnectablePair,
    counter: &mut InnovationCounter,
) -> Result<u64, String> {
    // ── 可选：投影块（形状不兼容时）──────────────────────────────
    let proj_output_id = if pair.from_shape == pair.to_shape {
        pair.from_id
    } else {
        let block_id = next_block_id(genome);
        let proj_nodes = match pair.domain {
            ShapeDomain::Flat => {
                let in_dim = *pair.from_shape.get(1).ok_or("from_shape 维度不足")?;
                let out_dim = *pair.to_shape.get(1).ok_or("to_shape 维度不足")?;
                expand_linear(pair.from_id, in_dim, out_dim, block_id, counter)
            }
            ShapeDomain::Spatial => {
                let in_ch = *pair.from_shape.get(1).ok_or("from_shape 维度不足")?;
                let out_ch = *pair.to_shape.get(1).ok_or("to_shape 维度不足")?;
                let h = *pair.from_shape.get(2).ok_or("from_shape 维度不足")?;
                let w = *pair.from_shape.get(3).ok_or("from_shape 维度不足")?;
                // kernel_size=1 的 Conv2d：仅改变通道数，不改变 H/W
                expand_conv2d(pair.from_id, in_ch, out_ch, 1, (h, w), block_id, counter)
            }
            ShapeDomain::Sequence => {
                return Err("序列域不支持跳跃连接".into());
            }
        };
        let proj_out = proj_nodes
            .last()
            .ok_or("投影块展开结果为空")?
            .innovation_number;
        genome.nodes_mut().extend(proj_nodes);
        proj_out
    };

    // ── 找入口节点的主路径父节点（将被 Add 聚合节点替代）────────────
    let param_ids: HashSet<u64> = genome
        .nodes()
        .iter()
        .filter(|n| n.is_parameter())
        .map(|n| n.innovation_number)
        .collect();

    let entry_main_parent = genome
        .nodes()
        .iter()
        .find(|n| n.innovation_number == pair.target_entry_id)
        .and_then(|n| {
            n.parents
                .iter()
                .find(|&&p| !param_ids.contains(&p))
                .copied()
        })
        .ok_or_else(|| {
            format!(
                "目标入口节点 {} 没有非参数父节点",
                pair.target_entry_id
            )
        })?;

    // ── 插入 Add 聚合节点 ─────────────────────────────────────────
    let agg_id = counter.next();
    genome.nodes_mut().push(NodeGene::new(
        agg_id,
        NodeTypeDescriptor::Add,
        pair.to_shape.clone(),
        vec![entry_main_parent, proj_output_id],
        None, // 独立跳跃聚合节点，不属于任何模板组
    ));

    // ── 更新入口节点：用 agg_id 替换主路径父节点 ─────────────────
    for node in genome.nodes_mut().iter_mut() {
        if node.innovation_number == pair.target_entry_id {
            for p in node.parents.iter_mut() {
                if *p == entry_main_parent {
                    *p = agg_id;
                    break;
                }
            }
            break;
        }
    }

    sync_computation_shapes(genome);
    Ok(agg_id)
}

/// 查找所有可移除的跳跃聚合节点（由 `add_skip_connection` 插入的独立 Add 节点）
///
/// 满足以下条件的节点视为可移除的跳跃聚合节点：
/// - 类型为 `Add` 或 `Maximum`
/// - `block_id = None`（非模板组节点）
/// - 具有 ≥2 个非参数父节点
pub fn find_removable_skip_connections(genome: &NetworkGenome) -> Vec<u64> {
    if !genome.is_node_level() {
        return vec![];
    }

    let param_ids: HashSet<u64> = genome
        .nodes()
        .iter()
        .filter(|n| n.is_parameter())
        .map(|n| n.innovation_number)
        .collect();

    genome
        .nodes()
        .iter()
        .filter(|n| {
            n.enabled
                && n.block_id.is_none()
                && matches!(
                    n.node_type,
                    NodeTypeDescriptor::Add | NodeTypeDescriptor::Maximum
                )
                && n.parents
                    .iter()
                    .filter(|&&p| !param_ids.contains(&p))
                    .count()
                    >= 2
        })
        .map(|n| n.innovation_number)
        .collect()
}

/// 移除一个跳跃聚合节点，将下游重新连线到主路径输入（第一个非参数父节点）
///
/// 移除后调用 `cleanup_orphan_nodes` 清理孤立的投影节点（若有）。
pub fn remove_skip_connection(genome: &mut NetworkGenome, agg_id: u64) -> Result<(), String> {
    let param_ids: HashSet<u64> = genome
        .nodes()
        .iter()
        .filter(|n| n.is_parameter())
        .map(|n| n.innovation_number)
        .collect();

    let agg_node = genome
        .nodes()
        .iter()
        .find(|n| n.innovation_number == agg_id)
        .ok_or_else(|| format!("聚合节点 {} 不存在", agg_id))?
        .clone();

    if !matches!(
        agg_node.node_type,
        NodeTypeDescriptor::Add | NodeTypeDescriptor::Maximum
    ) {
        return Err(format!(
            "节点 {} 类型不是 Add/Maximum，无法作为跳跃聚合节点移除",
            agg_id
        ));
    }

    let non_param_parents: Vec<u64> = agg_node
        .parents
        .iter()
        .filter(|&&p| !param_ids.contains(&p))
        .copied()
        .collect();

    if non_param_parents.len() < 2 {
        return Err(format!(
            "聚合节点 {} 的非参数父节点不足 2 个，无法移除跳跃连接",
            agg_id
        ));
    }

    // 保留第一个非参数父节点（视为主路径输入）
    let main_input = non_param_parents[0];

    // 将所有引用 agg_id 的节点改为引用 main_input
    for node in genome.nodes_mut().iter_mut() {
        for p in node.parents.iter_mut() {
            if *p == agg_id {
                *p = main_input;
            }
        }
    }

    // 删除聚合节点
    genome
        .nodes_mut()
        .retain(|n| n.innovation_number != agg_id);

    // 清理孤立节点（投影块等），同步形状
    cleanup_orphan_nodes(genome);
    sync_computation_shapes(genome);
    Ok(())
}

/// 清理孤立节点（不在主路径输出到 INPUT_INNOVATION 的可达路径上的节点）
///
/// 从主路径最后一个块的输出节点出发，向上 BFS 收集全部可达节点集合，
/// 删除不可达的孤立节点及其权重快照。
///
/// **设计要点**：不能用"所有无下游节点"作为末端种子——当跳跃连接被移除后，
/// 投影块尾节点也变成无下游，会被误判为合法末端导致无法清理。
/// 因此固定用主路径真正的输出节点作为唯一末端种子。
pub fn cleanup_orphan_nodes(genome: &mut NetworkGenome) {
    let nodes = genome.nodes();
    if nodes.is_empty() {
        return;
    }

    // 用主路径最后一个块的输出节点作为唯一合法末端
    let blocks = node_main_path(genome);
    let terminal_ids: Vec<u64> = if let Some(last_block) = blocks.last() {
        vec![last_block.output_id]
    } else {
        // 无主路径块时退化为旧逻辑（保底）
        let all_parent_ids: HashSet<u64> = nodes
            .iter()
            .flat_map(|n| n.parents.iter().copied())
            .collect();
        nodes
            .iter()
            .filter(|n| n.enabled && !all_parent_ids.contains(&n.innovation_number))
            .map(|n| n.innovation_number)
            .collect()
    };

    if terminal_ids.is_empty() {
        return;
    }

    // 从末端节点向上 BFS，收集所有可达节点（含虚拟输入 INPUT_INNOVATION）
    let nodes = genome.nodes(); // 重新借用（上面 node_main_path 可能触发不可变访问）
    let mut reachable: HashSet<u64> = HashSet::new();
    reachable.insert(INPUT_INNOVATION);
    let mut queue: VecDeque<u64> = terminal_ids.into_iter().collect();

    while let Some(id) = queue.pop_front() {
        if reachable.insert(id) {
            if let Some(node) = nodes.iter().find(|n| n.innovation_number == id) {
                for &pid in &node.parents {
                    queue.push_back(pid);
                }
            }
        }
    }

    // 删除不可达节点
    genome
        .nodes_mut()
        .retain(|n| reachable.contains(&n.innovation_number));

    // 同步清理不可达节点的权重快照
    if let GenomeRepr::NodeLevel {
        weight_snapshots, ..
    } = &mut genome.repr
    {
        weight_snapshots.retain(|k, _| reachable.contains(k));
    }
}

/// 修复所有跳跃连接的形状一致性（后置修复器）
///
/// 当 Grow/Shrink/Insert/Remove 等变异改变了主路径块的输出维度后，
/// 跳跃连接处的 Add 聚合节点的两个输入形状可能不一致。
///
/// 本函数遍历所有 `block_id=None` 的 Add/Maximum 聚合节点，检查其非参数父节点
/// 的形状是否一致。对于不一致的情况：
/// - 如果跳跃分支经过投影块，更新投影块的参数形状以匹配主路径输入形状
/// - 如果跳跃分支是直连且形状不兼容，删除该跳跃连接
///
/// **必须在 `sync_computation_shapes` 之后调用。**
pub fn repair_skip_connections(genome: &mut NetworkGenome) {
    if !genome.is_node_level() {
        return;
    }

    let mut agg_ids = find_removable_skip_connections(genome);
    if agg_ids.is_empty() {
        return;
    }

    let input_shape = genome_input_shape(genome);
    let input_domain = genome_input_domain(genome);

    // 按拓扑序排序 agg_ids（上游 SkipAgg 先修复），避免嵌套 SkipAgg 依赖时
    // 下游节点因上游尚未修复而无法推导 main_shape 导致被跳过（Bug M）。
    {
        let analysis =
            GenomeAnalysis::compute(genome.nodes(), INPUT_INNOVATION, input_shape.clone(), input_domain);
        let topo_pos: HashMap<u64, usize> = analysis
            .topo_order
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();
        agg_ids.sort_by_key(|&id| topo_pos.get(&id).copied().unwrap_or(usize::MAX));
    }

    // 收集需要删除的聚合节点（形状不兼容且无法修复的）
    let mut to_remove: Vec<u64> = Vec::new();

    for &agg_id in &agg_ids {
        // 重新分析（每轮修复后形状可能变）
        let analysis =
            GenomeAnalysis::compute(genome.nodes(), INPUT_INNOVATION, input_shape.clone(), input_domain);

        let agg_node = match genome.nodes().iter().find(|n| n.innovation_number == agg_id) {
            Some(n) => n.clone(),
            None => continue,
        };

        let param_ids: HashSet<u64> = genome
            .nodes()
            .iter()
            .filter(|n| n.is_parameter())
            .map(|n| n.innovation_number)
            .collect();

        let non_param_parents: Vec<u64> = agg_node
            .parents
            .iter()
            .filter(|&&p| !param_ids.contains(&p))
            .copied()
            .collect();

        if non_param_parents.len() < 2 {
            continue;
        }

        // 主路径输入 = 第一个非参数父节点，跳跃分支 = 第二个
        let main_parent = non_param_parents[0];
        let skip_parent = non_param_parents[1];

        let main_shape = match analysis.shape_of(main_parent) {
            Some(s) => s.clone(),
            None => continue,
        };
        // skip_shape 可能为 None（投影块参数过时导致 analysis 推导失败）
        let skip_shape = analysis.shape_of(skip_parent).cloned();

        // 尝试找跳跃分支的投影块（用原始节点信息，不依赖 analysis）
        let skip_node = genome.nodes().iter().find(|n| n.innovation_number == skip_parent).cloned();
        // 判断跳跃分支是否经过投影块。投影块的两个必要条件：
        // 1. block 内包含 MatMul 或 Conv2d 计算节点（是 Linear/Conv2d 块）
        // 2. skip_parent（block 输出节点）的所有下游都是 SkipAgg 节点
        //    （排除主路径上普通的 Linear/Conv2d 块被误判为投影块）
        let is_projection = skip_node.as_ref().map_or(false, |n| {
            if let Some(bid) = n.block_id {
                let has_computation = genome.nodes().iter().any(|sibling| {
                    sibling.block_id == Some(bid)
                        && matches!(
                            sibling.node_type,
                            NodeTypeDescriptor::MatMul | NodeTypeDescriptor::Conv2d { .. }
                        )
                });
                if !has_computation {
                    return false;
                }
                // skip_parent 的所有下游必须都是 SkipAgg（block_id=None 的 Add/Maximum）
                let sp_id = n.innovation_number;
                let downstream: Vec<&NodeGene> = genome
                    .nodes()
                    .iter()
                    .filter(|dn| dn.parents.contains(&sp_id))
                    .collect();
                !downstream.is_empty()
                    && downstream.iter().all(|dn| {
                        dn.block_id.is_none()
                            && matches!(
                                dn.node_type,
                                NodeTypeDescriptor::Add | NodeTypeDescriptor::Maximum
                            )
                    })
            } else {
                false
            }
        });

        if is_projection {
            // 始终修复投影块参数（不依赖 skip_shape，因其可能因参数过时而推导失败）
            // 找到投影块，更新其参数形状（同时修复输入维和输出维）
            let proj_bid = skip_node.as_ref().unwrap().block_id;
            let target_shape = &main_shape;

            // 找投影块的外部输入节点（= 跳跃源），以获取其真实输出形状
            let proj_block_ids: HashSet<u64> = genome
                .nodes()
                .iter()
                .filter(|n| n.block_id == proj_bid)
                .map(|n| n.innovation_number)
                .collect();

            let source_shape: Option<Vec<usize>> = genome
                .nodes()
                .iter()
                .filter(|n| proj_block_ids.contains(&n.innovation_number) && !n.is_leaf())
                .flat_map(|n| n.parents.iter())
                .find(|&&pid| !proj_block_ids.contains(&pid))
                .and_then(|&src_id| analysis.shape_of(src_id).cloned());

            // 更新投影块中所有参数节点的形状
            for node in genome.nodes_mut().iter_mut() {
                if node.block_id == proj_bid && node.is_parameter() {
                    match (node.output_shape.len(), target_shape.len()) {
                        (2, 2) => {
                            if node.output_shape[0] == 1 {
                                // bias: [1, old_out] → [1, new_out]
                                node.output_shape[1] = target_shape[1];
                            } else {
                                // weight W: [old_in, old_out] → [src_dim, target_dim]
                                if let Some(ref src) = source_shape {
                                    node.output_shape[0] = src[1]; // 输入维 = 跳跃源特征维
                                }
                                node.output_shape[1] = target_shape[1]; // 输出维 = 主路径特征维
                            }
                        }
                        (4, 4) => {
                            // Conv2d 1x1 projection kernel: [out_ch, in_ch, 1, 1]
                            if node.output_shape[2] == 1 && node.output_shape[3] == 1 {
                                node.output_shape[0] = target_shape[1]; // out_ch = 目标通道
                                if let Some(ref src) = source_shape {
                                    node.output_shape[1] = src[1]; // in_ch = 源通道
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }

            // 更新聚合节点的输出形状
            for node in genome.nodes_mut().iter_mut() {
                if node.innovation_number == agg_id {
                    node.output_shape = main_shape.clone();
                    break;
                }
            }
        } else if skip_shape.as_ref() != Some(&main_shape) {
            // 直连跳跃，形状不兼容（或推导失败）且无投影块 → 标记删除
            to_remove.push(agg_id);
        } else {
            // 直连跳跃，形状一致 → 更新聚合节点输出形状
            for node in genome.nodes_mut().iter_mut() {
                if node.innovation_number == agg_id {
                    node.output_shape = main_shape.clone();
                    break;
                }
            }
        }
    }

    // 删除无法修复的跳跃连接
    let mut removed_any = !to_remove.is_empty();
    for agg_id in to_remove {
        let _ = remove_skip_connection(genome, agg_id);
    }

    // 最终验证：修复一个投影可能破坏共享同一投影块的另一个 SkipAgg（Bug N），
    // 因此做一轮终态检查，删除仍然形状不匹配的 SkipAgg。
    {
        sync_computation_shapes(genome);
        let analysis = GenomeAnalysis::compute(
            genome.nodes(),
            INPUT_INNOVATION,
            input_shape,
            input_domain,
        );
        let param_ids: HashSet<u64> = genome
            .nodes()
            .iter()
            .filter(|n| n.is_parameter())
            .map(|n| n.innovation_number)
            .collect();
        let remaining_aggs = find_removable_skip_connections(genome);
        for agg_id in remaining_aggs {
            let agg = match genome.nodes().iter().find(|n| n.innovation_number == agg_id) {
                Some(n) => n.clone(),
                None => continue,
            };
            let npp: Vec<u64> = agg
                .parents
                .iter()
                .filter(|&&p| !param_ids.contains(&p))
                .copied()
                .collect();
            if npp.len() < 2 {
                continue;
            }
            let s0 = analysis.shape_of(npp[0]);
            let s1 = analysis.shape_of(npp[1]);
            if s0 != s1 {
                let _ = remove_skip_connection(genome, agg_id);
                removed_any = true;
            }
        }
    }

    // 移除跳跃连接后主路径拓扑可能改变，需重新修复参数维度链
    if removed_any {
        repair_param_input_dims_inner(genome);
    }

    // 最终同步形状
    sync_computation_shapes(genome);
}

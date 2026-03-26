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

use std::collections::{HashMap, HashSet};

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
        for &id in node_ids {
            if let Some(n) = node_map.get(&id) {
                // W 参数：2D，第一维 > 1（= in_dim，区别于 b 的第一维 = 1）
                if n.is_parameter() && n.output_shape.len() == 2 && n.output_shape[0] > 1 {
                    return NodeBlockKind::Linear {
                        out_features: n.output_shape[1],
                    };
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

/// 修复所有参数节点的输入维度（插入/删除块后调用）
///
/// 沿主路径正向推导，将每个 Linear/Conv2d 块的参数节点输入维度
/// 更新为前驱块的实际输出维度，最后调用 `sync_computation_shapes`。
pub fn repair_param_input_dims(genome: &mut NetworkGenome) {
    let blocks = node_main_path(genome);
    let mut prev_out = if genome.input_spatial.is_some() {
        genome.input_dim // in_channels
    } else {
        genome.input_dim // flat features
    };

    for block in &blocks {
        let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();
        match &block.kind {
            NodeBlockKind::Linear { out_features } => {
                let out = *out_features;
                for node in genome.nodes_mut().iter_mut() {
                    if bid_set.contains(&node.innovation_number) && node.is_parameter() {
                        if node.output_shape.len() == 2 && node.output_shape[0] > 1 {
                            // W: [old_in, out] → [prev_out, out]
                            node.output_shape[0] = prev_out;
                        }
                    }
                }
                prev_out = out;
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
                prev_out = out;
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
                    prev_out = flat_dim;
                }
            }
            // Pool2d、Activation、Dropout 不改变通道/特征维度
            _ => {}
        }
    }

    sync_computation_shapes(genome);
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
    let old_out = match &block.kind {
        NodeBlockKind::Linear { out_features } => *out_features,
        _ => return Err("不是 Linear 块".into()),
    };
    let bid_set: HashSet<u64> = block.node_ids.iter().copied().collect();

    // 更新本块的 W 和 b 参数形状
    for node in genome.nodes_mut().iter_mut() {
        if bid_set.contains(&node.innovation_number) && node.is_parameter() {
            match node.output_shape.len() {
                2 if node.output_shape[0] > 1 => {
                    // W: [in, old_out] → [in, new_out]
                    node.output_shape[1] = new_out;
                }
                2 if node.output_shape[0] == 1 => {
                    // b: [1, old_out] → [1, new_out]
                    node.output_shape[1] = new_out;
                }
                _ => {}
            }
        }
    }

    // 级联：更新下一个 Linear 块的 W 的第一维
    let blocks = node_main_path(genome);
    if let Some(idx) = blocks.iter().position(|b| b.block_id == block.block_id) {
        for next in blocks[idx + 1..].iter() {
            match &next.kind {
                NodeBlockKind::Linear { .. } => {
                    let next_bid_set: HashSet<u64> = next.node_ids.iter().copied().collect();
                    for node in genome.nodes_mut().iter_mut() {
                        if next_bid_set.contains(&node.innovation_number) && node.is_parameter() {
                            if node.output_shape.len() == 2 && node.output_shape[0] == old_out {
                                // W: [old_out, next_out] → [new_out, next_out]
                                node.output_shape[0] = new_out;
                                break;
                            }
                        }
                    }
                    break; // 只级联到紧邻的下一个 Linear
                }
                NodeBlockKind::Activation { .. } | NodeBlockKind::Dropout { .. } => continue,
                // Flatten 后尺寸计算方式不同，停止级联
                _ => break,
            }
        }
    }

    // 重新推导计算节点形状
    sync_computation_shapes(genome);
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

    // 级联：找下一个接受通道维度的块（跳过 Pool2d，在 Flatten 后停止）
    let blocks = node_main_path(genome);
    if let Some(idx) = blocks.iter().position(|b| b.block_id == block.block_id) {
        for next in blocks[idx + 1..].iter() {
            match &next.kind {
                NodeBlockKind::Conv2d { .. } => {
                    // 下一个 Conv2d 的 kernel 第二维 = in_ch
                    let next_bid_set: HashSet<u64> = next.node_ids.iter().copied().collect();
                    for node in genome.nodes_mut().iter_mut() {
                        if next_bid_set.contains(&node.innovation_number) && node.is_parameter() {
                            if node.output_shape.len() == 4 && node.output_shape[1] == old_ch {
                                node.output_shape[1] = new_ch;
                                break;
                            }
                        }
                    }
                    break;
                }
                NodeBlockKind::Pool2d { .. } | NodeBlockKind::Activation { .. } => continue,
                _ => break,
            }
        }
    }

    sync_computation_shapes(genome);
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

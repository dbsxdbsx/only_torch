/*
 * @Author       : 老董
 * @Date         : 2026-04-19
 * @Description  : Feature Map (FM) 粒度演化的辅助数据结构和查询函数
 *
 * FM 表示将 Spatial 域的卷积处理分解为：
 * - FM 节点（fm_id != None）：代表一个独立 feature map
 * - FM 边（fm_id = None, block_id 配对 kernel + op）：代表两个 FM 之间的连接
 *
 * 此模块提供 FM 子图的查询、分析和操作原语，
 * 供 builder、mutation、migration 使用。
 */

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::node_gene::NodeGene;

// ==================== FM 信息结构 ====================

/// 单个 Feature Map 的元信息
#[derive(Debug, Clone)]
pub struct FMNodeInfo {
    /// fm_id（全局唯一）
    pub fm_id: u64,
    /// 该 FM 中所有节点的创新号（聚合节点、激活节点等）
    pub node_ids: Vec<u64>,
    /// FM 的空间尺寸 (H, W)
    pub spatial_size: (usize, usize),
    /// FM 的输出节点创新号（FM 内拓扑序的最后一个计算节点）
    pub output_node_id: u64,
}

/// FM 边（两个 FM 之间的一条连接）的元信息
#[derive(Debug, Clone)]
pub struct FMEdgeInfo {
    /// 边的 block_id（kernel + op 共享同一 block_id）
    pub block_id: u64,
    /// 源 FM 的 fm_id
    pub src_fm_id: u64,
    /// 目标 FM 的 fm_id
    pub dst_fm_id: u64,
    /// op 节点的创新号
    pub op_node_id: u64,
    /// kernel Parameter 节点的创新号（Pool2d 类型为 None）
    pub kernel_node_id: Option<u64>,
    /// 边的操作类型
    pub edge_type: FMEdgeType,
}

/// FM 边操作类型
#[derive(Debug, Clone, PartialEq)]
pub enum FMEdgeType {
    Conv2d {
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    },
    ConvTranspose2d {
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    },
    MaxPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
    AvgPool2d {
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },
}

impl FMEdgeType {
    /// 从 NodeTypeDescriptor 推导 FM 边类型
    pub fn from_descriptor(desc: &NodeTypeDescriptor) -> Option<Self> {
        match desc {
            NodeTypeDescriptor::Conv2d { stride, padding, dilation } => Some(FMEdgeType::Conv2d {
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
            }),
            NodeTypeDescriptor::ConvTranspose2d { stride, padding, output_padding } => {
                Some(FMEdgeType::ConvTranspose2d {
                    stride: *stride,
                    padding: *padding,
                    output_padding: *output_padding,
                })
            }
            NodeTypeDescriptor::MaxPool2d { kernel_size, stride } => {
                Some(FMEdgeType::MaxPool2d {
                    kernel_size: *kernel_size,
                    stride: *stride,
                })
            }
            NodeTypeDescriptor::AvgPool2d { kernel_size, stride } => {
                Some(FMEdgeType::AvgPool2d {
                    kernel_size: *kernel_size,
                    stride: *stride,
                })
            }
            _ => None,
        }
    }

    /// 是否有可学习参数（Pool2d 无参数）
    pub fn has_learnable_params(&self) -> bool {
        matches!(self, FMEdgeType::Conv2d { .. } | FMEdgeType::ConvTranspose2d { .. })
    }
}

// ==================== FM 子图分析 ====================

/// FM 子图的完整分析结果
#[derive(Debug)]
pub struct FMSubgraphAnalysis {
    /// 所有 FM 节点信息（按 fm_id 排序）
    pub fm_nodes: BTreeMap<u64, FMNodeInfo>,
    /// 所有 FM 边信息
    pub fm_edges: Vec<FMEdgeInfo>,
    /// 输入 FM 的 fm_id 集合（无上游 FM 边的 FM）
    pub input_fm_ids: HashSet<u64>,
    /// 输出 FM 的 fm_id 集合（无下游 FM 边的 FM）
    pub output_fm_ids: HashSet<u64>,
}

/// 分析基因组中的 FM 子图
///
/// 扫描所有带 `fm_id` 的节点和所有 FM 边（通过 block_id 和 op 类型匹配），
/// 构建完整的 FM 拓扑视图。
pub fn analyze_fm_subgraph(nodes: &[NodeGene]) -> FMSubgraphAnalysis {
    let node_map: HashMap<u64, &NodeGene> = nodes
        .iter()
        .filter(|n| n.enabled)
        .map(|n| (n.innovation_number, n))
        .collect();

    // 1. 收集所有 FM 节点（按 fm_id 分组）
    let mut fm_nodes_map: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
    for n in nodes.iter().filter(|n| n.enabled) {
        if let Some(fid) = n.fm_id {
            fm_nodes_map.entry(fid).or_default().push(n.innovation_number);
        }
    }

    // 2. 构建 FM 节点信息
    let mut fm_nodes: BTreeMap<u64, FMNodeInfo> = BTreeMap::new();
    for (&fid, ids) in &fm_nodes_map {
        let output_node_id = find_fm_output_node(ids, &node_map);
        let spatial_size = infer_fm_spatial_size(output_node_id, &node_map);

        fm_nodes.insert(fid, FMNodeInfo {
            fm_id: fid,
            node_ids: ids.clone(),
            spatial_size,
            output_node_id,
        });
    }

    // 3. 收集所有 FM 边（识别连接两个 FM 的 conv/pool/deconv 操作）
    let fm_edges = find_fm_edges(nodes, &fm_nodes, &node_map);

    // 4. 确定输入/输出 FM
    let mut has_incoming: HashSet<u64> = HashSet::new();
    let mut has_outgoing: HashSet<u64> = HashSet::new();
    for edge in &fm_edges {
        has_incoming.insert(edge.dst_fm_id);
        has_outgoing.insert(edge.src_fm_id);
    }
    let all_fm_ids: HashSet<u64> = fm_nodes.keys().copied().collect();
    let input_fm_ids: HashSet<u64> = all_fm_ids.difference(&has_incoming).copied().collect();
    let output_fm_ids: HashSet<u64> = all_fm_ids.difference(&has_outgoing).copied().collect();

    FMSubgraphAnalysis {
        fm_nodes,
        fm_edges,
        input_fm_ids,
        output_fm_ids,
    }
}

/// 在一组 FM 节点中找到输出节点（被其他 FM 节点依赖的最终计算节点）
fn find_fm_output_node(node_ids: &[u64], node_map: &HashMap<u64, &NodeGene>) -> u64 {
    let id_set: HashSet<u64> = node_ids.iter().copied().collect();
    // 输出节点 = FM 内没有被其他 FM 内节点作为 parent 的节点
    let mut candidates: HashSet<u64> = id_set.clone();
    for &nid in node_ids {
        if let Some(n) = node_map.get(&nid) {
            for &pid in &n.parents {
                if id_set.contains(&pid) {
                    candidates.remove(&pid);
                }
            }
        }
    }
    // 如果多个候选者，取创新号最大的（最后创建的）
    candidates.into_iter().max().unwrap_or(node_ids[0])
}

/// 从 FM 的输出节点推导空间尺寸
fn infer_fm_spatial_size(output_node_id: u64, node_map: &HashMap<u64, &NodeGene>) -> (usize, usize) {
    if let Some(n) = node_map.get(&output_node_id) {
        let shape = &n.output_shape;
        if shape.len() >= 4 {
            return (shape[2], shape[3]);
        }
    }
    (0, 0)
}

/// 找出所有 FM 边：连接两个 FM 的 conv/pool/deconv 操作
fn find_fm_edges(
    nodes: &[NodeGene],
    fm_nodes: &BTreeMap<u64, FMNodeInfo>,
    node_map: &HashMap<u64, &NodeGene>,
) -> Vec<FMEdgeInfo> {
    let mut edges = Vec::new();

    // 建立 node_id → fm_id 的反向映射
    let mut node_to_fm: HashMap<u64, u64> = HashMap::new();
    for (fid, info) in fm_nodes {
        for &nid in &info.node_ids {
            node_to_fm.insert(nid, *fid);
        }
    }

    // 扫描所有启用的非 FM 节点，找 conv/pool/deconv op
    for n in nodes.iter().filter(|n| n.enabled && n.fm_id.is_none()) {
        let edge_type = match FMEdgeType::from_descriptor(&n.node_type) {
            Some(et) => et,
            None => continue,
        };

        // op 节点的 parents 应包含一个 FM 的输出（源 FM）和可能一个 kernel Parameter
        let mut src_fm_id = None;
        let mut kernel_node_id = None;

        for &pid in &n.parents {
            if let Some(&fid) = node_to_fm.get(&pid) {
                src_fm_id = Some(fid);
            } else if let Some(parent) = node_map.get(&pid) {
                if parent.is_parameter() && parent.fm_id.is_none() {
                    kernel_node_id = Some(pid);
                }
            }
        }

        let src_fm = match src_fm_id {
            Some(fid) => fid,
            None => continue,
        };

        // 找到 op 的下游消费者 → 哪个 FM 接收了它
        let dst_fm = find_destination_fm(n.innovation_number, nodes, &node_to_fm);
        let dst_fm = match dst_fm {
            Some(fid) => fid,
            None => continue,
        };

        let block_id = n.block_id.unwrap_or(n.innovation_number);

        edges.push(FMEdgeInfo {
            block_id,
            src_fm_id: src_fm,
            dst_fm_id: dst_fm,
            op_node_id: n.innovation_number,
            kernel_node_id,
            edge_type,
        });
    }

    edges
}

/// 找到一个 op 节点的输出流向哪个 FM
fn find_destination_fm(
    op_id: u64,
    nodes: &[NodeGene],
    node_to_fm: &HashMap<u64, u64>,
) -> Option<u64> {
    for n in nodes.iter().filter(|n| n.enabled) {
        if n.parents.contains(&op_id) {
            if let Some(&fid) = node_to_fm.get(&n.innovation_number) {
                return Some(fid);
            }
        }
    }
    None
}

// ==================== FM 操作原语 ====================

/// 获取基因组中下一个可用的 fm_id
pub fn next_fm_id(nodes: &[NodeGene]) -> u64 {
    nodes
        .iter()
        .filter_map(|n| n.fm_id)
        .max()
        .map(|m| m + 1)
        .unwrap_or(1)
}

/// 检测一组输出 FM 是否与输入 FM 全连接（且边类型完全一致）
///
/// 返回 `Some((in_count, out_count, edge_type))` 如果全连接，否则 `None`
pub fn detect_fully_connected(
    input_fm_ids: &[u64],
    output_fm_ids: &[u64],
    edges: &[FMEdgeInfo],
) -> Option<(usize, usize, FMEdgeType)> {
    if input_fm_ids.is_empty() || output_fm_ids.is_empty() {
        return None;
    }

    let in_set: HashSet<u64> = input_fm_ids.iter().copied().collect();
    let out_set: HashSet<u64> = output_fm_ids.iter().copied().collect();

    // 过滤出跨越 input→output 的边
    let crossing_edges: Vec<&FMEdgeInfo> = edges
        .iter()
        .filter(|e| in_set.contains(&e.src_fm_id) && out_set.contains(&e.dst_fm_id))
        .collect();

    let expected_count = input_fm_ids.len() * output_fm_ids.len();
    if crossing_edges.len() != expected_count {
        return None;
    }

    // 检查所有边类型一致
    let first_type = &crossing_edges[0].edge_type;
    if !crossing_edges.iter().all(|e| &e.edge_type == first_type) {
        return None;
    }

    // 检查所有 (src, dst) 对都有且仅有一条边
    let mut pairs: HashSet<(u64, u64)> = HashSet::new();
    for e in &crossing_edges {
        if !pairs.insert((e.src_fm_id, e.dst_fm_id)) {
            return None;
        }
    }

    Some((input_fm_ids.len(), output_fm_ids.len(), first_type.clone()))
}

/// 查找可以添加新边的 FM 对（src_fm, dst_fm）
///
/// 条件：src 和 dst 都是有效 FM，且当前没有直接连接
pub fn find_connectable_pairs(
    analysis: &FMSubgraphAnalysis,
) -> Vec<(u64, u64)> {
    let existing_pairs: HashSet<(u64, u64)> = analysis
        .fm_edges
        .iter()
        .map(|e| (e.src_fm_id, e.dst_fm_id))
        .collect();

    let fm_ids: Vec<u64> = analysis.fm_nodes.keys().copied().collect();
    let mut pairs = Vec::new();

    for &src in &fm_ids {
        for &dst in &fm_ids {
            if src != dst && !existing_pairs.contains(&(src, dst)) {
                pairs.push((src, dst));
            }
        }
    }

    pairs
}

/// 查找可以安全删除的 FM 边（删除后每个 FM 仍保留至少一条输入边）
pub fn find_removable_edges(analysis: &FMSubgraphAnalysis) -> Vec<&FMEdgeInfo> {
    // 统计每个 dst FM 的输入边数
    let mut incoming_count: HashMap<u64, usize> = HashMap::new();
    for edge in &analysis.fm_edges {
        *incoming_count.entry(edge.dst_fm_id).or_insert(0) += 1;
    }

    analysis
        .fm_edges
        .iter()
        .filter(|e| {
            // 只有当 dst FM 有多于 1 条输入边时才可删除
            incoming_count.get(&e.dst_fm_id).copied().unwrap_or(0) > 1
        })
        .collect()
}

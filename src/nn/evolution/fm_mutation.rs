/*
 * @Author       : 老董
 * @Date         : 2026-04-19
 * @Description  : FM（Feature Map）级别变异操作
 *
 * 10 种 FM 级别变异：
 *
 * 结构性变异（改变拓扑）：
 * 1. AddFeatureMap      - 添加新隐藏 FM
 * 2. RemoveFeatureMap   - 移除隐藏 FM
 * 3. AddFMEdge          - 添加 FM 间的边
 * 4. RemoveFMEdge       - 移除 FM 间的边
 * 5. SplitFMEdge        - 在边中间插入新 FM
 *
 * 参数变异（不改变拓扑）：
 * 6. ChangeFMEdgeType   - 切换边类型（conv/pool/deconv）
 * 7. MutateFMEdgeKernelSize - 改变 kernel 大小
 * 8. MutateFMEdgeStride     - 改变 stride
 * 9. MutateFMEdgeDilation   - 改变 dilation
 * 10. ChangeFeatureMapSize  - 改变 FM 空间尺寸
 */

use rand::Rng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::fm_ops::{
    FMEdgeInfo, FMSubgraphAnalysis, analyze_fm_subgraph, find_block_edge_op_ids,
    find_block_kernel_ids, find_connectable_pairs, find_removable_edges, next_fm_id,
    query_block_conv_params,
};
use crate::nn::evolution::gene::NetworkGenome;
use crate::nn::evolution::mutation::{Mutation, MutationError, SizeConstraints};
use crate::nn::evolution::node_gene::NodeGene;

fn has_fm_nodes(genome: &NetworkGenome) -> bool {
    genome.is_node_level()
        && genome.input_spatial.is_some()
        && genome
            .nodes()
            .iter()
            .any(|n| n.enabled && n.fm_id.is_some())
}

fn get_fm_analysis(genome: &NetworkGenome) -> FMSubgraphAnalysis {
    analyze_fm_subgraph(genome.nodes())
}

fn genome_next_innovation(genome: &mut NetworkGenome) -> u64 {
    genome.next_innovation_number()
}

// ==================== 1. AddFeatureMap ====================

pub struct AddFeatureMapMutation;

impl Mutation for AddFeatureMapMutation {
    fn name(&self) -> &str {
        "AddFeatureMap"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        has_fm_nodes(genome)
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        let fm_ids: Vec<u64> = analysis.fm_nodes.keys().copied().collect();
        if fm_ids.len() < 2 {
            return Err(MutationError::NotApplicable(
                "FM 节点不足以添加新 FM".into(),
            ));
        }

        // 随机选择一个上游 FM 和一个下游 FM
        let src_idx = rng.gen_range(0..fm_ids.len());
        let mut dst_idx = rng.gen_range(0..fm_ids.len());
        while dst_idx == src_idx {
            dst_idx = rng.gen_range(0..fm_ids.len());
        }
        let src_fm = fm_ids[src_idx];
        let dst_fm = fm_ids[dst_idx];

        let src_info = &analysis.fm_nodes[&src_fm];
        let dst_info = &analysis.fm_nodes[&dst_fm];

        // 新 FM 空间尺寸取平均
        let new_h = (src_info.spatial_size.0 + dst_info.spatial_size.0) / 2;
        let new_w = (src_info.spatial_size.1 + dst_info.spatial_size.1) / 2;
        let new_h = new_h.max(1);
        let new_w = new_w.max(1);

        let new_fm_id = next_fm_id(genome.nodes());

        // 创建新 FM 节点（Identity 作为 FM 标识）
        let fm_node_id = genome_next_innovation(genome);
        let mut fm_node = NodeGene::new(
            fm_node_id,
            NodeTypeDescriptor::Identity,
            vec![1, 1, new_h, new_w],
            vec![],
            None,
        );
        fm_node.fm_id = Some(new_fm_id);

        // 继承 src 侧已有边的结构参数（块内同构），找不到则用默认 3×3
        let (ks1, stride1, _pad1, dil1, is_deconv1) = query_block_conv_params(
            src_fm,
            &analysis,
            genome.nodes(),
        )
        .unwrap_or((3, (1, 1), (1, 1), (1, 1), false));
        let pad1 = (ks1 / 2, ks1 / 2);

        let edge_block_1 = genome_next_innovation(genome);
        let kernel_1_id = genome_next_innovation(genome);
        let conv_1_id = genome_next_innovation(genome);

        let src_output = src_info.output_node_id;
        let kernel_1 = NodeGene::new(
            kernel_1_id,
            NodeTypeDescriptor::Parameter,
            vec![1, 1, ks1, ks1],
            vec![],
            Some(edge_block_1),
        );
        let conv_1_type = if is_deconv1 {
            NodeTypeDescriptor::ConvTranspose2d {
                stride: stride1,
                padding: pad1,
                output_padding: (0, 0),
            }
        } else {
            NodeTypeDescriptor::Conv2d {
                stride: stride1,
                padding: pad1,
                dilation: dil1,
            }
        };
        let conv_1 = NodeGene::new(
            conv_1_id,
            conv_1_type,
            vec![1, 1, new_h, new_w],
            vec![src_output, kernel_1_id],
            Some(edge_block_1),
        );

        fm_node.parents = vec![conv_1_id];

        // 继承 dst 侧已有边的结构参数
        let (ks2, stride2, _pad2, dil2, is_deconv2) = query_block_conv_params(
            dst_fm,
            &analysis,
            genome.nodes(),
        )
        .unwrap_or((3, (1, 1), (1, 1), (1, 1), false));
        let pad2 = (ks2 / 2, ks2 / 2);

        let edge_block_2 = genome_next_innovation(genome);
        let kernel_2_id = genome_next_innovation(genome);
        let conv_2_id = genome_next_innovation(genome);

        let kernel_2 = NodeGene::new(
            kernel_2_id,
            NodeTypeDescriptor::Parameter,
            vec![1, 1, ks2, ks2],
            vec![],
            Some(edge_block_2),
        );
        let conv_2_type = if is_deconv2 {
            NodeTypeDescriptor::ConvTranspose2d {
                stride: stride2,
                padding: pad2,
                output_padding: (0, 0),
            }
        } else {
            NodeTypeDescriptor::Conv2d {
                stride: stride2,
                padding: pad2,
                dilation: dil2,
            }
        };
        let conv_2 = NodeGene::new(
            conv_2_id,
            conv_2_type,
            vec![1, 1, dst_info.spatial_size.0, dst_info.spatial_size.1],
            vec![fm_node_id, kernel_2_id],
            Some(edge_block_2),
        );

        // 在 dst FM 的聚合中添加 conv_2 作为额外输入
        let nodes = genome.nodes_mut();
        let dst_output_id = dst_info.output_node_id;
        if let Some(dst_node) = nodes
            .iter_mut()
            .find(|n| n.innovation_number == dst_output_id)
        {
            if matches!(dst_node.node_type, NodeTypeDescriptor::Add) {
                // 需要创建新的 Add 来包含 conv_2 和原 dst 输出
                // 简单做法：让 conv_2 作为额外 parent 添加到 dst 的 parents
                // 但 Add 只支持 2 个 parent，所以创建新 Add
            }
        }

        // 用更通用的方式：创建新 Add 聚合节点，parent = [dst_output_id, conv_2_id]
        let new_add_id = genome_next_innovation(genome);
        let mut new_add = NodeGene::new(
            new_add_id,
            NodeTypeDescriptor::Add,
            vec![1, 1, dst_info.spatial_size.0, dst_info.spatial_size.1],
            vec![dst_output_id, conv_2_id],
            None,
        );
        new_add.fm_id = Some(dst_fm);

        // 重定向下游引用 dst_output_id → new_add_id
        let nodes = genome.nodes_mut();
        for n in nodes.iter_mut() {
            if n.enabled && n.innovation_number != new_add_id {
                for pid in n.parents.iter_mut() {
                    if *pid == dst_output_id {
                        *pid = new_add_id;
                    }
                }
            }
        }

        nodes.push(kernel_1);
        nodes.push(conv_1);
        nodes.push(fm_node);
        nodes.push(kernel_2);
        nodes.push(conv_2);
        nodes.push(new_add);

        Ok(())
    }
}

// ==================== 2. RemoveFeatureMap ====================

pub struct RemoveFeatureMapMutation;

impl Mutation for RemoveFeatureMapMutation {
    fn name(&self) -> &str {
        "RemoveFeatureMap"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        // 需要至少有一个非输入/非输出的隐藏 FM
        analysis.fm_nodes.len() > 2
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        let fm_ids: Vec<u64> = analysis.fm_nodes.keys().copied().collect();

        // 找到可移除的 FM：非输入 FM（有 conv 边输入）且非唯一输出 FM
        // 简化：找不是 Concat 直接 parent 的 FM
        let nodes = genome.nodes();
        let concat_parent_fm_ids: std::collections::HashSet<u64> = nodes
            .iter()
            .filter(|n| n.enabled && matches!(n.node_type, NodeTypeDescriptor::Concat { .. }))
            .flat_map(|c| {
                c.parents.iter().filter_map(|&pid| {
                    nodes
                        .iter()
                        .find(|n| n.innovation_number == pid && n.fm_id.is_some())
                        .and_then(|n| n.fm_id)
                })
            })
            .collect();

        // 找输入 FM（Identity 且 parent 是 INPUT 或 Concat）
        let input_fm_ids: std::collections::HashSet<u64> = nodes
            .iter()
            .filter(|n| {
                n.enabled
                    && n.fm_id.is_some()
                    && matches!(n.node_type, NodeTypeDescriptor::Identity)
            })
            .filter_map(|n| n.fm_id)
            .collect();

        let removable: Vec<u64> = fm_ids
            .iter()
            .copied()
            .filter(|id| !input_fm_ids.contains(id) && !concat_parent_fm_ids.contains(id))
            .collect();

        if removable.is_empty() {
            return Err(MutationError::NotApplicable("没有可移除的隐藏 FM".into()));
        }

        let target_fm = removable[rng.gen_range(0..removable.len())];

        // 分两遍收集信息然后修改
        let nodes = genome.nodes_mut();

        // Pass 1: 收集要禁用的节点
        let target_node_ids: Vec<u64> = nodes
            .iter()
            .filter(|n| n.enabled && n.fm_id == Some(target_fm))
            .map(|n| n.innovation_number)
            .collect();

        let target_set: std::collections::HashSet<u64> = target_node_ids.iter().copied().collect();

        // 收集 FM 边节点（parent 指向 target FM）
        let edge_node_ids: Vec<u64> = nodes
            .iter()
            .filter(|n| {
                n.enabled && n.fm_id.is_none() && n.parents.iter().any(|p| target_set.contains(p))
            })
            .map(|n| n.innovation_number)
            .collect();

        // 收集边节点的 block_id（用于禁用同 block 的 kernel）
        let edge_set: std::collections::HashSet<u64> = edge_node_ids.iter().copied().collect();
        let edge_block_ids: std::collections::HashSet<u64> = nodes
            .iter()
            .filter(|n| edge_set.contains(&n.innovation_number))
            .filter_map(|n| n.block_id)
            .collect();

        // Pass 2: 禁用节点
        let mut to_disable = target_set;
        to_disable.extend(edge_node_ids.iter());
        for n in nodes.iter_mut() {
            if to_disable.contains(&n.innovation_number) {
                n.enabled = false;
            }
            if let Some(bid) = n.block_id {
                if edge_block_ids.contains(&bid) {
                    n.enabled = false;
                }
            }
        }

        // Pass 3: 清理 parents 引用并移除禁用节点
        let disabled: std::collections::HashSet<u64> = nodes
            .iter()
            .filter(|n| !n.enabled)
            .map(|n| n.innovation_number)
            .collect();

        for n in nodes.iter_mut() {
            if n.enabled {
                n.parents.retain(|p| !disabled.contains(p));
            }
        }

        nodes.retain(|n| n.enabled);

        Ok(())
    }
}

// ==================== 3. AddFMEdge ====================

pub struct AddFMEdgeMutation;

impl Mutation for AddFMEdgeMutation {
    fn name(&self) -> &str {
        "AddFMEdge"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        !find_connectable_pairs(&analysis).is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        let pairs = find_connectable_pairs(&analysis);
        if pairs.is_empty() {
            return Err(MutationError::NotApplicable("没有可添加边的 FM 对".into()));
        }

        let &(src_fm, dst_fm) = &pairs[rng.gen_range(0..pairs.len())];
        let src_info = &analysis.fm_nodes[&src_fm];
        let dst_info = &analysis.fm_nodes[&dst_fm];

        // 继承目标 FM 所在 block 已有边的结构参数（块内同构）
        let (ks, stride, _pad, dil, is_deconv) = query_block_conv_params(
            dst_fm,
            &analysis,
            genome.nodes(),
        )
        .unwrap_or((3, (1, 1), (1, 1), (1, 1), false));
        let pad = (ks / 2, ks / 2);

        let edge_block = genome_next_innovation(genome);
        let kernel_id = genome_next_innovation(genome);
        let conv_id = genome_next_innovation(genome);

        let kernel = NodeGene::new(
            kernel_id,
            NodeTypeDescriptor::Parameter,
            vec![1, 1, ks, ks],
            vec![],
            Some(edge_block),
        );

        let conv_type = if is_deconv {
            NodeTypeDescriptor::ConvTranspose2d {
                stride,
                padding: pad,
                output_padding: (0, 0),
            }
        } else {
            NodeTypeDescriptor::Conv2d {
                stride,
                padding: pad,
                dilation: dil,
            }
        };
        let conv = NodeGene::new(
            conv_id,
            conv_type,
            vec![1, 1, dst_info.spatial_size.0, dst_info.spatial_size.1],
            vec![src_info.output_node_id, kernel_id],
            Some(edge_block),
        );

        // 在 dst FM 添加 Add 聚合
        let new_add_id = genome_next_innovation(genome);
        let mut new_add = NodeGene::new(
            new_add_id,
            NodeTypeDescriptor::Add,
            vec![1, 1, dst_info.spatial_size.0, dst_info.spatial_size.1],
            vec![dst_info.output_node_id, conv_id],
            None,
        );
        new_add.fm_id = Some(dst_fm);

        // 重定向下游
        let dst_output_id = dst_info.output_node_id;
        let nodes = genome.nodes_mut();
        for n in nodes.iter_mut() {
            if n.enabled && n.innovation_number != new_add_id {
                for pid in n.parents.iter_mut() {
                    if *pid == dst_output_id {
                        *pid = new_add_id;
                    }
                }
            }
        }

        nodes.push(kernel);
        nodes.push(conv);
        nodes.push(new_add);

        Ok(())
    }
}

// ==================== 4. RemoveFMEdge ====================

pub struct RemoveFMEdgeMutation;

impl Mutation for RemoveFMEdgeMutation {
    fn name(&self) -> &str {
        "RemoveFMEdge"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        !find_removable_edges(&analysis).is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        let removable = find_removable_edges(&analysis);
        if removable.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有可安全移除的 FM 边".into(),
            ));
        }

        let edge = removable[rng.gen_range(0..removable.len())];
        let edge_block = edge.block_id;

        // 禁用 edge 块的所有节点
        let nodes = genome.nodes_mut();
        for n in nodes.iter_mut() {
            if n.block_id == Some(edge_block) {
                n.enabled = false;
            }
        }

        // 清理引用
        let disabled: std::collections::HashSet<u64> = nodes
            .iter()
            .filter(|n| !n.enabled)
            .map(|n| n.innovation_number)
            .collect();

        for n in nodes.iter_mut() {
            if n.enabled {
                n.parents.retain(|p| !disabled.contains(p));
            }
        }

        nodes.retain(|n| n.enabled);

        Ok(())
    }
}

// ==================== 5. SplitFMEdge ====================

pub struct SplitFMEdgeMutation;

impl Mutation for SplitFMEdgeMutation {
    fn name(&self) -> &str {
        "SplitFMEdge"
    }

    fn is_structural(&self) -> bool {
        true
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        !analysis.fm_edges.is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        if analysis.fm_edges.is_empty() {
            return Err(MutationError::NotApplicable("没有 FM 边可以分裂".into()));
        }

        let edge = &analysis.fm_edges[rng.gen_range(0..analysis.fm_edges.len())];
        let src_info = &analysis.fm_nodes[&edge.src_fm_id];
        let dst_info = &analysis.fm_nodes[&edge.dst_fm_id];

        // 新 FM 空间尺寸 = src 和 dst 的平均
        let new_h = ((src_info.spatial_size.0 + dst_info.spatial_size.0) / 2).max(1);
        let new_w = ((src_info.spatial_size.1 + dst_info.spatial_size.1) / 2).max(1);

        let new_fm_id = next_fm_id(genome.nodes());

        // 禁用原边
        let old_block = edge.block_id;
        let nodes = genome.nodes_mut();
        for n in nodes.iter_mut() {
            if n.block_id == Some(old_block) {
                n.enabled = false;
            }
        }

        // 创建新 FM
        let fm_node_id = genome_next_innovation(genome);
        let mut fm_node = NodeGene::new(
            fm_node_id,
            NodeTypeDescriptor::Identity,
            vec![1, 1, new_h, new_w],
            vec![],
            None,
        );
        fm_node.fm_id = Some(new_fm_id);

        // 继承被分裂边的结构参数（块内同构）
        let (orig_ks, orig_stride, _orig_pad, orig_dil, orig_is_deconv) = {
            let node_map: std::collections::HashMap<u64, &NodeGene> = genome
                .nodes()
                .iter()
                .filter(|n| n.enabled)
                .map(|n| (n.innovation_number, n))
                .collect();
            let ks = edge
                .kernel_node_id
                .and_then(|kid| node_map.get(&kid).map(|k| k.output_shape[2]))
                .unwrap_or(3);
            match &edge.edge_type {
                crate::nn::evolution::fm_ops::FMEdgeType::Conv2d {
                    stride, dilation, ..
                } => (ks, *stride, (ks / 2, ks / 2), *dilation, false),
                crate::nn::evolution::fm_ops::FMEdgeType::ConvTranspose2d { stride, .. } => {
                    (ks, *stride, (ks / 2, ks / 2), (1, 1), true)
                }
                _ => (3, (1, 1), (1, 1), (1, 1), false),
            }
        };
        let p = (orig_ks / 2, orig_ks / 2);

        // 边 A: src → new FM
        let block_a = genome_next_innovation(genome);
        let k_a = genome_next_innovation(genome);
        let c_a = genome_next_innovation(genome);

        let kernel_a = NodeGene::new(
            k_a,
            NodeTypeDescriptor::Parameter,
            vec![1, 1, orig_ks, orig_ks],
            vec![],
            Some(block_a),
        );
        let conv_a_type = if orig_is_deconv {
            NodeTypeDescriptor::ConvTranspose2d {
                stride: orig_stride,
                padding: p,
                output_padding: (0, 0),
            }
        } else {
            NodeTypeDescriptor::Conv2d {
                stride: orig_stride,
                padding: p,
                dilation: orig_dil,
            }
        };
        let conv_a = NodeGene::new(
            c_a,
            conv_a_type,
            vec![1, 1, new_h, new_w],
            vec![src_info.output_node_id, k_a],
            Some(block_a),
        );
        fm_node.parents = vec![c_a];

        // 边 B: new FM → dst（继承 dst 侧的参数以保持块内同构）
        let (ks_b, stride_b, _pad_b, dil_b, is_deconv_b) = query_block_conv_params(
            edge.dst_fm_id,
            &analysis,
            genome.nodes(),
        )
        .unwrap_or((orig_ks, orig_stride, p, orig_dil, orig_is_deconv));
        let p_b = (ks_b / 2, ks_b / 2);

        let block_b = genome_next_innovation(genome);
        let k_b = genome_next_innovation(genome);
        let c_b = genome_next_innovation(genome);

        let kernel_b = NodeGene::new(
            k_b,
            NodeTypeDescriptor::Parameter,
            vec![1, 1, ks_b, ks_b],
            vec![],
            Some(block_b),
        );
        let conv_b_type = if is_deconv_b {
            NodeTypeDescriptor::ConvTranspose2d {
                stride: stride_b,
                padding: p_b,
                output_padding: (0, 0),
            }
        } else {
            NodeTypeDescriptor::Conv2d {
                stride: stride_b,
                padding: p_b,
                dilation: dil_b,
            }
        };
        let conv_b = NodeGene::new(
            c_b,
            conv_b_type,
            vec![1, 1, dst_info.spatial_size.0, dst_info.spatial_size.1],
            vec![fm_node_id, k_b],
            Some(block_b),
        );

        // 将原边的 conv 输出替换为新 conv_b 的输出
        let old_conv_id = edge.op_node_id;
        let nodes = genome.nodes_mut();
        for n in nodes.iter_mut() {
            if n.enabled {
                for pid in n.parents.iter_mut() {
                    if *pid == old_conv_id {
                        *pid = c_b;
                    }
                }
            }
        }

        nodes.push(kernel_a);
        nodes.push(conv_a);
        nodes.push(fm_node);
        nodes.push(kernel_b);
        nodes.push(conv_b);

        nodes.retain(|n| n.enabled);

        Ok(())
    }
}

// ==================== 6. ChangeFMEdgeType ====================

pub struct ChangeFMEdgeTypeMutation;

impl Mutation for ChangeFMEdgeTypeMutation {
    fn name(&self) -> &str {
        "ChangeFMEdgeType"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        !analysis.fm_edges.is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        // 只选择 Conv2d/ConvTranspose2d 边（Pool 边不参与类型互换）
        let conv_edges: Vec<&FMEdgeInfo> = analysis
            .fm_edges
            .iter()
            .filter(|e| e.edge_type.has_learnable_params())
            .collect();
        if conv_edges.is_empty() {
            return Err(MutationError::NotApplicable(
                "没有 Conv/Deconv FM 边".into(),
            ));
        }

        let edge = conv_edges[rng.gen_range(0..conv_edges.len())];
        // per-block：找到同一 FM block 的所有边
        let block_op_ids = find_block_edge_op_ids(edge, &analysis, genome.nodes());
        let block_op_set: std::collections::HashSet<u64> = block_op_ids.iter().copied().collect();

        let nodes = genome.nodes_mut();
        for conv_node in nodes
            .iter_mut()
            .filter(|n| block_op_set.contains(&n.innovation_number))
        {
            match &conv_node.node_type {
                NodeTypeDescriptor::Conv2d {
                    stride, padding, ..
                } => {
                    let s = *stride;
                    let p = *padding;
                    conv_node.node_type = NodeTypeDescriptor::ConvTranspose2d {
                        stride: s,
                        padding: p,
                        output_padding: (0, 0),
                    };
                }
                NodeTypeDescriptor::ConvTranspose2d {
                    stride, padding, ..
                } => {
                    let s = *stride;
                    let p = *padding;
                    conv_node.node_type = NodeTypeDescriptor::Conv2d {
                        stride: s,
                        padding: p,
                        dilation: (1, 1),
                    };
                }
                _ => {}
            }
        }

        Ok(())
    }
}

// ==================== 7. MutateFMEdgeKernelSize ====================

pub struct MutateFMEdgeKernelSizeMutation;

impl Mutation for MutateFMEdgeKernelSizeMutation {
    fn name(&self) -> &str {
        "MutateFMEdgeKernelSize"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        analysis.fm_edges.iter().any(|e| e.kernel_node_id.is_some())
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        let conv_edges: Vec<&FMEdgeInfo> = analysis
            .fm_edges
            .iter()
            .filter(|e| e.kernel_node_id.is_some())
            .collect();

        if conv_edges.is_empty() {
            return Err(MutationError::NotApplicable("没有可变的 kernel 边".into()));
        }

        let edge = conv_edges[rng.gen_range(0..conv_edges.len())];
        let new_k = *[1usize, 3, 5, 7].choose(rng).unwrap();
        let new_padding = new_k / 2;

        // per-block：找到同一 FM block 的所有 kernel 和 op 节点
        let block_kernel_ids = find_block_kernel_ids(edge, &analysis, genome.nodes());
        let block_op_ids = find_block_edge_op_ids(edge, &analysis, genome.nodes());
        let kernel_set: std::collections::HashSet<u64> = block_kernel_ids.iter().copied().collect();
        let op_set: std::collections::HashSet<u64> = block_op_ids.iter().copied().collect();

        let nodes = genome.nodes_mut();

        // 更新所有 kernel shape
        for kernel_node in nodes
            .iter_mut()
            .filter(|n| kernel_set.contains(&n.innovation_number))
        {
            kernel_node.output_shape = vec![1, 1, new_k, new_k];
        }

        // 更新所有 conv 节点的 padding
        for conv_node in nodes
            .iter_mut()
            .filter(|n| op_set.contains(&n.innovation_number))
        {
            match &mut conv_node.node_type {
                NodeTypeDescriptor::Conv2d { padding, .. } => {
                    *padding = (new_padding, new_padding);
                }
                NodeTypeDescriptor::ConvTranspose2d { padding, .. } => {
                    *padding = (new_padding, new_padding);
                }
                _ => {}
            }
        }

        Ok(())
    }
}

// ==================== 8. MutateFMEdgeStride ====================

pub struct MutateFMEdgeStrideMutation;

impl Mutation for MutateFMEdgeStrideMutation {
    fn name(&self) -> &str {
        "MutateFMEdgeStride"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        !analysis.fm_edges.is_empty()
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        if analysis.fm_edges.is_empty() {
            return Err(MutationError::NotApplicable("没有 FM 边".into()));
        }

        let edge = &analysis.fm_edges[rng.gen_range(0..analysis.fm_edges.len())];
        // per-block：找到同一 FM block 的所有 op 节点
        let block_op_ids = find_block_edge_op_ids(edge, &analysis, genome.nodes());
        let op_set: std::collections::HashSet<u64> = block_op_ids.iter().copied().collect();

        // 根据目标边的当前 stride 决定新 stride
        let new_stride =
            {
                let target_node = genome
                    .nodes()
                    .iter()
                    .find(|n| n.innovation_number == edge.op_node_id);
                match target_node.map(|n| &n.node_type) {
                    Some(NodeTypeDescriptor::Conv2d { stride, .. })
                    | Some(NodeTypeDescriptor::ConvTranspose2d { stride, .. }) => {
                        if *stride == (1, 1) { (2, 2) } else { (1, 1) }
                    }
                    _ => (1, 1),
                }
            };

        let nodes = genome.nodes_mut();
        for conv_node in nodes
            .iter_mut()
            .filter(|n| op_set.contains(&n.innovation_number))
        {
            match &mut conv_node.node_type {
                NodeTypeDescriptor::Conv2d { stride, .. } => {
                    *stride = new_stride;
                }
                NodeTypeDescriptor::ConvTranspose2d { stride, .. } => {
                    *stride = new_stride;
                }
                _ => {}
            }
        }

        Ok(())
    }
}

// ==================== 9. MutateFMEdgeDilation ====================

pub struct MutateFMEdgeDilationMutation;

impl Mutation for MutateFMEdgeDilationMutation {
    fn name(&self) -> &str {
        "MutateFMEdgeDilation"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        analysis.fm_edges.iter().any(|e| {
            matches!(
                &e.edge_type,
                crate::nn::evolution::fm_ops::FMEdgeType::Conv2d { .. }
            )
        })
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        let conv_edges: Vec<&FMEdgeInfo> = analysis
            .fm_edges
            .iter()
            .filter(|e| {
                matches!(
                    &e.edge_type,
                    crate::nn::evolution::fm_ops::FMEdgeType::Conv2d { .. }
                )
            })
            .collect();

        if conv_edges.is_empty() {
            return Err(MutationError::NotApplicable("没有 Conv2d FM 边".into()));
        }

        let edge = conv_edges[rng.gen_range(0..conv_edges.len())];
        let dilations = [(1, 1), (2, 2), (3, 3)];
        let new_dilation = dilations[rng.gen_range(0..dilations.len())];

        // per-block：找到同一 FM block 的所有 op 节点
        let block_op_ids = find_block_edge_op_ids(edge, &analysis, genome.nodes());
        let op_set: std::collections::HashSet<u64> = block_op_ids.iter().copied().collect();

        let nodes = genome.nodes_mut();
        for conv_node in nodes
            .iter_mut()
            .filter(|n| op_set.contains(&n.innovation_number))
        {
            if let NodeTypeDescriptor::Conv2d { dilation, .. } = &mut conv_node.node_type {
                *dilation = new_dilation;
            }
        }

        Ok(())
    }
}

// ==================== 10. ChangeFeatureMapSize ====================

pub struct ChangeFeatureMapSizeMutation;

impl Mutation for ChangeFeatureMapSizeMutation {
    fn name(&self) -> &str {
        "ChangeFeatureMapSize"
    }

    fn is_applicable(&self, genome: &NetworkGenome, _constraints: &SizeConstraints) -> bool {
        if !has_fm_nodes(genome) {
            return false;
        }
        let analysis = get_fm_analysis(genome);
        // 需要有隐藏 FM（非输入/输出）
        analysis.fm_nodes.len() > 2
    }

    fn apply(
        &self,
        genome: &mut NetworkGenome,
        _constraints: &SizeConstraints,
        rng: &mut StdRng,
    ) -> Result<(), MutationError> {
        let analysis = get_fm_analysis(genome);
        let fm_ids: Vec<u64> = analysis.fm_nodes.keys().copied().collect();

        if fm_ids.is_empty() {
            return Err(MutationError::NotApplicable("没有 FM 节点".into()));
        }

        let target_fm = fm_ids[rng.gen_range(0..fm_ids.len())];
        let info = &analysis.fm_nodes[&target_fm];

        let delta_h: i32 = rng.gen_range(-3..=3);
        let delta_w: i32 = rng.gen_range(-3..=3);
        let new_h = (info.spatial_size.0 as i32 + delta_h).max(1) as usize;
        let new_w = (info.spatial_size.1 as i32 + delta_w).max(1) as usize;

        if new_h == info.spatial_size.0 && new_w == info.spatial_size.1 {
            return Err(MutationError::NotApplicable("尺寸未变化".into()));
        }

        // 更新该 FM 的所有节点的 output_shape
        let nodes = genome.nodes_mut();
        for n in nodes.iter_mut() {
            if n.enabled && n.fm_id == Some(target_fm) && n.output_shape.len() == 4 {
                n.output_shape[2] = new_h;
                n.output_shape[3] = new_w;
            }
        }

        Ok(())
    }
}

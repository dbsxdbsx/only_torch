/*
 * @Author       : 老董
 * @Date         : 2026-03-06
 * @Description  : Genome → Graph 转换 + Lamarckian 权重继承
 *
 * build() 统一走 NodeLevel → GraphDescriptor → Graph，
 * capture_weights() / restore_weights() 实现参数节点粒度的跨代权重复用。
 */

use std::collections::HashMap;

use rand::Rng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::{Graph, GraphError, Init, Var, VarActivationOps, VarMatrixOps, VarShapeOps};
use crate::tensor::Tensor;

use super::gene::{INPUT_INNOVATION, NetworkGenome, OutputHead};

// ==================== BuildResult ====================

/// FM 融合掩码信息（描述某个合成 kernel 中哪些通道对有边）
#[derive(Debug, Clone)]
pub struct FMMaskInfo {
    /// 有边的 (src_channel_idx, dst_channel_idx) 对
    pub connected_pairs: std::collections::HashSet<(usize, usize)>,
    /// 输入通道数
    pub in_ch: usize,
    /// 输出通道数
    pub out_ch: usize,
}

/// build() 的完整返回值
///
/// 将构建过程中产生的所有信息一次性交付给调用者，
/// 避免事后从 Graph 中重新收集。
pub struct BuildResult {
    /// 输入节点（用于设置训练/测试数据）
    pub input: Var,
    /// 默认输出节点（单输出时即唯一输出；多头时为 primary / inference head）
    pub output: Var,
    /// 所有输出节点，顺序与 `output_heads` 一致
    pub outputs: Vec<Var>,
    /// 输出 head 元数据。旧单输出模型会自动补成 `output`
    pub output_heads: Vec<OutputHead>,
    /// innovation_number → 该层的参数变量列表（如 Linear 有 [W, b]）
    ///
    /// capture_weights() / restore_weights() 按 innovation_number 匹配参数。
    /// Optimizer 所需的扁平参数列表通过 all_parameters() 派生。
    pub layer_params: HashMap<u64, Vec<Var>>,
    /// 内部引用的 Graph（保持 Graph 存活，防止 Var 中的 Weak 失效）
    pub graph: Graph,
    /// FM 融合掩码：merged_kernel_id → 连接掩码信息（用于权重归零）
    pub fm_masks: HashMap<u64, FMMaskInfo>,
}

impl BuildResult {
    /// 查找命名 head 的输出变量。
    pub fn output_by_name(&self, name: &str) -> Option<&Var> {
        self.output_heads
            .iter()
            .position(|head| head.name == name)
            .and_then(|idx| self.outputs.get(idx))
    }

    /// 返回默认推理 head 的索引。
    pub fn default_output_index(&self) -> usize {
        self.output_heads
            .iter()
            .position(|head| head.inference || head.primary)
            .unwrap_or(0)
    }

    /// 以引用形式返回所有输出，便于保存、导出和可视化。
    pub fn output_refs(&self) -> Vec<&Var> {
        self.outputs.iter().collect()
    }

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

    /// 对所有 FM 融合 kernel 施加连接掩码：将无边通道对的 kernel 切片置零
    ///
    /// kernel shape: [out_ch, in_ch, kH, kW]
    /// 对每个 (src_idx, dst_idx) 不在 connected_pairs 中的位置，
    /// 将 kernel[dst_idx, src_idx, :, :] 置零。
    pub fn apply_fm_masks(&self) -> Result<(), GraphError> {
        for (&kernel_id, mask_info) in &self.fm_masks {
            if let Some(params) = self.layer_params.get(&kernel_id) {
                if let Some(param) = params.first() {
                    if let Some(tensor) = param.value()? {
                        let shape = tensor.shape();
                        if shape.len() != 4 {
                            continue;
                        }
                        let flat = tensor.to_vec();
                        let (out_ch, in_ch, kh, kw) = (shape[0], shape[1], shape[2], shape[3]);
                        let mut new_data = flat;
                        let slice_size = kh * kw;
                        for dst in 0..out_ch.min(mask_info.out_ch) {
                            for src in 0..in_ch.min(mask_info.in_ch) {
                                if !mask_info.connected_pairs.contains(&(src, dst)) {
                                    let offset = ((dst * in_ch + src) * kh + 0) * kw;
                                    for i in 0..slice_size {
                                        new_data[offset + i] = 0.0;
                                    }
                                }
                            }
                        }
                        let masked = Tensor::new(&new_data, shape);
                        param.set_value(&masked)?;
                    }
                }
            }
        }
        Ok(())
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

// ==================== NetworkGenome 构建与权重管理 ====================

use super::gene::{GenomeRepr, ShapeDomain};
use super::node_gene::GenomeAnalysis;
use super::node_gene::NodeGene;
use crate::nn::descriptor::{GraphDescriptor, NodeDescriptor, NodeTypeDescriptor as NTD};

// ==================== FM Builder 分析 ====================

/// FM 融合合并后的合成节点
struct SyntheticNode {
    id: u64,
    name: String,
    node_type: NTD,
    shape: Vec<usize>,
    parents: Vec<u64>,
}

/// 合成 ID 乘数和偏移常量，用于从 Concat innovation 生成稳定的合成参数 ID
const SYNTH_ID_MULTIPLIER: u64 = 100_000;
const KERNEL_OFFSET: u64 = 1;
const CONV_OFFSET: u64 = 2;

/// FM 融合合并组信息（支持全连接和稀疏连接）
struct MergedFMGroup {
    /// 被合并替换的节点 ID 集合（FM Identity 输入、FM edge kernel/conv、FM Add 聚合）
    replaced_ids: std::collections::HashSet<u64>,
    /// 替换后的合成节点列表（kernel + Conv2d/ConvTranspose2d）
    synthetic_nodes: Vec<SyntheticNode>,
    /// Concat 节点的替代父节点（合并后 Concat 的唯一输出源）
    concat_replacement: Option<(u64, u64)>, // (concat_id, replacement_parent_id)
    /// 连接掩码：有边的 (src_channel_idx, dst_channel_idx) 对集合
    connected_pairs: std::collections::HashSet<(usize, usize)>,
    /// in_ch 和 out_ch
    in_ch: usize,
    out_ch: usize,
    /// 合成 kernel 的稳定 ID（基于 Concat innovation number）
    merged_kernel_id: u64,
}

/// FM 相关的构图分析
struct FMBuilderAnalysis {
    /// FM 输入 Identity 节点 → Narrow 通道索引
    fm_narrow_map: HashMap<u64, usize>,
    /// FM 融合合并组（支持全连接和稀疏连接）
    merged_groups: Vec<MergedFMGroup>,
}

impl FMBuilderAnalysis {
    fn analyze(nodes: &[NodeGene]) -> Self {
        let mut fm_narrow_map = HashMap::new();
        let mut merged_groups = Vec::new();

        let node_map: HashMap<u64, &NodeGene> = nodes
            .iter()
            .filter(|n| n.enabled)
            .map(|n| (n.innovation_number, n))
            .collect();

        // 识别 FM 输入 Identity 节点
        // 先按 parent 粗分组，再按下游 Concat 细分（处理并行分支共享上游的情况）
        let mut raw_parent_groups: HashMap<u64, Vec<(u64, u64)>> = HashMap::new(); // parent_id → [(node_id, fm_id)]
        for n in nodes.iter().filter(|n| n.enabled) {
            if n.fm_id.is_some() && matches!(n.node_type, NTD::Identity) && !n.parents.is_empty() {
                raw_parent_groups
                    .entry(n.parents[0])
                    .or_default()
                    .push((n.innovation_number, n.fm_id.unwrap()));
            }
        }

        // 按下游 Concat 细分：每个 Identity 节点追踪其 conv 边的 dst FM 到达的 Concat
        let mut final_groups: Vec<(u64, Vec<(u64, u64)>)> = Vec::new(); // (source_id, [(node_id, fm_id)])
        for (&parent_id, group) in &raw_parent_groups {
            // 对每个 Identity 节点，找到其 conv 边到达的下游 Concat
            let mut concat_sub: HashMap<Option<u64>, Vec<(u64, u64)>> = HashMap::new();
            for &(node_id, fm_id) in group {
                // 找该 Identity 直接连接的 conv 边的 dst FM
                let dst_concat = nodes
                    .iter()
                    .filter(|n| {
                        n.enabled
                            && matches!(
                                &n.node_type,
                                NTD::Conv2d { .. } | NTD::ConvTranspose2d { .. }
                            )
                            && n.parents.first() == Some(&node_id)
                    })
                    .find_map(|conv_node| {
                        // 从 conv 输出追踪到 dst FM → 追踪到 Concat
                        Self::trace_to_concat(conv_node.innovation_number, nodes)
                    });
                concat_sub
                    .entry(dst_concat)
                    .or_default()
                    .push((node_id, fm_id));
            }
            for (_concat_id, sub_group) in concat_sub {
                final_groups.push((parent_id, sub_group));
            }
        }

        for (_parent_id, group) in final_groups.iter_mut() {
            group.sort_by_key(|&(_, fm_id)| fm_id);
            for (channel_idx, &(node_id, _)) in group.iter().enumerate() {
                fm_narrow_map.insert(node_id, channel_idx);
            }
        }

        // FM 组融合检测：支持全连接和稀疏连接
        // 对每组 FM 输入节点，尝试将同构 Conv/Deconv 边融合为单个操作
        for (source_id, input_group) in &final_groups {
            let input_node_ids: Vec<u64> = input_group.iter().map(|&(id, _)| id).collect();
            let input_fm_ids: Vec<u64> = input_group.iter().map(|&(_, fid)| fid).collect();
            let in_ch = input_node_ids.len();

            if let Some(merge_result) = Self::try_merge_fm_group(
                *source_id,
                &input_node_ids,
                &input_fm_ids,
                in_ch,
                nodes,
                &node_map,
            ) {
                for &id in &input_node_ids {
                    fm_narrow_map.remove(&id);
                }
                merged_groups.push(merge_result);
            }
        }

        Self {
            fm_narrow_map,
            merged_groups,
        }
    }

    /// 尝试将 FM 组融合为单个 Conv2d/ConvTranspose2d 操作
    ///
    /// 支持全连接和稀疏连接模式。稀疏连接时，无边的通道对在 kernel 中为零。
    /// 仅融合 Conv2d 和 ConvTranspose2d 边，Pool 边排除。
    fn try_merge_fm_group(
        source_id: u64,
        input_node_ids: &[u64],
        input_fm_ids: &[u64],
        in_ch: usize,
        nodes: &[NodeGene],
        node_map: &HashMap<u64, &NodeGene>,
    ) -> Option<MergedFMGroup> {
        let input_set: std::collections::HashSet<u64> = input_node_ids.iter().copied().collect();
        // input fm_id → 通道索引
        let fm_to_src_idx: HashMap<u64, usize> = input_fm_ids
            .iter()
            .enumerate()
            .map(|(i, &fid)| (fid, i))
            .collect();

        // 找到所有 FM edge Conv2d/ConvTranspose2d，其第一个 parent 是 input FM 节点之一
        let mut edge_convs: Vec<&NodeGene> = Vec::new();
        for n in nodes.iter().filter(|n| n.enabled) {
            let is_conv_like = matches!(
                &n.node_type,
                NTD::Conv2d { .. } | NTD::ConvTranspose2d { .. }
            );
            if is_conv_like && n.parents.len() >= 2 && input_set.contains(&n.parents[0]) {
                if let Some(kn) = node_map.get(&n.parents[1]) {
                    if kn.is_parameter()
                        && kn.output_shape.len() == 4
                        && kn.output_shape[0] == 1
                        && kn.output_shape[1] == 1
                    {
                        edge_convs.push(n);
                    }
                }
            }
        }

        if edge_convs.is_empty() {
            return None;
        }

        // 检查所有边的 op 类型和参数是否一致（块内同构）
        let first_type = &edge_convs[0].node_type;
        if !edge_convs.iter().all(|e| &e.node_type == first_type) {
            return None;
        }

        // 检查 kernel_size 一致
        let first_kernel = node_map.get(&edge_convs[0].parents[1]).unwrap();
        let (kh, kw) = (first_kernel.output_shape[2], first_kernel.output_shape[3]);
        for e in &edge_convs {
            let k = node_map.get(&e.parents[1]).unwrap();
            if k.output_shape[2] != kh || k.output_shape[3] != kw {
                return None;
            }
        }

        // 沿拓扑向下追踪，找到当前 FM 组的 Concat 节点
        let edge_conv_ids: std::collections::HashSet<u64> =
            edge_convs.iter().map(|e| e.innovation_number).collect();
        let mut frontier: std::collections::HashSet<u64> = edge_conv_ids.clone();
        let mut visited: std::collections::HashSet<u64> = frontier.clone();
        let mut found_concat: Option<&NodeGene> = None;

        for _ in 0..10 {
            let mut next_frontier = std::collections::HashSet::new();
            for n in nodes.iter().filter(|n| n.enabled) {
                if !visited.contains(&n.innovation_number)
                    && n.parents.iter().any(|p| frontier.contains(p))
                {
                    if matches!(n.node_type, NTD::Concat { axis: 1 }) && n.output_shape.len() == 4 {
                        found_concat = Some(n);
                        break;
                    }
                    next_frontier.insert(n.innovation_number);
                    visited.insert(n.innovation_number);
                }
            }
            if found_concat.is_some() || next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }

        let concat_node = match found_concat {
            Some(c) => c,
            None => return None,
        };

        let out_ch = concat_node.parents.len();

        // 建立 output FM 通道索引映射
        // Concat 的 parents 是各输出 FM 的聚合节点，按顺序对应 out_ch 通道
        let mut dst_fm_to_idx: HashMap<u64, usize> = HashMap::new();
        for (idx, &parent_id) in concat_node.parents.iter().enumerate() {
            // 从 Concat parent 追溯到 FM 的 fm_id
            if let Some(pn) = node_map.get(&parent_id) {
                if let Some(fid) = pn.fm_id {
                    dst_fm_to_idx.insert(fid, idx);
                }
            }
        }

        // 建立连接掩码：(src_channel_idx, dst_channel_idx)
        let mut connected_pairs = std::collections::HashSet::new();
        for e in &edge_convs {
            let src_node_id = e.parents[0];
            // 找 src Identity 节点的 fm_id
            let src_fm_id = node_map.get(&src_node_id).and_then(|n| n.fm_id);
            // 找 dst fm_id：追踪 edge conv 的下游消费者到 FM 节点
            let dst_fm_id = Self::find_edge_dst_fm(e.innovation_number, nodes, node_map);

            if let (Some(src_fid), Some(dst_fid)) = (src_fm_id, dst_fm_id) {
                if let (Some(&src_idx), Some(&dst_idx)) =
                    (fm_to_src_idx.get(&src_fid), dst_fm_to_idx.get(&dst_fid))
                {
                    connected_pairs.insert((src_idx, dst_idx));
                }
            }
        }

        // 至少需要一条连接才值得融合
        if connected_pairs.is_empty() {
            return None;
        }

        // 提取 op 参数
        let merged_op_type = match first_type {
            NTD::Conv2d {
                stride,
                padding,
                dilation,
            } => NTD::Conv2d {
                stride: *stride,
                padding: *padding,
                dilation: *dilation,
            },
            NTD::ConvTranspose2d {
                stride,
                padding,
                output_padding,
            } => NTD::ConvTranspose2d {
                stride: *stride,
                padding: *padding,
                output_padding: *output_padding,
            },
            _ => return None,
        };

        // 收集所有被替换的节点 ID
        let mut replaced_ids = std::collections::HashSet::new();
        for &id in input_node_ids {
            replaced_ids.insert(id);
        }
        for e in &edge_convs {
            replaced_ids.insert(e.innovation_number);
            replaced_ids.insert(e.parents[1]); // kernel
        }
        for &parent_id in &concat_node.parents {
            if let Some(pn) = node_map.get(&parent_id) {
                if pn.fm_id.is_some() {
                    replaced_ids.insert(parent_id);
                    if matches!(pn.node_type, NTD::Add) {
                        Self::collect_fm_add_tree(parent_id, node_map, &mut replaced_ids);
                    }
                }
            }
        }

        let conv_output_shape = &edge_convs[0].output_shape;
        let (oh, ow) = (conv_output_shape[2], conv_output_shape[3]);

        // 稳定合成 ID：基于 Concat 节点的 innovation number
        let concat_inn = concat_node.innovation_number;
        let merged_kernel_id = concat_inn * SYNTH_ID_MULTIPLIER + KERNEL_OFFSET;
        let merged_conv_id = concat_inn * SYNTH_ID_MULTIPLIER + CONV_OFFSET;

        let synthetic_nodes = vec![
            SyntheticNode {
                id: merged_kernel_id,
                name: format!("evo_merged_kernel_{}", concat_inn),
                node_type: NTD::Parameter,
                shape: vec![out_ch, in_ch, kh, kw],
                parents: vec![],
            },
            SyntheticNode {
                id: merged_conv_id,
                name: format!("evo_merged_conv_{}", concat_inn),
                node_type: merged_op_type,
                shape: vec![1, out_ch, oh, ow],
                parents: vec![source_id, merged_kernel_id],
            },
        ];

        Some(MergedFMGroup {
            replaced_ids,
            synthetic_nodes,
            concat_replacement: Some((concat_node.innovation_number, merged_conv_id)),
            connected_pairs,
            in_ch,
            out_ch,
            merged_kernel_id,
        })
    }

    /// 从某个节点向下追踪，找到其输出最终汇入的 Concat 节点 ID
    fn trace_to_concat(start_id: u64, nodes: &[NodeGene]) -> Option<u64> {
        let mut frontier = std::collections::HashSet::new();
        frontier.insert(start_id);
        let mut visited = frontier.clone();

        for _ in 0..10 {
            let mut next = std::collections::HashSet::new();
            for n in nodes.iter().filter(|n| n.enabled) {
                if !visited.contains(&n.innovation_number)
                    && n.parents.iter().any(|p| frontier.contains(p))
                {
                    if matches!(n.node_type, NTD::Concat { axis: 1 }) {
                        return Some(n.innovation_number);
                    }
                    next.insert(n.innovation_number);
                    visited.insert(n.innovation_number);
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        None
    }

    /// 追踪 edge conv 输出流向哪个 FM（返回 fm_id）
    fn find_edge_dst_fm(
        edge_conv_id: u64,
        nodes: &[NodeGene],
        _node_map: &HashMap<u64, &NodeGene>,
    ) -> Option<u64> {
        // 找到直接消费此 edge_conv 输出的节点
        for n in nodes.iter().filter(|n| n.enabled) {
            if n.parents.contains(&edge_conv_id) {
                if let Some(fid) = n.fm_id {
                    return Some(fid);
                }
                // 如果是 Add 聚合节点（无 fm_id），递归向下找
                if matches!(n.node_type, NTD::Add) {
                    if let Some(fid) = Self::find_edge_dst_fm(n.innovation_number, nodes, _node_map)
                    {
                        return Some(fid);
                    }
                }
            }
        }
        None
    }

    /// 递归收集 FM Add 聚合树中的所有节点 ID
    fn collect_fm_add_tree(
        node_id: u64,
        node_map: &HashMap<u64, &NodeGene>,
        collected: &mut std::collections::HashSet<u64>,
    ) {
        if let Some(node) = node_map.get(&node_id) {
            for &parent_id in &node.parents {
                if let Some(parent) = node_map.get(&parent_id) {
                    if matches!(parent.node_type, NTD::Add) && parent.fm_id.is_some() {
                        collected.insert(parent_id);
                        Self::collect_fm_add_tree(parent_id, node_map, collected);
                    }
                }
            }
        }
    }

    /// 检查节点是否被全连接优化合并替换
    fn is_merged_away(&self, node_id: u64) -> bool {
        self.merged_groups
            .iter()
            .any(|g| g.replaced_ids.contains(&node_id))
    }

    /// 对单个节点进行 FM 感知的类型转换
    ///
    /// 返回 (node_type, output_shape, parents)
    fn transform_node(
        &self,
        node: &NodeGene,
        _all_nodes: &[NodeGene],
    ) -> (NTD, Vec<usize>, Vec<u64>) {
        let id = node.innovation_number;

        // FM 输入 Identity → Narrow
        if let Some(&channel_idx) = self.fm_narrow_map.get(&id) {
            return (
                NTD::Narrow {
                    axis: 1,
                    start: channel_idx,
                    length: 1,
                },
                node.output_shape.clone(), // [1, 1, H, W]
                node.parents.clone(),
            );
        }

        // Concat 替换检查（全连接优化时 Concat 变为 Identity）
        for merged in &self.merged_groups {
            if let Some(&(concat_id, replacement_id)) = merged.concat_replacement.as_ref() {
                if id == concat_id {
                    return (
                        NTD::Identity,
                        node.output_shape.clone(),
                        vec![replacement_id],
                    );
                }
            }
        }

        // 默认：不变
        (
            node.node_type.clone(),
            node.output_shape.clone(),
            node.parents.clone(),
        )
    }
}

impl NetworkGenome {
    /// 将当前基因组转换为 `GraphDescriptor`
    ///
    /// 仅支持 NodeLevel 基因组。
    ///
    /// 返回的 `GraphDescriptor` 可直接传入 `Graph::from_descriptor()` 构建计算图。
    pub fn to_graph_descriptor(
        &self,
    ) -> Result<GraphDescriptor, super::node_expansion::NodeExpansionError> {
        let nodes = match &self.repr {
            GenomeRepr::NodeLevel { nodes, .. } => nodes.as_slice(),
            GenomeRepr::LayerLevel { .. } => {
                return Err(super::node_expansion::NodeExpansionError::InvalidGenome(
                    "to_graph_descriptor 仅支持 NodeLevel 基因组".into(),
                ));
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

        // Conv2d/ConvTranspose2d 形状合法性检查：
        // saturating_sub + max(1) 在 infer_output_shape 中防止 panic，
        // 但此处需要拒绝实际会产生退化输出的无效配置。
        for &id in &analysis.topo_order {
            if let Some(node) = node_lookup.get(&id) {
                if let Some(inp_shape) = analysis
                    .output_shapes
                    .get(&node.parents.first().copied().unwrap_or(0))
                {
                    if inp_shape.len() >= 4 {
                        match &node.node_type {
                            NTD::Conv2d {
                                stride,
                                padding,
                                dilation,
                            } => {
                                if node.parents.len() >= 2 {
                                    if let Some(ker_shape) =
                                        analysis.output_shapes.get(&node.parents[1])
                                    {
                                        if ker_shape.len() >= 4 {
                                            let eff_kh = dilation.0 * (ker_shape[2] - 1) + 1;
                                            let eff_kw = dilation.1 * (ker_shape[3] - 1) + 1;
                                            let h_raw = inp_shape[2] + 2 * padding.0;
                                            let w_raw = inp_shape[3] + 2 * padding.1;
                                            if h_raw < eff_kh || w_raw < eff_kw {
                                                return Err(
                                                    super::node_expansion::NodeExpansionError::InvalidGenome(
                                                        format!(
                                                            "Conv2d 节点 {} 的输入 {}x{} 对 kernel {}x{}（dilation {:?}）太小",
                                                            id,
                                                            inp_shape[2],
                                                            inp_shape[3],
                                                            ker_shape[2],
                                                            ker_shape[3],
                                                            dilation
                                                        ),
                                                    ),
                                                );
                                            }
                                            let h_out = (h_raw - eff_kh) / stride.0 + 1;
                                            let w_out = (w_raw - eff_kw) / stride.1 + 1;
                                            if h_out == 0 || w_out == 0 {
                                                return Err(
                                                    super::node_expansion::NodeExpansionError::InvalidGenome(
                                                        format!(
                                                            "Conv2d 节点 {} 输出尺寸为 0 ({}x{})",
                                                            id, h_out, w_out
                                                        ),
                                                    ),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            NTD::ConvTranspose2d {
                                stride, padding, ..
                            } => {
                                if node.parents.len() >= 2 {
                                    if let Some(ker_shape) =
                                        analysis.output_shapes.get(&node.parents[1])
                                    {
                                        if ker_shape.len() >= 4 {
                                            let h_sum =
                                                (inp_shape[2].max(1) - 1) * stride.0 + ker_shape[2];
                                            let w_sum =
                                                (inp_shape[3].max(1) - 1) * stride.1 + ker_shape[3];
                                            if h_sum < 2 * padding.0 || w_sum < 2 * padding.1 {
                                                return Err(
                                                    super::node_expansion::NodeExpansionError::InvalidGenome(
                                                        format!(
                                                            "ConvTranspose2d 节点 {} 的参数导致输出尺寸为负（padding {:?} 过大）",
                                                            id, padding
                                                        ),
                                                    ),
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        // FM 感知分析：
        // 1. 识别 FM 输入 Identity 节点并映射到 Narrow 通道索引
        // 2. 检测全连接 FM 组用于合并优化
        let fm_analysis = FMBuilderAnalysis::analyze(&nodes);

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

        // 收集需要在特定节点之前插入的合成节点
        let mut inject_before: HashMap<u64, Vec<usize>> = HashMap::new(); // node_id → [merged_group_index]
        for (gi, merged) in fm_analysis.merged_groups.iter().enumerate() {
            if let Some(&(concat_id, _)) = merged.concat_replacement.as_ref() {
                inject_before.entry(concat_id).or_default().push(gi);
            }
        }

        // 按拓扑序添加所有启用的 NodeGene（父节点必须在子节点之前）
        for &id in &analysis.topo_order {
            if let Some(node) = node_lookup.get(&id) {
                // 全连接优化：跳过已被合并替换的节点
                if fm_analysis.is_merged_away(id) {
                    continue;
                }

                // 在 Concat 节点之前注入合成节点
                if let Some(group_indices) = inject_before.get(&id) {
                    for &gi in group_indices {
                        for synth_node in &fm_analysis.merged_groups[gi].synthetic_nodes {
                            let dynamic = synth_node.shape.first().map(|_| {
                                let mut d: Vec<Option<usize>> =
                                    synth_node.shape.iter().map(|&x| Some(x)).collect();
                                if !d.is_empty() {
                                    d[0] = None;
                                }
                                d
                            });
                            desc.add_node(NodeDescriptor::new(
                                synth_node.id,
                                &synth_node.name,
                                synth_node.node_type.clone(),
                                synth_node.shape.clone(),
                                dynamic,
                                synth_node.parents.clone(),
                            ));
                        }
                    }
                }

                // 确定最终的 node_type 和 output_shape
                let (final_type, final_shape, final_parents) =
                    fm_analysis.transform_node(node, &nodes);

                let dynamic = final_shape.first().map(|_| {
                    let mut d: Vec<Option<usize>> = final_shape.iter().map(|&x| Some(x)).collect();
                    if !d.is_empty() {
                        d[0] = None;
                    } // batch 维动态
                    d
                });
                desc.add_node(NodeDescriptor::new(
                    node.innovation_number,
                    &format!("evo_{}", node.innovation_number),
                    final_type,
                    final_shape,
                    dynamic,
                    final_parents,
                ));
            }
        }

        // 演化 genome 需要显式输出：多头使用 genome.output_heads，旧单输出模型
        // 回退到最后一个非参数节点。FM 分解/融合会产生额外无后继中间端点，
        // 不能依赖 GraphDescriptor 的“无后继 = 输出”启发式。
        if self.output_heads.is_empty() {
            desc.explicit_output_ids = analysis
                .topo_order
                .iter()
                .rev()
                .find(|&&id| {
                    node_lookup
                        .get(&id)
                        .map(|node| !node.is_parameter())
                        .unwrap_or(false)
                })
                .map(|&id| vec![id]);
        } else {
            let mut ids = Vec::with_capacity(self.output_heads.len());
            for head in &self.output_heads {
                let Some(node) = node_lookup.get(&head.node_id) else {
                    return Err(super::node_expansion::NodeExpansionError::InvalidGenome(
                        format!(
                            "输出 head '{}' 引用不存在的节点 {}",
                            head.name, head.node_id
                        ),
                    ));
                };
                if node.is_parameter() {
                    return Err(super::node_expansion::NodeExpansionError::InvalidGenome(
                        format!(
                            "输出 head '{}' 不能引用参数节点 {}",
                            head.name, head.node_id
                        ),
                    ));
                }
                ids.push(head.node_id);
            }
            desc.explicit_output_ids = Some(ids);
        }

        Ok(desc)
    }

    /// 从基因组构建计算图
    ///
    /// - NodeLevel 基因组：`to_graph_descriptor()` + `Graph::from_descriptor()`
    /// - 含 edge-based 循环边的 NodeLevel 基因组：时间步展开构图路径
    ///
    /// rng 用于派生 Graph seed，确保参数初始化受 Evolution seed 控制。
    pub fn build(&self, rng: &mut StdRng) -> Result<BuildResult, GraphError> {
        if self.is_node_level() {
            // 检测是否含 edge-based 循环边
            let has_recurrent = self
                .nodes()
                .iter()
                .any(|n| n.enabled && !n.recurrent_parents.is_empty());
            if has_recurrent {
                return self.build_recurrent_from_nodes(rng);
            }
            return self.build_from_nodes(rng);
        }
        let mut migrated = self.clone();
        migrated
            .migrate_to_node_level()
            .map_err(|e| GraphError::ComputationError(e.to_string()))?;
        migrated.build(rng)
    }

    /// NodeLevel 基因组的构图路径（无循环边）
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
        let outputs = rebuild.outputs.clone();
        if outputs.is_empty() {
            return Err(GraphError::ComputationError(
                "NodeLevel 基因组构图后无输出节点".into(),
            ));
        }
        let output_heads = self.resolved_output_heads_for_build(&desc, &outputs)?;
        let default_output_idx = output_heads
            .iter()
            .position(|head| head.inference || head.primary)
            .unwrap_or(0);
        let output = outputs.get(default_output_idx).cloned().ok_or_else(|| {
            GraphError::ComputationError("NodeLevel 基因组构图后无输出节点".into())
        })?;

        // 收集参数节点：param_innovation → [Var]
        let nodes = self.nodes();
        let mut layer_params: HashMap<u64, Vec<Var>> = nodes
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

        // 补充合成参数节点（FM 融合产生的 merged kernel）并收集掩码信息
        let fm_analysis = FMBuilderAnalysis::analyze(nodes);
        let mut fm_masks: HashMap<u64, FMMaskInfo> = HashMap::new();
        for merged in &fm_analysis.merged_groups {
            for synth in &merged.synthetic_nodes {
                if matches!(synth.node_type, NTD::Parameter) {
                    if let Some(var) = rebuild.node_map.get(&synth.id) {
                        layer_params.insert(synth.id, vec![var.clone()]);
                    }
                }
            }
            fm_masks.insert(
                merged.merged_kernel_id,
                FMMaskInfo {
                    connected_pairs: merged.connected_pairs.clone(),
                    in_ch: merged.in_ch,
                    out_ch: merged.out_ch,
                },
            );
        }

        // 回填 NodeGroupTag：将 NodeGene 的 block_id 映射为可视化 Cluster 标签
        backfill_node_group_tags(self, &rebuild.node_map);

        let result = BuildResult {
            input,
            output,
            outputs,
            output_heads,
            layer_params,
            graph: rebuild.graph,
            fm_masks,
        };

        // 构建后立即对合成 kernel 施加掩码（首次构建时零化无连接位置）
        result.apply_fm_masks()?;

        Ok(result)
    }

    fn resolved_output_heads_for_build(
        &self,
        desc: &GraphDescriptor,
        outputs: &[Var],
    ) -> Result<Vec<OutputHead>, GraphError> {
        if !self.output_heads.is_empty() {
            if self.output_heads.len() != outputs.len() {
                return Err(GraphError::ComputationError(format!(
                    "输出 head 数量({})与构图输出数量({})不一致",
                    self.output_heads.len(),
                    outputs.len()
                )));
            }
            return Ok(self.output_heads.clone());
        }

        let output_id = desc
            .explicit_output_ids
            .as_ref()
            .and_then(|ids| ids.first())
            .copied()
            .unwrap_or(0);
        let output_dim = outputs
            .first()
            .and_then(|output| output.value_expected_shape().last().copied())
            .unwrap_or(self.output_dim);
        Ok(vec![OutputHead::new(
            "output", output_id, output_dim, true, true,
        )])
    }

    /// 含 edge-based 循环边的 NodeLevel 基因组构图路径
    ///
    /// 按时间步展开（类似 RNN `forward_seq`），每步重用共享参数：
    /// 1. 创建输入 `[batch, seq_len, features]`
    /// 2. 创建共享 Parameter Var（跨时间步共享权重）
    /// 3. 每个时间步按拓扑序逐节点计算，注入循环贡献
    /// 4. 堆叠所有时间步输出 → `[batch, seq_len, output_dim]`
    fn build_recurrent_from_nodes(&self, rng: &mut StdRng) -> Result<BuildResult, GraphError> {
        use super::node_gene::GenomeAnalysis;
        use super::node_ops::{genome_input_domain, genome_input_shape};

        let seq_len = self.seq_len.ok_or_else(|| {
            GraphError::ComputationError("edge-based 循环基因组需要 seq_len".into())
        })?;

        let graph_seed: u64 = rng.r#gen();
        let graph = Graph::new_with_seed(graph_seed).with_model_name("EvolutionNet");

        // 序列输入 [batch, seq_len, features]
        let input = graph.input_shape(&[1, seq_len, self.input_dim], Some("evo_input"))?;
        input.set_value(&Tensor::zeros(&[1, seq_len, self.input_dim]))?;

        // 分析获取拓扑序
        let analysis = GenomeAnalysis::compute(
            self.nodes(),
            INPUT_INNOVATION,
            genome_input_shape(self),
            genome_input_domain(self),
        );
        if !analysis.is_valid {
            return Err(GraphError::ComputationError(format!(
                "循环基因组分析失败: {:?}",
                analysis.errors
            )));
        }

        let node_lookup: HashMap<u64, &super::node_gene::NodeGene> = self
            .nodes()
            .iter()
            .filter(|n| n.enabled)
            .map(|n| (n.innovation_number, n))
            .collect();

        // 创建共享 Parameter Vars（跨时间步复用）
        let mut param_vars: HashMap<u64, Var> = HashMap::new();
        for node in self
            .nodes()
            .iter()
            .filter(|n| n.enabled && n.is_parameter())
        {
            let var = graph.parameter(
                &node.output_shape,
                Init::Xavier,
                &format!("evo_{}", node.innovation_number),
            )?;
            param_vars.insert(node.innovation_number, var);
        }

        // 收集所有循环边的源节点 ID
        let recurrent_source_ids: std::collections::HashSet<u64> = self
            .nodes()
            .iter()
            .filter(|n| n.enabled)
            .flat_map(|n| n.recurrent_parents.iter().map(|e| e.source_id))
            .collect();

        // 初始化循环状态为零
        let mut prev_activations: HashMap<u64, Var> = HashMap::new();
        for &sid in &recurrent_source_ids {
            let dim = if let Some(shape) = analysis.output_shapes.get(&sid) {
                *shape.last().unwrap_or(&self.input_dim)
            } else {
                self.input_dim
            };
            let zeros = graph.zeros_like(&input, &[dim], None)?;
            prev_activations.insert(sid, zeros);
        }

        let mut all_outputs: Vec<Var> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // x_t = input[:, t, :] → [batch, features]
            let x_t = input.narrow(1, t, 1)?.squeeze(Some(1))?;

            // 当前时间步的 node_id → Var 映射
            let mut activations: HashMap<u64, Var> = HashMap::new();
            activations.insert(INPUT_INNOVATION, x_t);

            // 参数共享引用
            for (&id, var) in &param_vars {
                activations.insert(id, var.clone());
            }

            // 按拓扑序逐节点计算
            for &id in &analysis.topo_order {
                let node = match node_lookup.get(&id) {
                    Some(n) => *n,
                    None => continue,
                };

                if node.is_leaf() {
                    continue;
                }

                let parent_vars: Vec<Var> = node
                    .parents
                    .iter()
                    .filter_map(|pid| activations.get(pid).cloned())
                    .collect();

                if parent_vars.len() != node.parents.len() {
                    return Err(GraphError::ComputationError(format!(
                        "时间步 {t}：节点 {id} 缺少父节点 Var"
                    )));
                }

                let mut result =
                    evaluate_step_node(&graph, &node.node_type, &parent_vars, &node.output_shape)
                        .map_err(|e| {
                        GraphError::ComputationError(format!(
                            "时间步 {t}：节点 {id} ({:?}) 计算失败: {e}",
                            node.node_type
                        ))
                    })?;

                // 注入循环贡献：result += prev_h @ W^T
                for edge in &node.recurrent_parents {
                    if let Some(prev_h) = prev_activations.get(&edge.source_id) {
                        if let Some(w) = param_vars.get(&edge.weight_param_id) {
                            // prev_h: [batch, source_dim], w: [target_dim, source_dim]
                            let w_t = w.transpose(0, 1)?;
                            let contribution = prev_h.matmul(&w_t)?;
                            result = &result + &contribution;
                        }
                    }
                }

                activations.insert(id, result);
            }

            // 更新循环状态
            for &sid in &recurrent_source_ids {
                if let Some(act) = activations.get(&sid) {
                    prev_activations.insert(sid, act.clone());
                }
            }

            // 收集本时间步的输出（拓扑序最后一个节点）
            let output_id = analysis
                .topo_order
                .last()
                .copied()
                .ok_or_else(|| GraphError::ComputationError("拓扑序为空".into()))?;
            let step_output = activations.get(&output_id).cloned().ok_or_else(|| {
                GraphError::ComputationError(format!("时间步 {t}：输出节点 {output_id} 无值"))
            })?;
            all_outputs.push(step_output);
        }

        // 堆叠所有时间步输出 → [batch, seq_len, output_dim]
        let output_refs: Vec<&Var> = all_outputs.iter().collect();
        let final_output = Var::stack(&output_refs, 1)?;

        let layer_params: HashMap<u64, Vec<Var>> = param_vars
            .into_iter()
            .map(|(id, v)| (id, vec![v]))
            .collect();

        Ok(BuildResult {
            input,
            output: final_output.clone(),
            outputs: vec![final_output.clone()],
            output_heads: vec![OutputHead::new("output", 0, self.output_dim, true, true)],
            layer_params,
            graph,
            fm_masks: HashMap::new(),
        })
    }

    /// 将当前计算图的权重捕获到 Genome 的参数节点快照中。
    pub fn capture_weights(&mut self, build: &BuildResult) -> Result<(), GraphError> {
        // 保存前对 FM 融合 kernel 施加掩码，确保断开位置权重归零
        build.apply_fm_masks()?;
        if !self.is_node_level() {
            self.migrate_to_node_level()
                .map_err(|e| GraphError::ComputationError(e.to_string()))?;
        }

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
    ) -> Result<Self, super::node_expansion::NodeExpansionError> {
        use super::node_expansion::NodeExpansionError;

        // 找 BasicInput 节点（虚拟输入）
        let input_nd = desc
            .nodes
            .iter()
            .find(|n| matches!(n.node_type, NTD::BasicInput))
            .ok_or_else(|| {
                NodeExpansionError::DimensionError("GraphDescriptor 中没有 BasicInput 节点".into())
            })?;

        let original_input_id = input_nd.id;
        let input_shape = &input_nd.output_shape;

        // 从输入形状推导模式：[batch, features] / [batch,seq,feat] / [batch,C,H,W]
        let (input_dim, seq_len, input_spatial) = match input_shape.len() {
            2 => (input_shape[1], None, None),
            3 => (input_shape[2], Some(input_shape[1]), None),
            4 => (input_shape[1], None, Some((input_shape[2], input_shape[3]))),
            _ => {
                return Err(NodeExpansionError::DimensionError(format!(
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
            return Err(NodeExpansionError::DimensionError(
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
            .ok_or_else(|| NodeExpansionError::DimensionError("无法确定输出节点".into()))?;
        let output_dim = match output_shape.len() {
            n if n >= 2 => output_shape[output_shape.len() - 1],
            1 => output_shape[0],
            _ => {
                return Err(NodeExpansionError::DimensionError(
                    "输出节点形状为空".into(),
                ));
            }
        };

        Ok(Self {
            input_dim,
            output_dim,
            seq_len,
            input_spatial,
            training_config: super::gene::TrainingConfig::default(),
            generated_by: "from_graph_descriptor".to_string(),
            output_heads: Vec::new(),
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
    ) -> Result<Self, super::node_expansion::NodeExpansionError> {
        let import_result = crate::nn::graph::onnx_import::load_onnx(path).map_err(|e| {
            super::node_expansion::NodeExpansionError::DimensionError(format!("ONNX 导入失败: {e}"))
        })?;
        Self::from_graph_descriptor(&import_result.descriptor)
    }

    /// 从内存中的 .onnx 字节流构建 NetworkGenome
    pub fn from_onnx_bytes(
        bytes: &[u8],
    ) -> Result<Self, super::node_expansion::NodeExpansionError> {
        let import_result =
            crate::nn::graph::onnx_import::load_onnx_from_bytes(bytes).map_err(|e| {
                super::node_expansion::NodeExpansionError::DimensionError(format!(
                    "ONNX 导入失败: {e}"
                ))
            })?;
        Self::from_graph_descriptor(&import_result.descriptor)
    }

    /// 从参数节点快照恢复权重到当前计算图。
    ///
    /// - 形状完全相同 → 全量继承（`inherited`）
    /// - 仅一轴扩缩 → 部分继承，重叠区域保留旧值（`partially_inherited`）
    /// - 两轴均变化或无快照 → 保留随机初始化（`reinitialized`）
    pub fn restore_weights(&self, build: &BuildResult) -> Result<InheritReport, GraphError> {
        let mut inherited = 0usize;
        let mut partially_inherited = 0usize;
        let mut reinitialized = 0usize;

        if !self.is_node_level() {
            return Err(GraphError::ComputationError(
                "restore_weights 仅支持 NodeLevel 基因组".into(),
            ));
        }

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

        // 恢复后对 FM 融合 kernel 施加掩码，确保断开位置归零
        build.apply_fm_masks()?;

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
/// NodeLevel 经 descriptor rebuild 后无上下文标签；此函数利用
/// `NodeGene::block_id` 在构图完成后补填。
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
            NodeBlockKind::ConvTranspose2d { .. } => ("ConvTranspose2d", GroupStyle::Layer),
            NodeBlockKind::Pool2d { .. } => ("Pool2d", GroupStyle::Layer),
            NodeBlockKind::Flatten => ("Flatten", GroupStyle::Layer),
            NodeBlockKind::Dropout { .. } => ("Dropout", GroupStyle::Layer),
            NodeBlockKind::BatchNorm { .. } => ("BatchNorm", GroupStyle::Layer),
            NodeBlockKind::LayerNorm { .. } => ("LayerNorm", GroupStyle::Layer),
            NodeBlockKind::RMSNorm { .. } => ("RMSNorm", GroupStyle::Layer),
            NodeBlockKind::Activation { .. } => ("Activation", GroupStyle::Layer),
            NodeBlockKind::Rnn { .. } => ("RNN", GroupStyle::Recurrent),
            NodeBlockKind::Lstm { .. } => ("LSTM", GroupStyle::Recurrent),
            NodeBlockKind::Gru { .. } => ("GRU", GroupStyle::Recurrent),
            NodeBlockKind::SkipAgg | NodeBlockKind::Unknown => continue,
        };

        let fmt_shape = |s: &[usize]| -> String {
            let parts: Vec<String> = s
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
            format!("[{}]", parts.join(", "))
        };

        let input_shape = if block.input_id == INPUT_INNOVATION {
            Some(super::node_ops::genome_input_shape(genome))
        } else {
            genome
                .nodes()
                .iter()
                .find(|n| n.innovation_number == block.input_id)
                .map(|n| n.output_shape.clone())
        };
        let output_shape = genome
            .nodes()
            .iter()
            .find(|n| n.innovation_number == block.output_id)
            .map(|n| n.output_shape.clone());

        let description = match (input_shape, output_shape) {
            (Some(inp), Some(out)) => Some(format!("{} → {}", fmt_shape(&inp), fmt_shape(&out))),
            (None, Some(out)) => Some(fmt_shape(&out)),
            _ => None,
        };

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
                // Backfill 仅作为兜底：若节点已有 tag（由 unroll 期间的 NodeGroupContext
                // 权威标记，例如 Rnn::unroll 会给 W_ih/W_hh/b_h 打上含正确 instance_id
                // 的 Recurrent tag），此处不得覆盖——否则同一逻辑块会被拆成两个
                // instance_id 不同的 cluster，在可视化中显示为两个并列 box。
                if var.node().node_group_tag().is_some() {
                    continue;
                }
                var.node().set_node_group_tag(Some(tag.clone()));
            }
        }
    }
}

/// 循环展开构图中的单节点计算
///
/// 根据 `NodeTypeDescriptor` 和父节点 Var 计算当前节点的输出 Var。
/// 仅覆盖 Flat/Sequence 域中可能出现的节点类型。
fn evaluate_step_node(
    _graph: &Graph,
    node_type: &NodeTypeDescriptor,
    parents: &[Var],
    output_shape: &[usize],
) -> Result<Var, GraphError> {
    use NodeTypeDescriptor as NT;
    match node_type {
        // ── 算术 ──
        NT::MatMul => {
            require_parents(2, parents, "MatMul")?;
            parents[0].matmul(&parents[1])
        }
        NT::Add => {
            require_parents(2, parents, "Add")?;
            Ok(&parents[0] + &parents[1])
        }
        NT::Subtract => {
            require_parents(2, parents, "Subtract")?;
            Ok(&parents[0] - &parents[1])
        }
        NT::Multiply => {
            require_parents(2, parents, "Multiply")?;
            Ok(&parents[0] * &parents[1])
        }
        NT::Negate => {
            require_parents(1, parents, "Negate")?;
            Ok(-&parents[0])
        }

        // ── 激活函数 ──
        NT::ReLU => {
            require_parents(1, parents, "ReLU")?;
            Ok(parents[0].relu())
        }
        NT::Tanh => {
            require_parents(1, parents, "Tanh")?;
            Ok(parents[0].tanh())
        }
        NT::Sigmoid => {
            require_parents(1, parents, "Sigmoid")?;
            Ok(parents[0].sigmoid())
        }
        NT::LeakyReLU { alpha } => {
            require_parents(1, parents, "LeakyReLU")?;
            Ok(parents[0].leaky_relu(*alpha))
        }
        NT::Gelu => {
            require_parents(1, parents, "Gelu")?;
            Ok(parents[0].gelu())
        }
        NT::Swish => {
            require_parents(1, parents, "Swish")?;
            Ok(parents[0].silu())
        }
        NT::Elu { alpha } => {
            require_parents(1, parents, "Elu")?;
            Ok(parents[0].elu(*alpha))
        }
        NT::Selu => {
            require_parents(1, parents, "Selu")?;
            Ok(parents[0].selu())
        }
        NT::Mish => {
            require_parents(1, parents, "Mish")?;
            Ok(parents[0].mish())
        }
        NT::HardSwish => {
            require_parents(1, parents, "HardSwish")?;
            Ok(parents[0].hard_swish())
        }
        NT::HardSigmoid => {
            require_parents(1, parents, "HardSigmoid")?;
            Ok(parents[0].hard_sigmoid())
        }
        NT::SoftPlus => {
            require_parents(1, parents, "SoftPlus")?;
            Ok(parents[0].softplus())
        }
        NT::ReLU6 => {
            require_parents(1, parents, "ReLU6")?;
            Ok(parents[0].relu6())
        }

        // ── 形状变换 ──
        NT::Identity | NT::Detach => {
            require_parents(1, parents, "Identity")?;
            Ok(parents[0].clone())
        }
        NT::Flatten { .. } => {
            require_parents(1, parents, "Flatten")?;
            let flat_dim: usize = output_shape[1..].iter().product::<usize>().max(1);
            parents[0].reshape(&[output_shape[0].max(1), flat_dim])
        }

        // ── 聚合 ──
        NT::Maximum => {
            require_parents(2, parents, "Maximum")?;
            parents[0].maximum(&parents[1])
        }
        NT::Concat { axis } => {
            let refs: Vec<&Var> = parents.iter().collect();
            Var::concat(&refs, *axis)
        }

        // ── 不支持 ──
        other => Err(GraphError::ComputationError(format!(
            "循环展开构图不支持的节点类型: {:?}",
            other
        ))),
    }
}

/// 辅助：检查父节点数量
fn require_parents(expected: usize, parents: &[Var], op_name: &str) -> Result<(), GraphError> {
    if parents.len() < expected {
        return Err(GraphError::ComputationError(format!(
            "{op_name} 需要至少 {expected} 个父节点，实际 {}",
            parents.len()
        )));
    }
    Ok(())
}

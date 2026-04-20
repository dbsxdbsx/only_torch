//! Net2Net: 函数保持（function-preserving）的神经元扩宽。
//!
//! # 基本思想
//!
//! 当某一层输出维度 `h_old → h_new`（`h_new > h_old`）时：
//! - 前 `h_old` 个位置保持恒等映射：`mapping[i] = i`
//! - 后续新增位置从 `[0, h_old)` 随机选取：`mapping[i] = rng.choose`
//!
//! 对参数张量进行如下变换（下面以矩阵乘法 `y = W·x + b` 为例）：
//! - **Owner 块**（被扩宽的层）的 `W[:, j]` 与 `b[:, j]`：
//!   - `W_new[:, j] = W_old[:, mapping[j]]`（**复制**，不缩放）
//!   - `b_new[:, j] = b_old[:, mapping[j]]`
//! - **Consumer 块**（紧跟的下一层，其输入维度随 owner 的输出维度变化）：
//!   - `W_down_new[i, :] = W_down_old[mapping[i], :] / counts[mapping[i]]`
//!   - 其中 `counts[k]` = mapping 中值 `k` 出现的次数
//!
//! 这样可以保证任意输入 `x` 下，新网络与旧网络的输出严格相等（浮点精度内）。

use rand::Rng;
use rand::rngs::StdRng;
use std::collections::HashMap;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::NetworkGenome;
use crate::nn::evolution::node_gene::NodeGene;
use crate::nn::evolution::node_ops::{NodeBlock, NodeBlockKind, node_main_path};
use crate::tensor::Tensor;

// ==================== 原语：mapping / counts / gather ====================

/// 生成函数保持扩宽映射表。
pub(crate) fn widening_mapping(old_size: usize, new_size: usize, rng: &mut StdRng) -> Vec<usize> {
    assert!(new_size >= old_size, "Net2Net widening requires new >= old");
    let mut m = Vec::with_capacity(new_size);
    for i in 0..old_size {
        m.push(i);
    }
    if old_size > 0 {
        for _ in old_size..new_size {
            m.push(rng.gen_range(0..old_size));
        }
    }
    m
}

/// 计算每个源索引在 `mapping` 中出现的次数。
pub(crate) fn counts_of(mapping: &[usize], old_size: usize) -> Vec<usize> {
    let mut c = vec![0usize; old_size];
    for &j in mapping {
        if j < old_size {
            c[j] += 1;
        }
    }
    c
}

/// 沿指定轴按 `mapping` 重排（纯复制，不缩放）。
pub(crate) fn gather_along_axis(src: &Tensor, axis: usize, mapping: &[usize]) -> Tensor {
    assert!(!mapping.is_empty(), "gather_along_axis: empty mapping");
    let slices: Vec<Tensor> = mapping.iter().map(|&j| src.narrow(axis, j, 1)).collect();
    let refs: Vec<&Tensor> = slices.iter().collect();
    Tensor::concat(&refs, axis)
}

/// 沿指定轴按 `mapping` 重排并按 `counts` 缩放。
pub(crate) fn gather_along_axis_scaled(
    src: &Tensor,
    axis: usize,
    mapping: &[usize],
    counts: &[usize],
) -> Tensor {
    assert!(
        !mapping.is_empty(),
        "gather_along_axis_scaled: empty mapping"
    );
    let slices: Vec<Tensor> = mapping
        .iter()
        .map(|&j| {
            let s = src.narrow(axis, j, 1);
            let c = counts[j].max(1) as f32;
            &s * (1.0_f32 / c)
        })
        .collect();
    let refs: Vec<&Tensor> = slices.iter().collect();
    Tensor::concat(&refs, axis)
}

// ==================== 定位参数节点 ====================

/// 块内所有参数节点的创新号
fn block_param_ids(genome: &NetworkGenome, block: &NodeBlock) -> Vec<u64> {
    let nodes = genome.nodes();
    block
        .node_ids
        .iter()
        .copied()
        .filter(|&nid| {
            nodes
                .iter()
                .any(|n| n.innovation_number == nid && n.is_parameter())
        })
        .collect()
}

/// 定位 Linear 块的 W 参数（通过 MatMul 的父节点关系），返回 `(W_id, b_id)`
fn locate_linear_params(genome: &NetworkGenome, block: &NodeBlock) -> Option<(u64, u64)> {
    let nodes = genome.nodes();
    let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();
    let matmul = nodes.iter().find(|n| {
        bid_set.contains(&n.innovation_number) && matches!(n.node_type, NodeTypeDescriptor::MatMul)
    })?;
    let w_id = matmul.parents.iter().copied().find(|&pid| {
        bid_set.contains(&pid)
            && nodes
                .iter()
                .any(|n| n.innovation_number == pid && n.is_parameter())
    })?;
    let b_id = block_param_ids(genome, block)
        .into_iter()
        .find(|&id| id != w_id)?;
    Some((w_id, b_id))
}

/// 定位 Conv2d 块的 kernel 参数
fn locate_conv2d_kernel(genome: &NetworkGenome, block: &NodeBlock) -> Option<u64> {
    let nodes = genome.nodes();
    let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();
    let conv = nodes.iter().find(|n| {
        bid_set.contains(&n.innovation_number)
            && matches!(n.node_type, NodeTypeDescriptor::Conv2d { .. })
    })?;
    conv.parents.iter().copied().find(|&pid| {
        bid_set.contains(&pid)
            && nodes
                .iter()
                .any(|n| n.innovation_number == pid && n.is_parameter())
    })
}

/// 定位循环块的参数组，按门分组返回 `(W_ix, W_hx, b_x)`
fn locate_recurrent_cell_params(
    genome: &NetworkGenome,
    block: &NodeBlock,
) -> Option<Vec<(u64, u64, u64)>> {
    let nodes = genome.nodes();
    let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();
    let cell = nodes.iter().find(|n| {
        bid_set.contains(&n.innovation_number)
            && matches!(
                n.node_type,
                NodeTypeDescriptor::CellRnn { .. }
                    | NodeTypeDescriptor::CellLstm { .. }
                    | NodeTypeDescriptor::CellGru { .. }
            )
    })?;
    let param_ids: Vec<u64> = cell.parents.iter().skip(1).copied().collect();
    if param_ids.is_empty() || param_ids.len() % 3 != 0 {
        return None;
    }
    let mut gates = Vec::with_capacity(param_ids.len() / 3);
    for chunk in param_ids.chunks(3) {
        gates.push((chunk[0], chunk[1], chunk[2]));
    }
    Some(gates)
}

/// 判断是否是 "不影响特征维度" 的纯透传块
fn is_pass_through(kind: &NodeBlockKind) -> bool {
    matches!(
        kind,
        NodeBlockKind::Activation { .. }
            | NodeBlockKind::Dropout { .. }
            | NodeBlockKind::Pool2d { .. }
    )
}

// ==================== 主入口：对快照应用 Net2Net 扩宽 ====================

/// 对 `genome.node_weight_snapshots` 应用函数保持扩宽。
///
/// 调用时机：`resize_*_out` **之后**，此时 block 参数节点的 `output_shape` 已是新
/// 尺寸，但快照仍是旧尺寸。我们读旧快照 → 生成新张量 → 替换快照。
///
/// 返回 `Ok(true)` 表示成功；`Ok(false)` 表示无法安全应用（由外层清理并走朴素
/// 回退）；`Err` 表示内部不一致。
pub(crate) fn apply_widen_to_snapshots(
    genome: &mut NetworkGenome,
    block: &NodeBlock,
    old_size: usize,
    new_size: usize,
    rng: &mut StdRng,
) -> Result<bool, String> {
    if new_size <= old_size {
        return Ok(false);
    }
    if !genome.is_node_level() {
        return Ok(false);
    }

    let mapping = widening_mapping(old_size, new_size, rng);
    let counts = counts_of(&mapping, old_size);

    let mut new_snapshots: HashMap<u64, Tensor> = HashMap::new();
    let owner_ok = match &block.kind {
        NodeBlockKind::Linear { .. } => {
            widen_linear_owner(genome, block, old_size, &mapping, &mut new_snapshots)
        }
        NodeBlockKind::Conv2d { .. } => {
            widen_conv2d_owner(genome, block, old_size, &mapping, &mut new_snapshots)
        }
        NodeBlockKind::Rnn { .. } | NodeBlockKind::Lstm { .. } | NodeBlockKind::Gru { .. } => {
            widen_recurrent_owner(
                genome,
                block,
                old_size,
                &mapping,
                &counts,
                &mut new_snapshots,
            )
        }
        _ => return Ok(false),
    };
    if !owner_ok {
        return Ok(false);
    }

    let blocks = node_main_path(genome);
    let owner_idx = match blocks
        .iter()
        .position(|b| b.block_id == block.block_id && b.node_ids == block.node_ids)
    {
        Some(i) => i,
        None => return Ok(false),
    };

    let mut flatten_stride: Option<usize> = None;

    for down in blocks.iter().skip(owner_idx + 1) {
        match &down.kind {
            k if is_pass_through(k) => continue,
            NodeBlockKind::BatchNorm { .. }
            | NodeBlockKind::LayerNorm { .. }
            | NodeBlockKind::RMSNorm { .. } => {
                for pid in block_param_ids(genome, down) {
                    if let Some(old) = genome.node_weight_snapshots().get(&pid).cloned() {
                        let shape = old.shape().to_vec();
                        if shape.get(1).copied() == Some(old_size) {
                            let new_t = gather_along_axis(&old, 1, &mapping);
                            new_snapshots.insert(pid, new_t);
                        }
                    }
                }
                continue;
            }
            NodeBlockKind::Flatten => {
                if !matches!(&block.kind, NodeBlockKind::Conv2d { .. }) {
                    return Ok(false);
                }
                let flat = genome
                    .nodes()
                    .iter()
                    .find(|n| n.innovation_number == down.output_id)
                    .and_then(|n| n.output_shape.get(1).copied());
                match flat {
                    Some(c_h_w) if c_h_w % old_size == 0 => {
                        flatten_stride = Some(c_h_w / old_size);
                    }
                    _ => return Ok(false),
                }
                continue;
            }
            NodeBlockKind::SkipAgg => {
                return Ok(false);
            }
            NodeBlockKind::Linear { .. } => {
                if let Some((w_id, _b_id)) = locate_linear_params(genome, down) {
                    if let Some(old) = genome.node_weight_snapshots().get(&w_id).cloned() {
                        let shape = old.shape().to_vec();
                        match flatten_stride {
                            None => {
                                if shape.first().copied() != Some(old_size) {
                                    return Ok(false);
                                }
                                let new_t = gather_along_axis_scaled(&old, 0, &mapping, &counts);
                                new_snapshots.insert(w_id, new_t);
                            }
                            Some(stride) => {
                                let expected = old_size * stride;
                                if shape.first().copied() != Some(expected) {
                                    return Ok(false);
                                }
                                let new_t =
                                    gather_linear_in_with_flatten(&old, &mapping, &counts, stride);
                                new_snapshots.insert(w_id, new_t);
                            }
                        }
                    } else {
                        return Ok(false);
                    }
                } else {
                    return Ok(false);
                }
                break;
            }
            NodeBlockKind::Conv2d { .. } => {
                if let Some(k_id) = locate_conv2d_kernel(genome, down) {
                    if let Some(old) = genome.node_weight_snapshots().get(&k_id).cloned() {
                        let shape = old.shape().to_vec();
                        if shape.get(1).copied() != Some(old_size) {
                            return Ok(false);
                        }
                        let new_t = gather_along_axis_scaled(&old, 1, &mapping, &counts);
                        new_snapshots.insert(k_id, new_t);
                    } else {
                        return Ok(false);
                    }
                } else {
                    return Ok(false);
                }
                break;
            }
            NodeBlockKind::Rnn { .. } | NodeBlockKind::Lstm { .. } | NodeBlockKind::Gru { .. } => {
                let gates = match locate_recurrent_cell_params(genome, down) {
                    Some(g) => g,
                    None => return Ok(false),
                };
                for (w_ix, _w_hx, _b_x) in &gates {
                    if let Some(old) = genome.node_weight_snapshots().get(w_ix).cloned() {
                        let shape = old.shape().to_vec();
                        if shape.first().copied() != Some(old_size) {
                            return Ok(false);
                        }
                        let new_t = gather_along_axis_scaled(&old, 0, &mapping, &counts);
                        new_snapshots.insert(*w_ix, new_t);
                    } else {
                        return Ok(false);
                    }
                }
                break;
            }
            _ => return Ok(false),
        }
    }

    let snaps = genome.node_weight_snapshots_mut();
    for (id, t) in new_snapshots {
        snaps.insert(id, t);
    }
    Ok(true)
}

// ==================== Owner 扩宽：Linear ====================

fn widen_linear_owner(
    genome: &NetworkGenome,
    block: &NodeBlock,
    old_size: usize,
    mapping: &[usize],
    out: &mut HashMap<u64, Tensor>,
) -> bool {
    let (w_id, b_id) = match locate_linear_params(genome, block) {
        Some(x) => x,
        None => return false,
    };
    let snaps = genome.node_weight_snapshots();
    let w_old = match snaps.get(&w_id) {
        Some(t) => t.clone(),
        None => return false,
    };
    if w_old.shape().get(1).copied() != Some(old_size) {
        return false;
    }
    let w_new = gather_along_axis(&w_old, 1, mapping);
    let b_old = match snaps.get(&b_id) {
        Some(t) => t.clone(),
        None => return false,
    };
    if b_old.shape().get(1).copied() != Some(old_size) {
        return false;
    }
    let b_new = gather_along_axis(&b_old, 1, mapping);
    out.insert(w_id, w_new);
    out.insert(b_id, b_new);
    true
}

// ==================== Owner 扩宽：Conv2d ====================

fn widen_conv2d_owner(
    genome: &NetworkGenome,
    block: &NodeBlock,
    old_size: usize,
    mapping: &[usize],
    out: &mut HashMap<u64, Tensor>,
) -> bool {
    let k_id = match locate_conv2d_kernel(genome, block) {
        Some(x) => x,
        None => return false,
    };
    let snaps = genome.node_weight_snapshots();
    let k_old = match snaps.get(&k_id) {
        Some(t) => t.clone(),
        None => return false,
    };
    if k_old.shape().first().copied() != Some(old_size) {
        return false;
    }
    let k_new = gather_along_axis(&k_old, 0, mapping);
    out.insert(k_id, k_new);

    let other_ids: Vec<u64> = block_param_ids(genome, block)
        .into_iter()
        .filter(|&id| id != k_id)
        .collect();
    for pid in other_ids {
        if let Some(t) = snaps.get(&pid).cloned() {
            let shape = t.shape().to_vec();
            if shape.len() == 4 && shape.get(1).copied() == Some(old_size) {
                let new_t = gather_along_axis(&t, 1, mapping);
                out.insert(pid, new_t);
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    true
}

// ==================== Owner 扩宽：Rnn / Lstm / Gru ====================

fn widen_recurrent_owner(
    genome: &NetworkGenome,
    block: &NodeBlock,
    old_size: usize,
    mapping: &[usize],
    counts: &[usize],
    out: &mut HashMap<u64, Tensor>,
) -> bool {
    let gates = match locate_recurrent_cell_params(genome, block) {
        Some(g) => g,
        None => return false,
    };
    let snaps = genome.node_weight_snapshots();
    for (w_ix, w_hx, b_x) in &gates {
        let wix_old = match snaps.get(w_ix) {
            Some(t) => t.clone(),
            None => return false,
        };
        if wix_old.shape().get(1).copied() != Some(old_size) {
            return false;
        }
        let wix_new = gather_along_axis(&wix_old, 1, mapping);
        out.insert(*w_ix, wix_new);

        let whx_old = match snaps.get(w_hx) {
            Some(t) => t.clone(),
            None => return false,
        };
        let shape = whx_old.shape().to_vec();
        if shape.len() != 2 || shape[0] != old_size || shape[1] != old_size {
            return false;
        }
        // W_hh 两轴变换：axis=0 按 counts 缩放复制（行/输出轴），axis=1 纯复制（列/输入轴）
        //   W_hh_new[i, j] = W_hh[mapping[i], mapping[j]] / counts[mapping[i]]
        // 推导：希望 pre_new[j] = pre_old[mapping[j]]，即
        //   Σᵢ h_prev[mapping[i]] · W_hh_new[i, j] = Σₖ h_prev[k] · W_hh[k, mapping[j]]
        // 按 mapping[i]=k 分组（k 出现 counts[k] 次）即可证。
        let whx_mid = gather_along_axis_scaled(&whx_old, 0, mapping, counts);
        let whx_new = gather_along_axis(&whx_mid, 1, mapping);
        out.insert(*w_hx, whx_new);

        let bx_old = match snaps.get(b_x) {
            Some(t) => t.clone(),
            None => return false,
        };
        if bx_old.shape().get(1).copied() != Some(old_size) {
            return false;
        }
        let bx_new = gather_along_axis(&bx_old, 1, mapping);
        out.insert(*b_x, bx_new);
    }
    true
}

// ==================== Flatten → Linear 缩放复制 ====================

/// Owner 是 Conv2d 且下游经 Flatten 接 Linear 时的 W 输入维度扩展。
/// Linear W 输入行数为 `old_channels * stride`（`stride = H*W`）。
/// 每个新通道 `c_new` 取旧通道 `mapping[c_new]` 对应的 `stride` 行块，
/// 并除以 `counts[mapping[c_new]]`。
pub(crate) fn gather_linear_in_with_flatten(
    w_old: &Tensor,
    mapping: &[usize],
    counts: &[usize],
    stride: usize,
) -> Tensor {
    let slices: Vec<Tensor> = mapping
        .iter()
        .map(|&c_old| {
            let start = c_old * stride;
            let block = w_old.narrow(0, start, stride);
            let scale = 1.0_f32 / counts[c_old].max(1) as f32;
            &block * scale
        })
        .collect();
    let refs: Vec<&Tensor> = slices.iter().collect();
    Tensor::concat(&refs, 0)
}

// ==================== 便利函数 ====================

#[allow(dead_code)]
fn collect_params<'a>(genome: &'a NetworkGenome, block: &NodeBlock) -> Vec<&'a NodeGene> {
    let nodes = genome.nodes();
    let bid_set: std::collections::HashSet<u64> = block.node_ids.iter().copied().collect();
    nodes
        .iter()
        .filter(|n| bid_set.contains(&n.innovation_number) && n.is_parameter())
        .collect()
}

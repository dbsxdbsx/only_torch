/*
 * @Author       : 老董
 * @Date         : 2026-05-02
 * @Description  : CellAttention 演化原语单元测试
 *
 * 覆盖三件事：
 * 1. expand_attention 生成 9 个节点（8 参数 + 1 复合 cell），父子关系正确。
 * 2. infer_output_shape / infer_domain 对 CellAttention 的输出与域推导。
 * 3. infer_block_kind 能识别 CellAttention 块为 NodeBlockKind::Attention。
 */

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::{INPUT_INNOVATION, NetworkGenome, ShapeDomain};
use crate::nn::evolution::node_expansion::{InnovationCounter, expand_attention};
use crate::nn::evolution::node_gene::{infer_domain, infer_output_shape};
use crate::nn::evolution::node_ops::{
    NodeBlockKind, commit_counter, insert_after, make_counter, next_block_id, node_main_path,
    repair_param_input_dims,
};

const IN_DIM: usize = 12;
const EMBED_DIM: usize = 16;
const NUM_HEADS: usize = 4;
const SEQ_LEN: usize = 7;

#[test]
fn expand_attention_returns_9_nodes_with_correct_shapes() {
    let mut counter = InnovationCounter::new(INPUT_INNOVATION + 1);
    let nodes = expand_attention(
        INPUT_INNOVATION,
        IN_DIM,
        EMBED_DIM,
        NUM_HEADS,
        false,
        SEQ_LEN,
        0,
        &mut counter,
    );

    assert_eq!(nodes.len(), 9, "expand_attention 应返回 8 参数 + 1 cell");

    // 前 8 个全部是 Parameter，最后一个是 CellAttention
    for (i, n) in nodes.iter().take(8).enumerate() {
        assert!(
            n.is_parameter(),
            "第 {} 个节点应是 Parameter，实际 {:?}",
            i,
            n.node_type
        );
    }
    let cell = nodes.last().unwrap();
    assert!(
        matches!(cell.node_type, NodeTypeDescriptor::CellAttention { .. }),
        "最后一个节点应是 CellAttention"
    );

    // 形状校验：W_q/W_k/W_v: [in, embed]; W_o: [embed, embed]; bias 全部 [1, embed]
    let qkv_w = vec![IN_DIM, EMBED_DIM];
    let o_w = vec![EMBED_DIM, EMBED_DIM];
    let bias = vec![1, EMBED_DIM];
    assert_eq!(nodes[0].output_shape, qkv_w, "W_q");
    assert_eq!(nodes[1].output_shape, bias, "b_q");
    assert_eq!(nodes[2].output_shape, qkv_w, "W_k");
    assert_eq!(nodes[3].output_shape, bias, "b_k");
    assert_eq!(nodes[4].output_shape, qkv_w, "W_v");
    assert_eq!(nodes[5].output_shape, bias, "b_v");
    assert_eq!(nodes[6].output_shape, o_w, "W_o");
    assert_eq!(nodes[7].output_shape, bias, "b_o");

    // CellAttention 输出：!return_sequences → [1, embed_dim]
    assert_eq!(cell.output_shape, vec![1, EMBED_DIM]);

    // 父节点关系：[input, w_q, b_q, w_k, b_k, w_v, b_v, w_o, b_o]
    let expected_parents: Vec<u64> = std::iter::once(INPUT_INNOVATION)
        .chain(nodes.iter().take(8).map(|n| n.innovation_number))
        .collect();
    assert_eq!(cell.parents, expected_parents);

    // block_id 必须在所有节点上一致
    let bid = cell.block_id;
    for n in &nodes {
        assert_eq!(n.block_id, bid);
    }
}

#[test]
fn expand_attention_return_sequences_changes_output_shape() {
    let mut counter = InnovationCounter::new(INPUT_INNOVATION + 1);
    let nodes = expand_attention(
        INPUT_INNOVATION,
        IN_DIM,
        EMBED_DIM,
        NUM_HEADS,
        true, // return_sequences
        SEQ_LEN,
        0,
        &mut counter,
    );
    let cell = nodes.last().unwrap();
    assert_eq!(cell.output_shape, vec![1, SEQ_LEN, EMBED_DIM]);
}

#[test]
fn cell_attention_infer_output_shape_handles_both_return_modes() {
    // !return_sequences → [1, embed_dim]
    let nt = NodeTypeDescriptor::CellAttention {
        input_size: IN_DIM,
        embed_dim: EMBED_DIM,
        num_heads: NUM_HEADS,
        return_sequences: false,
        seq_len: SEQ_LEN,
    };
    let parent = vec![1, SEQ_LEN, IN_DIM];
    let parent_refs = vec![&parent];
    let shape = infer_output_shape(&nt, &parent_refs).unwrap();
    assert_eq!(shape, vec![1, EMBED_DIM]);

    // return_sequences → [1, seq_len, embed_dim]
    let nt = NodeTypeDescriptor::CellAttention {
        input_size: IN_DIM,
        embed_dim: EMBED_DIM,
        num_heads: NUM_HEADS,
        return_sequences: true,
        seq_len: SEQ_LEN,
    };
    let shape = infer_output_shape(&nt, &parent_refs).unwrap();
    assert_eq!(shape, vec![1, SEQ_LEN, EMBED_DIM]);
}

#[test]
fn cell_attention_infer_domain_follows_return_sequences() {
    let parents = vec![ShapeDomain::Sequence];

    let nt_seq = NodeTypeDescriptor::CellAttention {
        input_size: IN_DIM,
        embed_dim: EMBED_DIM,
        num_heads: NUM_HEADS,
        return_sequences: true,
        seq_len: SEQ_LEN,
    };
    assert_eq!(infer_domain(&nt_seq, &parents), ShapeDomain::Sequence);

    let nt_flat = NodeTypeDescriptor::CellAttention {
        input_size: IN_DIM,
        embed_dim: EMBED_DIM,
        num_heads: NUM_HEADS,
        return_sequences: false,
        seq_len: SEQ_LEN,
    };
    assert_eq!(infer_domain(&nt_flat, &parents), ShapeDomain::Flat);
}

#[test]
fn node_main_path_recognizes_attention_block() {
    // 构造一个 attention-only 序列基因组：input → CellAttention → Linear (output)
    let mut genome = NetworkGenome::minimal_sequential(IN_DIM, 2);
    genome.seq_len = Some(SEQ_LEN);

    // 删除原 RNN 主路径，用 attention 替换。简化做法：直接在 input 后插入 attention 块，
    // 然后让原有 Linear 块自动接到新 attention 输出。
    let mut counter = make_counter(&genome);
    let attn_nodes = expand_attention(
        INPUT_INNOVATION,
        IN_DIM,
        EMBED_DIM,
        NUM_HEADS,
        false,
        SEQ_LEN,
        next_block_id(&genome),
        &mut counter,
    );
    insert_after(&mut genome, INPUT_INNOVATION, attn_nodes).unwrap();
    commit_counter(&mut genome, &counter);
    repair_param_input_dims(&mut genome);

    let blocks = node_main_path(&genome);
    let attn_block = blocks
        .iter()
        .find(|b| matches!(b.kind, NodeBlockKind::Attention { .. }))
        .expect("主路径应包含 Attention 块");
    if let NodeBlockKind::Attention {
        embed_dim,
        num_heads,
    } = attn_block.kind
    {
        assert_eq!(embed_dim, EMBED_DIM);
        assert_eq!(num_heads, NUM_HEADS);
    }

    // 工具方法
    assert!(attn_block.kind.is_attention());
    assert!(attn_block.kind.is_sequence());
    assert!(!attn_block.kind.is_recurrent());
    assert!(attn_block.kind.is_resizable());
    assert_eq!(attn_block.kind.current_size(), Some(EMBED_DIM));
}

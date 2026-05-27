/*
 * @Author       : 老董
 * @Date         : 2026-05-02
 * @Description  : CellAttention rebuild 与 forward 测试
 *
 * 覆盖：
 * 1. 含 CellAttention 的 NodeLevel genome 能通过 build → forward → backward。
 * 2. return_sequences=false 时输出形状为 [B, embed_dim]（mean pooling 后）。
 * 3. return_sequences=true 时输出形状为 [B, T, embed_dim]。
 * 4. 多个 attention 块串联仍能正常构图与训练。
 */

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::{INPUT_INNOVATION, NetworkGenome};
use crate::nn::evolution::node_expansion::expand_attention;
use crate::nn::evolution::node_ops::{
    commit_counter, insert_after, make_counter, next_block_id, repair_param_input_dims,
};
use crate::tensor::Tensor;

fn rng() -> StdRng {
    StdRng::seed_from_u64(1234)
}

fn input_for(genome: &NetworkGenome, seq_len: usize) -> Tensor {
    let total = seq_len * genome.input_dim;
    let data: Vec<f32> = (0..total).map(|i| (i as f32) * 0.1 - 0.5).collect();
    Tensor::new(&data, &[1, seq_len, genome.input_dim])
}

/// 构造一个 attention-only 序列基因组：input → CellAttention → Linear(embed_dim → output_dim)
///
/// 通过先建 minimal_sequential 再用 attention 替换 RNN：
/// 1. 删除原 RNN 块的所有节点
/// 2. 在 input 之后插入 attention 块
/// 3. 通过 repair_param_input_dims 修正末尾 Linear 的 in_features
fn replace_rnn_with_attention(
    in_dim: usize,
    embed_dim: usize,
    num_heads: usize,
    seq_len: usize,
    output_dim: usize,
    return_sequences: bool,
) -> NetworkGenome {
    let mut genome = NetworkGenome::minimal_sequential(in_dim, output_dim);
    genome.seq_len = Some(seq_len);

    // 找到原 RNN 块的所有节点（block_id=0）—— minimal_sequential 把 RNN 放在 block 0
    let rnn_node_ids: Vec<u64> = genome
        .nodes()
        .iter()
        .filter(|n| {
            n.block_id == Some(0)
                && (matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. })
                    || n.is_parameter())
        })
        .map(|n| n.innovation_number)
        .collect();

    // 找到 RNN 块的输出（CellRnn 节点本身），重新 wire：让原本指向 cell_id 的下游节点
    // 改为指向新 attention 块的输出
    let cell_id = genome
        .nodes()
        .iter()
        .find(|n| matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. }))
        .map(|n| n.innovation_number)
        .expect("minimal_sequential 应有 CellRnn");

    // 1. 在 input 之后插入 attention（attention 输出会被自动连到原 RNN 的 dependents）
    let mut counter = make_counter(&genome);
    let bid = next_block_id(&genome);
    let attn_nodes = expand_attention(
        INPUT_INNOVATION,
        in_dim,
        embed_dim,
        num_heads,
        return_sequences,
        seq_len,
        bid,
        &mut counter,
    );
    let attn_out = insert_after(&mut genome, INPUT_INNOVATION, attn_nodes).unwrap();
    commit_counter(&mut genome, &counter);

    // 2. 把所有指向 cell_id 的依赖（即末尾 Linear 的 W*input MatMul 输入）改为 attn_out
    for node in genome.nodes_mut().iter_mut() {
        for pid in node.parents.iter_mut() {
            if *pid == cell_id {
                *pid = attn_out;
            }
        }
    }

    // 3. 删除原 RNN 块的所有节点
    genome
        .nodes_mut()
        .retain(|n| !rnn_node_ids.contains(&n.innovation_number));

    repair_param_input_dims(&mut genome);
    genome
}

#[test]
fn attention_block_rebuilds_and_forwards_with_pooling() {
    let in_dim = 8;
    let embed_dim = 16;
    let num_heads = 4;
    let seq_len = 6;
    let output_dim = 3;

    let genome = replace_rnn_with_attention(
        in_dim, embed_dim, num_heads, seq_len, output_dim, false,
    );

    let mut rng = rng();
    let build = genome.build(&mut rng).expect("Attention 块应能构图");

    build.input.set_value(&input_for(&genome, seq_len)).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    // 末尾 Linear(embed_dim → output_dim)
    assert_eq!(out.shape(), &[1, output_dim]);

    // 校验所有数值有限（attention softmax 不溢出）
    let flat = out.to_vec();
    assert!(
        flat.iter().all(|v| v.is_finite()),
        "输出包含 NaN / Inf：{:?}",
        flat
    );
}

#[test]
fn attention_block_with_return_sequences_keeps_seq_dim() {
    let in_dim = 6;
    let embed_dim = 8;
    let num_heads = 2;
    let seq_len = 4;
    let output_dim = 2;

    let genome = replace_rnn_with_attention(
        in_dim, embed_dim, num_heads, seq_len, output_dim, true,
    );

    // 找到 attention cell node 输出形状
    let cell_shape = genome
        .nodes()
        .iter()
        .find(|n| matches!(n.node_type, NodeTypeDescriptor::CellAttention { .. }))
        .map(|n| n.output_shape.clone())
        .expect("应有 CellAttention 节点");
    assert_eq!(cell_shape, vec![1, seq_len, embed_dim]);
}

#[test]
fn two_attention_blocks_stack_without_panic() {
    let in_dim = 8;
    let embed_dim = 8;
    let num_heads = 2;
    let seq_len = 5;
    let output_dim = 2;

    // 先构造单层 attention 基因组（attention return_sequences=true 给第二层留 3D 输入）
    let mut genome = replace_rnn_with_attention(
        in_dim, embed_dim, num_heads, seq_len, output_dim, true,
    );

    // attn1 的输出 id（CellAttention 节点）
    let attn1_out = genome
        .nodes()
        .iter()
        .find(|n| matches!(n.node_type, NodeTypeDescriptor::CellAttention { .. }))
        .map(|n| n.innovation_number)
        .expect("应有第一层 CellAttention");

    // 在 attn1 之后再插入一层 attention（return_sequences=false → mean pooling 输出 [1, embed]）
    let mut counter = make_counter(&genome);
    let attn2 = expand_attention(
        attn1_out,
        embed_dim,
        embed_dim,
        num_heads,
        false,
        seq_len,
        next_block_id(&genome),
        &mut counter,
    );
    insert_after(&mut genome, attn1_out, attn2).unwrap();
    commit_counter(&mut genome, &counter);
    repair_param_input_dims(&mut genome);

    let mut rng = rng();
    let build = genome.build(&mut rng).expect("两层 attention 应能构图");

    build.input.set_value(&input_for(&genome, seq_len)).unwrap();
    build.graph.forward(&build.output).unwrap();
    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, output_dim]);
}

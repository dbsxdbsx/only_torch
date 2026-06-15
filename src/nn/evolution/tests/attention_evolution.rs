/*
 * @Author       : 老董
 * @Date         : 2026-05-02
 * @Description  : Attention 演化集成测试
 *
 * 覆盖：
 * 1. SequenceOpSet::AttentionOnly：InsertLayer 在序列任务能产生 Attention 块。
 * 2. resize_attention_out：通过 GrowHiddenSize 路径把 attention block 扩到合法对齐尺寸。
 * 3. SequenceOpSet::default() = Recurrent：不会自动插入 attention（向后兼容）。
 */

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::{
    GrowHiddenSizeMutation, InsertLayerMutation, Mutation, SequenceOpSet, SizeConstraints,
};
use crate::nn::evolution::node_expansion::expand_attention;
use crate::nn::evolution::node_ops::{
    NodeBlockKind, commit_counter, insert_after, make_counter, next_block_id, node_main_path,
    repair_param_input_dims,
};

fn rng() -> StdRng {
    StdRng::seed_from_u64(2026)
}

fn attention_constraints() -> SizeConstraints {
    SizeConstraints {
        max_layers: 16,
        max_hidden_size: 64,
        max_total_params: 1_000_000,
        min_hidden_size: 8,
        size_strategy: crate::nn::evolution::SizeStrategy::AlignTo(8),
        sequence_ops: SequenceOpSet::AttentionOnly,
        attention_num_heads_candidates: vec![2, 4, 8],
    }
}

/// 构造 input → CellAttention → Linear 的 attention-only 序列基因组（用 attention 替换 minimal_sequential 的 RNN）
fn build_attention_genome() -> NetworkGenome {
    let in_dim = 8;
    let embed_dim = 16;
    let num_heads = 4;
    let seq_len = 6;
    let output_dim = 2;

    let mut genome = NetworkGenome::minimal_sequential(in_dim, output_dim);
    genome.seq_len = Some(seq_len);

    let rnn_node_ids: Vec<u64> = genome
        .nodes()
        .iter()
        .filter(|n| {
            n.block_id == Some(0)
                && (matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. }) || n.is_parameter())
        })
        .map(|n| n.innovation_number)
        .collect();
    let cell_id = genome
        .nodes()
        .iter()
        .find(|n| matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. }))
        .map(|n| n.innovation_number)
        .expect("minimal_sequential 应有 CellRnn");

    let mut counter = make_counter(&genome);
    let bid = next_block_id(&genome);
    let attn_nodes = expand_attention(
        INPUT_INNOVATION,
        in_dim,
        embed_dim,
        num_heads,
        false,
        seq_len,
        bid,
        &mut counter,
    );
    let attn_out = insert_after(&mut genome, INPUT_INNOVATION, attn_nodes).unwrap();
    commit_counter(&mut genome, &counter);

    for node in genome.nodes_mut().iter_mut() {
        for pid in node.parents.iter_mut() {
            if *pid == cell_id {
                *pid = attn_out;
            }
        }
    }
    genome
        .nodes_mut()
        .retain(|n| !rnn_node_ids.contains(&n.innovation_number));

    repair_param_input_dims(&mut genome);
    genome
}

#[test]
fn insert_layer_under_attention_only_eventually_inserts_attention() {
    // 在序列任务中，AttentionOnly 模式下 InsertLayer 多次重试足以触发至少一次 attention 插入。
    // 这里通过多个不同 seed 跑若干次，断言至少有一次产生 NodeBlockKind::Attention。
    let mut found_attention = false;
    for seed in 0..32u64 {
        let mut genome = NetworkGenome::minimal_sequential(8, 2);
        genome.seq_len = Some(6);

        let mut rng = StdRng::seed_from_u64(seed);
        let mutation = InsertLayerMutation::default();

        // is_applicable 仅做粗筛，apply 内部还会按概率走激活 / 归一化 / 序列算子分支
        if !mutation.is_applicable(&genome, &attention_constraints()) {
            continue;
        }
        if mutation
            .apply(&mut genome, &attention_constraints(), &mut rng)
            .is_err()
        {
            continue;
        }

        if node_main_path(&genome)
            .iter()
            .any(|b| matches!(b.kind, NodeBlockKind::Attention { .. }))
        {
            found_attention = true;
            break;
        }
    }
    assert!(
        found_attention,
        "AttentionOnly 模式下 32 次 InsertLayer 应至少出现一次 Attention 块"
    );
}

#[test]
fn default_sequence_ops_does_not_insert_attention() {
    // 默认 SequenceOpSet::Recurrent 不应自动产生 attention 块。
    let constraints = SizeConstraints::default();
    for seed in 0..16u64 {
        let mut genome = NetworkGenome::minimal_sequential(8, 2);
        genome.seq_len = Some(6);

        let mut rng = StdRng::seed_from_u64(seed);
        let mutation = InsertLayerMutation::default();
        if !mutation.is_applicable(&genome, &constraints) {
            continue;
        }
        let _ = mutation.apply(&mut genome, &constraints, &mut rng);

        let has_attention = node_main_path(&genome)
            .iter()
            .any(|b| matches!(b.kind, NodeBlockKind::Attention { .. }));
        assert!(
            !has_attention,
            "默认 Recurrent 模式下 InsertLayer 不应产生 Attention 块（seed={seed}）"
        );
    }
}

#[test]
fn grow_attention_block_aligns_to_num_heads_and_remains_buildable() {
    let mut genome = build_attention_genome();
    let mut rng = rng();
    let constraints = attention_constraints();

    let initial_embed = node_main_path(&genome)
        .iter()
        .find_map(|b| {
            if let NodeBlockKind::Attention { embed_dim, .. } = b.kind {
                Some(embed_dim)
            } else {
                None
            }
        })
        .expect("初始 genome 应包含 Attention 块");

    let mutation = GrowHiddenSizeMutation;
    // 多次尝试以增大命中 attention 块的概率
    let mut grew_attention = false;
    for _ in 0..16 {
        let snapshot = genome.clone();
        if mutation.apply(&mut genome, &constraints, &mut rng).is_err() {
            continue;
        }
        // 检查 attention 块的 embed_dim 是否变化
        if let Some(new_embed) = node_main_path(&genome).iter().find_map(|b| {
            if let NodeBlockKind::Attention { embed_dim, .. } = b.kind {
                Some(embed_dim)
            } else {
                None
            }
        }) {
            if new_embed != initial_embed {
                // 校验对齐：必须能被 num_heads=4 整除
                assert!(
                    new_embed % 4 == 0,
                    "新 embed_dim={} 必须能被 num_heads=4 整除",
                    new_embed
                );
                // 校验新 genome 仍可构图
                let mut local_rng = StdRng::seed_from_u64(99);
                genome.build(&mut local_rng).expect("Grow 后应能构图");
                grew_attention = true;
                break;
            }
        }
        // 还没改到 attention 块？回滚继续尝试
        genome = snapshot;
    }
    assert!(
        grew_attention,
        "16 次 Grow 应至少有一次命中 Attention 块（block 是该序列基因组中唯一可 resize 的块）"
    );
}

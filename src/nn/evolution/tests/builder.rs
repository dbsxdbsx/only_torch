use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::builder::BuildResult;
use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::{GrowHiddenSizeMutation, Mutation, SizeConstraints};
use crate::nn::evolution::node_expansion::{expand_activation, expand_conv2d, expand_linear};
use crate::nn::evolution::node_ops::{
    commit_counter, insert_after, make_counter, next_block_id, node_main_path,
    repair_param_input_dims,
};
use crate::tensor::Tensor;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn build(genome: &NetworkGenome) -> BuildResult {
    let mut rng = rng();
    genome.build(&mut rng).expect("NodeLevel build 应成功")
}

fn hidden_mlp(input_dim: usize, hidden: usize, output_dim: usize) -> NetworkGenome {
    let mut genome = NetworkGenome::minimal(input_dim, output_dim);
    let mut counter = make_counter(&genome);
    let hidden_nodes = expand_linear(
        INPUT_INNOVATION,
        input_dim,
        hidden,
        next_block_id(&genome),
        &mut counter,
    );
    let hidden_out = insert_after(&mut genome, INPUT_INNOVATION, hidden_nodes).unwrap();
    let act_nodes = expand_activation(
        hidden_out,
        vec![1, hidden],
        &ActivationType::ReLU,
        &mut counter,
    );
    insert_after(&mut genome, hidden_out, act_nodes).unwrap();
    commit_counter(&mut genome, &counter);
    repair_param_input_dims(&mut genome);
    genome
}

fn param_shapes(build: &BuildResult) -> Vec<Vec<usize>> {
    let mut shapes: Vec<Vec<usize>> = build
        .all_parameters()
        .iter()
        .map(|p| p.value().unwrap().unwrap().shape().to_vec())
        .collect();
    shapes.sort();
    shapes
}

#[test]
fn test_minimal_constructor_is_node_level() {
    let genome = NetworkGenome::minimal(2, 1);

    assert!(genome.is_node_level());
    assert!(!genome.nodes().is_empty());
    assert!(genome.layers().is_empty());
    assert!(genome.skip_edges().is_empty());
}

#[test]
fn test_build_minimal_forward_and_params() {
    let genome = NetworkGenome::minimal(2, 1);
    let build = build(&genome);

    build
        .input
        .set_value(&Tensor::new(&[1.0, -1.0], &[1, 2]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    assert_eq!(build.output.value().unwrap().unwrap().shape(), &[1, 1]);
    assert_eq!(build.all_parameters().len(), 2);
    assert_eq!(param_shapes(&build), vec![vec![1, 1], vec![2, 1]]);
}

#[test]
fn test_build_multi_head_flat_exposes_named_outputs() {
    let genome = NetworkGenome::minimal_multi_head_flat(
        2,
        &[
            ("class".to_string(), 1, true, true),
            ("radius".to_string(), 1, false, false),
        ],
    );
    let build = build(&genome);

    assert_eq!(build.outputs.len(), 2);
    assert_eq!(build.output_heads.len(), 2);
    assert_eq!(build.output_heads[0].name, "class");
    assert_eq!(build.output_heads[1].name, "radius");
    assert!(build.output_by_name("class").is_some());
    assert!(build.output_by_name("radius").is_some());

    build
        .input
        .set_value(&Tensor::new(&[0.25, 0.75], &[1, 2]))
        .unwrap();
    for output in &build.outputs {
        build.graph.forward(output).unwrap();
        assert_eq!(output.value().unwrap().unwrap().shape(), &[1, 1]);
    }
}

#[test]
fn test_build_hidden_mlp_forward_and_block_shapes() {
    let genome = hidden_mlp(3, 8, 2);
    let build = build(&genome);

    build
        .input
        .set_value(&Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    assert_eq!(build.output.value().unwrap().unwrap().shape(), &[1, 2]);
    assert_eq!(build.all_parameters().len(), 4);
    assert_eq!(
        param_shapes(&build),
        vec![vec![1, 2], vec![1, 8], vec![3, 8], vec![8, 2]]
    );

    let blocks = node_main_path(&genome);
    assert!(
        blocks.len() >= 3,
        "hidden MLP 应保留 Linear / Activation / Linear block 语义"
    );
}

#[test]
fn test_capture_and_restore_weights_roundtrip() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = rng();

    let build1 = genome.build(&mut rng).unwrap();
    let first_param_id = *build1.layer_params.keys().min().unwrap();
    let original = build1.layer_params[&first_param_id][0]
        .value()
        .unwrap()
        .unwrap();
    genome.capture_weights(&build1).unwrap();

    assert!(genome.has_weight_snapshots());
    assert!(genome.node_weight_snapshots().contains_key(&first_param_id));

    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    assert_eq!(report.inherited, build2.all_parameters().len());
    assert_eq!(report.reinitialized, 0);
    assert_eq!(
        original,
        build2.layer_params[&first_param_id][0]
            .value()
            .unwrap()
            .unwrap()
    );
}

#[test]
fn test_restore_without_snapshot_reinitializes_params() {
    let genome = NetworkGenome::minimal(2, 1);
    let build = build(&genome);
    let report = genome.restore_weights(&build).unwrap();

    assert_eq!(report.inherited, 0);
    assert_eq!(report.reinitialized, build.all_parameters().len());
}

#[test]
fn test_sequential_minimal_forward() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.seq_len = Some(5);
    let build = build(&genome);

    let data: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 5, 3]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    assert_eq!(build.output.value().unwrap().unwrap().shape(), &[1, 2]);
    assert!(build.all_parameters().len() >= 5);
}

#[test]
fn test_sequential_cell_descriptor_uses_genome_seq_len_for_visualization() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.seq_len = Some(5);

    let desc = genome.to_graph_descriptor().unwrap();
    let cell = desc
        .nodes
        .iter()
        .find(|node| {
            matches!(
                node.node_type,
                NodeTypeDescriptor::CellRnn { .. }
                    | NodeTypeDescriptor::CellLstm { .. }
                    | NodeTypeDescriptor::CellGru { .. }
            )
        })
        .expect("minimal_sequential 应包含 memory cell");

    let seq_len = match &cell.node_type {
        NodeTypeDescriptor::CellRnn { seq_len, .. }
        | NodeTypeDescriptor::CellLstm { seq_len, .. }
        | NodeTypeDescriptor::CellGru { seq_len, .. } => *seq_len,
        _ => unreachable!(),
    };
    assert_eq!(seq_len, 5);

    let build = build(&genome);
    let dot = build.output.to_dot();
    assert!(
        dot.contains("×5"),
        "memory cell 应按真实序列长度折叠：\n{dot}"
    );
    assert!(
        dot.contains("#E67E22") && dot.contains("t=0") && dot.contains("peripheries=2"),
        "memory cell 可视化应保留橙色时序边和双框标识：\n{dot}"
    );
}

#[test]
fn test_spatial_minimal_forward() {
    let genome = NetworkGenome::minimal_spatial(1, 3, (8, 8));
    let build = build(&genome);

    build.input.set_value(&Tensor::ones(&[1, 1, 8, 8])).unwrap();
    build.graph.forward(&build.output).unwrap();

    assert_eq!(build.output.value().unwrap().unwrap().shape(), &[1, 3]);
}

#[test]
fn test_spatial_flat_mlp_seed_forward() {
    let genome = NetworkGenome::spatial_flat_mlp(1, 3, (8, 8), 16);
    let build = build(&genome);

    build.input.set_value(&Tensor::ones(&[1, 1, 8, 8])).unwrap();
    build.graph.forward(&build.output).unwrap();

    assert_eq!(build.output.value().unwrap().unwrap().shape(), &[1, 3]);
    assert_eq!(build.all_parameters().len(), 4);
}

#[test]
fn test_spatial_lenet_tiny_seed_forward() {
    let genome = NetworkGenome::spatial_lenet_tiny(1, 3, (8, 8));
    let build = build(&genome);

    build.input.set_value(&Tensor::ones(&[1, 1, 8, 8])).unwrap();
    build.graph.forward(&build.output).unwrap();

    assert_eq!(build.output.value().unwrap().unwrap().shape(), &[1, 3]);
    assert_eq!(build.all_parameters().len(), 8);
}

#[test]
fn test_spatial_segmentation_forward() {
    let genome = NetworkGenome::minimal_spatial_segmentation(1, 2, (8, 8));
    let build = build(&genome);

    build.input.set_value(&Tensor::ones(&[1, 1, 8, 8])).unwrap();
    build.graph.forward(&build.output).unwrap();

    assert_eq!(
        build.output.value().unwrap().unwrap().shape(),
        &[1, 2, 8, 8]
    );
}

#[test]
fn test_node_group_tags_are_backfilled_for_layer_blocks() {
    let genome = NetworkGenome::minimal_spatial(1, 3, (8, 8));
    let build = build(&genome);

    let block_count = node_main_path(&genome)
        .into_iter()
        .filter(|block| block.block_id.is_some())
        .count();

    assert!(block_count > 0);
    assert!(!build.layer_params.is_empty());
}

#[test]
fn test_node_level_partial_inherit_after_grow() {
    let mut genome = hidden_mlp(4, 8, 3);
    let mut rng = rng();

    let build1 = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build1).unwrap();

    let mutation = GrowHiddenSizeMutation;
    mutation
        .apply(&mut genome, &SizeConstraints::default(), &mut rng)
        .expect("GrowHiddenSize 应适用于 NodeLevel hidden MLP");
    repair_param_input_dims(&mut genome);

    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    assert!(
        report.inherited + report.partially_inherited > 0,
        "扩宽后应至少继承部分 NodeLevel 参数"
    );
}

#[test]
fn test_hand_built_conv_block_keeps_block_id() {
    let genome = NetworkGenome::minimal(4, 1);
    let mut counter = make_counter(&genome);
    let conv = expand_conv2d(
        INPUT_INNOVATION,
        1,
        2,
        3,
        (4, 4),
        next_block_id(&genome),
        &mut counter,
    );

    assert!(conv.iter().all(|n| n.block_id.is_some()));
}

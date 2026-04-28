use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::*;
use crate::nn::evolution::node_expansion::{
    expand_activation, expand_deformable_conv2d, expand_dropout, expand_linear,
};
use crate::nn::evolution::node_ops::{
    NodeBlockKind, commit_counter, insert_after, make_counter, next_block_id, node_main_path,
    repair_param_input_dims,
};
use crate::tensor::Tensor;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn constraints() -> SizeConstraints {
    SizeConstraints::default()
}

fn assert_buildable(genome: &NetworkGenome) {
    let mut rng = rng();
    let build = genome
        .build(&mut rng)
        .expect("变异后的 NodeLevel genome 应可构图");
    assert!(!build.all_parameters().is_empty());
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

#[test]
fn test_insert_layer_mutates_node_blocks() {
    let mut genome = NetworkGenome::minimal(4, 2);
    let before = node_main_path(&genome).len();
    let mut rng = rng();
    let mutation = InsertLayerMutation::default();

    assert!(mutation.is_applicable(&genome, &constraints()));
    mutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();

    assert!(genome.is_node_level());
    assert!(node_main_path(&genome).len() > before);
    assert_buildable(&genome);
}

#[test]
fn test_remove_layer_keeps_output_head_buildable() {
    let mut genome = hidden_mlp(4, 8, 2);
    let before = node_main_path(&genome).len();
    let mut rng = rng();
    let mutation = RemoveLayerMutation;

    assert!(mutation.is_applicable(&genome, &constraints()));
    mutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();

    assert!(node_main_path(&genome).len() < before);
    assert_buildable(&genome);
}

#[test]
fn test_replace_layer_type_preserves_node_level_buildability() {
    let mut genome = hidden_mlp(4, 8, 2);
    let mut rng = rng();
    let mutation = ReplaceLayerTypeMutation::default();

    assert!(mutation.is_applicable(&genome, &constraints()));
    mutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();

    assert!(genome.is_node_level());
    assert_buildable(&genome);
}

#[test]
fn test_grow_and_shrink_hidden_size_update_node_shapes() {
    let mut genome = hidden_mlp(4, 8, 2);
    let mut rng = rng();
    let c = constraints();

    let grow = GrowHiddenSizeMutation;
    assert!(grow.is_applicable(&genome, &c));
    grow.apply(&mut genome, &c, &mut rng).unwrap();
    repair_param_input_dims(&mut genome);
    assert_buildable(&genome);

    let shrink = ShrinkHiddenSizeMutation;
    assert!(shrink.is_applicable(&genome, &c));
    shrink.apply(&mut genome, &c, &mut rng).unwrap();
    repair_param_input_dims(&mut genome);
    assert_buildable(&genome);
}

#[test]
fn test_mutate_layer_param_touches_node_level_activation() {
    let mut genome = hidden_mlp(4, 8, 2);
    let mut counter = make_counter(&genome);
    let last_hidden = node_main_path(&genome)
        .into_iter()
        .find(|block| matches!(block.kind, NodeBlockKind::Activation { .. }))
        .unwrap()
        .output_id;
    let dropout = expand_dropout(last_hidden, vec![1, 8], 0.25, &mut counter);
    insert_after(&mut genome, last_hidden, dropout).unwrap();
    commit_counter(&mut genome, &counter);
    repair_param_input_dims(&mut genome);

    let mut rng = rng();
    let mutation = MutateLayerParamMutation;

    assert!(mutation.is_applicable(&genome, &constraints()));
    mutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();

    assert_buildable(&genome);
}

#[test]
fn test_add_and_remove_connection_are_node_level_mutations() {
    let mut genome = hidden_mlp(4, 8, 2);
    let mut rng = rng();
    let c = constraints();

    let add = AddConnectionMutation;
    assert!(add.is_applicable(&genome, &c));
    let before_nodes = genome.nodes().len();
    add.apply(&mut genome, &c, &mut rng).unwrap();
    assert!(genome.nodes().len() > before_nodes);
    assert_buildable(&genome);

    let remove = RemoveConnectionMutation;
    assert!(remove.is_applicable(&genome, &c));
    remove.apply(&mut genome, &c, &mut rng).unwrap();
    assert_buildable(&genome);
}

#[test]
fn test_node_level_registry_does_not_expose_legacy_skip_edge_mutations() {
    let registry = MutationRegistry::default_registry(&TaskMetric::R2, false, false);
    let names = registry.mutation_names();

    assert!(names.contains(&"AddConnection"));
    assert!(names.contains(&"RemoveConnection"));
    assert!(!names.contains(&"AddSkipEdge"));
    assert!(!names.contains(&"RemoveSkipEdge"));
    assert!(!names.contains(&"MutateAggregateStrategy"));
}

#[test]
fn test_registry_apply_random_keeps_node_level_buildable() {
    let mut genome = hidden_mlp(4, 8, 2);
    let mut rng = rng();
    let registry = MutationRegistry::default_registry(&TaskMetric::R2, false, false);

    for _ in 0..8 {
        let name = registry
            .apply_random(&mut genome, &constraints(), &mut rng)
            .expect("registry 应能选到适用的 NodeLevel 变异");
        assert!(!name.contains("SkipEdge"));
        repair_param_input_dims(&mut genome);
        assert_buildable(&genome);
    }
}

#[test]
fn test_sequential_cell_type_mutation_keeps_recurrent_genome_buildable() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.seq_len = Some(5);
    let mut rng = rng();
    let mutation = MutateCellTypeMutation;

    assert!(mutation.is_applicable(&genome, &constraints()));
    mutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();
    repair_param_input_dims(&mut genome);
    assert_buildable(&genome);
}

#[test]
fn test_spatial_insert_layer_keeps_segmentation_shape_domain() {
    let mut genome = NetworkGenome::minimal_spatial_segmentation(1, 2, (8, 8));
    let mut rng = rng();
    let mutation = InsertLayerMutation::spatial_preserving(vec![ActivationType::ReLU]);

    assert!(mutation.is_applicable(&genome, &constraints()));
    mutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();

    assert_eq!(genome.input_spatial, Some((8, 8)));
    assert_buildable(&genome);
}

#[test]
fn test_deformable_conv2d_block_builds_in_spatial_segmentation_genome() {
    let mut genome = NetworkGenome::minimal_spatial_segmentation(1, 2, (8, 8));
    let mut counter = make_counter(&genome);
    let block_id = next_block_id(&genome);
    let nodes =
        expand_deformable_conv2d(INPUT_INNOVATION, 1, 4, 3, (8, 8), 1, block_id, &mut counter);
    insert_after(&mut genome, INPUT_INNOVATION, nodes).unwrap();
    commit_counter(&mut genome, &counter);
    repair_param_input_dims(&mut genome);

    assert!(genome.analyze().is_valid);
    assert!(genome.nodes().iter().any(|node| {
        matches!(
            node.node_type,
            NodeTypeDescriptor::DeformableConv2d {
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
                deformable_groups: 1,
            }
        )
    }));
    assert!(node_main_path(&genome).iter().any(|block| {
        matches!(
            block.kind,
            NodeBlockKind::DeformableConv2d {
                out_channels: 4,
                kernel_size: 3,
            }
        )
    }));

    let mut rng = rng();
    let build = genome.build(&mut rng).unwrap();
    build
        .input
        .set_value(&Tensor::zeros(&[1, 1, 8, 8]))
        .unwrap();
    build.output.forward().unwrap();
    assert_eq!(
        build.output.value().unwrap().unwrap().shape(),
        &[1, 2, 8, 8]
    );
}

#[test]
fn test_insert_encoder_decoder_skip_adds_unet_style_block() {
    let mut genome = NetworkGenome::minimal_spatial_segmentation(1, 2, (8, 8));
    let mut rng = rng();
    let mutation = InsertEncoderDecoderSkipMutation;

    assert!(mutation.is_applicable(&genome, &constraints()));
    mutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();

    let desc = genome
        .to_graph_descriptor()
        .expect("encoder-decoder skip 变异后应能生成 GraphDescriptor");
    let output_id = desc
        .explicit_output_ids
        .as_ref()
        .and_then(|ids| ids.first())
        .copied()
        .expect("segmentation genome 应有显式输出");
    let output = desc
        .nodes
        .iter()
        .find(|node| node.id == output_id)
        .expect("显式输出节点应存在");

    assert_eq!(output.output_shape, vec![1, 2, 8, 8]);
    assert!(
        desc.nodes.iter().any(|node| matches!(
            &node.node_type,
            NodeTypeDescriptor::MaxPool2d { .. } | NodeTypeDescriptor::AvgPool2d { .. }
        )),
        "encoder-decoder skip 变异应包含 Pool2d"
    );
    assert!(
        desc.nodes
            .iter()
            .any(|node| matches!(&node.node_type, NodeTypeDescriptor::ConvTranspose2d { .. })),
        "encoder-decoder skip 变异应包含 ConvTranspose2d"
    );
    assert!(
        desc.nodes
            .iter()
            .any(|node| matches!(&node.node_type, NodeTypeDescriptor::Concat { axis: 1 })),
        "encoder-decoder skip 变异应包含 channel 维 Concat"
    );
    assert_buildable(&genome);
}

#[test]
fn test_segmentation_registry_contains_encoder_decoder_skip_mutation() {
    let registry = MutationRegistry::phase1_registry(&TaskMetric::MeanIoU, false, true);
    let names = registry.mutation_names();

    assert!(names.contains(&"InsertEncoderDecoderSkip"));
    assert!(!names.contains(&"AddFeatureMap"));
}

#[test]
fn test_optimizer_and_lr_mutations_do_not_change_node_repr() {
    let mut genome = NetworkGenome::minimal(4, 2);
    let mut rng = rng();

    MutateOptimizerMutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();
    MutateLearningRateMutation
        .apply(&mut genome, &constraints(), &mut rng)
        .unwrap();

    assert!(genome.is_node_level());
    assert_buildable(&genome);
}

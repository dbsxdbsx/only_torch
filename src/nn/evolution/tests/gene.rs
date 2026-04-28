use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::{
    GrowHiddenSizeMutation, InsertLayerMutation, Mutation, SizeConstraints,
};
use crate::nn::evolution::node_ops::{node_main_path, repair_param_input_dims};

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn assert_buildable(genome: &NetworkGenome) {
    let mut rng = rng();
    genome.build(&mut rng).expect("NodeLevel genome 应可构图");
}

#[test]
fn test_minimal_creates_node_level_genome() {
    let genome = NetworkGenome::minimal(2, 1);

    assert!(genome.is_node_level());
    assert!(!genome.is_layer_level());
    assert_eq!(genome.input_dim, 2);
    assert_eq!(genome.output_dim, 1);
    assert!(genome.layers().is_empty());
    assert!(genome.skip_edges().is_empty());
    assert!(!genome.nodes().is_empty());
    assert_buildable(&genome);
}

#[test]
fn test_minimal_sequential_creates_recurrent_node_block() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.seq_len = Some(5);

    assert!(genome.is_node_level());
    assert!(genome.seq_len.is_some());
    assert!(node_main_path(&genome).len() >= 2);
    assert_buildable(&genome);
}

#[test]
fn test_minimal_spatial_creates_conv_blocks() {
    let genome = NetworkGenome::minimal_spatial(1, 10, (8, 8));

    assert!(genome.is_node_level());
    assert_eq!(genome.input_spatial, Some((8, 8)));
    assert!(node_main_path(&genome).iter().any(|b| b.block_id.is_some()));
    assert_buildable(&genome);
}

#[test]
fn test_minimal_spatial_segmentation_keeps_dense_output() {
    let genome = NetworkGenome::minimal_spatial_segmentation(1, 3, (8, 8));
    let mut rng = rng();
    let build = genome.build(&mut rng).unwrap();

    assert_eq!(genome.input_spatial, Some((8, 8)));
    build
        .input
        .set_value(&crate::tensor::Tensor::ones(&[1, 1, 8, 8]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();
    assert_eq!(
        build.output.value().unwrap().unwrap().shape(),
        &[1, 3, 8, 8]
    );
}

#[test]
fn test_migrate_to_node_level_is_noop_for_node_level() {
    let mut genome = NetworkGenome::minimal(2, 1);

    genome.migrate_to_node_level().unwrap();

    assert!(genome.is_node_level());
}

#[test]
fn test_migrate_to_node_level_rejects_layer_level() {
    let mut genome = NetworkGenome {
        input_dim: 2,
        output_dim: 1,
        seq_len: None,
        input_spatial: None,
        training_config: TrainingConfig::default(),
        generated_by: "legacy".to_string(),
        output_heads: Vec::new(),
        repr: GenomeRepr::LayerLevel {
            layers: vec![LayerGene {
                innovation_number: 1,
                layer_config: LayerConfig::Linear { out_features: 1 },
                enabled: true,
            }],
            skip_edges: Vec::new(),
            next_innovation: 2,
            weight_snapshots: std::collections::HashMap::new(),
        },
    };

    let err = genome.migrate_to_node_level().unwrap_err().to_string();
    assert!(err.contains("LayerLevel"));
}

#[test]
fn test_node_level_total_params_and_layer_count_are_positive() {
    let genome = NetworkGenome::minimal(3, 2);

    assert_eq!(genome.layer_count(), node_main_path(&genome).len());
    assert_eq!(genome.total_params().unwrap(), 8);
}

#[test]
fn test_node_level_main_path_summary_is_analysis_based() {
    let genome = NetworkGenome::minimal(2, 1);
    let summary = genome.main_path_summary();

    assert!(summary.contains("nodes="));
    assert!(summary.contains("params="));
    assert_eq!(summary, format!("{}", genome));
}

#[test]
fn test_node_weight_snapshots_are_captured_and_cloned() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = rng();
    let build = genome.build(&mut rng).unwrap();

    assert!(!genome.has_weight_snapshots());
    genome.capture_weights(&build).unwrap();

    assert!(genome.has_weight_snapshots());
    assert_eq!(
        genome.node_weight_snapshots().len(),
        build.all_parameters().len()
    );
    assert_eq!(
        genome.clone().node_weight_snapshots(),
        genome.node_weight_snapshots()
    );
}

#[test]
fn test_grow_hidden_updates_node_level_analysis() {
    let mut genome = NetworkGenome::minimal(4, 2);
    let mut rng = rng();
    InsertLayerMutation::default()
        .apply(&mut genome, &SizeConstraints::default(), &mut rng)
        .unwrap();
    repair_param_input_dims(&mut genome);
    let before = genome.total_params().unwrap();

    GrowHiddenSizeMutation
        .apply(&mut genome, &SizeConstraints::default(), &mut rng)
        .unwrap();
    repair_param_input_dims(&mut genome);

    assert!(genome.total_params().unwrap() >= before);
    assert_buildable(&genome);
}

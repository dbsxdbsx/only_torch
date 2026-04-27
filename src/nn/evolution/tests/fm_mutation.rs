/*
 * @Author       : 老董
 * @Date         : 2026-04-19
 * @Description  : FM（Feature Map）级别变异的单元测试
 *
 * 测试策略：
 * 1. 对每种 FM 变异测试 is_applicable + apply
 * 2. 验证结构不变量（节点数变化、fm_id 正确性、parent 引用一致性）
 * 3. 使用 minimal_spatial genome + migrate_to_node_level + migrate_to_fm_level 构造测试基因组
 */

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::fm_mutation::*;
use crate::nn::evolution::fm_ops::analyze_fm_subgraph;
use crate::nn::evolution::gene::NetworkGenome;
use crate::nn::evolution::mutation::{Mutation, SizeConstraints};

/// 创建一个已 FM 化的空间基因组用于测试
fn make_fm_genome() -> NetworkGenome {
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (8, 8));
    genome.migrate_to_node_level().unwrap();
    genome.migrate_to_fm_level();
    assert!(genome.is_node_level());
    assert!(genome.nodes().iter().any(|n| n.fm_id.is_some()));
    genome
}

/// 创建一个多通道 FM 基因组（更多 FM 边可操作）
fn make_multichannel_fm_genome() -> NetworkGenome {
    let mut genome = NetworkGenome::minimal_spatial(2, 10, (8, 8));
    genome.migrate_to_node_level().unwrap();
    genome.migrate_to_fm_level();
    genome
}

fn default_constraints() -> SizeConstraints {
    SizeConstraints::default()
}

/// 验证基因组的基本一致性
fn assert_genome_consistent(genome: &NetworkGenome) {
    let nodes = genome.nodes();

    // 所有 parent 引用的节点应存在或为 INPUT_INNOVATION(0)
    let node_ids: std::collections::HashSet<u64> = nodes
        .iter()
        .filter(|n| n.enabled)
        .map(|n| n.innovation_number)
        .collect();

    for n in nodes.iter().filter(|n| n.enabled) {
        for &pid in &n.parents {
            assert!(
                pid == 0 || node_ids.contains(&pid),
                "节点 {} 引用不存在的 parent {}",
                n.innovation_number,
                pid
            );
        }
    }
}

// ==================== 1. AddFeatureMap ====================

#[test]
fn test_add_feature_map_applicable() {
    let genome = make_fm_genome();
    let mutation = AddFeatureMapMutation;
    assert!(
        mutation.is_applicable(&genome, &default_constraints()),
        "FM 基因组应该可以添加 FM"
    );
}

#[test]
fn test_add_feature_map_apply() {
    let mut genome = make_fm_genome();
    let mutation = AddFeatureMapMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    let original_fm_count = genome.nodes().iter().filter(|n| n.fm_id.is_some()).count();
    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    // 可能成功也可能 NotApplicable（取决于随机选择）
    if result.is_ok() {
        let new_fm_count = genome.nodes().iter().filter(|n| n.fm_id.is_some()).count();
        assert!(
            new_fm_count > original_fm_count,
            "添加 FM 后 FM 节点数应增加: {} -> {}",
            original_fm_count,
            new_fm_count
        );
        assert_genome_consistent(&genome);
    }
}

#[test]
fn test_add_feature_map_is_structural() {
    assert!(AddFeatureMapMutation.is_structural());
}

// ==================== 2. RemoveFeatureMap ====================

#[test]
fn test_remove_feature_map_applicable_multichannel() {
    let genome = make_multichannel_fm_genome();
    let mutation = RemoveFeatureMapMutation;
    // 多通道基因组应有可移除的隐藏 FM
    let analysis = analyze_fm_subgraph(genome.nodes());
    // 需要至少 3 个 FM 才可移除
    if analysis.fm_nodes.len() > 2 {
        assert!(mutation.is_applicable(&genome, &default_constraints()));
    }
}

#[test]
fn test_remove_feature_map_apply() {
    let mut genome = make_multichannel_fm_genome();
    let mutation = RemoveFeatureMapMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    let analysis = analyze_fm_subgraph(genome.nodes());
    if analysis.fm_nodes.len() <= 2 {
        return; // 无法测试
    }

    let result = mutation.apply(&mut genome, &constraints, &mut rng);
    if result.is_ok() {
        assert_genome_consistent(&genome);
    }
}

// ==================== 3. AddFMEdge ====================

#[test]
fn test_add_fm_edge_applicable() {
    let genome = make_fm_genome();
    let mutation = AddFMEdgeMutation;
    // FM 基因组应有可添加边的 FM 对
    let applicable = mutation.is_applicable(&genome, &default_constraints());
    // 不一定为 true（如果所有 FM 对已全连接），但不应 panic
    let _ = applicable;
}

#[test]
fn test_add_fm_edge_apply() {
    let mut genome = make_fm_genome();
    let mutation = AddFMEdgeMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    let original_node_count = genome.nodes().len();
    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    if result.is_ok() {
        // 添加边应增加节点（kernel + conv + add）
        assert!(
            genome.nodes().len() > original_node_count,
            "添加 FM 边后节点数应增加"
        );
        assert_genome_consistent(&genome);
    }
}

#[test]
fn test_add_fm_edge_is_structural() {
    assert!(AddFMEdgeMutation.is_structural());
}

// ==================== 4. RemoveFMEdge ====================

#[test]
fn test_remove_fm_edge_applicable() {
    let genome = make_multichannel_fm_genome();
    let mutation = RemoveFMEdgeMutation;
    // 多通道基因组可能有可移除的边
    let _ = mutation.is_applicable(&genome, &default_constraints());
}

#[test]
fn test_remove_fm_edge_apply() {
    let mut genome = make_multichannel_fm_genome();
    let mutation = RemoveFMEdgeMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    let result = mutation.apply(&mut genome, &constraints, &mut rng);
    if result.is_ok() {
        assert_genome_consistent(&genome);
    }
}

// ==================== 5. SplitFMEdge ====================

#[test]
fn test_split_fm_edge_applicable() {
    let genome = make_fm_genome();
    let mutation = SplitFMEdgeMutation;
    let analysis = analyze_fm_subgraph(genome.nodes());
    if !analysis.fm_edges.is_empty() {
        assert!(mutation.is_applicable(&genome, &default_constraints()));
    }
}

#[test]
fn test_split_fm_edge_apply() {
    let mut genome = make_fm_genome();
    let mutation = SplitFMEdgeMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    let original_fm_ids: std::collections::HashSet<u64> =
        genome.nodes().iter().filter_map(|n| n.fm_id).collect();

    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    if result.is_ok() {
        let new_fm_ids: std::collections::HashSet<u64> =
            genome.nodes().iter().filter_map(|n| n.fm_id).collect();
        // 分裂应产生新 FM
        assert!(
            new_fm_ids.len() > original_fm_ids.len(),
            "SplitFMEdge 应创建新 FM: {} -> {}",
            original_fm_ids.len(),
            new_fm_ids.len()
        );
        assert_genome_consistent(&genome);
    }
}

#[test]
fn test_split_fm_edge_is_structural() {
    assert!(SplitFMEdgeMutation.is_structural());
}

// ==================== 6. ChangeFMEdgeType ====================

#[test]
fn test_change_fm_edge_type_applicable() {
    let genome = make_fm_genome();
    let mutation = ChangeFMEdgeTypeMutation;
    let analysis = analyze_fm_subgraph(genome.nodes());
    if !analysis.fm_edges.is_empty() {
        assert!(mutation.is_applicable(&genome, &default_constraints()));
    }
}

#[test]
fn test_change_fm_edge_type_apply() {
    let mut genome = make_fm_genome();
    let mutation = ChangeFMEdgeTypeMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    // 记录原始边类型
    let analysis_before = analyze_fm_subgraph(genome.nodes());
    let edge_count_before = analysis_before.fm_edges.len();

    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    if result.is_ok() {
        // 边数不应变化（只改类型，不增删）
        let analysis_after = analyze_fm_subgraph(genome.nodes());
        assert_eq!(analysis_after.fm_edges.len(), edge_count_before);
        assert_genome_consistent(&genome);
    }
}

#[test]
fn test_change_fm_edge_type_not_structural() {
    assert!(!ChangeFMEdgeTypeMutation.is_structural());
}

// ==================== 7. MutateFMEdgeKernelSize ====================

#[test]
fn test_mutate_fm_edge_kernel_size_applicable() {
    let genome = make_fm_genome();
    let mutation = MutateFMEdgeKernelSizeMutation;
    let analysis = analyze_fm_subgraph(genome.nodes());
    let has_conv_edges = analysis.fm_edges.iter().any(|e| e.kernel_node_id.is_some());
    if has_conv_edges {
        assert!(mutation.is_applicable(&genome, &default_constraints()));
    }
}

#[test]
fn test_mutate_fm_edge_kernel_size_apply() {
    let mut genome = make_fm_genome();
    let mutation = MutateFMEdgeKernelSizeMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    if result.is_ok() {
        // 验证某个 kernel 的 shape 是合法的 [1,1,k,k]
        let kernel_shapes: Vec<&Vec<usize>> = genome
            .nodes()
            .iter()
            .filter(|n| {
                n.enabled
                    && matches!(n.node_type, NodeTypeDescriptor::Parameter)
                    && n.output_shape.len() == 4
                    && n.output_shape[0] == 1
                    && n.output_shape[1] == 1
            })
            .map(|n| &n.output_shape)
            .collect();

        for shape in kernel_shapes {
            let k = shape[2];
            assert!(
                [1, 3, 5, 7].contains(&k),
                "kernel size {} 应在 [1,3,5,7] 中",
                k
            );
            assert_eq!(shape[2], shape[3], "kernel 应为正方形");
        }

        assert_genome_consistent(&genome);
    }
}

// ==================== 8. MutateFMEdgeStride ====================

#[test]
fn test_mutate_fm_edge_stride_applicable() {
    let genome = make_fm_genome();
    let mutation = MutateFMEdgeStrideMutation;
    let analysis = analyze_fm_subgraph(genome.nodes());
    if !analysis.fm_edges.is_empty() {
        assert!(mutation.is_applicable(&genome, &default_constraints()));
    }
}

#[test]
fn test_mutate_fm_edge_stride_apply() {
    let mut genome = make_fm_genome();
    let mutation = MutateFMEdgeStrideMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    if result.is_ok() {
        // 验证至少有一个 Conv2d 的 stride 被修改
        let strides: Vec<(usize, usize)> = genome
            .nodes()
            .iter()
            .filter_map(|n| match &n.node_type {
                NodeTypeDescriptor::Conv2d { stride, .. } => Some(*stride),
                NodeTypeDescriptor::ConvTranspose2d { stride, .. } => Some(*stride),
                _ => None,
            })
            .collect();

        // stride 应为 (1,1) 或 (2,2)
        for s in strides {
            assert!(
                s == (1, 1) || s == (2, 2),
                "stride {:?} 应为 (1,1) 或 (2,2)",
                s
            );
        }

        assert_genome_consistent(&genome);
    }
}

// ==================== 9. MutateFMEdgeDilation ====================

#[test]
fn test_mutate_fm_edge_dilation_applicable() {
    let genome = make_fm_genome();
    let mutation = MutateFMEdgeDilationMutation;
    let analysis = analyze_fm_subgraph(genome.nodes());
    let has_conv = analysis.fm_edges.iter().any(|e| {
        matches!(
            &e.edge_type,
            crate::nn::evolution::fm_ops::FMEdgeType::Conv2d { .. }
        )
    });
    if has_conv {
        assert!(mutation.is_applicable(&genome, &default_constraints()));
    }
}

#[test]
fn test_mutate_fm_edge_dilation_apply() {
    let mut genome = make_fm_genome();
    let mutation = MutateFMEdgeDilationMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    if result.is_ok() {
        let dilations: Vec<(usize, usize)> = genome
            .nodes()
            .iter()
            .filter_map(|n| match &n.node_type {
                NodeTypeDescriptor::Conv2d { dilation, .. } => Some(*dilation),
                _ => None,
            })
            .collect();

        for d in dilations {
            assert!(
                d == (1, 1) || d == (2, 2) || d == (3, 3),
                "dilation {:?} 应为 (1,1), (2,2), 或 (3,3)",
                d
            );
        }

        assert_genome_consistent(&genome);
    }
}

// ==================== 10. ChangeFeatureMapSize ====================

#[test]
fn test_change_feature_map_size_applicable() {
    let genome = make_multichannel_fm_genome();
    let mutation = ChangeFeatureMapSizeMutation;
    let analysis = analyze_fm_subgraph(genome.nodes());
    if analysis.fm_nodes.len() > 2 {
        assert!(mutation.is_applicable(&genome, &default_constraints()));
    }
}

#[test]
fn test_change_feature_map_size_apply() {
    let mut genome = make_multichannel_fm_genome();
    let mutation = ChangeFeatureMapSizeMutation;
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(42);

    if !mutation.is_applicable(&genome, &constraints) {
        return;
    }

    let result = mutation.apply(&mut genome, &constraints, &mut rng);

    if result.is_ok() {
        // 验证 FM 节点的空间尺寸至少有一个 > 0
        let fm_spatial_sizes: Vec<(usize, usize)> = genome
            .nodes()
            .iter()
            .filter(|n| n.fm_id.is_some() && n.output_shape.len() == 4)
            .map(|n| (n.output_shape[2], n.output_shape[3]))
            .collect();

        for (h, w) in fm_spatial_sizes {
            assert!(h >= 1, "FM 空间高度应 >= 1, got {}", h);
            assert!(w >= 1, "FM 空间宽度应 >= 1, got {}", w);
        }

        assert_genome_consistent(&genome);
    }
}

// ==================== 非 FM 基因组不适用 ====================

#[test]
fn test_fm_mutations_not_applicable_on_flat() {
    let genome = NetworkGenome::minimal(4, 2);
    let constraints = default_constraints();

    assert!(!AddFeatureMapMutation.is_applicable(&genome, &constraints));
    assert!(!RemoveFeatureMapMutation.is_applicable(&genome, &constraints));
    assert!(!AddFMEdgeMutation.is_applicable(&genome, &constraints));
    assert!(!RemoveFMEdgeMutation.is_applicable(&genome, &constraints));
    assert!(!SplitFMEdgeMutation.is_applicable(&genome, &constraints));
    assert!(!ChangeFMEdgeTypeMutation.is_applicable(&genome, &constraints));
    assert!(!MutateFMEdgeKernelSizeMutation.is_applicable(&genome, &constraints));
    assert!(!MutateFMEdgeStrideMutation.is_applicable(&genome, &constraints));
    assert!(!MutateFMEdgeDilationMutation.is_applicable(&genome, &constraints));
    assert!(!ChangeFeatureMapSizeMutation.is_applicable(&genome, &constraints));
}

// ==================== 变异名称和属性测试 ====================

#[test]
fn test_fm_mutation_names() {
    let expected = [
        ("AddFeatureMap", true),
        ("RemoveFeatureMap", true),
        ("AddFMEdge", true),
        ("RemoveFMEdge", true),
        ("SplitFMEdge", true),
        ("ChangeFMEdgeType", false),
        ("MutateFMEdgeKernelSize", false),
        ("MutateFMEdgeStride", false),
        ("MutateFMEdgeDilation", false),
        ("ChangeFeatureMapSize", false),
    ];

    let mutations: Vec<Box<dyn Mutation>> = vec![
        Box::new(AddFeatureMapMutation),
        Box::new(RemoveFeatureMapMutation),
        Box::new(AddFMEdgeMutation),
        Box::new(RemoveFMEdgeMutation),
        Box::new(SplitFMEdgeMutation),
        Box::new(ChangeFMEdgeTypeMutation),
        Box::new(MutateFMEdgeKernelSizeMutation),
        Box::new(MutateFMEdgeStrideMutation),
        Box::new(MutateFMEdgeDilationMutation),
        Box::new(ChangeFeatureMapSizeMutation),
    ];

    for (i, (expected_name, expected_structural)) in expected.iter().enumerate() {
        assert_eq!(
            mutations[i].name(),
            *expected_name,
            "变异 #{} 名称不匹配",
            i
        );
        assert_eq!(
            mutations[i].is_structural(),
            *expected_structural,
            "变异 {} 的 is_structural 不匹配",
            expected_name
        );
    }
}

// ==================== 多轮连续变异稳定性 ====================

#[test]
fn test_multiple_mutations_stability() {
    let mut genome = make_fm_genome();
    let constraints = default_constraints();
    let mut rng = StdRng::seed_from_u64(123);

    let mutations: Vec<Box<dyn Mutation>> = vec![
        Box::new(AddFMEdgeMutation),
        Box::new(ChangeFMEdgeTypeMutation),
        Box::new(MutateFMEdgeKernelSizeMutation),
        Box::new(MutateFMEdgeStrideMutation),
        Box::new(MutateFMEdgeDilationMutation),
    ];

    // 执行 20 轮随机变异
    for round in 0..20 {
        let mutation = &mutations[round % mutations.len()];
        if mutation.is_applicable(&genome, &constraints) {
            let _ = mutation.apply(&mut genome, &constraints, &mut rng);
        }
        assert_genome_consistent(&genome);
    }

    // 变异后基因组仍有 FM 节点
    assert!(
        genome.nodes().iter().any(|n| n.fm_id.is_some()),
        "多轮变异后应仍有 FM 节点"
    );
}

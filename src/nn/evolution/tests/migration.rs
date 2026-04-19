/*
 * @Author       : 老董
 * @Date         : 2026-03-25
 * @Description  : migration.rs 的单元测试
 *
 * 覆盖：
 * 1. InnovationCounter 基础行为
 * 2. 各 LayerConfig 变体的展开结构验证
 * 3. GenomeAnalysis 对展开结果的合法性验证
 * 4. migrate_network_genome：单层、多层、带 skip edge
 * 5. Rnn/Lstm/Gru deferred 路径
 */

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::gene::ShapeDomain;
use crate::nn::evolution::gene::{
    ActivationType, AggregateStrategy, INPUT_INNOVATION, LayerConfig, LayerGene, NetworkGenome,
    PoolType, SkipEdge,
};
use crate::nn::evolution::migration::{
    InnovationCounter, expand_activation, expand_conv2d, expand_dropout, expand_flatten,
    expand_gru, expand_linear, expand_lstm, expand_pool2d, expand_rnn, migrate_conv2d_to_feature_maps,
    migrate_network_genome,
};
use crate::nn::evolution::node_gene::{GenomeAnalysis, NodeGene};

// ==================== InnovationCounter ====================

#[test]
fn innovation_counter_sequential() {
    let mut c = InnovationCounter::new(1);
    assert_eq!(c.next(), 1);
    assert_eq!(c.next(), 2);
    assert_eq!(c.next(), 3);
    assert_eq!(c.peek(), 4);
}

#[test]
fn innovation_counter_custom_start() {
    let mut c = InnovationCounter::new(100);
    assert_eq!(c.next(), 100);
    assert_eq!(c.peek(), 101);
}

// ==================== expand_linear ====================

#[test]
fn expand_linear_produces_4_nodes() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_linear(0, 4, 8, 0, &mut c);
    assert_eq!(nodes.len(), 4);
    // 节点顺序：W, MatMul, b, Add
    assert!(nodes[0].is_parameter(), "节点 0 应为参数节点 W");
    assert!(
        matches!(nodes[1].node_type, NodeTypeDescriptor::MatMul),
        "节点 1 应为 MatMul"
    );
    assert!(nodes[2].is_parameter(), "节点 2 应为参数节点 b");
    assert!(
        matches!(nodes[3].node_type, NodeTypeDescriptor::Add),
        "节点 3 应为 Add"
    );
}

#[test]
fn expand_linear_shapes_correct() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_linear(0, 4, 8, 0, &mut c);
    assert_eq!(nodes[0].output_shape, vec![4, 8], "W 形状应为 [in,out]");
    assert_eq!(nodes[1].output_shape, vec![1, 8], "MatMul 输出应为 [1,out]");
    assert_eq!(nodes[2].output_shape, vec![1, 8], "b 形状应为 [1,out]");
    assert_eq!(nodes[3].output_shape, vec![1, 8], "Add 输出应为 [1,out]");
}

#[test]
fn expand_linear_parent_connections_correct() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_linear(0, 4, 8, 0, &mut c);
    let w_id = nodes[0].innovation_number;
    let mm_id = nodes[1].innovation_number;
    let b_id = nodes[2].innovation_number;
    let _add_id = nodes[3].innovation_number; // innovation 号本身不用于断言，仅占位避免警告

    // W 和 b 无父节点（叶节点）
    assert!(nodes[0].parents.is_empty(), "W 无父节点");
    assert!(nodes[2].parents.is_empty(), "b 无父节点");
    // MatMul: [input, W]
    assert_eq!(nodes[1].parents, vec![0, w_id], "MatMul 父节点：[input, W]");
    // Add: [MatMul, b]
    assert_eq!(
        nodes[3].parents,
        vec![mm_id, b_id],
        "Add 父节点：[MatMul, b]"
    );
}

#[test]
fn expand_linear_all_same_block_id() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_linear(0, 4, 8, 42, &mut c);
    for node in &nodes {
        assert_eq!(node.block_id, Some(42), "所有节点应共享 block_id");
    }
}

// ==================== RNN/LSTM/GRU 的 NodeLevel 迁移 ====================

#[test]
fn expand_rnn_produces_parameter_nodes_and_cell() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_rnn(0, 4, 6, false, 5, 7, &mut c);
    assert_eq!(nodes.len(), 4);
    assert!(nodes[0].is_parameter());
    assert!(nodes[1].is_parameter());
    assert!(nodes[2].is_parameter());
    assert!(matches!(
        nodes[3].node_type,
        NodeTypeDescriptor::CellRnn {
            input_size: 4,
            hidden_size: 6,
            return_sequences: false,
            seq_len: 5
        }
    ));
    assert_eq!(nodes[3].output_shape, vec![1, 6]);
    assert!(nodes.iter().all(|n| n.block_id == Some(7)));
}

#[test]
fn expand_lstm_and_gru_return_sequence_shape() {
    let mut c1 = InnovationCounter::new(1);
    let lstm_nodes = expand_lstm(0, 3, 8, true, 9, 11, &mut c1);
    assert_eq!(lstm_nodes.len(), 13);
    assert!(matches!(
        lstm_nodes.last().unwrap().node_type,
        NodeTypeDescriptor::CellLstm {
            input_size: 3,
            hidden_size: 8,
            return_sequences: true,
            seq_len: 9
        }
    ));
    assert_eq!(lstm_nodes.last().unwrap().output_shape, vec![1, 9, 8]);

    let mut c2 = InnovationCounter::new(1);
    let gru_nodes = expand_gru(0, 3, 8, true, 9, 12, &mut c2);
    assert_eq!(gru_nodes.len(), 10);
    assert!(matches!(
        gru_nodes.last().unwrap().node_type,
        NodeTypeDescriptor::CellGru {
            input_size: 3,
            hidden_size: 8,
            return_sequences: true,
            seq_len: 9
        }
    ));
    assert_eq!(gru_nodes.last().unwrap().output_shape, vec![1, 9, 8]);
}

#[test]
fn migrate_sequential_genome_no_longer_deferred() {
    let genome = NetworkGenome::minimal_sequential(4, 2);
    let out = migrate_network_genome(&genome).unwrap();
    assert!(out.deferred.is_empty(), "循环层展开为 NodeLevel 后不应再 deferred");
    assert!(
        out.nodes
            .iter()
            .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. }))
    );

    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 0, 4], ShapeDomain::Sequence);
    assert!(
        analysis.is_valid,
        "序列 genome 迁移后应静态合法: {:?}",
        analysis.errors
    );
}

#[test]
fn migrate_stacked_recurrent_genome_marks_return_sequences_correctly() {
    let mut genome = NetworkGenome::minimal_sequential(4, 2);
    let inn = genome.next_innovation_number();
    genome.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Lstm { hidden_size: 4 },
            enabled: true,
        },
    );

    let out = migrate_network_genome(&genome).unwrap();
    let cell_nodes: Vec<_> = out
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.node_type,
                NodeTypeDescriptor::CellRnn { .. }
                    | NodeTypeDescriptor::CellLstm { .. }
                    | NodeTypeDescriptor::CellGru { .. }
            )
        })
        .collect();
    assert_eq!(cell_nodes.len(), 2);

    assert!(matches!(
        cell_nodes[0].node_type,
        NodeTypeDescriptor::CellRnn {
            return_sequences: true,
            ..
        }
    ));
    assert!(matches!(
        cell_nodes[1].node_type,
        NodeTypeDescriptor::CellLstm {
            return_sequences: false,
            ..
        }
    ));
}

#[test]
fn expand_linear_analysis_valid() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_linear(0, 4, 8, 0, &mut c);
    let analysis = GenomeAnalysis::compute(&nodes, 0, vec![1, 4], ShapeDomain::Flat);
    assert!(
        analysis.is_valid,
        "Linear 展开应通过合法性校验：{:?}",
        analysis.errors
    );
    assert_eq!(analysis.param_count, 4 * 8 + 1 * 8, "参数量 = W + b = 40");
}

// ==================== expand_activation ====================

#[test]
fn expand_activation_single_node() {
    let mut c = InnovationCounter::new(10);
    let nodes = expand_activation(5, vec![1, 8], &ActivationType::ReLU, &mut c);
    assert_eq!(nodes.len(), 1);
    assert!(matches!(nodes[0].node_type, NodeTypeDescriptor::ReLU));
    assert_eq!(nodes[0].parents, vec![5]);
    assert_eq!(nodes[0].block_id, None, "激活节点无 block_id");
}

#[test]
fn expand_activation_all_types() {
    let input_shape = vec![1, 4];
    let cases = [
        ActivationType::ReLU,
        ActivationType::Tanh,
        ActivationType::Sigmoid,
        ActivationType::GELU,
        ActivationType::SiLU,
        ActivationType::Softplus,
        ActivationType::ReLU6,
        ActivationType::SELU,
        ActivationType::Mish,
        ActivationType::HardSwish,
        ActivationType::HardSigmoid,
        ActivationType::LeakyReLU { alpha: 0.01 },
        ActivationType::ELU { alpha: 1.0 },
    ];
    for act in cases {
        let mut c = InnovationCounter::new(1);
        let nodes = expand_activation(0, input_shape.clone(), &act, &mut c);
        assert_eq!(nodes.len(), 1, "每种激活应展开为 1 个节点：{act:?}");
        assert_eq!(
            nodes[0].output_shape, input_shape,
            "激活不改变形状：{act:?}"
        );
    }
}

// ==================== expand_conv2d ====================

#[test]
fn expand_conv2d_four_nodes_with_bias() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_conv2d(0, 1, 8, 3, (28, 28), 0, &mut c);
    assert_eq!(nodes.len(), 4);
    assert!(nodes[0].is_parameter(), "节点 0 应为 kernel Parameter");
    assert!(
        matches!(nodes[1].node_type, NodeTypeDescriptor::Conv2d { .. }),
        "节点 1 应为 Conv2d"
    );
    assert!(nodes[2].is_parameter(), "节点 2 应为 bias Parameter");
    assert!(
        matches!(nodes[3].node_type, NodeTypeDescriptor::Add),
        "节点 3 应为 bias Add"
    );
}

#[test]
fn expand_conv2d_kernel_shape() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_conv2d(0, 1, 8, 3, (28, 28), 0, &mut c);
    assert_eq!(
        nodes[0].output_shape,
        vec![8, 1, 3, 3],
        "kernel 形状 [out_ch,in_ch,k,k]"
    );
    assert_eq!(
        nodes[2].output_shape,
        vec![1, 8, 1, 1],
        "bias 形状 [1,out_ch,1,1]"
    );
}

#[test]
fn expand_conv2d_same_padding_preserves_spatial() {
    let mut c = InnovationCounter::new(1);
    // k=3, padding=1, stride=1 → H/W 不变
    let nodes = expand_conv2d(0, 1, 8, 3, (28, 28), 0, &mut c);
    assert_eq!(
        nodes[1].output_shape,
        vec![1, 8, 28, 28],
        "conv 输出应保持 H/W"
    );
    assert_eq!(
        nodes[3].output_shape,
        vec![1, 8, 28, 28],
        "bias add 输出应保持 H/W"
    );
}

#[test]
fn expand_conv2d_all_same_block_id() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_conv2d(0, 1, 8, 3, (28, 28), 7, &mut c);
    for node in &nodes {
        assert_eq!(node.block_id, Some(7));
    }
}

// ==================== expand_pool2d ====================

#[test]
fn expand_maxpool_single_node() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_pool2d(0, PoolType::Max, 2, 2, (28, 28), 8, &mut c);
    assert_eq!(nodes.len(), 1);
    assert!(matches!(
        nodes[0].node_type,
        NodeTypeDescriptor::MaxPool2d { .. }
    ));
    assert_eq!(
        nodes[0].output_shape,
        vec![1, 8, 14, 14],
        "MaxPool 2x2/s2 输出"
    );
}

#[test]
fn expand_avgpool_single_node() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_pool2d(0, PoolType::Avg, 2, 2, (28, 28), 4, &mut c);
    assert_eq!(nodes.len(), 1);
    assert!(matches!(
        nodes[0].node_type,
        NodeTypeDescriptor::AvgPool2d { .. }
    ));
}

// ==================== expand_flatten ====================

#[test]
fn expand_flatten_spatial_to_flat() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_flatten(0, 8, Some((14, 14)), &mut c);
    assert_eq!(nodes.len(), 1);
    assert!(matches!(
        nodes[0].node_type,
        NodeTypeDescriptor::Flatten { .. }
    ));
    assert_eq!(nodes[0].output_shape, vec![1, 8 * 14 * 14]);
}

#[test]
fn expand_flatten_already_flat() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_flatten(0, 128, None, &mut c);
    assert_eq!(nodes[0].output_shape, vec![1, 128]);
}

// ==================== expand_dropout ====================

#[test]
fn expand_dropout_single_node() {
    let mut c = InnovationCounter::new(1);
    let nodes = expand_dropout(0, vec![1, 16], 0.5, &mut c);
    assert_eq!(nodes.len(), 1);
    assert!(
        matches!(nodes[0].node_type, NodeTypeDescriptor::Dropout { p } if (p - 0.5).abs() < 1e-6)
    );
    assert_eq!(nodes[0].output_shape, vec![1, 16]);
}

// ==================== migrate_network_genome ====================

/// 辅助：构建最小 MLP genome（Input → Linear(out)）
fn minimal_genome(input_dim: usize, output_dim: usize) -> NetworkGenome {
    NetworkGenome::minimal(input_dim, output_dim)
}

/// 辅助：构建含隐藏层的 MLP genome（Input → Linear(hidden) → ReLU → Linear(out)）
fn mlp_genome(input_dim: usize, hidden: usize, output_dim: usize) -> NetworkGenome {
    let mut g = NetworkGenome::minimal(input_dim, output_dim);
    // 在输出头前插入 Linear(hidden) + ReLU
    let hidden_layer = LayerGene {
        innovation_number: g.next_innovation_number(),
        layer_config: LayerConfig::Linear {
            out_features: hidden,
        },
        enabled: true,
    };
    let relu_layer = LayerGene {
        innovation_number: g.next_innovation_number(),
        layer_config: LayerConfig::Activation {
            activation_type: ActivationType::ReLU,
        },
        enabled: true,
    };
    // 插入到输出头前面
    g.layers_mut().insert(0, relu_layer);
    g.layers_mut().insert(0, hidden_layer);
    g
}

#[test]
fn migrate_minimal_genome_xor() {
    // XOR: Input(2) → Linear(1)
    let genome = minimal_genome(2, 1);
    let out = migrate_network_genome(&genome).expect("迁移不应失败");

    // Linear(1) 展开为 4 个节点
    assert_eq!(out.nodes.len(), 4, "1 个 Linear → 4 节点");
    assert!(out.deferred.is_empty(), "不应有 deferred 层");
    assert_eq!(
        out.output_innovation,
        out.nodes.last().unwrap().innovation_number
    );
}

#[test]
fn migrate_minimal_genome_analysis_valid() {
    let genome = minimal_genome(4, 3); // Iris
    let out = migrate_network_genome(&genome).unwrap();
    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 4], ShapeDomain::Flat);
    assert!(
        analysis.is_valid,
        "Iris 最小 genome 迁移应合法：{:?}",
        analysis.errors
    );
    assert_eq!(
        analysis.param_count,
        4 * 3 + 1 * 3,
        "参数量 = W[4,3] + b[1,3] = 15"
    );
}

#[test]
fn migrate_mlp_genome_node_count() {
    // Input(4) → Linear(8) → ReLU → Linear(3)
    let genome = mlp_genome(4, 8, 3);
    let out = migrate_network_genome(&genome).unwrap();
    // Linear(8): 4 节点，ReLU: 1 节点，Linear(3): 4 节点 = 9 节点
    assert_eq!(out.nodes.len(), 9, "2 个 Linear + 1 ReLU = 9 节点");
    assert!(out.deferred.is_empty());
}

#[test]
fn migrate_mlp_genome_analysis_valid() {
    let genome = mlp_genome(2, 4, 1);
    let out = migrate_network_genome(&genome).unwrap();
    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 2], ShapeDomain::Flat);
    assert!(analysis.is_valid, "MLP 迁移应合法：{:?}", analysis.errors);
    // W1[2,4]+b1[1,4]+W2[4,1]+b2[1,1] = 8+4+4+1 = 17
    assert_eq!(
        analysis.param_count,
        2 * 4 + 1 * 4 + 4 * 1 + 1 * 1,
        "参数量应为 17"
    );
}

#[test]
fn migrate_spatial_genome_flatten_linear() {
    // minimal_spatial: Conv(1→8) → MaxPool2d → Flatten → Linear(10)（小 CNN 起点）
    let genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    let out = migrate_network_genome(&genome).unwrap();
    // Conv2d(4) + MaxPool(1) + Flatten(1) + Linear(4) = 10 节点
    assert_eq!(out.nodes.len(), 10, "Conv+Pool+Flatten+Linear = 10 节点");
    assert!(out.deferred.is_empty());

    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 1, 28, 28], ShapeDomain::Spatial);
    assert!(
        analysis.is_valid,
        "空间 genome 迁移应合法：{:?}",
        analysis.errors
    );
}

#[test]
fn migrate_genome_with_conv2d() {
    // 在 minimal_spatial 主链前再叠一层 Conv2d(8)（[Conv(新), Conv(种), Pool, F, L]）
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    let conv_layer = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Conv2d {
            out_channels: 8,
            kernel_size: 3,
        },
        enabled: true,
    };
    genome.layers_mut().insert(0, conv_layer);

    let out = migrate_network_genome(&genome).unwrap();
    // 双 Conv2d(各 4) + MaxPool(1) + Flatten(1) + Linear(4) = 14 节点
    assert_eq!(out.nodes.len(), 14, "双 Conv+Pool+Flatten+Linear = 14 节点");
    assert!(out.deferred.is_empty());
}

#[test]
fn migrate_rnn_genome_deferred() {
    // 序列 genome（含 Rnn 层）应直接展开为 CellRnn，不再 deferred
    let genome = NetworkGenome::minimal_sequential(4, 2);
    let out = migrate_network_genome(&genome).unwrap();
    assert!(out.deferred.is_empty(), "Rnn 展开为 CellRnn 后不应再 deferred");
    assert!(
        out.nodes
            .iter()
            .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. })),
        "迁移结果应包含 CellRnn 节点"
    );
}

#[test]
fn migrate_genome_with_skip_edge_add() {
    // 使用 input_dim=hidden=4，确保 Add 聚合的两个输入维度相同：
    // Input(4) → Linear(4) → ReLU →(+skip from INPUT, dim=4) → Linear(1)
    let mut genome = mlp_genome(4, 4, 1);

    // 给最后一层（Linear(1)，输出头）加一条从 INPUT 的 skip edge（Add 策略）
    // INPUT 维度=4，ReLU 输出维度=4，两者相同，Add 合法
    let output_head_innov = genome
        .layers()
        .iter()
        .rev()
        .find(|l| l.enabled)
        .unwrap()
        .innovation_number;

    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: 100, // dummy，仅迁移时用 from/to
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_head_innov,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let out = migrate_network_genome(&genome).unwrap();
    // Linear(4): 4 节点, ReLU: 1, Agg-Add: 1, Linear(1): 4 节点 = 10 节点
    assert_eq!(out.nodes.len(), 10, "带 skip edge 应多一个聚合节点");
    assert!(out.deferred.is_empty());
}

#[test]
fn migrate_innovation_numbers_monotonic() {
    let genome = mlp_genome(2, 4, 1);
    let out = migrate_network_genome(&genome).unwrap();
    let ids: Vec<u64> = out.nodes.iter().map(|n| n.innovation_number).collect();
    assert!(ids.iter().all(|&id| id > 0));
    for w in ids.windows(2) {
        assert!(w[0] < w[1], "创新号应严格单调递增");
    }
    assert_eq!(out.next_innovation, *ids.last().unwrap() + 1);
}

// ============================================================
// 补充测试（P0 + P1 覆盖差距 — 2026-03-25）
// ============================================================

// ==================== P0: Roundtrip 形状一致性 ====================

/// 验证展开函数设定的 output_shape 与 GenomeAnalysis 推导出的形状完全一致
#[test]
fn migrate_mlp_shapes_consistent_with_analysis() {
    let genome = mlp_genome(4, 8, 3);
    let out = migrate_network_genome(&genome).unwrap();
    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 4], ShapeDomain::Flat);
    assert!(analysis.is_valid, "{:?}", analysis.errors);
    for node in &out.nodes {
        if let Some(inferred) = analysis.shape_of(node.innovation_number) {
            assert_eq!(
                inferred, &node.output_shape,
                "节点 {} 声明形状与推导形状不一致 ({:?})",
                node.innovation_number, node.node_type
            );
        }
    }
}

// ==================== P0: 多层 CNN 管道迁移 ====================

/// Conv2d → Pool2d → Conv2d → Flatten → Linear，空间维度逐层缩减验证
#[test]
fn migrate_cnn_pipeline_conv_pool_conv_flatten_linear() {
    // 目标: [Conv(1→8,k=3), MaxPool(2,2), Conv(8→16,k=3), Flatten, Linear(10)]
    // 在 minimal 的 [Conv(1→8), MaxPool(2,2), ...] 上，仅在 Pool 后、Flatten 前插入第二段 Conv(8→16)
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    let conv2 = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Conv2d {
            out_channels: 16,
            kernel_size: 3,
        },
        enabled: true,
    };
    genome.layers_mut().insert(2, conv2);

    let out = migrate_network_genome(&genome).unwrap();
    // 双 Conv2d(各 4) + MaxPool(1) + Flatten(1) + Linear(4) = 14 节点
    assert_eq!(out.nodes.len(), 14, "CNN 管道应展开为 14 节点");
    assert!(out.deferred.is_empty());

    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 1, 28, 28], ShapeDomain::Spatial);
    assert!(analysis.is_valid, "CNN 迁移应合法：{:?}", analysis.errors);

    // 验证空间维度逐层缩减的形状一致性
    for node in &out.nodes {
        if let Some(inferred) = analysis.shape_of(node.innovation_number) {
            assert_eq!(
                inferred, &node.output_shape,
                "CNN 节点 {} 形状不一致：{:?}",
                node.innovation_number, node.node_type
            );
        }
    }
}

#[test]
fn migrate_genome_with_skip_edge_mean_preserves_mean_semantics() {
    let mut genome = mlp_genome(4, 4, 1);
    let output_head_innov = genome
        .layers()
        .iter()
        .rev()
        .find(|l| l.enabled)
        .unwrap()
        .innovation_number;

    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: 201,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_head_innov,
        strategy: AggregateStrategy::Mean,
        enabled: true,
    });

    let out = migrate_network_genome(&genome).unwrap();
    assert!(out.deferred.is_empty(), "Mean 迁移不应留下 deferred 警告");
    assert!(
        out.nodes
            .iter()
            .any(|n| matches!(n.node_type, NodeTypeDescriptor::Stack { axis: 0 })),
        "Mean 迁移应插入 Stack(axis=0)"
    );
    assert!(
        out.nodes
            .iter()
            .any(|n| matches!(n.node_type, NodeTypeDescriptor::Mean { axis: Some(0) })),
        "Mean 迁移应插入 Mean(axis=0)"
    );

    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 4], ShapeDomain::Flat);
    assert!(
        analysis.is_valid,
        "Mean skip edge 迁移应合法：{:?}",
        analysis.errors
    );
}

// ==================== P1: Concat SkipEdge ====================

/// Concat 策略的 skip edge 迁移验证
#[test]
fn migrate_genome_with_skip_edge_concat() {
    // Input(2) → Linear(2) → ReLU → [Concat(main=2, skip=INPUT=2) → 得 4] → Linear(1, in=4)
    let mut genome = mlp_genome(2, 2, 1);
    let output_head_innov = genome
        .layers()
        .iter()
        .rev()
        .find(|l| l.enabled)
        .unwrap()
        .innovation_number;

    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: 200,
        from_innovation: INPUT_INNOVATION,
        to_innovation: output_head_innov,
        strategy: AggregateStrategy::Concat { dim: 1 }, // 沿 features 轴拼接
        enabled: true,
    });

    let out = migrate_network_genome(&genome).unwrap();
    // Linear(2): 4, ReLU: 1, Concat agg: 1, Linear(1) in_dim=4: 4 = 10 节点
    assert_eq!(out.nodes.len(), 10, "Concat skip edge 应多一个聚合节点");
    assert!(out.deferred.is_empty());

    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 2], ShapeDomain::Flat);
    assert!(
        analysis.is_valid,
        "Concat skip edge 迁移应合法：{:?}",
        analysis.errors
    );
}

// ==================== P1: Disabled 层 + SkipEdge ====================

/// skip edge 来源指向禁用层时，迁移不应 panic
#[test]
fn migrate_skip_edge_from_disabled_layer_no_panic() {
    let mut genome = mlp_genome(2, 4, 1);
    let first_layer_innov = genome.layers()[0].innovation_number;
    genome.layers_mut()[0].enabled = false; // 禁用 Linear(4)

    let output_head_innov = genome
        .layers()
        .iter()
        .rev()
        .find(|l| l.enabled)
        .unwrap()
        .innovation_number;
    genome.skip_edges_mut().push(SkipEdge {
        innovation_number: 300,
        from_innovation: first_layer_innov, // 指向被禁用的层
        to_innovation: output_head_innov,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    // resolve_dimensions 会检测到 skip edge 源创新号不在维度表中而返回 InvalidSkipEdge。
    // 迁移层将其包装为 DimensionError 返回——这是正确行为（genome 本身不合法）。
    // 测试目标：不应 panic，应返回包含原因的 Err。
    let result = migrate_network_genome(&genome);
    assert!(
        result.is_err(),
        "含指向禁用层的 skip edge 的 genome 不合法，应返回 Err"
    );
    if let Err(e) = result {
        let msg = e.to_string();
        assert!(
            msg.contains("维度") || msg.contains("skip") || msg.contains("invalid"),
            "错误消息应描述原因，实际：{msg}"
        );
    }
}

// ==================== P1: Spatial 域完整链路 ====================

/// Conv2d(Spatial) → Pool2d(Spatial) → Flatten(Flat) → Linear(Flat) 的域传播验证
#[test]
fn migrate_spatial_full_domain_chain() {
    // 构建包含 Conv + Pool 的空间 CNN
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    let pool = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Pool2d {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 2,
        },
        enabled: true,
    };
    let conv = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Conv2d {
            out_channels: 8,
            kernel_size: 3,
        },
        enabled: true,
    };
    genome.layers_mut().insert(0, pool);
    genome.layers_mut().insert(0, conv);
    // 层序: [Conv(1→8), Pool(2,2), Flatten, Linear(10)]

    let out = migrate_network_genome(&genome).unwrap();
    let analysis = GenomeAnalysis::compute(&out.nodes, 0, vec![1, 1, 28, 28], ShapeDomain::Spatial);
    assert!(analysis.is_valid, "{:?}", analysis.errors);

    for node in &out.nodes {
        let domain = analysis.domain_of(node.innovation_number).unwrap();
        let nt = &node.node_type;
        match nt {
            // Conv2d 和 MaxPool2d 节点应为 Spatial 域
            NodeTypeDescriptor::Conv2d { .. } | NodeTypeDescriptor::MaxPool2d { .. } => {
                assert_eq!(
                    domain,
                    ShapeDomain::Spatial,
                    "节点 {} ({nt:?}) 应为 Spatial 域",
                    node.innovation_number
                );
            }
            // Flatten 节点应切换到 Flat 域
            NodeTypeDescriptor::Flatten { .. } => {
                assert_eq!(
                    domain,
                    ShapeDomain::Flat,
                    "节点 {} ({nt:?}) 应为 Flat 域",
                    node.innovation_number
                );
            }
            // MatMul 始终位于 Flatten 后，应为 Flat 域
            NodeTypeDescriptor::MatMul => {
                assert_eq!(
                    domain,
                    ShapeDomain::Flat,
                    "节点 {} ({nt:?}) 应为 Flat 域",
                    node.innovation_number
                );
            }
            // Add 既可能是 Conv2d bias add（Spatial），也可能是 Linear bias add（Flat）
            NodeTypeDescriptor::Add => {
                let expected = if node.output_shape.len() == 4 {
                    ShapeDomain::Spatial
                } else {
                    ShapeDomain::Flat
                };
                assert_eq!(
                    domain, expected,
                    "节点 {} ({nt:?}) 域不符合其输出形状 {:?}",
                    node.innovation_number, node.output_shape
                );
            }
            // Parameter 节点始终为 Flat
            NodeTypeDescriptor::Parameter => {
                assert_eq!(domain, ShapeDomain::Flat);
            }
            _ => {} // 其他节点不检查
        }
    }
}

// ==================== C.2: Conv2d → FM 分解迁移 ====================

/// 辅助函数：创建一个最小的 Conv2d 模板块 (kernel+conv+bias+add)，返回节点列表
fn make_conv2d_block(
    input_id: u64,
    in_ch: usize,
    out_ch: usize,
    kernel_size: usize,
    spatial: (usize, usize),
    block_id: u64,
    counter: &mut InnovationCounter,
) -> Vec<NodeGene> {
    expand_conv2d(input_id, in_ch, out_ch, kernel_size, spatial, block_id, counter)
}

/// 辅助函数：创建一个带输入节点的完整 NodeLevel 节点列表（spatial input + conv block + output）
fn make_simple_spatial_nodes(
    in_ch: usize,
    out_ch: usize,
    spatial: (usize, usize),
) -> (Vec<NodeGene>, InnovationCounter) {
    let mut counter = InnovationCounter::new(1);

    let input_id = INPUT_INNOVATION; // 0 — 隐式输入，不创建节点

    let block_id = counter.next();
    let block_nodes = make_conv2d_block(
        input_id,
        in_ch,
        out_ch,
        3,
        spatial,
        block_id,
        &mut counter,
    );

    let last_id = block_nodes.last().unwrap().innovation_number;

    let output_id = counter.next();
    let output_node = NodeGene::new(
        output_id,
        NodeTypeDescriptor::Identity,
        vec![1, out_ch, spatial.0, spatial.1],
        vec![last_id],
        None,
    );

    let mut nodes = block_nodes;
    nodes.push(output_node);

    (nodes, counter)
}

#[test]
fn fm_migration_empty_genome_is_noop() {
    let mut counter = InnovationCounter::new(100);
    let mut nodes: Vec<NodeGene> = vec![];
    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);
    assert!(nodes.is_empty());
}

#[test]
fn fm_migration_no_conv_block_is_noop() {
    let mut counter = InnovationCounter::new(100);
    let mut nodes = vec![
        NodeGene::new(1, NodeTypeDescriptor::Identity, vec![1, 1, 8, 8], vec![INPUT_INNOVATION], None),
        NodeGene::new(2, NodeTypeDescriptor::Identity, vec![1, 1, 8, 8], vec![1], None),
    ];
    let original_len = nodes.len();
    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);
    assert_eq!(nodes.len(), original_len);
}

#[test]
fn fm_migration_basic_1ch_to_1ch() {
    let (mut nodes, mut counter) = make_simple_spatial_nodes(1, 1, (8, 8));
    let original_len = nodes.len();

    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);

    // 原模板块节点应被移除（kernel, conv, bias, add）
    // 新增：1 input FM (Identity) + 1 kernel + 1 conv + 1 concat + 输出重定向
    // 没有 Add（只有 1 条输入边），所以 agg_id = conv_id 本身
    assert!(
        nodes.len() > original_len - 4, // 移除了旧块，加入了新节点
        "迁移后节点数应变化: {} -> {}",
        original_len,
        nodes.len()
    );

    // 验证没有残留的旧模板块节点（所有 enabled 节点的 block_id 如果有值，则应该有 fm_id 或者是新 edge 块）
    let fm_nodes: Vec<&NodeGene> = nodes.iter().filter(|n| n.fm_id.is_some()).collect();
    assert!(
        !fm_nodes.is_empty(),
        "迁移后应有 fm_id 标记的节点"
    );

    // 验证 Concat 节点存在
    let concat_nodes: Vec<&NodeGene> = nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Concat { .. }))
        .collect();
    assert_eq!(concat_nodes.len(), 1, "应有 1 个 Concat 节点");

    // Concat 的输出 shape 应为 [1, 1, H, W]
    let concat = concat_nodes[0];
    assert_eq!(concat.output_shape[1], 1, "Concat 输出通道应为 1");
}

#[test]
fn fm_migration_2ch_to_3ch() {
    let (mut nodes, mut counter) = make_simple_spatial_nodes(2, 3, (8, 8));

    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);

    // 验证 FM 节点数：2 输入 FM + 聚合相关
    let fm_nodes: Vec<&NodeGene> = nodes.iter().filter(|n| n.fm_id.is_some()).collect();
    assert!(
        fm_nodes.len() >= 2,
        "至少应有 2 个输入 FM 节点, got {}",
        fm_nodes.len()
    );

    // 唯一的 fm_id 集合大小
    let fm_ids: std::collections::HashSet<u64> = fm_nodes.iter().map(|n| n.fm_id.unwrap()).collect();
    // 至少 2（输入）+ 3（输出）= 5 个不同的 FM
    assert!(
        fm_ids.len() >= 5,
        "应有至少 5 个不同 fm_id（2 输入 + 3 输出）, got {}",
        fm_ids.len()
    );

    // 验证 Concat 节点
    let concat_nodes: Vec<&NodeGene> = nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Concat { .. }))
        .collect();
    assert_eq!(concat_nodes.len(), 1, "应有 1 个 Concat 节点");
    assert_eq!(
        concat_nodes[0].output_shape[1], 3,
        "Concat 输出通道应为 3"
    );

    // 验证 FM 边数量：2 × 3 = 6 条 Conv2d 边
    let conv_edge_nodes: Vec<&NodeGene> = nodes
        .iter()
        .filter(|n| {
            matches!(n.node_type, NodeTypeDescriptor::Conv2d { .. })
                && n.output_shape == vec![1, 1, 8, 8]
        })
        .collect();
    assert_eq!(
        conv_edge_nodes.len(),
        6,
        "2 输入 × 3 输出 = 6 条 FM 边 (Conv2d), got {}",
        conv_edge_nodes.len()
    );

    // 每条 FM 边的 kernel 应为 [1, 1, 3, 3]
    for edge in &conv_edge_nodes {
        assert_eq!(edge.parents.len(), 2, "Conv2d 边应有 2 个 parent");
        let kernel_id = edge.parents[1];
        let kernel = nodes.iter().find(|n| n.innovation_number == kernel_id).unwrap();
        assert_eq!(
            kernel.output_shape,
            vec![1, 1, 3, 3],
            "FM 边 kernel 应为 [1, 1, 3, 3]"
        );
    }
}

#[test]
fn fm_migration_preserves_downstream_connectivity() {
    let (mut nodes, mut counter) = make_simple_spatial_nodes(1, 2, (8, 8));

    // 找到迁移前输出节点的 parent（应该是 Add/conv 块的输出）
    let output_node_id = nodes.last().unwrap().innovation_number;
    let old_parent = nodes.last().unwrap().parents[0];

    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);

    // 迁移后输出节点的 parent 应指向 Concat（而非旧的 block 输出）
    let output_node = nodes
        .iter()
        .find(|n| n.innovation_number == output_node_id)
        .unwrap();
    let new_parent = output_node.parents[0];
    assert_ne!(
        new_parent, old_parent,
        "输出节点 parent 应从旧 block 输出重定向到 Concat"
    );

    // 新 parent 应该是 Concat 节点
    let parent_node = nodes
        .iter()
        .find(|n| n.innovation_number == new_parent)
        .unwrap();
    assert!(
        matches!(parent_node.node_type, NodeTypeDescriptor::Concat { .. }),
        "输出节点的新 parent 应为 Concat 节点"
    );
}

#[test]
fn fm_migration_old_template_nodes_removed() {
    let (mut nodes, mut counter) = make_simple_spatial_nodes(1, 2, (8, 8));

    // 记录原始 block_id
    let old_block_ids: Vec<Option<u64>> = nodes.iter().map(|n| n.block_id).collect();
    let original_block_id = old_block_ids.iter().find(|b| b.is_some()).unwrap().unwrap();

    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);

    // 旧模板块的 kernel/conv/bias/add 应全部被移除（enabled=false → retain 后消失）
    let old_block_nodes: Vec<&NodeGene> = nodes
        .iter()
        .filter(|n| n.block_id == Some(original_block_id) && n.fm_id.is_none())
        .collect();
    assert!(
        old_block_nodes.is_empty(),
        "旧模板块节点应全部被移除，残留 {} 个",
        old_block_nodes.len()
    );
}

#[test]
fn fm_migration_idempotent() {
    let (mut nodes, mut counter) = make_simple_spatial_nodes(1, 2, (8, 8));

    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);
    let nodes_after_first = nodes.clone();
    let counter_after_first = counter.peek();

    // 再次迁移应为 no-op（没有新的 Conv2d 模板块）
    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);

    assert_eq!(
        nodes.len(),
        nodes_after_first.len(),
        "第二次迁移应为 no-op"
    );
    assert_eq!(counter.peek(), counter_after_first, "计数器不应变化");
}

#[test]
fn fm_migration_add_aggregation_tree() {
    // 测试多输入通道时的 Add 聚合树
    let (mut nodes, mut counter) = make_simple_spatial_nodes(4, 1, (8, 8));

    migrate_conv2d_to_feature_maps(&mut nodes, &mut counter);

    // 1 个输出 FM 有 4 条输入边，需要 Add 聚合树
    // 4 → 2 pairs → 2 Adds → 1 pair → 1 Add = 3 Adds
    let add_nodes: Vec<&NodeGene> = nodes
        .iter()
        .filter(|n| {
            matches!(n.node_type, NodeTypeDescriptor::Add)
                && n.fm_id.is_some()
        })
        .collect();
    assert_eq!(
        add_nodes.len(),
        3,
        "4 条输入边需要 3 个 Add 节点聚合, got {}",
        add_nodes.len()
    );
}

#[test]
fn fm_migration_genome_level_spatial() {
    // 通过 NetworkGenome 层面调用 migrate_to_fm_level
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (8, 8));
    let _ = genome.migrate_to_node_level();
    genome.migrate_to_fm_level();

    let nodes = genome.nodes();

    // 验证有 FM 节点
    let fm_count = nodes.iter().filter(|n| n.fm_id.is_some()).count();
    assert!(fm_count > 0, "FM 级别迁移后应有 fm_id 节点");

    // 验证有 Concat 节点
    let concat_count = nodes
        .iter()
        .filter(|n| matches!(n.node_type, NodeTypeDescriptor::Concat { .. }))
        .count();
    assert!(concat_count > 0, "FM 级别迁移后应有 Concat 节点");

    // 验证 migrate_to_fm_level 是幂等的
    let count_before = nodes.len();
    genome.migrate_to_fm_level();
    assert_eq!(
        genome.nodes().len(),
        count_before,
        "重复调用 migrate_to_fm_level 应为 no-op"
    );
}

#[test]
fn fm_migration_non_spatial_is_noop() {
    // Flat 模式不应触发 FM 迁移
    let mut genome = NetworkGenome::minimal(10, 5);
    let _ = genome.migrate_to_node_level();
    let count_before = genome.nodes().len();
    genome.migrate_to_fm_level();
    assert_eq!(
        genome.nodes().len(),
        count_before,
        "Flat 模式 migrate_to_fm_level 应为 no-op"
    );
}

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
    expand_gru, expand_linear, expand_lstm, expand_pool2d, expand_rnn, migrate_network_genome,
};
use crate::nn::evolution::node_gene::GenomeAnalysis;

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

// ==================== Phase 8: RNN/LSTM/GRU NodeLevel 迁移 ====================

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
    assert!(out.deferred.is_empty(), "Phase 8 后循环层不应再 deferred");
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
    // 空间输入 → Flatten → Linear(10)（MNIST 最小结构）
    let genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    let out = migrate_network_genome(&genome).unwrap();
    // Flatten: 1 节点，Linear(10): 4 节点 = 5 节点
    assert_eq!(out.nodes.len(), 5, "Flatten + Linear = 5 节点");
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
    // Conv2d → Flatten → Linear（简单 CNN）
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    let conv_layer = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Conv2d {
            out_channels: 8,
            kernel_size: 3,
        },
        enabled: true,
    };
    // 插入到 Flatten 前面
    genome.layers_mut().insert(0, conv_layer);

    let out = migrate_network_genome(&genome).unwrap();
    // Conv2d: 4 节点（kernel/conv/bias/add），Flatten: 1 节点，Linear(10): 4 节点 = 9 节点
    assert_eq!(out.nodes.len(), 9, "Conv+Flatten+Linear = 9 节点");
    assert!(out.deferred.is_empty());
}

#[test]
fn migrate_rnn_genome_deferred() {
    // 阶段 8 后：序列 genome（含 Rnn 层）应直接展开为 CellRnn，不再 deferred
    let genome = NetworkGenome::minimal_sequential(4, 2);
    let out = migrate_network_genome(&genome).unwrap();
    assert!(out.deferred.is_empty(), "阶段 8 后 Rnn 不应再 deferred");
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
    let mut genome = NetworkGenome::minimal_spatial(1, 10, (28, 28));
    // 初始: [Flatten, Linear(10)]
    // 目标层序: [Conv(1→8,k=3), Pool(2,2), Conv(8→16,k=3), Flatten, Linear(10)]
    let pool = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Pool2d {
            pool_type: PoolType::Max,
            kernel_size: 2,
            stride: 2,
        },
        enabled: true,
    };
    let conv2 = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Conv2d {
            out_channels: 16,
            kernel_size: 3,
        },
        enabled: true,
    };
    let conv1 = LayerGene {
        innovation_number: genome.next_innovation_number(),
        layer_config: LayerConfig::Conv2d {
            out_channels: 8,
            kernel_size: 3,
        },
        enabled: true,
    };
    // 插入到 Flatten 前：[conv1, pool, conv2, Flatten, Linear(10)]
    genome.layers_mut().insert(0, pool);
    genome.layers_mut().insert(0, conv2);
    genome.layers_mut().insert(0, conv1);

    let out = migrate_network_genome(&genome).unwrap();
    // Conv(1→8): 4, Pool: 1, Conv(8→16): 4, Flatten: 1, Linear(10): 4 = 14 节点
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

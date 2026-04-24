/*
 * @Author       : 老董
 * @Date         : 2026-03-25
 * @Description  : node_gene.rs 的单元测试
 *
 * 覆盖：
 * 1. NodeGene 数据结构与辅助方法
 * 2. infer_output_shape — 覆盖全部 NodeTypeDescriptor 变体
 * 3. GenomeAnalysis::compute — 正常拓扑、错误检测
 */

use crate::nn::descriptor::NodeTypeDescriptor;
use crate::nn::evolution::node_gene::{
    infer_domain, infer_output_shape, AnalysisError, GenomeAnalysis, GenomeKind, NodeGene,
};
use crate::nn::evolution::gene::ShapeDomain;
use crate::nn::nodes::raw_node::Reduction;

// ==================== 辅助宏 ====================

macro_rules! shape {
    ($($x:expr),+) => { vec![$($x),+] }
}

fn infer(nt: NodeTypeDescriptor, inputs: Vec<Vec<usize>>) -> Result<Vec<usize>, String> {
    let refs: Vec<&Vec<usize>> = inputs.iter().collect();
    infer_output_shape(&nt, &refs)
}

// ==================== NodeGene 结构测试 ====================

#[test]
fn node_gene_new_enabled_by_default() {
    let g = NodeGene::new(1, NodeTypeDescriptor::ReLU, shape![4, 8], vec![0], None);
    assert_eq!(g.innovation_number, 1);
    assert!(g.enabled);
    assert_eq!(g.block_id, None);
    assert!(!g.is_parameter());
    assert!(!g.is_input());
    assert!(!g.is_state());
    assert!(!g.is_leaf());
}

#[test]
fn node_gene_parameter_is_leaf() {
    let g = NodeGene::new(
        2,
        NodeTypeDescriptor::Parameter,
        shape![3, 4],
        vec![],
        Some(10),
    );
    assert!(g.is_parameter());
    assert!(g.is_leaf());
    assert_eq!(g.param_count(), 12);
    assert_eq!(g.block_id, Some(10));
}

#[test]
fn node_gene_input_is_leaf() {
    let g = NodeGene::new(
        0,
        NodeTypeDescriptor::BasicInput,
        shape![1, 2],
        vec![],
        None,
    );
    assert!(g.is_input());
    assert!(g.is_leaf());
    assert_eq!(g.param_count(), 0);
}

#[test]
fn node_gene_state_is_leaf() {
    let g = NodeGene::new(5, NodeTypeDescriptor::State, shape![1, 16], vec![], None);
    assert!(g.is_state());
    assert!(g.is_leaf());
}

#[test]
fn genome_kind_variants() {
    assert_ne!(GenomeKind::LayerLevel, GenomeKind::NodeLevel);
}

// ==================== 叶节点形状推导（应返回 Err）====================

#[test]
fn infer_leaf_nodes_return_err() {
    for nt in [
        NodeTypeDescriptor::Parameter,
        NodeTypeDescriptor::BasicInput,
        NodeTypeDescriptor::TargetInput,
        NodeTypeDescriptor::State,
    ] {
        assert!(
            infer(nt, vec![]).is_err(),
            "叶节点应返回 Err 提示调用方使用存储的形状"
        );
    }
}

// ==================== 恒等透传 ====================

#[test]
fn infer_identity() {
    assert_eq!(
        infer(NodeTypeDescriptor::Identity, vec![shape![2, 3]]).unwrap(),
        shape![2, 3]
    );
}

#[test]
fn infer_detach() {
    assert_eq!(
        infer(NodeTypeDescriptor::Detach, vec![shape![4]]).unwrap(),
        shape![4]
    );
}

// ==================== 一元逐元素激活/变换 ====================

#[test]
fn infer_unary_elementwise() {
    let input = shape![2, 3, 4];
    let cases: Vec<NodeTypeDescriptor> = vec![
        NodeTypeDescriptor::Sigmoid,
        NodeTypeDescriptor::Softmax,
        NodeTypeDescriptor::Tanh,
        NodeTypeDescriptor::LeakyReLU { alpha: 0.01 },
        NodeTypeDescriptor::Ln,
        NodeTypeDescriptor::LogSoftmax,
        NodeTypeDescriptor::Dropout { p: 0.5 },
        NodeTypeDescriptor::Sign,
        NodeTypeDescriptor::Abs,
        NodeTypeDescriptor::SoftPlus,
        NodeTypeDescriptor::Step,
        NodeTypeDescriptor::Negate,
        NodeTypeDescriptor::ReLU,
        NodeTypeDescriptor::Gelu,
        NodeTypeDescriptor::Swish,
        NodeTypeDescriptor::Elu { alpha: 1.0 },
        NodeTypeDescriptor::Selu,
        NodeTypeDescriptor::Mish,
        NodeTypeDescriptor::HardSwish,
        NodeTypeDescriptor::HardSigmoid,
        NodeTypeDescriptor::ReLU6,
        NodeTypeDescriptor::HardTanh {
            min_val: -1.0,
            max_val: 1.0,
        },
        NodeTypeDescriptor::Exp,
        NodeTypeDescriptor::Sqrt,
        NodeTypeDescriptor::Log10,
        NodeTypeDescriptor::Log2,
        NodeTypeDescriptor::Pow { exponent: 2.0 },
        NodeTypeDescriptor::Square,
        NodeTypeDescriptor::Reciprocal,
        NodeTypeDescriptor::Clip { min: 0.0, max: 1.0 },
    ];
    for nt in cases {
        let result = infer(nt.clone(), vec![input.clone()]);
        assert_eq!(result.unwrap(), input, "一元逐元素操作应保持形状：{nt:?}");
    }
}

// ==================== 归一化操作 ====================

#[test]
fn infer_batch_norm() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::BatchNormOp {
                eps: 1e-5,
                momentum: 0.1,
                num_features: 4
            },
            vec![shape![2, 4, 8]]
        )
        .unwrap(),
        shape![2, 4, 8]
    );
}

#[test]
fn infer_layer_norm() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::LayerNormOp {
                normalized_dims: 2,
                eps: 1e-5
            },
            vec![shape![3, 5]]
        )
        .unwrap(),
        shape![3, 5]
    );
}

#[test]
fn infer_rms_norm() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::RMSNormOp {
                normalized_dims: 1,
                eps: 1e-6
            },
            vec![shape![4, 10]]
        )
        .unwrap(),
        shape![4, 10]
    );
}

// ==================== ZerosLike ====================

#[test]
fn infer_zeros_like() {
    assert_eq!(
        infer(NodeTypeDescriptor::ZerosLike, vec![shape![2, 3, 4]]).unwrap(),
        shape![2, 3, 4]
    );
}

// ==================== 二元逐元素操作 ====================

#[test]
fn infer_binary_elementwise_matching() {
    let s = shape![2, 3];
    for nt in [
        NodeTypeDescriptor::Add,
        NodeTypeDescriptor::Subtract,
        NodeTypeDescriptor::Divide,
        NodeTypeDescriptor::Multiply,
        NodeTypeDescriptor::Maximum,
        NodeTypeDescriptor::Minimum,
    ] {
        assert_eq!(
            infer(nt.clone(), vec![s.clone(), s.clone()]).unwrap(),
            s,
            "二元逐元素操作应保持形状：{nt:?}"
        );
    }
}

#[test]
fn infer_add_shape_mismatch_returns_err() {
    assert!(infer(NodeTypeDescriptor::Add, vec![shape![2, 3], shape![2, 4]]).is_err());
}

#[test]
fn infer_add_broadcast_shape() {
    assert_eq!(
        infer(NodeTypeDescriptor::Add, vec![shape![1, 8, 28, 28], shape![1, 8, 1, 1]]).unwrap(),
        shape![1, 8, 28, 28]
    );
}

// ==================== WhereCond ====================

#[test]
fn infer_where_cond() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::WhereCond {
                condition_data: vec![1.0, 0.0],
                condition_shape: vec![2],
            },
            vec![shape![2, 3]]
        )
        .unwrap(),
        shape![2, 3]
    );
}

// ==================== MatMul ====================

#[test]
fn infer_matmul_2d() {
    assert_eq!(
        infer(NodeTypeDescriptor::MatMul, vec![shape![4, 3], shape![3, 5]]).unwrap(),
        shape![4, 5]
    );
}

#[test]
fn infer_matmul_batched() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::MatMul,
            vec![shape![2, 4, 3], shape![2, 3, 5]]
        )
        .unwrap(),
        shape![2, 4, 5]
    );
}

#[test]
fn infer_matmul_inner_mismatch_returns_err() {
    assert!(infer(NodeTypeDescriptor::MatMul, vec![shape![4, 3], shape![4, 5]]).is_err());
}

// ==================== Sum / Mean ====================

#[test]
fn infer_sum_global() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Sum { axis: None },
            vec![shape![3, 4, 5]]
        )
        .unwrap(),
        shape![1]
    );
}

#[test]
fn infer_sum_axis() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Sum { axis: Some(1) },
            vec![shape![3, 4, 5]]
        )
        .unwrap(),
        shape![3, 5]
    );
}

#[test]
fn infer_mean_global() {
    assert_eq!(
        infer(NodeTypeDescriptor::Mean { axis: None }, vec![shape![2, 3]]).unwrap(),
        shape![1]
    );
}

#[test]
fn infer_mean_axis() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Mean { axis: Some(0) },
            vec![shape![4, 6]]
        )
        .unwrap(),
        shape![6]
    );
}

// ==================== Amax / Amin ====================

#[test]
fn infer_amax() {
    assert_eq!(
        infer(NodeTypeDescriptor::Amax { axis: 1 }, vec![shape![3, 4, 5]]).unwrap(),
        shape![3, 5]
    );
}

#[test]
fn infer_amin() {
    assert_eq!(
        infer(NodeTypeDescriptor::Amin { axis: 0 }, vec![shape![3, 4]]).unwrap(),
        shape![4]
    );
}

// ==================== Reshape ====================

#[test]
fn infer_reshape_valid() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Reshape {
                target_shape: vec![6, 4]
            },
            vec![shape![2, 3, 4]]
        )
        .unwrap(),
        shape![6, 4]
    );
}

#[test]
fn infer_reshape_size_mismatch_returns_err() {
    assert!(
        infer(
            NodeTypeDescriptor::Reshape {
                target_shape: vec![10]
            },
            vec![shape![2, 3, 4]]
        )
        .is_err()
    );
}

// ==================== Flatten ====================

#[test]
fn infer_flatten_keep_first() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Flatten {
                keep_first_dim: true
            },
            vec![shape![2, 3, 4]]
        )
        .unwrap(),
        shape![2, 12]
    );
}

#[test]
fn infer_flatten_all() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Flatten {
                keep_first_dim: false
            },
            vec![shape![2, 3, 4]]
        )
        .unwrap(),
        shape![24]
    );
}

// ==================== Conv2d ====================

#[test]
fn infer_conv2d_same_padding() {
    // 输入 [1,1,4,4]，核 [8,1,3,3]，stride=1，padding=1 → 输出 [1,8,4,4]
    assert_eq!(
        infer(
            NodeTypeDescriptor::Conv2d {
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
            },
            vec![shape![1, 1, 4, 4], shape![8, 1, 3, 3]]
        )
        .unwrap(),
        shape![1, 8, 4, 4]
    );
}

#[test]
fn infer_conv2d_strided() {
    // 输入 [1,3,6,6]，核 [16,3,3,3]，stride=2，padding=0 → H=(6-3)/2+1=2
    assert_eq!(
        infer(
            NodeTypeDescriptor::Conv2d {
                stride: (2, 2),
                padding: (0, 0),
                dilation: (1, 1),
            },
            vec![shape![1, 3, 6, 6], shape![16, 3, 3, 3]]
        )
        .unwrap(),
        shape![1, 16, 2, 2]
    );
}

#[test]
fn infer_conv2d_dilated() {
    // 输入 [1,1,7,7]，核 [4,1,3,3]，dilation=2 → eff_k=5 → H=(7+0-5)/1+1=3
    assert_eq!(
        infer(
            NodeTypeDescriptor::Conv2d {
                stride: (1, 1),
                padding: (0, 0),
                dilation: (2, 2),
            },
            vec![shape![1, 1, 7, 7], shape![4, 1, 3, 3]]
        )
        .unwrap(),
        shape![1, 4, 3, 3]
    );
}

// ==================== MaxPool2d / AvgPool2d ====================

#[test]
fn infer_maxpool2d() {
    // 输入 [1,8,4,4]，kernel=2，stride=2 → [1,8,2,2]
    assert_eq!(
        infer(
            NodeTypeDescriptor::MaxPool2d {
                kernel_size: (2, 2),
                stride: (2, 2),
                padding: (0, 0),
                ceil_mode: false,
            },
            vec![shape![1, 8, 4, 4]]
        )
        .unwrap(),
        shape![1, 8, 2, 2]
    );
}

#[test]
fn infer_avgpool2d() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::AvgPool2d {
                kernel_size: (3, 3),
                stride: (1, 1)
            },
            vec![shape![2, 4, 5, 5]]
        )
        .unwrap(),
        shape![2, 4, 3, 3]
    );
}

// ==================== Select ====================

#[test]
fn infer_select_removes_dim() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Select { axis: 1, index: 2 },
            vec![shape![3, 5, 7]]
        )
        .unwrap(),
        shape![3, 7]
    );
}

// ==================== Gather ====================

#[test]
fn infer_gather() {
    // data [4,6], indices [2,3], dim=1 → [4,2,3]
    assert_eq!(
        infer(
            NodeTypeDescriptor::Gather { dim: 1 },
            vec![shape![4, 6], shape![2, 3]]
        )
        .unwrap(),
        shape![4, 2, 3]
    );
}

// ==================== Stack ====================

#[test]
fn infer_stack_new_dim() {
    // 3 个 [2,3] 在 axis=0 stack → [3,2,3]
    assert_eq!(
        infer(
            NodeTypeDescriptor::Stack { axis: 0 },
            vec![shape![2, 3], shape![2, 3], shape![2, 3]]
        )
        .unwrap(),
        shape![3, 2, 3]
    );
}

#[test]
fn infer_stack_mid_axis() {
    // 2 个 [3,4] 在 axis=1 stack → [3,2,4]
    assert_eq!(
        infer(
            NodeTypeDescriptor::Stack { axis: 1 },
            vec![shape![3, 4], shape![3, 4]]
        )
        .unwrap(),
        shape![3, 2, 4]
    );
}

// ==================== Concat ====================

#[test]
fn infer_concat_axis0() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Concat { axis: 0 },
            vec![shape![2, 4], shape![3, 4]]
        )
        .unwrap(),
        shape![5, 4]
    );
}

#[test]
fn infer_concat_axis1() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Concat { axis: 1 },
            vec![shape![2, 3], shape![2, 5]]
        )
        .unwrap(),
        shape![2, 8]
    );
}

#[test]
fn infer_concat_three_inputs() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Concat { axis: 1 },
            vec![shape![2, 3], shape![2, 4], shape![2, 5]]
        )
        .unwrap(),
        shape![2, 12]
    );
}

// ==================== 损失函数 ====================

#[test]
fn infer_loss_functions_scalar() {
    let input = shape![4, 10];
    for nt in [
        NodeTypeDescriptor::BCE {
            reduction: Reduction::Mean,
        },
        NodeTypeDescriptor::MAE {
            reduction: Reduction::Sum,
        },
        NodeTypeDescriptor::MSE {
            reduction: Reduction::Mean,
        },
        NodeTypeDescriptor::Huber {
            delta: 1.0,
            reduction: Reduction::Mean,
        },
        NodeTypeDescriptor::SoftmaxCrossEntropy,
    ] {
        assert_eq!(
            infer(nt.clone(), vec![input.clone(), input.clone()]).unwrap(),
            shape![1],
            "损失节点应输出标量：{nt:?}"
        );
    }
}

// ==================== Narrow ====================

#[test]
fn infer_narrow() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Narrow {
                axis: 1,
                start: 2,
                length: 3
            },
            vec![shape![4, 10, 6]]
        )
        .unwrap(),
        shape![4, 3, 6]
    );
}

// ==================== Permute ====================

#[test]
fn infer_permute() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Permute {
                dims: vec![2, 0, 1]
            },
            vec![shape![3, 4, 5]]
        )
        .unwrap(),
        shape![5, 3, 4]
    );
}

// ==================== Pad ====================

#[test]
fn infer_pad() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Pad {
                paddings: vec![(1, 1), (2, 2)],
                pad_value: 0.0
            },
            vec![shape![3, 4]]
        )
        .unwrap(),
        shape![5, 8]
    );
}

// ==================== Repeat ====================

#[test]
fn infer_repeat() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::Repeat {
                repeats: vec![2, 3]
            },
            vec![shape![4, 5]]
        )
        .unwrap(),
        shape![8, 15]
    );
}

// ==================== TopK ====================

#[test]
fn infer_topk() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::TopK {
                k: 3,
                axis: 1,
                sorted: true
            },
            vec![shape![2, 10, 4]]
        )
        .unwrap(),
        shape![2, 3, 4]
    );
}

// ==================== SortNode ====================

#[test]
fn infer_sort_preserves_shape() {
    assert_eq!(
        infer(
            NodeTypeDescriptor::SortNode {
                axis: 0,
                descending: false
            },
            vec![shape![5, 3]]
        )
        .unwrap(),
        shape![5, 3]
    );
}

// ==================== GenomeAnalysis 测试 ====================

/// 构建一个简单线性链：Input(0) → W(1) → MatMul(2) → b(3) → Add(4)
/// 模拟一个最小 Linear 层
fn minimal_linear_nodes() -> Vec<NodeGene> {
    vec![
        NodeGene::new(
            1,
            NodeTypeDescriptor::Parameter,
            shape![4, 8],
            vec![],
            Some(1),
        ),
        NodeGene::new(
            2,
            NodeTypeDescriptor::MatMul,
            shape![1, 8],
            vec![0, 1], // input(0), W(1)
            Some(1),
        ),
        NodeGene::new(
            3,
            NodeTypeDescriptor::Parameter,
            shape![1, 8],
            vec![],
            Some(1),
        ),
        NodeGene::new(
            4,
            NodeTypeDescriptor::Add,
            shape![1, 8],
            vec![2, 3], // matmul(2), b(3)
            Some(1),
        ),
    ]
}

#[test]
fn genome_analysis_linear_chain_valid() {
    let nodes = minimal_linear_nodes();
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);

    assert!(
        analysis.is_valid,
        "线性链应通过合法性校验，错误：{:?}",
        analysis.errors
    );
    assert_eq!(analysis.param_node_count, 2, "应有 W 和 b 两个参数节点");
    assert_eq!(analysis.param_count, 4 * 8 + 1 * 8, "参数量应为 40");
    assert_eq!(analysis.topo_order.len(), 4);

    // 验证 Add 节点的推导形状
    assert_eq!(analysis.shape_of(4).unwrap(), &shape![1, 8]);
}

#[test]
fn genome_analysis_summary_format() {
    let nodes = minimal_linear_nodes();
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);
    let s = analysis.summary();
    assert!(s.starts_with("nodes="), "summary 应以 nodes= 开头");
    assert!(s.contains("active="), "summary 应包含 active=");
    assert!(s.contains("params="), "summary 应包含 params=");
}

#[test]
fn genome_analysis_snapshot_remains_immutable_after_genome_mutation() {
    use crate::nn::evolution::gene::NetworkGenome;

    let mut genome = NetworkGenome::minimal(4, 3);
    genome.migrate_to_node_level().expect("迁移到 NodeLevel 不应失败");

    let before = genome.analyze();
    assert!(before.is_valid, "初始分析应合法：{:?}", before.errors);

    let disabled_id = genome
        .nodes()
        .iter()
        .find(|n| !n.is_parameter())
        .map(|n| n.innovation_number)
        .expect("最小 NodeLevel genome 应存在非参数节点");
    let old_enabled = before.enabled_node_count;
    let old_param_count = before.param_count;

    genome
        .nodes_mut()
        .iter_mut()
        .find(|n| n.innovation_number == disabled_id)
        .unwrap()
        .enabled = false;

    let after = genome.analyze();
    assert!(after.enabled_node_count < old_enabled, "重新分析后启用节点数应减少");

    assert_eq!(before.enabled_node_count, old_enabled, "旧快照不应被污染");
    assert_eq!(before.param_count, old_param_count, "旧快照参数量不应被污染");
    assert!(
        before.shape_of(disabled_id).is_some(),
        "旧快照仍应保留被禁用前节点的形状"
    );
}

#[test]
fn genome_analysis_empty_nodes_returns_error() {
    let analysis = GenomeAnalysis::compute(&[], 0, shape![1, 4], ShapeDomain::Flat);
    assert!(!analysis.is_valid);
    assert!(analysis.errors.contains(&AnalysisError::Empty));
}

#[test]
fn genome_analysis_missing_parent_detected() {
    // 节点 2 引用了不存在的父节点 99
    let nodes = vec![
        NodeGene::new(1, NodeTypeDescriptor::Parameter, shape![4, 8], vec![], None),
        NodeGene::new(
            2,
            NodeTypeDescriptor::MatMul,
            shape![1, 8],
            vec![0, 99],
            None,
        ),
    ];
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);
    assert!(!analysis.is_valid);
    assert!(
        analysis
            .errors
            .iter()
            .any(|e| matches!(e, AnalysisError::MissingParent { parent_id: 99, .. }))
    );
}

#[test]
fn genome_analysis_shape_mismatch_detected() {
    // 两个 Add 的输入形状不一致
    let nodes = vec![
        NodeGene::new(1, NodeTypeDescriptor::Parameter, shape![1, 4], vec![], None),
        NodeGene::new(2, NodeTypeDescriptor::Parameter, shape![1, 8], vec![], None),
        NodeGene::new(3, NodeTypeDescriptor::Add, shape![1, 4], vec![1, 2], None), // [1,4] + [1,8] 不兼容
    ];
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);
    assert!(!analysis.is_valid);
    assert!(
        analysis
            .errors
            .iter()
            .any(|e| matches!(e, AnalysisError::IncompatibleShapes { .. }))
    );
}

#[test]
fn genome_analysis_disabled_nodes_excluded() {
    let mut nodes = minimal_linear_nodes();
    // 禁用节点 3（参数 b）
    nodes[2].enabled = false;
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);
    // 启用节点只有 W(1), MatMul(2), Add(4) 共 3 个
    assert_eq!(analysis.enabled_node_count, 3);
    assert_eq!(analysis.param_node_count, 1); // 只有 W
    assert_eq!(analysis.param_count, 4 * 8); // 只有 W 的参数
}

#[test]
fn genome_analysis_diamond_topology() {
    // Input(0) → A(1) → B(2) → D(4)
    //                  → C(3) → D(4)
    let nodes = vec![
        NodeGene::new(1, NodeTypeDescriptor::ReLU, shape![1, 4], vec![0], None),
        NodeGene::new(2, NodeTypeDescriptor::ReLU, shape![1, 4], vec![1], None),
        NodeGene::new(3, NodeTypeDescriptor::ReLU, shape![1, 4], vec![1], None),
        NodeGene::new(4, NodeTypeDescriptor::Add, shape![1, 4], vec![2, 3], None),
    ];
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);
    assert!(analysis.is_valid);
    assert_eq!(analysis.topo_order.len(), 4);
    assert_eq!(analysis.shape_of(4).unwrap(), &shape![1, 4]);
}

#[test]
fn genome_analysis_domain_spatial_propagation() {
    // 模拟 Conv2d 节点的域推导：输出应为 Spatial
    let w = NodeGene::new(
        1,
        NodeTypeDescriptor::Parameter,
        shape![8, 1, 3, 3],
        vec![],
        None,
    );
    let conv = NodeGene::new(
        2,
        NodeTypeDescriptor::Conv2d {
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
        },
        shape![1, 8, 4, 4],
        vec![0, 1], // input(0), kernel(1)
        None,
    );
    let analysis = GenomeAnalysis::compute(&[w, conv], 0, shape![1, 1, 4, 4], ShapeDomain::Spatial);
    assert!(analysis.is_valid, "{:?}", analysis.errors);
    assert_eq!(
        analysis.domain_of(2),
        Some(ShapeDomain::Spatial),
        "Conv2d 输出域应为 Spatial"
    );
}

// ==================== infer_output_shape 错误路径 ====================

#[test]
fn infer_unary_insufficient_parents_err() {
    // 一元操作需要 1 个父节点，不传入 → Err
    assert!(infer(NodeTypeDescriptor::ReLU, vec![]).is_err());
    assert!(infer(NodeTypeDescriptor::Sigmoid, vec![]).is_err());
    assert!(infer(NodeTypeDescriptor::Tanh, vec![]).is_err());
    assert!(infer(NodeTypeDescriptor::Identity, vec![]).is_err());
    assert!(infer(NodeTypeDescriptor::Detach, vec![]).is_err());
    assert!(infer(NodeTypeDescriptor::BatchNormOp { eps: 1e-5, momentum: 0.1, num_features: 4 }, vec![]).is_err());
    assert!(infer(NodeTypeDescriptor::ZerosLike, vec![]).is_err());
}

#[test]
fn infer_binary_insufficient_parents_err() {
    // 二元操作需要 2 个父节点，只传入 1 个 → Err
    for nt in [
        NodeTypeDescriptor::Add,
        NodeTypeDescriptor::Subtract,
        NodeTypeDescriptor::Multiply,
        NodeTypeDescriptor::Maximum,
        NodeTypeDescriptor::Minimum,
    ] {
        assert!(infer(nt, vec![shape![2, 3]]).is_err(), "少于 2 个父节点应返回 Err");
    }
}

#[test]
fn infer_matmul_1d_operand_err() {
    // MatMul 要求两个操作数都至少 2D
    assert!(infer(NodeTypeDescriptor::MatMul, vec![shape![3], shape![3, 4]]).is_err());
    assert!(infer(NodeTypeDescriptor::MatMul, vec![shape![2, 3], shape![4]]).is_err());
}

#[test]
fn infer_sum_axis_out_of_bounds_err() {
    assert!(infer(NodeTypeDescriptor::Sum { axis: Some(3) }, vec![shape![2, 3]]).is_err());
    assert!(infer(NodeTypeDescriptor::Mean { axis: Some(2) }, vec![shape![2, 3]]).is_err());
}

#[test]
fn infer_amax_amin_axis_oob_err() {
    assert!(infer(NodeTypeDescriptor::Amax { axis: 3 }, vec![shape![2, 3]]).is_err());
    assert!(infer(NodeTypeDescriptor::Amin { axis: 3 }, vec![shape![2, 3]]).is_err());
}

#[test]
fn infer_conv2d_insufficient_dims_err() {
    // 输入少于 4D → Err
    assert!(infer(
        NodeTypeDescriptor::Conv2d { stride: (1, 1), padding: (0, 0), dilation: (1, 1) },
        vec![shape![3, 4], shape![8, 3, 3, 3]]
    ).is_err());
    // 权重少于 4D → Err
    assert!(infer(
        NodeTypeDescriptor::Conv2d { stride: (1, 1), padding: (0, 0), dilation: (1, 1) },
        vec![shape![1, 3, 4, 4], shape![8, 3, 3]]
    ).is_err());
}

#[test]
fn infer_pool2d_insufficient_dims_err() {
    assert!(infer(
        NodeTypeDescriptor::MaxPool2d { kernel_size: (2, 2), stride: (2, 2), padding: (0, 0), ceil_mode: false },
        vec![shape![3, 4]]
    ).is_err());
    assert!(infer(
        NodeTypeDescriptor::AvgPool2d { kernel_size: (2, 2), stride: (2, 2) },
        vec![shape![1, 4, 4]]
    ).is_err());
}

#[test]
fn infer_pool2d_kernel_exceeds_spatial_soft_fail() {
    // kernel=5 > H/W=3 → h_out/w_out 软降级到 1（不返回错误）
    let r = infer(
        NodeTypeDescriptor::MaxPool2d { kernel_size: (5, 5), stride: (1, 1), padding: (0, 0), ceil_mode: false },
        vec![shape![1, 4, 3, 3]]
    ).unwrap();
    assert_eq!(r, shape![1, 4, 1, 1], "kernel 超出应软降级到 1");
}

#[test]
fn infer_select_axis_oob_err() {
    assert!(infer(
        NodeTypeDescriptor::Select { axis: 5, index: 0 },
        vec![shape![3, 4, 5]]
    ).is_err());
}

#[test]
fn infer_gather_dim_oob_err() {
    assert!(infer(
        NodeTypeDescriptor::Gather { dim: 5 },
        vec![shape![3, 4], shape![2]]
    ).is_err());
}

#[test]
fn infer_stack_inconsistent_shapes_err() {
    assert!(infer(
        NodeTypeDescriptor::Stack { axis: 0 },
        vec![shape![2, 3], shape![2, 4]]
    ).is_err());
}

#[test]
fn infer_stack_axis_oob_err() {
    // axis 允许最大为 ndim（在末尾插入），超出则错误
    assert!(infer(
        NodeTypeDescriptor::Stack { axis: 5 },
        vec![shape![2, 3]]
    ).is_err());
}

#[test]
fn infer_concat_dim_count_mismatch_err() {
    // 两输入维度数不一致
    assert!(infer(
        NodeTypeDescriptor::Concat { axis: 0 },
        vec![shape![2, 3], shape![2, 3, 4]]
    ).is_err());
}

#[test]
fn infer_concat_non_concat_dim_mismatch_err() {
    // axis=0，非拼接维度 1 不一致：[2,3] vs [2,4]
    assert!(infer(
        NodeTypeDescriptor::Concat { axis: 0 },
        vec![shape![2, 3], shape![2, 4]]
    ).is_err());
}

#[test]
fn infer_narrow_axis_oob_err() {
    assert!(infer(
        NodeTypeDescriptor::Narrow { axis: 3, start: 0, length: 2 },
        vec![shape![3, 4]]
    ).is_err());
}

#[test]
fn infer_permute_dims_mismatch_err() {
    // dims 长度与输入维度数不一致
    assert!(infer(
        NodeTypeDescriptor::Permute { dims: vec![0, 1] },
        vec![shape![3, 4, 5]]
    ).is_err());
}

#[test]
fn infer_pad_paddings_mismatch_err() {
    assert!(infer(
        NodeTypeDescriptor::Pad { paddings: vec![(1, 1)], pad_value: 0.0 },
        vec![shape![3, 4]]
    ).is_err());
}

#[test]
fn infer_repeat_repeats_mismatch_err() {
    assert!(infer(
        NodeTypeDescriptor::Repeat { repeats: vec![2] },
        vec![shape![3, 4]]
    ).is_err());
}

#[test]
fn infer_topk_axis_oob_err() {
    assert!(infer(
        NodeTypeDescriptor::TopK { k: 3, axis: 5, sorted: true },
        vec![shape![2, 4]]
    ).is_err());
}

// ==================== infer_domain 直接测试 ====================

#[test]
fn infer_domain_conv2d_to_spatial() {
    let d = infer_domain(
        &NodeTypeDescriptor::Conv2d { stride: (1, 1), padding: (0, 0), dilation: (1, 1) },
        &[ShapeDomain::Flat],
    );
    assert_eq!(d, ShapeDomain::Spatial, "Conv2d 应输出 Spatial域");
}

#[test]
fn infer_domain_pool2d_to_spatial() {
    assert_eq!(
        infer_domain(&NodeTypeDescriptor::MaxPool2d { kernel_size: (2, 2), stride: (2, 2), padding: (0, 0), ceil_mode: false }, &[ShapeDomain::Spatial]),
        ShapeDomain::Spatial
    );
    assert_eq!(
        infer_domain(&NodeTypeDescriptor::AvgPool2d { kernel_size: (2, 2), stride: (2, 2) }, &[ShapeDomain::Spatial]),
        ShapeDomain::Spatial
    );
}

#[test]
fn infer_domain_flatten_to_flat() {
    // Flatten 无论父节点域是什么，输出始终是 Flat
    let d = infer_domain(
        &NodeTypeDescriptor::Flatten { keep_first_dim: true },
        &[ShapeDomain::Spatial],
    );
    assert_eq!(d, ShapeDomain::Flat, "Flatten 应从 Spatial 切换到 Flat");
}

#[test]
fn infer_domain_default_passthrough_flat() {
    // 普通计算节点透传父节点的域
    let d = infer_domain(&NodeTypeDescriptor::ReLU, &[ShapeDomain::Flat]);
    assert_eq!(d, ShapeDomain::Flat);
}

#[test]
fn infer_domain_sequence_passthrough() {
    // 序列域透传：RNN 输出后的激活层保持 Sequence 域
    let d = infer_domain(&NodeTypeDescriptor::Tanh, &[ShapeDomain::Sequence]);
    assert_eq!(d, ShapeDomain::Sequence, "激活层应透传 Sequence 域");
}

#[test]
fn infer_domain_no_parent_default_flat() {
    // 没有父节点时默认返回 Flat
    let d = infer_domain(&NodeTypeDescriptor::ReLU, &[]);
    assert_eq!(d, ShapeDomain::Flat);
}

// ==================== GenomeAnalysis 导入补充 ====================

#[test]
fn genome_analysis_cycle_detected() {
    // 1 依赖5 2，2 依赖5 1 → 环
    let nodes = vec![
        NodeGene::new(1, NodeTypeDescriptor::ReLU, shape![1, 4], vec![2], None),
        NodeGene::new(2, NodeTypeDescriptor::ReLU, shape![1, 4], vec![1], None),
    ];
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);
    assert!(!analysis.is_valid);
    assert!(
        analysis.errors.contains(&AnalysisError::CycleDetected),
        "应检测到 CycleDetected，实际错误：{:?}",
        analysis.errors
    );
}

#[test]
fn genome_analysis_multiple_errors_accumulate() {
    // 两个节点分别引用不存在的父节点，应累积两条错误
    let nodes = vec![
        NodeGene::new(1, NodeTypeDescriptor::ReLU, shape![1, 4], vec![99], None),
        NodeGene::new(2, NodeTypeDescriptor::ReLU, shape![1, 4], vec![98], None),
    ];
    let analysis = GenomeAnalysis::compute(&nodes, 0, shape![1, 4], ShapeDomain::Flat);
    assert!(!analysis.is_valid);
    assert!(
        analysis.errors.len() >= 2,
        "应累积至少 2 条错误，实际：{:?}",
        analysis.errors
    );
}

// ==================== SkipEdge 聚合 ====================

/// 辅助：构建含一个 skip edge 的基因组
///
/// 结构：Input(2) → Linear(4)[inn=2] → ReLU[inn=3] → Linear(1)[inn=1]
/// skip edge: 从 INPUT(0) 到 Linear(1)[inn=1]，使用指定策略
fn genome_with_skip(strategy: AggregateStrategy) -> NetworkGenome {
    let mut g = NetworkGenome::minimal(2, 1);
    // 隐藏层
    let inn_h = g.next_innovation_number(); // 2
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn_h,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn_act = g.next_innovation_number(); // 3
    g.layers.insert(
        1,
        LayerGene {
            innovation_number: inn_act,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    // skip edge: INPUT(0) → 输出头(1)
    // Add/Mean/Max 需 in_dim 一致；INPUT dim=2, main path dim=4 ⇒ 不匹配
    // 所以 Add/Mean/Max 的 skip 需要 dim 匹配的拓扑
    // 这里仅用于 Concat (dim=1)，其余策略用专门的辅助函数
    let se_inn = g.next_innovation_number(); // 4
    g.skip_edges.push(SkipEdge {
        innovation_number: se_inn,
        from_innovation: INPUT_INNOVATION, // 0
        to_innovation: 1,                  // 输出头
        strategy,
        enabled: true,
    });
    g
}

/// 辅助：构建 Add/Mean/Max 兼容的 skip 基因组（维度对齐）
///
/// 结构：Input(4) → Linear(4)[inn=2] → ReLU[inn=3] → Linear(1)[inn=1]
/// skip edge: INPUT(0, dim=4) → ReLU 之后、输出头(1) 之前，main path dim=4
fn genome_with_same_dim_skip(strategy: AggregateStrategy) -> NetworkGenome {
    let mut g = NetworkGenome::minimal(4, 1);
    let inn_h = g.next_innovation_number(); // 2
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn_h,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn_act = g.next_innovation_number(); // 3
    g.layers.insert(
        1,
        LayerGene {
            innovation_number: inn_act,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    let se_inn = g.next_innovation_number(); // 4
    g.skip_edges.push(SkipEdge {
        innovation_number: se_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: 1, // 输出头
        strategy,
        enabled: true,
    });
    g
}

#[test]
fn test_build_skip_edge_add() {
    let genome = genome_with_same_dim_skip(AggregateStrategy::Add);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

#[test]
fn test_build_skip_edge_concat() {
    // Input(2) → Linear(4) → ReLU → [concat with Input(2)] → Linear(1)
    // 输出头接收 main_dim=4 + skip_dim=2 = 6
    let genome = genome_with_skip(AggregateStrategy::Concat { dim: 1 });
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0], &[1, 2]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);

    // 输出头的 W 形状 = [6, 1]（因为 Concat 后 in_dim=6）
    let w = build.layer_params[&1][0].value().unwrap().unwrap();
    assert_eq!(w.shape(), &[6, 1]);
}

#[test]
fn test_build_skip_edge_mean() {
    let genome = genome_with_same_dim_skip(AggregateStrategy::Mean);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

#[test]
fn test_build_skip_edge_max() {
    let genome = genome_with_same_dim_skip(AggregateStrategy::Max);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

#[test]
fn test_build_skip_edge_multi_path() {
    // Input(4) → Linear(4)[inn=2] → ReLU[inn=3] → Linear(4)[inn=4] → Tanh[inn=5] → Linear(1)[inn=1]
    // skip edge 1: INPUT(0) → inn=1 (Add)
    // skip edge 2: inn=2 → inn=1 (Add) — 隐藏层输出也跳到输出头
    // main path 到输出头时 dim=4, INPUT dim=4, inn=2 out=4 → 全部 dim=4 → Add 兼容
    let mut g = NetworkGenome::minimal(4, 1);
    let inn_h1 = g.next_innovation_number(); // 2
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn_h1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn_act1 = g.next_innovation_number(); // 3
    g.layers.insert(
        1,
        LayerGene {
            innovation_number: inn_act1,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    let inn_h2 = g.next_innovation_number(); // 4
    g.layers.insert(
        2,
        LayerGene {
            innovation_number: inn_h2,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn_act2 = g.next_innovation_number(); // 5
    g.layers.insert(
        3,
        LayerGene {
            innovation_number: inn_act2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );

    // 两条 skip edges 都指向输出头
    let se1 = g.next_innovation_number(); // 6
    g.skip_edges.push(SkipEdge {
        innovation_number: se1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: 1,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });
    let se2 = g.next_innovation_number(); // 7
    g.skip_edges.push(SkipEdge {
        innovation_number: se2,
        from_innovation: inn_h1, // Linear(4) 输出
        to_innovation: 1,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let mut rng = StdRng::seed_from_u64(42);
    let build = g.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

#[test]
fn test_build_skip_edge_backward() {
    // 验证 skip edge 路径支持反向传播
    let genome = genome_with_same_dim_skip(AggregateStrategy::Add);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    // 反向传播
    build.graph.backward(&build.output).unwrap();

    // 所有参数都应有梯度
    for param in build.all_parameters() {
        let grad = param.grad().unwrap();
        assert!(grad.is_some(), "参数应该有梯度");
    }
}

#[test]
fn test_build_skip_edge_disabled() {
    // disabled skip edge 不参与构建
    let mut genome = genome_with_same_dim_skip(AggregateStrategy::Add);
    genome.skip_edges[0].enabled = false;

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 输出头的 W 形状 = [4, 1]（无聚合，正常的 main path）
    let w = build.layer_params[&1][0].value().unwrap().unwrap();
    assert_eq!(w.shape(), &[4, 1]);

    let input_data = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

use crate::nn::evolution::gene::*;
use crate::tensor::Tensor;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ==================== 基本构建 ====================

#[test]
fn test_build_minimal() {
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 设置输入数据并前向传播
    let input_data = Tensor::new(&[1.0, 2.0], &[1, 2]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 1]);
}

#[test]
fn test_build_minimal_layer_params_keys() {
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 最小基因组只有一个 Linear 输出头（innovation=1）
    assert_eq!(build.layer_params.len(), 1);
    assert!(build.layer_params.contains_key(&1));

    // Linear 有 [W, b] 两个参数
    assert_eq!(build.layer_params[&1].len(), 2);
}

#[test]
fn test_build_all_parameters_count() {
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 1 个 Linear → [W, b] = 2 个 Var
    assert_eq!(build.all_parameters().len(), 2);
}

#[test]
fn test_build_with_hidden_layers() {
    let mut genome = NetworkGenome::minimal(2, 1);

    // 手动插入隐藏层：Linear(4) → ReLU
    let inn_hidden = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: inn_hidden,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn_relu = genome.next_innovation_number();
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: inn_relu,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 2 个 Linear 层有参数（隐藏层 + 输出头），Activation 无参数
    assert_eq!(build.layer_params.len(), 2);
    assert!(build.layer_params.contains_key(&inn_hidden));
    assert!(build.layer_params.contains_key(&1)); // 输出头

    // 隐藏层 Linear(in=2, out=4) → [W, b]，输出头 Linear(in=4, out=1) → [W, b]
    assert_eq!(build.all_parameters().len(), 4);

    // 前向传播：输出形状 = [1, 1]
    let input_data = Tensor::new(&[1.0, 2.0], &[1, 2]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 1]);
}

#[test]
fn test_build_output_shape_multi_dim() {
    // output_dim > 1
    let genome = NetworkGenome::minimal(3, 5);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 5]);
}

// ==================== disabled 层处理 ====================

#[test]
fn test_build_skips_disabled_layers() {
    let mut genome = NetworkGenome::minimal(2, 1);

    // 插入一个 disabled 的隐藏层
    let inn = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: false,
        },
    );

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // disabled 层不出现在 layer_params 中
    assert!(!build.layer_params.contains_key(&inn));
    assert_eq!(build.layer_params.len(), 1); // 只有输出头

    // 前向传播仍然工作（Input(2) → Linear(1)，跳过 disabled 层）
    let input_data = Tensor::new(&[1.0, 2.0], &[1, 2]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 1]);
}

// ==================== 变异后构建 ====================

#[test]
fn test_build_after_mutation() {
    use crate::nn::evolution::mutation::*;

    let mut genome = NetworkGenome::minimal(2, 1);
    let constraints = SizeConstraints::default();
    let mut rng = StdRng::seed_from_u64(42);

    // 执行若干变异
    let registry = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    for _ in 0..5 {
        let _ = registry.apply_random(&mut genome, &constraints, &mut rng);
    }

    // 变异后仍然可以构建
    let build = genome.build(&mut rng).unwrap();

    let input_data = Tensor::new(&[1.0, 2.0], &[1, 2]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    // output_dim 不变
    assert_eq!(output_val.shape()[1], 1);
}

// ==================== 权重捕获与恢复 ====================

#[test]
fn test_capture_weights() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    assert!(!genome.has_weight_snapshots());

    genome.capture_weights(&build).unwrap();

    assert!(genome.has_weight_snapshots());

    // 快照包含输出头的参数
    let snapshots = genome.weight_snapshots();
    assert!(snapshots.contains_key(&1));
    assert_eq!(snapshots[&1].len(), 2); // [W, b]

    // W 形状 = [2, 1]，b 形状 = [1, 1]
    assert_eq!(snapshots[&1][0].shape(), &[2, 1]);
    assert_eq!(snapshots[&1][1].shape(), &[1, 1]);
}

#[test]
fn test_restore_weights_roundtrip() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);

    // 第一次构建并捕获权重
    let build1 = genome.build(&mut rng).unwrap();
    let original_w = build1.layer_params[&1][0].value().unwrap().unwrap();
    let original_b = build1.layer_params[&1][1].value().unwrap().unwrap();
    genome.capture_weights(&build1).unwrap();

    // 第二次构建（参数被重新初始化，与第一次不同）
    let build2 = genome.build(&mut rng).unwrap();
    let reinit_w = build2.layer_params[&1][0].value().unwrap().unwrap();
    // 新初始化的权重极大概率与原始不同
    assert_ne!(original_w, reinit_w);

    // 恢复权重
    let report = genome.restore_weights(&build2).unwrap();
    assert_eq!(report.inherited, 2); // W + b
    assert_eq!(report.reinitialized, 0);

    // 恢复后的权重与原始一致
    let restored_w = build2.layer_params[&1][0].value().unwrap().unwrap();
    let restored_b = build2.layer_params[&1][1].value().unwrap().unwrap();
    assert_eq!(original_w, restored_w);
    assert_eq!(original_b, restored_b);
}

#[test]
fn test_restore_weights_shape_mismatch() {
    let mut genome = NetworkGenome::minimal(2, 1);

    // 插入隐藏层 Linear(4)
    let inn_hidden = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: inn_hidden,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let mut rng = StdRng::seed_from_u64(42);

    // 构建并捕获（隐藏层 in=2, out=4）
    let build1 = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build1).unwrap();

    // 增大隐藏层维度（4 → 8），改变参数形状
    genome.layers[0].layer_config = LayerConfig::Linear { out_features: 8 };

    // 重新构建（隐藏层 in=2, out=8，形状变了）
    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    // 隐藏层的 W、b 形状不匹配 → reinitialized
    // 输出头的 W 形状也变了（in_dim 从 4 变成 8）→ reinitialized
    // 输出头的 b 形状不变 [1, 1] → inherited
    assert!(report.reinitialized > 0);
    assert_eq!(report.inherited + report.reinitialized, 4); // 总共 4 个参数张量
}

#[test]
fn test_restore_weights_no_snapshot() {
    let genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 未捕获过权重，所有参数保留初始化值
    let report = genome.restore_weights(&build).unwrap();
    assert_eq!(report.inherited, 0);
    assert_eq!(report.reinitialized, 2); // W + b
}

#[test]
fn test_inherit_report_accuracy() {
    let mut genome = NetworkGenome::minimal(2, 1);

    // 插入两个隐藏层
    let inn1 = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: inn1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn2 = genome.next_innovation_number();
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: inn2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build).unwrap();

    // 不改结构，rebuild 后 restore
    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    // 所有参数形状不变，全部继承
    // 2 个 Linear × [W, b] = 4 个参数张量
    assert_eq!(report.inherited, 4);
    assert_eq!(report.reinitialized, 0);
    assert_eq!(
        report.inherited + report.reinitialized,
        build2.all_parameters().len()
    );
}

// ==================== Clone 独立性 ====================

#[test]
fn test_clone_weight_snapshots_independent() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);

    let build = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build).unwrap();

    // Clone genome
    let cloned = genome.clone();

    // 修改原件的快照（清空）
    genome.set_weight_snapshots(std::collections::HashMap::new());
    assert!(!genome.has_weight_snapshots());

    // 克隆体不受影响
    assert!(cloned.has_weight_snapshots());
    let snap = cloned.weight_snapshots();
    assert!(snap.contains_key(&1));
}

#[test]
fn test_clone_restore_independent() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);

    let build1 = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build1).unwrap();
    let original_w = genome.weight_snapshots()[&1][0].clone();

    let cloned = genome.clone();

    // 克隆体可以独立 build + restore
    let build2 = cloned.build(&mut rng).unwrap();
    let report = cloned.restore_weights(&build2).unwrap();
    assert_eq!(report.inherited, 2);

    let restored_w = build2.layer_params[&1][0].value().unwrap().unwrap();
    assert_eq!(original_w, restored_w);
}

// ==================== 错误处理 ====================

#[test]
fn test_build_empty_genome_error() {
    let mut genome = NetworkGenome::minimal(2, 1);
    // 禁用所有层
    genome.layers[0].enabled = false;

    let mut rng = StdRng::seed_from_u64(42);
    let result = genome.build(&mut rng);
    assert!(result.is_err());
}

// ==================== seed 可复现性 ====================

#[test]
fn test_build_deterministic_with_same_seed() {
    let genome = NetworkGenome::minimal(2, 1);

    let mut rng1 = StdRng::seed_from_u64(42);
    let build1 = genome.build(&mut rng1).unwrap();
    let w1 = build1.layer_params[&1][0].value().unwrap().unwrap();

    let mut rng2 = StdRng::seed_from_u64(42);
    let build2 = genome.build(&mut rng2).unwrap();
    let w2 = build2.layer_params[&1][0].value().unwrap().unwrap();

    assert_eq!(w1, w2);
}

#[test]
fn test_build_different_with_different_seed() {
    let genome = NetworkGenome::minimal(2, 1);

    let mut rng1 = StdRng::seed_from_u64(42);
    let build1 = genome.build(&mut rng1).unwrap();
    let w1 = build1.layer_params[&1][0].value().unwrap().unwrap();

    let mut rng2 = StdRng::seed_from_u64(99);
    let build2 = genome.build(&mut rng2).unwrap();
    let w2 = build2.layer_params[&1][0].value().unwrap().unwrap();

    assert_ne!(w1, w2);
}

// ==================== 各种激活函数 ====================

#[test]
fn test_build_with_all_activation_types() {
    let activations = vec![
        ActivationType::ReLU,
        ActivationType::LeakyReLU { alpha: 0.01 },
        ActivationType::Tanh,
        ActivationType::Sigmoid,
        ActivationType::GELU,
        ActivationType::SiLU,
        ActivationType::Softplus,
        ActivationType::ReLU6,
        ActivationType::ELU { alpha: 1.0 },
        ActivationType::SELU,
        ActivationType::Mish,
        ActivationType::HardSwish,
        ActivationType::HardSigmoid,
    ];

    for act in activations {
        let mut genome = NetworkGenome::minimal(2, 1);
        let inn_linear = genome.next_innovation_number();
        genome.layers.insert(
            0,
            LayerGene {
                innovation_number: inn_linear,
                layer_config: LayerConfig::Linear { out_features: 4 },
                enabled: true,
            },
        );
        let inn_act = genome.next_innovation_number();
        genome.layers.insert(
            1,
            LayerGene {
                innovation_number: inn_act,
                layer_config: LayerConfig::Activation {
                    activation_type: act,
                },
                enabled: true,
            },
        );

        let mut rng = StdRng::seed_from_u64(42);
        let build = genome.build(&mut rng).unwrap();

        let input_data = Tensor::new(&[0.5, -0.5], &[1, 2]);
        build.input.set_value(&input_data).unwrap();
        build.graph.forward(&build.output).unwrap();

        let output_val = build.output.value().unwrap().unwrap();
        assert_eq!(
            output_val.shape(),
            &[1, 1],
            "激活函数 {:?} 构建后输出形状错误",
            act
        );
    }
}

// ==================== 多层复杂结构 ====================

#[test]
fn test_build_deep_network() {
    let mut genome = NetworkGenome::minimal(3, 2);

    // 插入多层：Linear(8) → ReLU → Linear(4) → Tanh
    let layers_to_insert = vec![
        LayerConfig::Linear { out_features: 8 },
        LayerConfig::Activation {
            activation_type: ActivationType::ReLU,
        },
        LayerConfig::Linear { out_features: 4 },
        LayerConfig::Activation {
            activation_type: ActivationType::Tanh,
        },
    ];

    for (i, config) in layers_to_insert.into_iter().enumerate() {
        let inn = genome.next_innovation_number();
        genome.layers.insert(
            i,
            LayerGene {
                innovation_number: inn,
                layer_config: config,
                enabled: true,
            },
        );
    }

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 3 个 Linear 层有参数
    assert_eq!(build.layer_params.len(), 3);
    // 3 × [W, b] = 6 个参数 Var
    assert_eq!(build.all_parameters().len(), 6);

    let input_data = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
    build.input.set_value(&input_data).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[1, 2]);
}

// ==================== 快照在旧 Graph 释放后的持久性 ====================

#[test]
fn test_snapshot_survives_graph_drop() {
    let mut genome = NetworkGenome::minimal(2, 1);
    let mut rng = StdRng::seed_from_u64(42);

    let build1 = genome.build(&mut rng).unwrap();
    let original_w = build1.layer_params[&1][0].value().unwrap().unwrap();
    let original_b = build1.layer_params[&1][1].value().unwrap().unwrap();
    genome.capture_weights(&build1).unwrap();

    drop(build1);

    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();
    assert_eq!(report.inherited, 2);
    assert_eq!(report.reinitialized, 0);

    let restored_w = build2.layer_params[&1][0].value().unwrap().unwrap();
    let restored_b = build2.layer_params[&1][1].value().unwrap().unwrap();
    assert_eq!(original_w, restored_w);
    assert_eq!(original_b, restored_b);
}

// ==================== batch_size > 1 ====================

#[test]
fn test_build_forward_batch() {
    let mut genome = NetworkGenome::minimal(2, 1);

    let inn = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 4 个样本的 batch（模拟 XOR 全量输入）
    let batch_input = Tensor::new(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2]);
    build.input.set_value(&batch_input).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[4, 1]);
}

// ==================== 序列网络构建 ====================

#[test]
fn test_build_rnn_layer() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.seq_len = Some(5);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 输入 [1, 5, 3]
    let data: Vec<f32> = (0..15).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 5, 3]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 2]); // [batch, output_dim]

    // Rnn 层有 3 个参数 (w_ih, w_hh, b_h)
    let rnn_inn = genome.layers[0].innovation_number;
    assert_eq!(build.layer_params[&rnn_inn].len(), 3);
}

#[test]
fn test_build_lstm_layer() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.layers[0].layer_config = LayerConfig::Lstm { hidden_size: 2 };
    genome.seq_len = Some(4);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 4, 3]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    // Lstm 有 12 个参数
    let lstm_inn = genome.layers[0].innovation_number;
    assert_eq!(build.layer_params[&lstm_inn].len(), 12);
}

#[test]
fn test_build_gru_layer() {
    let mut genome = NetworkGenome::minimal_sequential(3, 2);
    genome.layers[0].layer_config = LayerConfig::Gru { hidden_size: 2 };
    genome.seq_len = Some(4);
    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 4, 3]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 2]);

    // Gru 有 9 个参数
    let gru_inn = genome.layers[0].innovation_number;
    assert_eq!(build.layer_params[&gru_inn].len(), 9);
}

#[test]
fn test_build_stacked_rnn() {
    // Rnn(4, return_seq) → Lstm(4, last_hidden) → [Linear(1)]
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.layers[0].layer_config = LayerConfig::Rnn { hidden_size: 4 };
    genome.seq_len = Some(3);

    let lstm_inn = genome.next_innovation_number();
    // 在 Rnn 之后、输出头之前插入 Lstm
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: lstm_inn,
            layer_config: LayerConfig::Lstm { hidden_size: 4 },
            enabled: true,
        },
    );

    assert!(genome.is_domain_valid());

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 3, 2]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

#[test]
fn test_build_sequential_with_activation() {
    // Rnn(4) → Tanh → [Linear(1)]
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.layers[0].layer_config = LayerConfig::Rnn { hidden_size: 4 };
    genome.seq_len = Some(3);

    let act_inn = genome.next_innovation_number();
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 3, 2]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

#[test]
fn test_capture_restore_rnn_weights() {
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(3);
    let mut rng = StdRng::seed_from_u64(42);

    let build1 = genome.build(&mut rng).unwrap();
    let rnn_inn = genome.layers[0].innovation_number;
    let original_params: Vec<_> = build1.layer_params[&rnn_inn]
        .iter()
        .map(|p| p.value().unwrap().unwrap())
        .collect();
    genome.capture_weights(&build1).unwrap();

    // 重建并恢复
    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    // Rnn 3 params + Linear 2 params = 5
    assert_eq!(report.inherited + report.reinitialized, 5);
    assert!(report.inherited >= 3, "Rnn 参数应被继承");

    let restored_params: Vec<_> = build2.layer_params[&rnn_inn]
        .iter()
        .map(|p| p.value().unwrap().unwrap())
        .collect();
    for (orig, rest) in original_params.iter().zip(restored_params.iter()) {
        assert_eq!(orig, rest);
    }
}

#[test]
fn test_capture_restore_lstm_weights() {
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.layers[0].layer_config = LayerConfig::Lstm { hidden_size: 1 };
    genome.seq_len = Some(3);
    let mut rng = StdRng::seed_from_u64(42);

    let build1 = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build1).unwrap();

    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    // Lstm 12 params + Linear 2 params = 14
    assert_eq!(report.inherited + report.reinitialized, 14);
    assert!(report.inherited >= 12, "Lstm 参数应被继承");
}

#[test]
fn test_restore_weights_cell_type_change() {
    // Rnn → Lstm 切换后，旧权重形状不匹配，应重初始化
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(3);
    let mut rng = StdRng::seed_from_u64(42);

    let build1 = genome.build(&mut rng).unwrap();
    genome.capture_weights(&build1).unwrap();

    // 切换为 Lstm（参数数量从 3 变成 12）
    genome.layers[0].layer_config = LayerConfig::Lstm { hidden_size: 1 };
    // 清除旧快照（模拟 MutateCellType 的行为）
    let inn = genome.layers[0].innovation_number;
    genome.weight_snapshots.remove(&inn);

    let build2 = genome.build(&mut rng).unwrap();
    let report = genome.restore_weights(&build2).unwrap();

    // Lstm 的 12 个参数无快照 → reinitialized
    assert!(report.reinitialized >= 12);
}

// ==================== 序列模型 + Flat 域 skip edge ====================

#[test]
fn test_build_sequential_with_flat_skip_edge() {
    // Input(seq×2) → Rnn(1) → Tanh → Linear(4) → [Linear(1)]
    // skip: Rnn(1) → [Linear(1)] (Concat)
    // 序列模型中 skip edge 仅在 Flat 域内，验证 build + 多次 forward 成功
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(5);
    let rnn_inn = genome.layers[0].innovation_number;
    let out_inn = genome.layers[1].innovation_number;

    // 插入 Tanh + Linear(4)
    let act_inn = genome.next_innovation_number();
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    let lin_inn = genome.next_innovation_number();
    genome.layers.insert(
        2,
        LayerGene {
            innovation_number: lin_inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    // skip: Rnn(1) → 输出头 (Concat)。
    // Rnn out_dim=1，main path at 输出头=4，Concat 后 in_dim=5
    let se_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
        innovation_number: se_inn,
        from_innovation: rnn_inn,
        to_innovation: out_inn,
        strategy: AggregateStrategy::Concat { dim: 1 },
        enabled: true,
    });

    assert!(genome.resolve_dimensions().is_ok());
    assert!(genome.is_domain_valid());

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    // 第一次 forward
    let data: Vec<f32> = (0..10).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 5, 2]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();
    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);

    // 第二次 forward（模拟 predict 场景——用新数据重新前向）
    let data2: Vec<f32> = (0..10).map(|i| (i as f32 + 5.0) * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data2, &[1, 5, 2]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();
    let out2 = build.output.value().unwrap().unwrap();
    assert_eq!(out2.shape(), &[1, 1]);
}

#[test]
fn test_build_sequential_with_flat_skip_edge_add() {
    // Input(seq×2) → Rnn(4) → Linear(4) → [Linear(1)]
    // skip: Rnn(4) → Linear(4) (Add，维度匹配)
    let mut genome = NetworkGenome::minimal_sequential(2, 1);
    genome.seq_len = Some(3);
    // minimal_sequential 的 Rnn hidden_size = output_dim = 1，这里需要 4 才能与 Linear(4) 做 Add
    genome.layers[0].layer_config = LayerConfig::Rnn { hidden_size: 4 };
    let rnn_inn = genome.layers[0].innovation_number;

    let lin_inn = genome.next_innovation_number();
    let out_inn = genome.layers[1].innovation_number;
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: lin_inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    // skip: Rnn(4) → 输出头(main=4)，Add(4,4) 兼容
    let se_inn = genome.next_innovation_number();
    genome.skip_edges.push(SkipEdge {
        innovation_number: se_inn,
        from_innovation: rnn_inn,
        to_innovation: out_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    // 注意：输出头 out_features=1，但 Add 后 in_dim=4，所以需调整
    // resolve_dimensions 会计算 Add(4, 4)=4 → 输出头 in=4, out=1
    assert!(genome.resolve_dimensions().is_ok());

    let mut rng = StdRng::seed_from_u64(42);
    let build = genome.build(&mut rng).unwrap();

    let data: Vec<f32> = (0..6).map(|i| i as f32 * 0.1).collect();
    build
        .input
        .set_value(&Tensor::new(&data, &[1, 3, 2]))
        .unwrap();
    build.graph.forward(&build.output).unwrap();
    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 1]);
}

// ==================== Spatial 模式构建测试 ====================

#[test]
fn test_build_spatial_minimal_forward() {
    // Flatten → Linear(2)
    // 输入: [1, 3, 8, 8]（3 channels, 8×8）→ Flatten(192) → Linear(2)
    let genome = NetworkGenome::minimal_spatial(3, 2, (8, 8));
    let mut r = StdRng::seed_from_u64(42);
    let build = genome.build(&mut r).unwrap();

    // 设置输入并前向
    let input = Tensor::ones(&[1, 3, 8, 8]);
    build.input.set_value(&input).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 2]); // batch=1, output_dim=2
}

#[test]
fn test_build_spatial_with_pool_forward() {
    // Conv2d(out=4, k=3) → Pool2d(Max, k=2, s=2) → Flatten → Linear(2)
    // 输入: [1, 1, 8, 8]（1 channel, 8×8）
    let mut genome = NetworkGenome::minimal_spatial(1, 2, (8, 8));
    let conv_inn = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: conv_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 4,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let pool_inn = genome.next_innovation_number();
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: pool_inn,
            layer_config: LayerConfig::Pool2d {
                pool_type: PoolType::Max,
                kernel_size: 2,
                stride: 2,
            },
            enabled: true,
        },
    );

    let mut r = StdRng::seed_from_u64(42);
    let build = genome.build(&mut r).unwrap();

    let input = Tensor::ones(&[1, 1, 8, 8]);
    build.input.set_value(&input).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 2]);
}

#[test]
fn test_build_spatial_avgpool_forward() {
    // Conv2d(out=2, k=1) → Pool2d(Avg, k=2, s=2) → Flatten → Linear(3)
    let mut genome = NetworkGenome::minimal_spatial(1, 3, (4, 4));
    let conv_inn = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: conv_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 2,
                kernel_size: 1,
            },
            enabled: true,
        },
    );
    let pool_inn = genome.next_innovation_number();
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: pool_inn,
            layer_config: LayerConfig::Pool2d {
                pool_type: PoolType::Avg,
                kernel_size: 2,
                stride: 2,
            },
            enabled: true,
        },
    );

    let mut r = StdRng::seed_from_u64(42);
    let build = genome.build(&mut r).unwrap();

    let input = Tensor::ones(&[1, 1, 4, 4]);
    build.input.set_value(&input).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 3]);
}

#[test]
fn test_build_spatial_batch_forward() {
    // batch_size=4
    let genome = NetworkGenome::minimal_spatial(1, 2, (4, 4));
    let mut r = StdRng::seed_from_u64(42);
    let build = genome.build(&mut r).unwrap();

    let input = Tensor::ones(&[4, 1, 4, 4]);
    build.input.set_value(&input).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[4, 2]);
}

#[test]
fn test_build_spatial_weight_capture_restore() {
    // 构建 → capture → 再构建 → restore → 验证权重一致
    let genome = NetworkGenome::minimal_spatial(1, 2, (4, 4));
    let mut r = StdRng::seed_from_u64(42);
    let build1 = genome.build(&mut r).unwrap();

    // 设置输入获得第一次输出
    let input = Tensor::ones(&[1, 1, 4, 4]);
    build1.input.set_value(&input).unwrap();
    build1.graph.forward(&build1.output).unwrap();
    let out1 = build1.output.value().unwrap().unwrap();

    // capture 权重
    let mut genome2 = genome.clone();
    genome2.capture_weights(&build1).unwrap();

    // 用新 rng 构建（随机初始化不同），但 restore 权重后应输出一致
    let mut r2 = StdRng::seed_from_u64(999);
    let build2 = genome2.build(&mut r2).unwrap();
    genome2.restore_weights(&build2).unwrap();
    build2.input.set_value(&input).unwrap();
    build2.graph.forward(&build2.output).unwrap();
    let out2 = build2.output.value().unwrap().unwrap();

    // capture 后 restore 的权重应与原来相同
    assert_eq!(out1.shape(), out2.shape());
    let diff = (out1 - out2).abs().sum().to_vec()[0];
    assert!(diff < 1e-6, "权重 restore 后输出不一致: diff={diff}");
}

#[test]
fn test_build_spatial_multi_conv_forward() {
    // Conv2d(out=4, k=3) → Conv2d(out=8, k=3) → Flatten → Linear(2)
    let mut genome = NetworkGenome::minimal_spatial(1, 2, (8, 8));
    let conv1_inn = genome.next_innovation_number();
    genome.layers.insert(
        0,
        LayerGene {
            innovation_number: conv1_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 4,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    let conv2_inn = genome.next_innovation_number();
    genome.layers.insert(
        1,
        LayerGene {
            innovation_number: conv2_inn,
            layer_config: LayerConfig::Conv2d {
                out_channels: 8,
                kernel_size: 3,
            },
            enabled: true,
        },
    );

    let mut r = StdRng::seed_from_u64(42);
    let build = genome.build(&mut r).unwrap();

    let input = Tensor::ones(&[1, 1, 8, 8]);
    build.input.set_value(&input).unwrap();
    build.graph.forward(&build.output).unwrap();

    let out = build.output.value().unwrap().unwrap();
    assert_eq!(out.shape(), &[1, 2]);
}

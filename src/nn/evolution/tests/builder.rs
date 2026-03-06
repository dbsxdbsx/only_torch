use crate::nn::evolution::gene::*;
use crate::tensor::Tensor;
use rand::rngs::StdRng;
use rand::SeedableRng;

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
    let registry = MutationRegistry::default_registry(&TaskMetric::Accuracy);
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
    let batch_input = Tensor::new(
        &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        &[4, 2],
    );
    build.input.set_value(&batch_input).unwrap();
    build.graph.forward(&build.output).unwrap();

    let output_val = build.output.value().unwrap().unwrap();
    assert_eq!(output_val.shape(), &[4, 1]);
}

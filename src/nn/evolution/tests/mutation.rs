use crate::nn::evolution::gene::*;
use crate::nn::evolution::mutation::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn constraints() -> SizeConstraints {
    SizeConstraints::default()
}

/// 构造含隐藏层的基因组：Input(2) → Linear(4) → ReLU → [Linear(1)]
fn genome_with_hidden() -> NetworkGenome {
    let mut g = NetworkGenome::minimal(2, 1);
    let i1 = g.next_innovation_number();
    let i2 = g.next_innovation_number();
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.layers.insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    g
}

// ==================== InsertLayerMutation ====================

#[test]
fn test_insert_layer_happy_path() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let m = InsertLayerMutation::default();
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    assert!(g.layer_count() >= 2);
    assert!(g.resolve_dimensions().is_ok());
    // 输出头仍是最后一个 enabled 层
    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

#[test]
fn test_insert_layer_max_layers_reached() {
    let g = genome_with_hidden();
    let c = SizeConstraints {
        max_layers: 3,
        ..constraints()
    };
    let m = InsertLayerMutation::default();
    assert!(!m.is_applicable(&g, &c));
}

#[test]
fn test_insert_layer_no_consecutive_activation() {
    // 从 minimal 出发插入多次，不应出现连续 Activation
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let m = InsertLayerMutation::default();
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let enabled: Vec<&LayerConfig> = g
        .layers
        .iter()
        .filter(|l| l.enabled)
        .map(|l| &l.layer_config)
        .collect();
    for w in enabled.windows(2) {
        let both_act = matches!(w[0], LayerConfig::Activation { .. })
            && matches!(w[1], LayerConfig::Activation { .. });
        assert!(!both_act, "发现连续 Activation: {:?} {:?}", w[0], w[1]);
    }
}

// ==================== RemoveLayerMutation ====================

#[test]
fn test_remove_layer_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = RemoveLayerMutation;
    let c = constraints();

    let before = g.layer_count();
    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.layer_count(), before - 1);
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_remove_layer_minimal_not_applicable() {
    let g = NetworkGenome::minimal(2, 1);
    let m = RemoveLayerMutation;
    assert!(!m.is_applicable(&g, &constraints()));
}

#[test]
fn test_remove_layer_preserves_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = RemoveLayerMutation;
    let c = constraints();

    // 连续删除直到只剩输出头
    while m.is_applicable(&g, &c) {
        m.apply(&mut g, &c, &mut r).unwrap();
    }

    assert_eq!(g.layer_count(), 1);
    let last = g.layers.iter().find(|l| l.enabled).unwrap();
    assert_eq!(
        last.layer_config,
        LayerConfig::Linear {
            out_features: g.output_dim
        }
    );
}

// ==================== ReplaceLayerTypeMutation ====================

#[test]
fn test_replace_layer_type_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ReplaceLayerTypeMutation::default();
    let c = constraints();

    // genome_with_hidden 有一个 ReLU
    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    // 确认 Activation 层被替换为不同的类型
    let act_layer = g
        .layers
        .iter()
        .find(|l| matches!(l.layer_config, LayerConfig::Activation { .. }))
        .unwrap();
    // 可能替换为 ReLU 以外的任何激活
    assert!(g.resolve_dimensions().is_ok());
    assert_ne!(
        act_layer.layer_config,
        LayerConfig::Activation {
            activation_type: ActivationType::ReLU,
        }
    );
}

#[test]
fn test_replace_layer_type_no_activation_not_applicable() {
    // 只有 Linear 层（无 Activation），不可替换
    let mut g = NetworkGenome::minimal(2, 1);
    let inn = g.next_innovation_number();
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let m = ReplaceLayerTypeMutation::default();
    assert!(!m.is_applicable(&g, &constraints()));
}

#[test]
fn test_replace_no_alternative_returns_error() {
    let mut g = genome_with_hidden(); // 含一个 ReLU 层
    let mut r = rng();
    // 可用列表只有 ReLU，与当前层相同 → 无替代
    let m = ReplaceLayerTypeMutation::new(vec![ActivationType::ReLU]);
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    assert!(m.apply(&mut g, &c, &mut r).is_err());
}

#[test]
fn test_replace_does_not_touch_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ReplaceLayerTypeMutation::default();
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

// ==================== GrowHiddenSizeMutation ====================

#[test]
fn test_grow_hidden_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = GrowHiddenSizeMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    let linear = g
        .layers
        .iter()
        .find(|l| {
            l.enabled
                && matches!(l.layer_config, LayerConfig::Linear { out_features } if out_features != g.output_dim)
        })
        .unwrap();
    if let LayerConfig::Linear { out_features } = linear.layer_config {
        assert!(out_features > 4);
    }
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_grow_hidden_max_reached_not_applicable() {
    let mut g = genome_with_hidden();
    // 设置隐藏层已达上限
    g.layers[0].layer_config = LayerConfig::Linear { out_features: 64 };
    let c = SizeConstraints {
        max_hidden_size: 64,
        ..constraints()
    };
    let m = GrowHiddenSizeMutation;
    assert!(!m.is_applicable(&g, &c));
}

#[test]
fn test_grow_hidden_max_params_violation() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = GrowHiddenSizeMutation;
    // 非常低的 max_total_params，增长必定超标
    let c = SizeConstraints {
        max_total_params: 1,
        ..constraints()
    };
    let result = m.apply(&mut g, &c, &mut r);
    // 要么 is_applicable=false（因为增长后必超标），要么 apply 返回 ConstraintViolation
    // 不管哪种情况，基因组不应被修改
    assert!(result.is_err() || !m.is_applicable(&g, &c));
}

#[test]
fn test_grow_does_not_touch_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = GrowHiddenSizeMutation;
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

// ==================== ShrinkHiddenSizeMutation ====================

#[test]
fn test_shrink_hidden_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ShrinkHiddenSizeMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    let linear = g
        .layers
        .iter()
        .find(|l| {
            l.enabled
                && matches!(l.layer_config, LayerConfig::Linear { out_features } if out_features != g.output_dim)
        })
        .unwrap();
    if let LayerConfig::Linear { out_features } = linear.layer_config {
        assert!(out_features < 4);
    }
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_shrink_hidden_min_reached_not_applicable() {
    let mut g = genome_with_hidden();
    g.layers[0].layer_config = LayerConfig::Linear { out_features: 1 };
    let c = SizeConstraints {
        min_hidden_size: 1,
        ..constraints()
    };
    let m = ShrinkHiddenSizeMutation;
    assert!(!m.is_applicable(&g, &c));
}

#[test]
fn test_shrink_does_not_touch_output_head() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = ShrinkHiddenSizeMutation;
    let c = constraints();

    for _ in 0..20 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

// ==================== MutateLayerParamMutation ====================

#[test]
fn test_mutate_layer_param_leaky_relu() {
    let mut g = NetworkGenome::minimal(2, 1);
    let inn = g.next_innovation_number();
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::LeakyReLU { alpha: 0.01 },
            },
            enabled: true,
        },
    );

    let mut r = rng();
    let m = MutateLayerParamMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    if let LayerConfig::Activation {
        activation_type: ActivationType::LeakyReLU { alpha },
    } = g.layers[0].layer_config
    {
        assert!((0.001..=0.5).contains(&alpha));
    } else {
        panic!("层类型应保持为 LeakyReLU");
    }
}

#[test]
fn test_mutate_layer_param_not_applicable_without_parameterized() {
    let g = genome_with_hidden(); // 只有 ReLU（无参数可变异）
    let m = MutateLayerParamMutation;
    // ReLU 不是参数化的激活函数
    assert!(!m.is_applicable(&g, &constraints()));
}

// ==================== MutateLossFunctionMutation ====================

#[test]
fn test_mutate_loss_function_happy_path() {
    let mut g = NetworkGenome::minimal(2, 1); // binary classification
    let mut r = rng();
    let m = MutateLossFunctionMutation {
        task_metric: TaskMetric::Accuracy,
    };
    let c = constraints();

    // Accuracy + output_dim=1 → [BCE, MSE]，可变异
    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    assert!(g.training_config.loss_override.is_some());
    let loss = g.training_config.loss_override.unwrap();
    // 默认推断是 BCE，变异后应该不同（MSE）
    assert_ne!(loss, LossType::BCE);
    assert_eq!(loss, LossType::MSE);
}

#[test]
fn test_mutate_loss_not_applicable_single_loss() {
    let g = NetworkGenome::minimal(2, 3); // multiclass
    let m = MutateLossFunctionMutation {
        task_metric: TaskMetric::Accuracy,
    };
    // Accuracy + output_dim=3 → 只有 [CrossEntropy]，无法变异
    assert!(!m.is_applicable(&g, &constraints()));
}

#[test]
fn test_mutate_loss_not_applicable_r2() {
    let g = NetworkGenome::minimal(2, 1);
    let m = MutateLossFunctionMutation {
        task_metric: TaskMetric::R2,
    };
    // R2 → 只有 [MSE]，无法变异
    assert!(!m.is_applicable(&g, &constraints()));
}

// ==================== MutationRegistry ====================

#[test]
fn test_default_registry_has_12_mutations() {
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
    assert_eq!(reg.len(), 12);
}

#[test]
fn test_registry_apply_random_no_applicable() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    // 只注册一个不可能成功的变异
    let mut reg = MutationRegistry::new();
    reg.register(1.0, RemoveLayerMutation);
    // minimal 只有输出头，RemoveLayer 不适用
    let result = reg.apply_random(&mut g, &constraints(), &mut r);
    assert!(result.is_err());
}

#[test]
fn test_registry_retries_on_apply_failure() {
    let mut g = genome_with_hidden();
    let mut r = rng();

    let mut reg = MutationRegistry::new();
    reg.register(1000.0, GrowHiddenSizeMutation);
    reg.register(0.001, ShrinkHiddenSizeMutation);

    // max_total_params=20 使 GrowHidden 的 apply 必定失败（当前 17，任何增长都超 20）
    // 但 is_applicable 仍返回 true（out_features=4 < max_hidden_size=64）
    // 重试后应选中 ShrinkHidden 并成功
    let c = SizeConstraints {
        max_total_params: 20,
        ..constraints()
    };

    let name = reg.apply_random(&mut g, &c, &mut r).unwrap();
    assert_eq!(name, "ShrinkHiddenSize");
}

#[test]
fn test_registry_apply_random_returns_name() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
    let name = reg.apply_random(&mut g, &constraints(), &mut r).unwrap();
    assert!(!name.is_empty());
    assert_eq!(g.generated_by, name);
}

// ==================== 输出头保护 ====================

#[test]
fn test_insert_never_after_output_head() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let m = InsertLayerMutation::default();
    let c = constraints();

    for _ in 0..50 {
        if m.is_applicable(&g, &c) {
            m.apply(&mut g, &c, &mut r).unwrap();
        }
    }

    let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(
        last.layer_config,
        LayerConfig::Linear {
            out_features: g.output_dim
        }
    );
}

#[test]
fn test_minimal_only_insert_and_add_skip_applicable() {
    let g = NetworkGenome::minimal(2, 1);
    let c = constraints();

    assert!(InsertLayerMutation::default().is_applicable(&g, &c));
    assert!(!RemoveLayerMutation.is_applicable(&g, &c));
    assert!(!ReplaceLayerTypeMutation::default().is_applicable(&g, &c));
    assert!(!GrowHiddenSizeMutation.is_applicable(&g, &c));
    assert!(!ShrinkHiddenSizeMutation.is_applicable(&g, &c));
    assert!(!MutateLayerParamMutation.is_applicable(&g, &c));
    // minimal 有 INPUT(0) → 输出头(1) 候选对，所以 AddSkipEdge 可用
    assert!(AddSkipEdgeMutation.is_applicable(&g, &c));
    assert!(!RemoveSkipEdgeMutation.is_applicable(&g, &c));
    assert!(!MutateAggregateStrategyMutation.is_applicable(&g, &c));
    // MutateLearningRate 始终可用，MutateOptimizer 仅输出头时不可用
    assert!(MutateLearningRateMutation.is_applicable(&g, &c));
    assert!(!MutateOptimizerMutation.is_applicable(&g, &c));
}

// ==================== AddSkipEdgeMutation ====================

#[test]
fn test_add_skip_edge_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = AddSkipEdgeMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    assert_eq!(g.skip_edges.len(), 1);
    assert!(g.skip_edges[0].enabled);
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_add_skip_edge_dag_validity() {
    // 添加多条 skip edge，全部必须是前向连接
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = AddSkipEdgeMutation;
    let c = constraints();

    for _ in 0..10 {
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
    }

    let enabled: Vec<u64> = g
        .layers
        .iter()
        .filter(|l| l.enabled)
        .map(|l| l.innovation_number)
        .collect();

    for edge in &g.skip_edges {
        if !edge.enabled {
            continue;
        }
        if edge.from_innovation == INPUT_INNOVATION {
            // INPUT 始终在所有层之前，DAG 合法
            continue;
        }
        let from_pos = enabled
            .iter()
            .position(|&inn| inn == edge.from_innovation)
            .expect("from 必须在 enabled 层中");
        let to_pos = enabled
            .iter()
            .position(|&inn| inn == edge.to_innovation)
            .expect("to 必须在 enabled 层中");
        assert!(
            from_pos < to_pos,
            "skip edge 必须是前向连接: from={} (pos={}) → to={} (pos={})",
            edge.from_innovation,
            from_pos,
            edge.to_innovation,
            to_pos
        );
    }
}

#[test]
fn test_add_skip_edge_forward_direction_only() {
    // 手动构造 genome，确认只有前向连接被添加
    let mut g = NetworkGenome::minimal(4, 1);
    // 仅有输出头 (inn=1)，所以唯一候选对是 INPUT(0)→inn=1
    let mut r = rng();
    let m = AddSkipEdgeMutation;
    let c = constraints();

    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.skip_edges[0].from_innovation, INPUT_INNOVATION);
    assert_eq!(g.skip_edges[0].to_innovation, 1);
}

// ==================== RemoveSkipEdgeMutation ====================

#[test]
fn test_remove_skip_edge_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let c = constraints();

    // 先添加一条
    AddSkipEdgeMutation.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.skip_edges.len(), 1);

    // 移除
    RemoveSkipEdgeMutation.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.skip_edges.len(), 0);
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_remove_skip_edge_no_edges_not_applicable() {
    let g = genome_with_hidden(); // 无 skip edges
    let c = constraints();
    assert!(!RemoveSkipEdgeMutation.is_applicable(&g, &c));
}

// ==================== MutateAggregateStrategyMutation ====================

#[test]
fn test_mutate_aggregate_strategy_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let c = constraints();

    // 添加 skip edge
    AddSkipEdgeMutation.apply(&mut g, &c, &mut r).unwrap();

    // 变异策略（可能多次尝试以确保成功）
    for _ in 0..10 {
        let result = MutateAggregateStrategyMutation.apply(&mut g, &c, &mut r);
        if result.is_ok() {
            break;
        }
    }
    // 策略应该被改变了（极高概率，因为有3个替代选项）
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_mutate_aggregate_strategy_same_target_unified() {
    // 同一目标层的所有 skip edge 应统一切换策略
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

    // 两条 skip edge 都指向输出头(1)
    let se1 = g.next_innovation_number();
    g.skip_edges.push(SkipEdge {
        innovation_number: se1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: 1,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });
    let se2 = g.next_innovation_number();
    g.skip_edges.push(SkipEdge {
        innovation_number: se2,
        from_innovation: inn_h,
        to_innovation: 1,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let mut r = rng();
    let c = constraints();

    // 变异策略（多次尝试，部分策略可能因维度不兼容被拒绝）
    for _ in 0..10 {
        let _ = MutateAggregateStrategyMutation.apply(&mut g, &c, &mut r);
    }

    // 核心断言：同目标层的所有 skip edge 必须使用相同策略
    let to_1_edges: Vec<_> = g
        .skip_edges
        .iter()
        .filter(|e| e.enabled && e.to_innovation == 1)
        .collect();
    if to_1_edges.len() >= 2 {
        let first_strategy = &to_1_edges[0].strategy;
        for edge in &to_1_edges[1..] {
            assert_eq!(
                &edge.strategy, first_strategy,
                "同目标层的 skip edge 必须使用相同策略"
            );
        }
    }
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_skip_edge_is_not_cycle() {
    // skip edge 是前向连接，不是环路（DAG 拓扑排序证明）
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = AddSkipEdgeMutation;
    let c = constraints();

    // 添加多条 skip edge
    for _ in 0..20 {
        let _ = m.apply(&mut g, &c, &mut r);
    }

    // 所有 skip edge 的 from 必须在 to 之前（或为 INPUT）
    // 这意味着没有环路
    let enabled_order: Vec<u64> = g
        .layers
        .iter()
        .filter(|l| l.enabled)
        .map(|l| l.innovation_number)
        .collect();

    for edge in &g.skip_edges {
        if !edge.enabled || edge.from_innovation == INPUT_INNOVATION {
            continue;
        }
        let from_pos = enabled_order
            .iter()
            .position(|&inn| inn == edge.from_innovation);
        let to_pos = enabled_order
            .iter()
            .position(|&inn| inn == edge.to_innovation);
        assert!(
            from_pos < to_pos,
            "发现后向 skip edge: from={} → to={}",
            edge.from_innovation,
            edge.to_innovation
        );
    }

    // 基因组仍然合法
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_skip_edge_dimension_check() {
    // 手动构造维度不兼容的 skip edge（Add 要求同维度）→ resolve_dimensions 报错
    let mut g = NetworkGenome::minimal(2, 1);
    let inn_h = g.next_innovation_number(); // 2
    g.layers.insert(
        0,
        LayerGene {
            innovation_number: inn_h,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    // INPUT dim=2, main path at 输出头 dim=4 → Add 不兼容
    let se_inn = g.next_innovation_number();
    g.skip_edges.push(SkipEdge {
        innovation_number: se_inn,
        from_innovation: INPUT_INNOVATION,
        to_innovation: 1, // 输出头
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    assert!(g.resolve_dimensions().is_err());
}

// ==================== 组合鲁棒性 ====================

#[test]
fn test_random_mutations_keep_genome_valid() {
    let mut g = NetworkGenome::minimal(2, 1);
    let mut r = rng();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
    let c = constraints();

    for _ in 0..50 {
        let _ = reg.apply_random(&mut g, &c, &mut r);

        // 每次变异后，基因组必须合法
        assert!(g.resolve_dimensions().is_ok(), "维度链断裂: {g}");
        assert!(g.total_params().is_ok(), "参数量计算失败: {g}");
        assert!(g.layer_count() >= 1, "层数为零: {g}");

        // 输出头完整
        let last = g.layers.iter().rev().find(|l| l.enabled).unwrap();
        assert_eq!(
            last.layer_config,
            LayerConfig::Linear {
                out_features: g.output_dim
            },
            "输出头被破坏: {g}"
        );

        // 创新号唯一
        let inns: Vec<u64> = g.layers.iter().map(|l| l.innovation_number).collect();
        let unique: std::collections::HashSet<u64> = inns.iter().copied().collect();
        assert_eq!(inns.len(), unique.len(), "创新号重复: {g}");
    }
}

#[test]
fn test_seed_reproducibility() {
    let run = |seed: u64| -> String {
        let mut g = NetworkGenome::minimal(2, 1);
        let mut r = StdRng::seed_from_u64(seed);
        let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy);
        let c = constraints();

        for _ in 0..20 {
            let _ = reg.apply_random(&mut g, &c, &mut r);
        }
        format!("{g}")
    };

    assert_eq!(run(123), run(123));
    assert_eq!(run(999), run(999));
}

// ==================== MutateLearningRateMutation ====================

#[test]
fn test_mutate_lr_result_is_valid_ladder_value() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = MutateLearningRateMutation;
    let c = constraints();

    for _ in 0..50 {
        m.apply(&mut g, &c, &mut r).unwrap();
        let lr = g.training_config.learning_rate;
        assert!(
            LR_LADDER.iter().any(|&v| (v - lr).abs() < 1e-10),
            "变异后 lr={lr} 不在 LR_LADDER 中"
        );
    }
}

#[test]
fn test_mutate_lr_boundary_clamp_bottom() {
    // lr=1e-5（最小值）只能上移
    let m = MutateLearningRateMutation;
    let c = constraints();

    for seed in 0..200u64 {
        let mut g = NetworkGenome::minimal(2, 1);
        g.training_config.learning_rate = 1e-5;
        let mut r = StdRng::seed_from_u64(seed);
        m.apply(&mut g, &c, &mut r).unwrap();
        assert!(
            g.training_config.learning_rate >= 1e-5,
            "lr 不应低于下界: {}",
            g.training_config.learning_rate
        );
    }
}

#[test]
fn test_mutate_lr_boundary_clamp_top() {
    // lr=1e-1（最大值）只能下移
    let m = MutateLearningRateMutation;
    let c = constraints();

    for seed in 0..200u64 {
        let mut g = NetworkGenome::minimal(2, 1);
        g.training_config.learning_rate = 1e-1;
        let mut r = StdRng::seed_from_u64(seed);
        m.apply(&mut g, &c, &mut r).unwrap();
        assert!(
            g.training_config.learning_rate <= 1e-1,
            "lr 不应超过上界: {}",
            g.training_config.learning_rate
        );
    }
}

#[test]
fn test_mutate_lr_step_distribution() {
    // 统计 1000 次变异：约80% 移动 1 步、20% 移动 2 步
    let m = MutateLearningRateMutation;
    let c = constraints();
    let start_idx = 6; // 1e-3，中间位置避免边界影响

    let mut one_step = 0;
    let mut two_step = 0;

    for seed in 0..1000u64 {
        let mut g = NetworkGenome::minimal(2, 1);
        g.training_config.learning_rate = LR_LADDER[start_idx];
        let mut r = StdRng::seed_from_u64(seed);
        m.apply(&mut g, &c, &mut r).unwrap();

        let new_idx = snap_to_nearest_index(g.training_config.learning_rate, LR_LADDER);
        let diff = (new_idx as i32 - start_idx as i32).unsigned_abs();
        match diff {
            1 => one_step += 1,
            2 => two_step += 1,
            _ => panic!("意外的步长: {diff}"),
        }
    }

    let one_pct = one_step as f64 / 1000.0;
    let two_pct = two_step as f64 / 1000.0;
    assert!(
        (0.75..=0.85).contains(&one_pct),
        "1-step 比例 {one_pct:.3} 不在 [0.75, 0.85]"
    );
    assert!(
        (0.15..=0.25).contains(&two_pct),
        "2-step 比例 {two_pct:.3} 不在 [0.15, 0.25]"
    );
}

#[test]
fn test_mutate_lr_snap_non_ladder_value() {
    // 0.007 不在 ladder 上，应先 snap 到 5e-3（index=8）再移动
    let m = MutateLearningRateMutation;
    let c = constraints();

    // 确认 snap 逻辑
    let idx = snap_to_nearest_index(0.007, LR_LADDER);
    assert_eq!(idx, 8, "0.007 应 snap 到 5e-3 (index=8)");

    // 变异后结果必须在 ladder 上
    let mut g = NetworkGenome::minimal(2, 1);
    g.training_config.learning_rate = 0.007;
    let mut r = rng();
    m.apply(&mut g, &c, &mut r).unwrap();
    assert!(
        LR_LADDER.iter().any(|&v| (v - g.training_config.learning_rate).abs() < 1e-10),
        "变异后 lr={} 不在 LR_LADDER 中",
        g.training_config.learning_rate
    );
}

#[test]
fn test_mutate_lr_is_not_structural() {
    let m = MutateLearningRateMutation;
    assert!(!m.is_structural());
}

// ==================== MutateOptimizerMutation ====================

#[test]
fn test_mutate_optimizer_adam_to_sgd() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = MutateOptimizerMutation;
    let c = constraints();

    assert_eq!(g.training_config.optimizer_type, OptimizerType::Adam);
    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.training_config.optimizer_type, OptimizerType::SGD);
}

#[test]
fn test_mutate_optimizer_sgd_to_adam() {
    let mut g = genome_with_hidden();
    g.training_config.optimizer_type = OptimizerType::SGD;
    g.training_config.learning_rate = 5e-2;
    let mut r = rng();
    let m = MutateOptimizerMutation;
    let c = constraints();

    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.training_config.optimizer_type, OptimizerType::Adam);
}

#[test]
fn test_mutate_optimizer_adam_to_sgd_lr_snap() {
    // Adam→SGD: lr=1e-4 在 SGD band [5e-3, 1e-1] 外，应 snap 到 5e-3
    let mut g = genome_with_hidden();
    g.training_config.learning_rate = 1e-4;
    let mut r = rng();
    let m = MutateOptimizerMutation;
    let c = constraints();

    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.training_config.optimizer_type, OptimizerType::SGD);
    assert!(
        (g.training_config.learning_rate - 5e-3).abs() < 1e-10,
        "lr 应 snap 到 5e-3，实际为 {}",
        g.training_config.learning_rate
    );
}

#[test]
fn test_mutate_optimizer_sgd_to_adam_lr_snap() {
    // SGD→Adam: lr=1e-1 在 Adam band [1e-4, 1e-2] 外，应 snap 到 1e-2
    let mut g = genome_with_hidden();
    g.training_config.optimizer_type = OptimizerType::SGD;
    g.training_config.learning_rate = 1e-1;
    let mut r = rng();
    let m = MutateOptimizerMutation;
    let c = constraints();

    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.training_config.optimizer_type, OptimizerType::Adam);
    assert!(
        (g.training_config.learning_rate - 1e-2).abs() < 1e-10,
        "lr 应 snap 到 1e-2，实际为 {}",
        g.training_config.learning_rate
    );
}

#[test]
fn test_mutate_optimizer_lr_in_band_intersection_unchanged() {
    // 5e-3 同时在 Adam band [1e-4, 1e-2] 和 SGD band [5e-3, 1e-1] 中
    // 切换时 lr 保持不变
    let mut g = genome_with_hidden();
    g.training_config.learning_rate = 5e-3;
    let mut r = rng();
    let m = MutateOptimizerMutation;
    let c = constraints();

    m.apply(&mut g, &c, &mut r).unwrap();
    assert!(
        (g.training_config.learning_rate - 5e-3).abs() < 1e-10,
        "band 交集内的 lr 应保持不变，实际为 {}",
        g.training_config.learning_rate
    );
}

#[test]
fn test_mutate_optimizer_not_applicable_minimal() {
    let g = NetworkGenome::minimal(2, 1);
    let c = constraints();
    assert!(!MutateOptimizerMutation.is_applicable(&g, &c));
}

#[test]
fn test_mutate_optimizer_applicable_with_hidden() {
    let g = genome_with_hidden();
    let c = constraints();
    assert!(MutateOptimizerMutation.is_applicable(&g, &c));
}

#[test]
fn test_mutate_optimizer_is_not_structural() {
    let m = MutateOptimizerMutation;
    assert!(!m.is_structural());
}

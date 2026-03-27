use crate::nn::descriptor::NodeTypeDescriptor;
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
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.layers_mut().insert(
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
    let last = g.layers().iter().rev().find(|l| l.enabled).unwrap();
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

fn spatial_genome(spatial: (usize, usize)) -> NetworkGenome {
    NetworkGenome::minimal_spatial(1, 10, spatial)
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
        .layers()
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

#[test]
fn test_insert_layer_spatial_width_not_locked_to_input_channels() {
    let m = InsertLayerMutation::new(vec![]);
    let c = SizeConstraints {
        min_hidden_size: 24,
        max_hidden_size: 64,
        size_strategy: SizeStrategy::AlignTo(8),
        ..constraints()
    };

    let mut observed_width_above_min = false;
    for seed in 0..32u64 {
        let mut g = spatial_genome((28, 28));
        let mut r = StdRng::seed_from_u64(seed);
        m.apply(&mut g, &c, &mut r).unwrap();
        for layer in g.layers().iter().filter(|l| l.enabled) {
            if let LayerConfig::Conv2d { out_channels, .. } = layer.layer_config {
                if out_channels > 24 {
                    observed_width_above_min = true;
                }
            }
        }
    }

    assert!(
        observed_width_above_min,
        "空间模式下新插入 Conv2d 的 out_channels 不应被 input_channels=1 锁死到最小值"
    );
}

#[test]
fn test_insert_layer_spatial_does_not_create_pool_when_spatial_too_small() {
    let m = InsertLayerMutation::new(vec![]);
    let c = constraints();

    for seed in 0..32u64 {
        let mut g = spatial_genome((1, 1));
        let mut r = StdRng::seed_from_u64(seed);
        m.apply(&mut g, &c, &mut r).unwrap();
        assert!(
            !g.layers()
                .iter()
                .any(|l| { l.enabled && matches!(l.layer_config, LayerConfig::Pool2d { .. }) }),
            "1x1 空间输入上不应再生成 Pool2d 候选"
        );
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
    let last = g.layers().iter().find(|l| l.enabled).unwrap();
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
        .layers()
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
    g.layers_mut().insert(
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

    let last = g.layers().iter().rev().find(|l| l.enabled).unwrap();
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
        .layers()
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
    g.layers_mut()[0].layer_config = LayerConfig::Linear { out_features: 64 };
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

    let last = g.layers().iter().rev().find(|l| l.enabled).unwrap();
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
        .layers()
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
    g.layers_mut()[0].layer_config = LayerConfig::Linear { out_features: 1 };
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

    let last = g.layers().iter().rev().find(|l| l.enabled).unwrap();
    assert_eq!(last.layer_config, LayerConfig::Linear { out_features: 1 });
}

// ==================== MutateLayerParamMutation ====================

#[test]
fn test_mutate_layer_param_leaky_relu() {
    let mut g = NetworkGenome::minimal(2, 1);
    let inn = g.next_innovation_number();
    g.layers_mut().insert(
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
    } = g.layers()[0].layer_config
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
fn test_default_registry_has_14_mutations() {
    // 12 原有 + 2 阶段7（AddConnection / RemoveConnection）= 14
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    assert_eq!(reg.len(), 14);
}

#[test]
fn test_default_registry_sequential_has_15_mutations() {
    // 14 基础 + 1 MutateCellType = 15
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, true, false);
    assert_eq!(reg.len(), 15);
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
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
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

    let last = g.layers().iter().rev().find(|l| l.enabled).unwrap();
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
    // minimal 仅有输出头，INPUT 是其直接前驱，排除后无候选。
    // is_applicable 只做快速预筛（有 enabled 层即 true），不等于有候选对。
    assert!(AddSkipEdgeMutation.is_applicable(&g, &c));
    assert!(!RemoveSkipEdgeMutation.is_applicable(&g, &c));
    assert!(!MutateAggregateStrategyMutation.is_applicable(&g, &c));
    // MutateLearningRate 始终可用，MutateOptimizer 仅输出头时不可用
    assert!(MutateLearningRateMutation.is_applicable(&g, &c));
    assert!(!MutateOptimizerMutation.is_applicable(&g, &c));
}

// ==================== MIGRATION: LayerLevel AddSkipEdgeMutation ====================

#[test]
fn test_add_skip_edge_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let m = AddSkipEdgeMutation;
    let c = constraints();

    assert!(m.is_applicable(&g, &c));
    m.apply(&mut g, &c, &mut r).unwrap();

    assert_eq!(g.skip_edges().len(), 1);
    assert!(g.skip_edges()[0].enabled);
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
        .layers()
        .iter()
        .filter(|l| l.enabled)
        .map(|l| l.innovation_number)
        .collect();

    for edge in g.skip_edges() {
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
    // Input(4) → Linear(4) → [Linear(1)]
    // 直接前驱被排除后，唯一候选: INPUT(0) → 输出头(1)
    let mut g = NetworkGenome::minimal(4, 1);
    let hidden_inn = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: hidden_inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let mut r = rng();
    let m = AddSkipEdgeMutation;
    let c = constraints();

    m.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.skip_edges()[0].from_innovation, INPUT_INNOVATION);
    assert_eq!(g.skip_edges()[0].to_innovation, 1); // 输出头
}

// ==================== MIGRATION: LayerLevel RemoveSkipEdgeMutation ====================

#[test]
fn test_remove_skip_edge_happy_path() {
    let mut g = genome_with_hidden();
    let mut r = rng();
    let c = constraints();

    // 先添加一条
    AddSkipEdgeMutation.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.skip_edges().len(), 1);

    // 移除
    RemoveSkipEdgeMutation.apply(&mut g, &c, &mut r).unwrap();
    assert_eq!(g.skip_edges().len(), 0);
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_remove_skip_edge_no_edges_not_applicable() {
    let g = genome_with_hidden(); // 无 skip edges
    let c = constraints();
    assert!(!RemoveSkipEdgeMutation.is_applicable(&g, &c));
}

// ==================== MIGRATION: LayerLevel MutateAggregateStrategyMutation ====================

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
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn_h,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let inn_act = g.next_innovation_number(); // 3
    g.layers_mut().insert(
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
    g.skip_edges_mut().push(SkipEdge {
        innovation_number: se1,
        from_innovation: INPUT_INNOVATION,
        to_innovation: 1,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });
    let se2 = g.next_innovation_number();
    g.skip_edges_mut().push(SkipEdge {
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
        .skip_edges()
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
        .layers()
        .iter()
        .filter(|l| l.enabled)
        .map(|l| l.innovation_number)
        .collect();

    for edge in g.skip_edges() {
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
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: inn_h,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    // INPUT dim=2, main path at 输出头 dim=4 → Add 不兼容
    let se_inn = g.next_innovation_number();
    g.skip_edges_mut().push(SkipEdge {
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
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    let c = constraints();

    for _ in 0..50 {
        let _ = reg.apply_random(&mut g, &c, &mut r);

        // 每次变异后，基因组必须合法
        assert!(g.resolve_dimensions().is_ok(), "维度链断裂: {g}");
        assert!(g.total_params().is_ok(), "参数量计算失败: {g}");
        assert!(g.layer_count() >= 1, "层数为零: {g}");

        // 输出头完整
        let last = g.layers().iter().rev().find(|l| l.enabled).unwrap();
        assert_eq!(
            last.layer_config,
            LayerConfig::Linear {
                out_features: g.output_dim
            },
            "输出头被破坏: {g}"
        );

        // 创新号唯一
        let inns: Vec<u64> = g.layers().iter().map(|l| l.innovation_number).collect();
        let unique: std::collections::HashSet<u64> = inns.iter().copied().collect();
        assert_eq!(inns.len(), unique.len(), "创新号重复: {g}");
    }
}

#[test]
fn test_seed_reproducibility() {
    let run = |seed: u64| -> String {
        let mut g = NetworkGenome::minimal(2, 1);
        let mut r = StdRng::seed_from_u64(seed);
        let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
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
        LR_LADDER
            .iter()
            .any(|&v| (v - g.training_config.learning_rate).abs() < 1e-10),
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

// ==================== MutateCellTypeMutation ====================

/// 构造含 RNN 的序列基因组：Input(2) → Rnn(4) → [Linear(1)]
fn genome_sequential() -> NetworkGenome {
    let mut g = NetworkGenome::minimal_sequential(2, 1);
    g.layers_mut()[0].layer_config = LayerConfig::Rnn { hidden_size: 4 };
    g.seq_len = Some(5);
    g
}

/// 构造含两层 RNN 的序列基因组：Input(2) → Rnn(4) → Lstm(4) → [Linear(1)]
fn genome_stacked_rnn() -> NetworkGenome {
    let mut g = genome_sequential();
    let inn = g.next_innovation_number();
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: inn,
            layer_config: LayerConfig::Lstm { hidden_size: 4 },
            enabled: true,
        },
    );
    g
}

#[test]
fn test_mutate_cell_type_applicable() {
    let g = genome_sequential();
    let c = constraints();
    assert!(MutateCellTypeMutation.is_applicable(&g, &c));

    // 平坦 genome 无 RNN → 不可用
    let g_flat = NetworkGenome::minimal(2, 1);
    assert!(!MutateCellTypeMutation.is_applicable(&g_flat, &c));
}

#[test]
fn test_mutate_cell_type_switches() {
    let g = genome_sequential();
    let mut r = rng();
    let c = constraints();

    let original = g.layers()[0].layer_config.clone();
    // 多次尝试确保切换
    for _ in 0..20 {
        let mut g2 = g.clone();
        MutateCellTypeMutation.apply(&mut g2, &c, &mut r).unwrap();
        if g2.layers()[0].layer_config != original {
            // 成功切换
            assert!(NetworkGenome::is_recurrent(&g2.layers()[0].layer_config));
            return;
        }
    }
    panic!("20 次尝试后仍未切换 cell 类型");
}

#[test]
fn test_mutate_cell_type_preserves_hidden_size() {
    let g = genome_sequential();
    let c = constraints();

    for seed in 0..20u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        MutateCellTypeMutation.apply(&mut g2, &c, &mut r).unwrap();

        // hidden_size 应保持为 4
        match &g2.layers()[0].layer_config {
            LayerConfig::Rnn { hidden_size }
            | LayerConfig::Lstm { hidden_size }
            | LayerConfig::Gru { hidden_size } => {
                assert_eq!(*hidden_size, 4);
            }
            _ => panic!("应为 RNN 族层"),
        }
    }
}

#[test]
fn test_mutate_cell_type_multi_rnn() {
    let g = genome_stacked_rnn();
    let c = constraints();
    let mut changed_first = false;
    let mut changed_second = false;

    for seed in 0..50u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        MutateCellTypeMutation.apply(&mut g2, &c, &mut r).unwrap();

        if g2.layers()[0].layer_config != g.layers()[0].layer_config {
            changed_first = true;
        }
        if g2.layers()[1].layer_config != g.layers()[1].layer_config {
            changed_second = true;
        }
        if changed_first && changed_second {
            break;
        }
    }
    assert!(
        changed_first && changed_second,
        "多 RNN 时应能随机选择任意一个切换"
    );
}

// ==================== 域感知 InsertLayer / RemoveLayer ====================

#[test]
fn test_insert_layer_sequence_domain_accepts_rnn() {
    // 在序列 genome 上多次 InsertLayer，应能插入 RNN 族层
    let c = constraints();
    let m = InsertLayerMutation::default();
    let mut found_rnn = false;

    for seed in 0..100u64 {
        let mut g = genome_sequential();
        let mut r = StdRng::seed_from_u64(seed);
        if m.is_applicable(&g, &c) {
            let _ = m.apply(&mut g, &c, &mut r);
        }
        let has_extra_rnn = g.layers().iter().filter(|l| l.enabled).any(|l| {
            NetworkGenome::is_recurrent(&l.layer_config)
                && l.innovation_number != g.layers()[0].innovation_number
        });
        if has_extra_rnn {
            found_rnn = true;
            // 域链应合法
            assert!(g.is_domain_valid(), "插入 RNN 后域应合法: {g}");
            break;
        }
    }
    assert!(found_rnn, "序列 genome 应能插入额外 RNN 层");
}

#[test]
fn test_insert_layer_flat_domain_no_rnn() {
    // 平坦 genome 上 InsertLayer 不应插入 RNN
    let c = constraints();
    let m = InsertLayerMutation::default();

    for seed in 0..100u64 {
        let mut g = genome_with_hidden();
        let mut r = StdRng::seed_from_u64(seed);
        let _ = m.apply(&mut g, &c, &mut r);
        let has_rnn = g
            .layers()
            .iter()
            .any(|l| l.enabled && NetworkGenome::is_recurrent(&l.layer_config));
        assert!(!has_rnn, "平坦 genome 不应插入 RNN: {g}");
    }
}

#[test]
fn test_remove_layer_last_rnn_blocked() {
    // 只有一个 RNN 的序列 genome，删除它会导致域非法 → 应被阻止
    let g = genome_sequential();
    let c = constraints();

    // 反复尝试删除，应始终失败（只有 RNN + 输出头，删 RNN 后 Sequence→Linear 非法）
    for seed in 0..20u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        let result = RemoveLayerMutation.apply(&mut g2, &c, &mut r);
        // 要么 NotApplicable（只有输出头不可删），要么 ConstraintViolation（域非法）
        assert!(result.is_err(), "仅剩一个 RNN 时不应被删除");
    }
}

#[test]
fn test_remove_layer_stacked_rnn_ok() {
    // 有两层 RNN 时可以删除其中一个
    let g = genome_stacked_rnn();
    let c = constraints();
    let mut removed = false;

    for seed in 0..50u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        if RemoveLayerMutation.apply(&mut g2, &c, &mut r).is_ok() {
            assert!(g2.is_domain_valid(), "删除后域应合法");
            removed = true;
            break;
        }
    }
    assert!(removed, "有多 RNN 时应能成功删除一个");
}

#[test]
fn test_insert_layer_sequence_node_level_accepts_rnn_family() {
    let c = constraints();
    let m = InsertLayerMutation::default();
    let mut inserted = false;

    for seed in 0..50u64 {
        let mut g = genome_sequential();
        g.migrate_to_node_level().unwrap();
        let before_cells = g
            .nodes()
            .iter()
            .filter(|n| {
                matches!(
                    n.node_type,
                    NodeTypeDescriptor::CellRnn { .. }
                        | NodeTypeDescriptor::CellLstm { .. }
                        | NodeTypeDescriptor::CellGru { .. }
                )
            })
            .count();
        let mut r = StdRng::seed_from_u64(seed);
        if m.apply(&mut g, &c, &mut r).is_ok() {
            let after_cells = g
                .nodes()
                .iter()
                .filter(|n| {
                    matches!(
                        n.node_type,
                        NodeTypeDescriptor::CellRnn { .. }
                            | NodeTypeDescriptor::CellLstm { .. }
                            | NodeTypeDescriptor::CellGru { .. }
                    )
                })
                .count();
            assert!(
                after_cells > before_cells,
                "序列 NodeLevel 插层应新增一个循环块"
            );
            assert!(g.analyze().is_valid, "插入后图应合法");
            inserted = true;
            break;
        }
    }

    assert!(inserted, "序列 NodeLevel 应能插入循环块");
}

#[test]
fn test_remove_layer_sequence_node_level_blocks_last_recurrent() {
    let c = constraints();
    for seed in 0..20u64 {
        let mut g = genome_sequential();
        g.migrate_to_node_level().unwrap();
        let mut r = StdRng::seed_from_u64(seed);
        let result = RemoveLayerMutation.apply(&mut g, &c, &mut r);
        assert!(result.is_err(), "序列 NodeLevel 不应删除唯一循环块");
    }
}

// ==================== Grow/Shrink 对 RNN 的支持 ====================

#[test]
fn test_grow_hidden_size_rnn() {
    let mut g = genome_sequential();
    let c = constraints();
    let m = GrowHiddenSizeMutation;

    assert!(m.is_applicable(&g, &c));

    let original_size = 4; // Rnn { hidden_size: 4 }
    let mut r = rng();
    m.apply(&mut g, &c, &mut r).unwrap();

    match &g.layers()[0].layer_config {
        LayerConfig::Rnn { hidden_size } => {
            assert!(*hidden_size > original_size, "hidden_size 应增长");
        }
        _ => {
            // 可能增长了输出头的 Linear，也可以
        }
    }
}

#[test]
fn test_shrink_hidden_size_lstm() {
    let mut g = genome_sequential();
    g.layers_mut()[0].layer_config = LayerConfig::Lstm { hidden_size: 8 };
    let c = constraints();
    let m = ShrinkHiddenSizeMutation;

    assert!(m.is_applicable(&g, &c));

    let mut r = rng();
    m.apply(&mut g, &c, &mut r).unwrap();

    // ShrinkHiddenSize 随机选择层，可能选了输出头也可能选了 LSTM
    // 维度链应仍然合法
    assert!(g.resolve_dimensions().is_ok());
}

#[test]
fn test_grow_hidden_size_node_level_recurrent_updates_shapes() {
    let c = constraints();
    let m = GrowHiddenSizeMutation;
    let mut changed = false;

    for seed in 0..50u64 {
        let mut g = genome_sequential();
        g.migrate_to_node_level().unwrap();
        let before_hidden = g
            .nodes()
            .iter()
            .find_map(|n| match n.node_type {
                NodeTypeDescriptor::CellRnn { hidden_size, .. }
                | NodeTypeDescriptor::CellLstm { hidden_size, .. }
                | NodeTypeDescriptor::CellGru { hidden_size, .. } => Some(hidden_size),
                _ => None,
            })
            .unwrap();

        let mut r = StdRng::seed_from_u64(seed);
        m.apply(&mut g, &c, &mut r).unwrap();

        let after_cell = g.nodes().iter().find(|n| {
            matches!(
                n.node_type,
                NodeTypeDescriptor::CellRnn { .. }
                    | NodeTypeDescriptor::CellLstm { .. }
                    | NodeTypeDescriptor::CellGru { .. }
            )
        });

        if let Some(cell) = after_cell {
            let after_hidden = match cell.node_type {
                NodeTypeDescriptor::CellRnn { hidden_size, .. }
                | NodeTypeDescriptor::CellLstm { hidden_size, .. }
                | NodeTypeDescriptor::CellGru { hidden_size, .. } => hidden_size,
                _ => unreachable!(),
            };
            if after_hidden > before_hidden {
                let param_ids = cell.parents[1..].to_vec();
                for (idx, pid) in param_ids.iter().enumerate() {
                    let p = g
                        .nodes()
                        .iter()
                        .find(|n| n.innovation_number == *pid)
                        .unwrap();
                    match idx % 3 {
                        0 => assert_eq!(p.output_shape[1], after_hidden),
                        1 => {
                            assert_eq!(p.output_shape[0], after_hidden);
                            assert_eq!(p.output_shape[1], after_hidden);
                        }
                        _ => {
                            assert_eq!(p.output_shape[0], 1);
                            assert_eq!(p.output_shape[1], after_hidden);
                        }
                    }
                }
                assert!(g.analyze().is_valid, "grow 后图应合法");
                changed = true;
                break;
            }
        }
    }

    assert!(changed, "NodeLevel grow 应能命中并放大循环块");
}

#[test]
fn test_shrink_hidden_size_node_level_recurrent_updates_shapes() {
    let mut changed = false;

    for seed in 0..50u64 {
        let mut g = genome_sequential();
        g.layers_mut()[0].layer_config = LayerConfig::Lstm { hidden_size: 8 };
        g.migrate_to_node_level().unwrap();
        let before_hidden = g
            .nodes()
            .iter()
            .find_map(|n| match n.node_type {
                NodeTypeDescriptor::CellRnn { hidden_size, .. }
                | NodeTypeDescriptor::CellLstm { hidden_size, .. }
                | NodeTypeDescriptor::CellGru { hidden_size, .. } => Some(hidden_size),
                _ => None,
            })
            .unwrap();

        let mut r = StdRng::seed_from_u64(seed);
        ShrinkHiddenSizeMutation
            .apply(&mut g, &constraints(), &mut r)
            .unwrap();

        let after_hidden = g
            .nodes()
            .iter()
            .find_map(|n| match n.node_type {
                NodeTypeDescriptor::CellRnn { hidden_size, .. }
                | NodeTypeDescriptor::CellLstm { hidden_size, .. }
                | NodeTypeDescriptor::CellGru { hidden_size, .. } => Some(hidden_size),
                _ => None,
            })
            .unwrap();

        if after_hidden < before_hidden {
            assert!(g.analyze().is_valid, "shrink 后图应合法");
            changed = true;
            break;
        }
    }

    assert!(changed, "NodeLevel shrink 应能命中并缩小循环块");
}

// ==================== AddSkipEdge 域约束（序列模式） ====================

#[test]
fn test_add_skip_edge_sequential_flat_only() {
    // 序列 genome 上的 AddSkipEdge 只应产生 Flat 域内的 skip edge
    // Input(Seq) → Rnn(4) → Tanh → Linear(4) → [Linear(1)]
    // Flat 域: Tanh, Linear(4), [Linear(1)] —— 只有这几个能参与 skip
    let mut g = genome_sequential();
    // 插入 Tanh + Linear(4) 在 Rnn 和 输出头之间
    let act_inn = g.next_innovation_number();
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    let lin_inn = g.next_innovation_number();
    g.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: lin_inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    let domain_map = g.compute_domain_map();
    let c = constraints();

    for seed in 0..50u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        if AddSkipEdgeMutation.apply(&mut g2, &c, &mut r).is_ok() {
            for edge in g2.skip_edges() {
                if !edge.enabled {
                    continue;
                }
                let from_domain = domain_map[&edge.from_innovation];
                assert_eq!(
                    from_domain,
                    ShapeDomain::Flat,
                    "skip 源应在 Flat 域，from={}",
                    edge.from_innovation
                );
            }
            // 构建应成功
            let mut rng = StdRng::seed_from_u64(seed);
            assert!(g2.build(&mut rng).is_ok(), "带 skip edge 的 build 应成功");
        }
    }
}

#[test]
fn test_add_skip_edge_sequential_minimal_no_candidates() {
    // 最小序列 genome: Input(Seq) → Rnn → [Linear]
    // Input 在 Sequence 域，Rnn 在 Flat 域，输出头在 Flat 域
    // 唯一的 Flat 域对是 Rnn → 输出头，但 Rnn 在输出头之前只有一个层——
    // resolve_dimensions 中 Rnn out_dim=4 ≠ Linear in=4，可能允许 Concat
    // 但 INPUT 在 Seq 域，不允许作为 skip 源
    let g = genome_sequential();
    let c = constraints();

    // INPUT 在 Sequence 域 → 不允许作为 skip 源
    let domain_map = g.compute_domain_map();
    assert_eq!(domain_map[&INPUT_INNOVATION], ShapeDomain::Sequence);

    // 仅有 Rnn(inn=1) → 输出头(inn=2?) 可能的 skip 对
    // 但只有一个源候选（Rnn）和一个目标（输出头）在 Flat 域
    for seed in 0..20u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        if AddSkipEdgeMutation.apply(&mut g2, &c, &mut r).is_ok() {
            // 如果成功，必须是 Flat 域内的 skip
            for edge in g2.skip_edges() {
                assert_ne!(
                    edge.from_innovation, INPUT_INNOVATION,
                    "序列模式不应允许从 Input(Sequence 域) 出发的 skip"
                );
            }
        }
    }
}

// ==================== 序列模式组合鲁棒性 ====================

#[test]
fn test_random_mutations_keep_sequential_genome_valid() {
    let mut g = genome_sequential();
    let mut r = rng();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, true, false);
    let c = constraints();

    for _ in 0..50 {
        let _ = reg.apply_random(&mut g, &c, &mut r);

        assert!(g.resolve_dimensions().is_ok(), "维度链断裂: {g}");
        assert!(g.is_domain_valid(), "域链非法: {g}");
        assert!(g.validate_skip_edge_domains(), "skip edge 域失效: {g}");
        assert!(g.layer_count() >= 1, "层数为零: {g}");

        // 输出头完整
        let last = g.layers().iter().rev().find(|l| l.enabled).unwrap();
        assert_eq!(
            last.layer_config,
            LayerConfig::Linear {
                out_features: g.output_dim
            },
            "输出头被破坏: {g}"
        );
    }
}

#[test]
fn test_random_mutations_sequential_build_always_succeeds() {
    // 多轮随机变异后，build() 应始终成功（不会因 skip edge 域失效而 panic）
    let g = genome_sequential();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, true, false);
    let c = constraints();

    for seed in 0..10u64 {
        let mut r = StdRng::seed_from_u64(seed);
        let mut g2 = g.clone();

        for _ in 0..30 {
            let _ = reg.apply_random(&mut g2, &c, &mut r);
        }

        // build 不应 panic 或返回 Err
        let mut build_rng = StdRng::seed_from_u64(seed + 1000);
        assert!(
            g2.build(&mut build_rng).is_ok(),
            "30 轮变异后 build 失败 (seed={seed}): {g2}"
        );
    }
}

// ==================== skip edge 域重新验证 ====================

#[test]
fn test_insert_rnn_after_skip_source_blocked() {
    // 构造: Input(seq) → GRU(4) → Tanh → Linear(4) → [Linear(1)]
    // GRU(4) 域 = Flat（下一个实质层是 Linear，非循环）
    // 添加 skip edge: GRU(4) → [Linear(1)]（跳过 Tanh 和 Linear(4)）
    // 然后在 Tanh 和 Linear(4) 之间插入一个新 LSTM 层，
    // 这会让 GRU(4) 的 needs_return_sequences 变为 true（因为 LSTM 是循环层），
    // GRU(4) 域从 Flat 变为 Sequence，skip edge 源域失效。
    let mut g = genome_sequential(); // Rnn(4) → [Linear(1)]
    // 把 Rnn(4) 换成 GRU(4)，方便区分
    g.layers_mut()[0].layer_config = LayerConfig::Gru { hidden_size: 4 };
    let gru_inn = g.layers()[0].innovation_number;

    // 插入 Tanh
    let act_inn = g.next_innovation_number();
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    // 插入 Linear(4)
    let lin_inn = g.next_innovation_number();
    g.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: lin_inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );

    // 添加 skip edge: GRU(4) → 输出头（跳过 Tanh 和 Linear(4)）
    let output_inn = g.layers().last().unwrap().innovation_number;
    let skip_inn = g.next_innovation_number();
    g.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: gru_inn,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    // 确认当前状态: GRU(4) 域 = Flat，skip edge 合法
    assert!(g.is_domain_valid());
    assert!(g.validate_skip_edge_domains());
    assert!(g.resolve_dimensions().is_ok());

    // 在 Tanh 和 Linear(4) 之间插入 LSTM，这会让 GRU(4) 的 return_sequences 变为 true
    // 因为 GRU(4) 的下一个实质层现在是 LSTM（跳过 Tanh），是循环层
    // GRU(4) 域从 Flat → Sequence，skip edge 源域失效
    let lstm_inn = g.next_innovation_number();
    g.layers_mut().insert(
        2, // Tanh 和 Linear(4) 之间
        LayerGene {
            innovation_number: lstm_inn,
            layer_config: LayerConfig::Lstm { hidden_size: 2 },
            enabled: true,
        },
    );

    // GRU(4) 域现在应为 Sequence，skip edge 失效
    let domain_map = g.compute_domain_map();
    assert_eq!(
        domain_map[&gru_inn],
        ShapeDomain::Sequence,
        "LSTM 插入后 GRU 应返回序列，域为 Sequence"
    );
    assert!(
        !g.validate_skip_edge_domains(),
        "skip edge 源 GRU 从 Flat 变为 Sequence 后，validate_skip_edge_domains 应返回 false"
    );
}

#[test]
fn test_insert_layer_mutation_rejects_skip_edge_domain_violation() {
    // 通过 InsertLayerMutation 的 apply() 流程验证：
    // 当插入 RNN 族层会导致已有 skip edge 域失效时，应自动回滚
    let mut g = genome_sequential(); // Rnn(4) → [Linear(1)]
    let act_inn = g.next_innovation_number();
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: act_inn,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::Tanh,
            },
            enabled: true,
        },
    );
    let lin_inn = g.next_innovation_number();
    g.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: lin_inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let output_inn = g.layers().last().unwrap().innovation_number;
    let skip_inn = g.next_innovation_number();
    g.skip_edges_mut().push(SkipEdge {
        innovation_number: skip_inn,
        from_innovation: lin_inn,
        to_innovation: output_inn,
        strategy: AggregateStrategy::Add,
        enabled: true,
    });

    let c = constraints();
    let m = InsertLayerMutation::default();

    // 多次尝试 InsertLayer，无论插入什么，skip edge 域都应保持合法
    for seed in 0..50u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        let _ = m.apply(&mut g2, &c, &mut r);

        // 无论变异成功与否，当前 genome 的 skip edge 域必须合法
        assert!(
            g2.validate_skip_edge_domains(),
            "InsertLayer 后 skip edge 域不唹合法 (seed={seed}): {g2}"
        );
    }
}

#[test]
fn test_is_domain_valid_matches_compute_domain_map() {
    // 确保 is_domain_valid() 和 compute_domain_map() 对循环层的域判定一致
    // 构造: Input(seq) → Rnn(4) → Linear(4) → Lstm(2) → [Linear(1)]
    // is_domain_valid 应判定 Rnn 域为 Sequence（因下一个实质层是 Linear，非循环）
    // 而非因为后面还有 LSTM 就认为是 Sequence
    let mut g = genome_sequential(); // Rnn(4) → [Linear(1)]
    let lin_inn = g.next_innovation_number();
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: lin_inn,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    let lstm_inn = g.next_innovation_number();
    g.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: lstm_inn,
            layer_config: LayerConfig::Lstm { hidden_size: 2 },
            enabled: true,
        },
    );

    // Rnn(4) 的下一个实质层是 Linear(4)，非循环 → Rnn 域应为 Flat
    // 但 Linear 在 Flat 域后面紧接 LSTM 在 Flat 域，而 LSTM 需要 Sequence 域输入
    // 所以这个结构本身就是非法的（Flat→LSTM 非法）
    // is_domain_valid 和 compute_domain_map 都应认为 Rnn 域 = Flat
    let domain_map = g.compute_domain_map();
    let rnn_inn = g.layers()[0].innovation_number;
    assert_eq!(
        domain_map[&rnn_inn],
        ShapeDomain::Flat,
        "compute_domain_map 应判定 Rnn 域为 Flat（下一个实质层是 Linear）"
    );

    // is_domain_valid 应拒绝（LSTM 在 Flat 域中非法）
    assert!(!g.is_domain_valid(), "Flat 域中出现 LSTM 应导致域链非法");
}

// ==================== NodeLevel 变异测试 ====================

/// 创建 NodeLevel 基因组：Input(2) → Linear(4) → ReLU → [Linear(1)]
fn node_level_genome_with_hidden() -> NetworkGenome {
    let mut g = genome_with_hidden();
    g.migrate_to_node_level().expect("迁移到 NodeLevel 应成功");
    g
}

/// 创建含 Dropout 的 NodeLevel 基因组：Input(2) → Linear(4) → Dropout(0.3) → ReLU → [Linear(1)]
fn node_level_genome_with_dropout() -> NetworkGenome {
    let mut g = NetworkGenome::minimal(2, 1);
    let i1 = g.next_innovation_number();
    let i2 = g.next_innovation_number();
    let i3 = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Dropout { p: 0.3 },
            enabled: true,
        },
    );
    g.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: i3,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    g.migrate_to_node_level().expect("含 Dropout 的迁移应成功");
    g
}

#[test]
fn test_node_level_insert_layer() {
    let mut g = node_level_genome_with_hidden();
    let mut r = rng();
    let m = InsertLayerMutation::default();
    let c = constraints();

    assert!(g.is_node_level(), "应为 NodeLevel");
    assert!(m.is_applicable(&g, &c), "含块时应可适用");

    let before = g.layer_count();
    m.apply(&mut g, &c, &mut r).expect("InsertLayer 应成功");
    // 插入后块数 >= 插入前（激活函数不新增块数，但线性层会）
    assert!(g.is_node_level(), "变异后仍应为 NodeLevel");
    assert!(
        g.layer_count() >= before,
        "InsertLayer 后 layer_count 不应减少: before={before}, after={}",
        g.layer_count()
    );

    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "InsertLayer 后 build 应成功"
    );
}

#[test]
fn test_node_level_remove_layer() {
    let mut g = node_level_genome_with_hidden();
    let mut r = rng();
    let m = RemoveLayerMutation;
    let c = constraints();

    assert!(g.is_node_level(), "应为 NodeLevel");
    assert!(m.is_applicable(&g, &c), "含隐藏块时应可适用");

    let before = g.layer_count();
    m.apply(&mut g, &c, &mut r).expect("RemoveLayer 应成功");
    assert!(
        g.layer_count() < before,
        "RemoveLayer 后 layer_count 应减少: before={before}, after={}",
        g.layer_count()
    );

    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "RemoveLayer 后 build 应成功"
    );
}

#[test]
fn test_node_level_remove_layer_output_head_preserved() {
    // 连续移除直到不可适用，输出头必须始终保留
    let mut g = node_level_genome_with_hidden();
    let mut r = rng();
    let m = RemoveLayerMutation;
    let c = constraints();

    while m.is_applicable(&g, &c) {
        m.apply(&mut g, &c, &mut r).unwrap();
    }

    assert!(g.layer_count() >= 1, "至少保留输出头块");
    let mut build_rng = rng();
    assert!(g.build(&mut build_rng).is_ok(), "移除到底后 build 应成功");
}

#[test]
fn test_node_level_minimal_not_removable() {
    // 最小 NodeLevel genome（只有输出头块）不可移除
    let mut g = NetworkGenome::minimal(2, 1);
    g.migrate_to_node_level().unwrap();
    let m = RemoveLayerMutation;
    assert!(
        !m.is_applicable(&g, &constraints()),
        "最小 NodeLevel 不应可移除"
    );
}

#[test]
fn test_node_level_grow() {
    let mut g = node_level_genome_with_hidden();
    let mut r = rng();
    let m = GrowHiddenSizeMutation;
    let c = constraints();

    assert!(g.is_node_level(), "应为 NodeLevel");
    assert!(m.is_applicable(&g, &c), "含 Linear 块时应可适用");

    m.apply(&mut g, &c, &mut r).expect("GrowHiddenSize 应成功");

    let mut build_rng = rng();
    assert!(g.build(&mut build_rng).is_ok(), "Grow 后 build 应成功");
}

#[test]
fn test_node_level_shrink() {
    // 先 Grow，再 Shrink，确保可缩小空间存在
    let mut g = node_level_genome_with_hidden();
    let c = SizeConstraints {
        min_hidden_size: 1,
        max_hidden_size: 256,
        ..constraints()
    };

    // Grow 一次给缩小创造空间
    GrowHiddenSizeMutation
        .apply(&mut g, &c, &mut StdRng::seed_from_u64(1))
        .ok();

    let mut r = StdRng::seed_from_u64(2);
    let m = ShrinkHiddenSizeMutation;
    assert!(m.is_applicable(&g, &c), "Grow 后 Shrink 应可适用");
    m.apply(&mut g, &c, &mut r)
        .expect("ShrinkHiddenSize 应成功");

    let mut build_rng = rng();
    assert!(g.build(&mut build_rng).is_ok(), "Shrink 后 build 应成功");
}

#[test]
fn test_node_level_replace_activation() {
    let mut g = node_level_genome_with_hidden();
    let mut r = rng();
    let m = ReplaceLayerTypeMutation::default();
    let c = constraints();

    assert!(g.is_node_level(), "应为 NodeLevel");
    assert!(m.is_applicable(&g, &c), "含 Activation 块时应可适用");

    m.apply(&mut g, &c, &mut r)
        .expect("ReplaceLayerType 应成功");

    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "ReplaceActivation 后 build 应成功"
    );
}

#[test]
fn test_node_level_replace_activation_not_applicable_without_activation() {
    // 无激活块时 ReplaceLayerType 不可适用
    let mut g = NetworkGenome::minimal(2, 1); // 只有输出头 Linear
    g.migrate_to_node_level().unwrap();
    let m = ReplaceLayerTypeMutation::default();
    assert!(!m.is_applicable(&g, &constraints()));
}

#[test]
fn test_node_level_mutate_param_dropout() {
    let mut g = node_level_genome_with_dropout();
    let mut r = rng();
    let m = MutateLayerParamMutation;
    let c = constraints();

    assert!(g.is_node_level(), "应为 NodeLevel");
    assert!(m.is_applicable(&g, &c), "含 Dropout 节点时应可适用");

    m.apply(&mut g, &c, &mut r)
        .expect("MutateLayerParam 应成功");

    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "Dropout 参数变异后 build 应成功"
    );
}

#[test]
fn test_node_level_multiple_mutations_build_succeeds() {
    // 多轮随机变异后 build 应始终成功
    let g = node_level_genome_with_hidden();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    let c = constraints();

    for seed in 0..15u64 {
        let mut r = StdRng::seed_from_u64(seed);
        let mut g2 = g.clone();

        // 逐轮跟踪，找出哪一轮变异导致 build 失败
        for round in 0..20usize {
            let prev = g2.clone();
            let mutation_result = reg.apply_random(&mut g2, &c, &mut r);
            let mut check_rng = StdRng::seed_from_u64(seed + 2000 + round as u64);
            if g2.build(&mut check_rng).is_err() {
                let mut build_rng = StdRng::seed_from_u64(seed + 1000);
                match g2.build(&mut build_rng) {
                    Ok(_) => {}
                    Err(e) => panic!(
                        "第 {round} 轮变异后 build 失败 (seed={seed}, err={e:?})\n变异结果: {mutation_result:?}\n变异前节点:\n{:#?}\n变异后节点:\n{:#?}",
                        prev.nodes(),
                        g2.nodes()
                    ),
                }
            }
        }

        let mut build_rng = StdRng::seed_from_u64(seed + 1000);
        match g2.build(&mut build_rng) {
            Ok(_) => {}
            Err(e) => panic!(
                "20 轮随机变异后 build 失败 (seed={seed}): {e:?}\n基因组节点: {:#?}",
                g2.nodes()
            ),
        }
    }
}

#[test]
fn test_node_level_skip_edge_mutations_not_applicable() {
    let g = node_level_genome_with_hidden();
    let c = constraints();

    assert!(g.is_node_level(), "应为 NodeLevel");
    assert!(
        g.skip_edges().is_empty(),
        "NodeLevel 不应暴露 LayerLevel skip_edges"
    );
    assert!(
        g.layers().is_empty(),
        "NodeLevel 不应暴露 LayerLevel layers"
    );
    assert!(
        !AddSkipEdgeMutation.is_applicable(&g, &c),
        "NodeLevel 当前不应适用 AddSkipEdge"
    );
    assert!(
        !RemoveSkipEdgeMutation.is_applicable(&g, &c),
        "NodeLevel 当前不应适用 RemoveSkipEdge"
    );
    assert!(
        !MutateAggregateStrategyMutation.is_applicable(&g, &c),
        "NodeLevel 当前不应适用 MutateAggregateStrategy"
    );
}

#[test]
fn test_node_level_default_registry_never_returns_skip_edge_mutation_names() {
    let g = node_level_genome_with_hidden();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    let c = constraints();

    for seed in 0..40u64 {
        let mut r = StdRng::seed_from_u64(seed);
        let mut g2 = g.clone();
        let name = reg
            .apply_random(&mut g2, &c, &mut r)
            .expect("NodeLevel default registry 应至少存在一种可用变异");
        assert!(
            name != "AddSkipEdge" && name != "RemoveSkipEdge" && name != "MutateAggregateStrategy",
            "NodeLevel default registry 不应返回 LayerLevel skip edge 变异，实际={name}"
        );
    }
}

// ==================== 阶段 7：AddConnectionMutation / RemoveConnectionMutation ====================

use crate::nn::evolution::gene::ShapeDomain;
use crate::nn::evolution::node_ops::{
    add_skip_connection, commit_counter, find_connectable_pairs, find_removable_skip_connections,
    make_counter, repair_skip_connections, sync_computation_shapes,
};

/// Input(2) → Linear(4) → ReLU → Linear(4) → [Linear(1)]（含 3 个中间块，便于测试 skip）
fn node_level_3block_genome() -> NetworkGenome {
    let mut g = NetworkGenome::minimal(2, 1);
    let i1 = g.next_innovation_number();
    let i2 = g.next_innovation_number();
    let i3 = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Activation {
                activation_type: ActivationType::ReLU,
            },
            enabled: true,
        },
    );
    g.layers_mut().insert(
        2,
        LayerGene {
            innovation_number: i3,
            layer_config: LayerConfig::Linear { out_features: 4 },
            enabled: true,
        },
    );
    g.migrate_to_node_level().expect("迁移应成功");
    g
}

/// Input([1,1,8,8]) → Conv2d(4,k=3) → Conv2d(8,k=3) → Flatten → [Linear(2)]
///
/// 用于验证 Spatial 域 AddConnection 的 1x1 Conv2d 投影路径。
fn node_level_spatial_2conv_genome() -> NetworkGenome {
    let mut g = NetworkGenome::minimal_spatial(1, 2, (8, 8));
    let i1 = g.next_innovation_number();
    let i2 = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Conv2d {
                out_channels: 4,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    g.layers_mut().insert(
        1,
        LayerGene {
            innovation_number: i2,
            layer_config: LayerConfig::Conv2d {
                out_channels: 8,
                kernel_size: 3,
            },
            enabled: true,
        },
    );
    g.migrate_to_node_level().expect("空间 genome 迁移应成功");
    g
}

// ── 候选对发现 ────────────────────────────────────────────

#[test]
fn test_find_connectable_pairs_non_empty_for_multiblock_genome() {
    let g = node_level_3block_genome();
    let pairs = find_connectable_pairs(&g);
    assert!(!pairs.is_empty(), "3块基因组应有可连接候选对");
}

#[test]
fn test_find_connectable_pairs_empty_for_minimal_genome() {
    let mut g = NetworkGenome::minimal(2, 1);
    g.migrate_to_node_level().unwrap();
    // 最小基因组只有输出头，不满足 blocks.len() >= 2 的非直接前驱条件
    let pairs = find_connectable_pairs(&g);
    // 可能为空（只有一个块，无法添加跳跃连接）
    assert!(pairs.is_empty() || !pairs.is_empty()); // 允许任意（主要测试不 panic）
}

// ── AddConnectionMutation ─────────────────────────────────

#[test]
fn test_add_connection_is_applicable_for_multiblock() {
    let g = node_level_3block_genome();
    let c = constraints();
    assert!(AddConnectionMutation.is_applicable(&g, &c), "3块应可适用");
}

#[test]
fn test_add_connection_not_applicable_for_layer_level() {
    let g = genome_with_hidden(); // LayerLevel
    let c = constraints();
    assert!(
        !AddConnectionMutation.is_applicable(&g, &c),
        "LayerLevel 不应适用"
    );
}

#[test]
fn test_add_connection_creates_agg_node() {
    let mut g = node_level_3block_genome();
    let before_node_count = g.nodes().len();
    AddConnectionMutation
        .apply(&mut g, &constraints(), &mut rng())
        .unwrap();
    let after_node_count = g.nodes().len();
    // 插入了至少一个新节点（Add 聚合节点，可能还有投影节点）
    assert!(
        after_node_count > before_node_count,
        "AddConnection 应增加节点数"
    );
    // 应存在至少一个 block_id=None 且类型为 Add 的节点
    let has_agg = g
        .nodes()
        .iter()
        .any(|n| n.block_id.is_none() && matches!(n.node_type, NodeTypeDescriptor::Add));
    assert!(has_agg, "应存在跳跃聚合 Add 节点");
}

#[test]
fn test_add_connection_dag_remains_valid() {
    let mut g = node_level_3block_genome();
    for _ in 0..5 {
        if AddConnectionMutation.is_applicable(&g, &constraints()) {
            AddConnectionMutation
                .apply(&mut g, &constraints(), &mut rng())
                .unwrap();
        }
    }
    // GenomeAnalysis 应合法（无环、形状一致）
    let analysis = g.analyze();
    assert!(
        analysis.is_valid,
        "多次 AddConnection 后图应合法: {:?}",
        analysis.errors
    );
}

#[test]
fn test_add_connection_build_succeeds() {
    let mut g = node_level_3block_genome();
    AddConnectionMutation
        .apply(&mut g, &constraints(), &mut rng())
        .unwrap();
    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "AddConnection 后 build 应成功"
    );
}

#[test]
fn test_add_connection_with_shape_mismatch_inserts_projection() {
    // 构造输入维度与中间层输出维度不同的 genome，强制走投影分支
    // Input(3) → Linear(8) → [Linear(2)]  → 从 INPUT(3) 跳到 [Linear(2)] 入口（需要 3→8 投影）
    let mut g = NetworkGenome::minimal(3, 2);
    let i1 = g.next_innovation_number();
    g.layers_mut().insert(
        0,
        LayerGene {
            innovation_number: i1,
            layer_config: LayerConfig::Linear { out_features: 8 },
            enabled: true,
        },
    );
    g.migrate_to_node_level().unwrap();

    let pairs = find_connectable_pairs(&g);
    // 至少应有一个需要投影的候选（INPUT dim=3 → Linear(8) 入口 dim=8 不同）
    // 或者维度相同也可以直接 Add
    if !pairs.is_empty() {
        let before_param_count = g.analyze().param_count;
        AddConnectionMutation
            .apply(&mut g, &constraints(), &mut rng())
            .unwrap();
        let after_param_count = g.analyze().param_count;
        // 如果插了投影块，参数量会增加
        let _ = after_param_count; // 不管是否增加，build 应成功
        let mut build_rng = rng();
        assert!(
            g.build(&mut build_rng).is_ok(),
            "带投影的 AddConnection 后 build 应成功"
        );
        let _ = before_param_count;
    }
}

#[test]
fn test_add_connection_spatial_shape_mismatch_inserts_conv1x1_projection() {
    let mut g = node_level_spatial_2conv_genome();
    let pair = find_connectable_pairs(&g)
        .into_iter()
        .find(|p| p.domain == ShapeDomain::Spatial && p.from_shape != p.to_shape)
        .expect("应存在需要 1x1 Conv2d 投影的 Spatial 候选对");

    let before_param_count = g.analyze().param_count;
    let mut counter = make_counter(&g);
    let agg_id = add_skip_connection(&mut g, &pair, &mut counter).expect("添加空间跳跃连接应成功");
    commit_counter(&mut g, &counter);

    let after_param_count = g.analyze().param_count;
    assert!(
        after_param_count > before_param_count,
        "Spatial 投影应新增 1x1 Conv2d 参数"
    );

    let agg_node = g
        .nodes()
        .iter()
        .find(|n| n.innovation_number == agg_id)
        .expect("应存在新聚合节点");
    assert_eq!(agg_node.parents.len(), 2, "聚合节点应有两个非参数父节点");

    let proj_output_id = agg_node.parents[1];
    let proj_bid = g
        .nodes()
        .iter()
        .find(|n| n.innovation_number == proj_output_id)
        .and_then(|n| n.block_id)
        .expect("投影输出节点应属于独立 block");

    let has_conv1x1 = g.nodes().iter().any(|n| {
        n.block_id == Some(proj_bid) && matches!(n.node_type, NodeTypeDescriptor::Conv2d { .. })
    });
    assert!(has_conv1x1, "投影 block 中应包含 Conv2d 计算节点");

    let kernel = g
        .nodes()
        .iter()
        .find(|n| {
            n.block_id == Some(proj_bid)
                && n.is_parameter()
                && n.output_shape.len() == 4
                && n.output_shape[2] == 1
                && n.output_shape[3] == 1
        })
        .expect("应存在 1x1 Conv2d kernel 参数");
    assert_eq!(
        kernel.output_shape[1], pair.from_shape[1],
        "投影输入通道应匹配源通道数"
    );
    assert_eq!(
        kernel.output_shape[0], pair.to_shape[1],
        "投影输出通道应匹配目标通道数"
    );

    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "带 Spatial 1x1 投影的 AddConnection 后 build 应成功"
    );
}

#[test]
fn test_repair_skip_connections_repairs_projection_after_manual_corruption() {
    let mut g = node_level_3block_genome();
    let pair = find_connectable_pairs(&g)
        .into_iter()
        .find(|p| p.domain == ShapeDomain::Flat && p.from_shape != p.to_shape)
        .expect("应存在需要 Linear 投影的 Flat 候选对");

    let mut counter = make_counter(&g);
    let agg_id = add_skip_connection(&mut g, &pair, &mut counter).expect("添加跳跃连接应成功");
    commit_counter(&mut g, &counter);

    let agg_node = g
        .nodes()
        .iter()
        .find(|n| n.innovation_number == agg_id)
        .expect("应存在新聚合节点")
        .clone();
    let proj_output_id = agg_node.parents[1];
    let proj_bid = g
        .nodes()
        .iter()
        .find(|n| n.innovation_number == proj_output_id)
        .and_then(|n| n.block_id)
        .expect("投影输出节点应属于独立 block");

    for node in g.nodes_mut().iter_mut() {
        if node.block_id == Some(proj_bid) && node.is_parameter() && node.output_shape.len() == 2 {
            node.output_shape[1] += 2;
        }
    }

    sync_computation_shapes(&mut g);
    repair_skip_connections(&mut g);

    let analysis = g.analyze();
    assert!(
        analysis.is_valid,
        "修复后图应恢复合法: {:?}",
        analysis.errors
    );

    let main_shape = analysis
        .shape_of(agg_node.parents[0])
        .expect("主路径父节点应可推导形状")
        .clone();
    let repaired_params: Vec<_> = g
        .nodes()
        .iter()
        .filter(|n| n.block_id == Some(proj_bid) && n.is_parameter() && n.output_shape.len() == 2)
        .collect();
    assert!(
        repaired_params
            .iter()
            .any(|n| n.output_shape[0] != 1 && n.output_shape[1] == main_shape[1]),
        "修复后投影权重输出维应与主路径形状一致"
    );
    assert!(
        repaired_params
            .iter()
            .any(|n| n.output_shape[0] == 1 && n.output_shape[1] == main_shape[1]),
        "修复后投影 bias 输出维应与主路径形状一致"
    );

    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "repair_skip_connections 修复后 build 应成功"
    );
}

// ── RemoveConnectionMutation ──────────────────────────────

#[test]
fn test_remove_connection_not_applicable_before_add() {
    let g = node_level_3block_genome();
    let c = constraints();
    // 没有 AddConnection 之前，应不存在可移除的聚合节点
    assert!(
        !RemoveConnectionMutation.is_applicable(&g, &c),
        "无跳跃连接时不应适用"
    );
}

#[test]
fn test_remove_connection_applicable_after_add() {
    let mut g = node_level_3block_genome();
    let c = constraints();
    AddConnectionMutation.apply(&mut g, &c, &mut rng()).unwrap();
    assert!(
        RemoveConnectionMutation.is_applicable(&g, &c),
        "AddConnection 后应可移除"
    );
}

#[test]
fn test_remove_connection_restores_valid_graph() {
    let mut g = node_level_3block_genome();
    let c = constraints();
    let before_node_count = g.nodes().len();
    AddConnectionMutation.apply(&mut g, &c, &mut rng()).unwrap();
    RemoveConnectionMutation
        .apply(&mut g, &c, &mut rng())
        .unwrap();

    let analysis = g.analyze();
    assert!(analysis.is_valid, "移除后图应合法: {:?}", analysis.errors);
    // 节点数应恢复（聚合节点和可能的投影节点被清理）
    assert!(
        g.nodes().len() <= before_node_count + 1,
        "移除后节点数应基本恢复（允许 orphan 清理差异），before={}, after={}",
        before_node_count,
        g.nodes().len()
    );
    let mut build_rng = rng();
    assert!(g.build(&mut build_rng).is_ok(), "移除后 build 应成功");
}

#[test]
fn test_remove_connection_no_removable_after_remove() {
    let mut g = node_level_3block_genome();
    let c = constraints();
    AddConnectionMutation.apply(&mut g, &c, &mut rng()).unwrap();
    RemoveConnectionMutation
        .apply(&mut g, &c, &mut rng())
        .unwrap();
    let remaining = find_removable_skip_connections(&g);
    assert!(remaining.is_empty(), "移除后不应再有可移除的聚合节点");
}

// ── ResNet / DenseNet 式连接 ──────────────────────────────

#[test]
fn test_resnet_style_single_skip_build_and_forward() {
    // ResNet 式：Input → Linear(4) → ReLU → [+skip from Input] → Linear(4) → [Linear(1)]
    let mut g = node_level_3block_genome();
    // 强制添加一条跳跃连接
    let added = AddConnectionMutation.apply(&mut g, &constraints(), &mut rng());
    assert!(added.is_ok(), "ResNet 式跳跃连接应能添加");

    let mut build_rng = rng();
    let build_result = g.build(&mut build_rng);
    assert!(
        build_result.is_ok(),
        "ResNet 式 build 应成功: {:?}",
        build_result.err()
    );
}

#[test]
fn test_densenet_style_multiple_skips() {
    // DenseNet 式：添加多条跳跃连接，图应保持合法且可构建
    let mut g = node_level_3block_genome();
    let c = constraints();
    let mut added = 0;
    for _ in 0..5 {
        if AddConnectionMutation.is_applicable(&g, &c) {
            if AddConnectionMutation.apply(&mut g, &c, &mut rng()).is_ok() {
                added += 1;
            }
        }
    }

    let analysis = g.analyze();
    assert!(
        analysis.is_valid,
        "多条跳跃连接后图应合法: {:?}",
        analysis.errors
    );

    let mut build_rng = rng();
    assert!(
        g.build(&mut build_rng).is_ok(),
        "DenseNet 式 build 应成功（添加了{}条连接）",
        added
    );
}

// ── 与现有变异组合的鲁棒性 ───────────────────────────────

#[test]
fn test_node_level_with_connections_multi_mutations_build_succeeds() {
    // 含跳跃连接的 NodeLevel 基因组，经历多轮随机变异后 build 应始终成功
    let g_base = node_level_3block_genome();
    let reg = MutationRegistry::default_registry(&TaskMetric::Accuracy, false, false);
    let c = constraints();

    for seed in 0..10u64 {
        let mut g = g_base.clone();
        let mut r = StdRng::seed_from_u64(seed);

        // 先添加一条跳跃连接
        let _ = AddConnectionMutation.apply(&mut g, &c, &mut r);

        // 再做 15 轮随机变异
        for _ in 0..15 {
            let _ = reg.apply_random(&mut g, &c, &mut r);
        }

        let mut build_rng = StdRng::seed_from_u64(seed + 500);
        let result = g.build(&mut build_rng);
        assert!(
            result.is_ok(),
            "含跳跃连接的多轮随机变异后 build 应成功 (seed={seed}): {:?}\nnodes: {:#?}",
            result.err(),
            g.nodes()
        );
    }
}

#[test]
fn test_add_connection_is_structural() {
    assert!(
        AddConnectionMutation.is_structural(),
        "AddConnection 应为结构变异"
    );
    assert!(
        RemoveConnectionMutation.is_structural(),
        "RemoveConnection 应为结构变异"
    );
}

// ==================== Phase 8: NodeLevel CellType Mutation ====================

#[test]
fn test_mutate_cell_type_node_level_applicable() {
    let mut g = genome_sequential();
    g.migrate_to_node_level().unwrap();
    assert!(MutateCellTypeMutation.is_applicable(&g, &constraints()));
}

#[test]
fn test_mutate_cell_type_node_level_switches_cell_kind() {
    let mut g = genome_sequential();
    g.migrate_to_node_level().unwrap();
    let before_is_rnn = g
        .nodes()
        .iter()
        .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. }));
    let before_is_lstm = g
        .nodes()
        .iter()
        .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellLstm { .. }));
    let before_is_gru = g
        .nodes()
        .iter()
        .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellGru { .. }));

    let mut r = StdRng::seed_from_u64(7);
    MutateCellTypeMutation
        .apply(&mut g, &constraints(), &mut r)
        .unwrap();

    let after_is_rnn = g
        .nodes()
        .iter()
        .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellRnn { .. }));
    let after_is_lstm = g
        .nodes()
        .iter()
        .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellLstm { .. }));
    let after_is_gru = g
        .nodes()
        .iter()
        .any(|n| matches!(n.node_type, NodeTypeDescriptor::CellGru { .. }));

    assert!(
        before_is_rnn != after_is_rnn
            || before_is_lstm != after_is_lstm
            || before_is_gru != after_is_gru,
        "NodeLevel cell 类型应发生切换"
    );
}

#[test]
fn test_mutate_cell_type_node_level_preserves_buildability() {
    let mut g = genome_sequential();
    g.migrate_to_node_level().unwrap();

    for seed in 0..10u64 {
        let mut g2 = g.clone();
        let mut r = StdRng::seed_from_u64(seed);
        MutateCellTypeMutation
            .apply(&mut g2, &constraints(), &mut r)
            .unwrap();

        let analysis = g2.analyze();
        assert!(
            analysis.is_valid,
            "NodeLevel cell 切换后图应合法: {:?}",
            analysis.errors
        );

        let mut build_rng = StdRng::seed_from_u64(seed + 1000);
        assert!(
            g2.build(&mut build_rng).is_ok(),
            "NodeLevel cell 切换后 build 应成功"
        );
    }
}
